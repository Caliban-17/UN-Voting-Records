import ast
from pathlib import Path

WEB_APP_PATH = Path("web_app.py")
MAIN_PATH = Path("src/main.py")
MODEL_PATH = Path("src/model.py")
DATA_PROCESSING_PATH = Path("src/data_processing.py")


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function_decorators(path: Path) -> dict[str, list[str]]:
    tree = _parse_module(path)
    decorators: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        names: list[str] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                names.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                base = (
                    decorator.value.id if isinstance(decorator.value, ast.Name) else ""
                )
                names.append(f"{base}.{decorator.attr}".strip("."))
            elif isinstance(decorator, ast.Call):
                fn = decorator.func
                if isinstance(fn, ast.Name):
                    names.append(fn.id)
                elif isinstance(fn, ast.Attribute):
                    base = fn.value.id if isinstance(fn.value, ast.Name) else ""
                    names.append(f"{base}.{fn.attr}".strip("."))

        decorators[node.name] = names

    return decorators


def _imports_streamlit(path: Path) -> bool:
    tree = _parse_module(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "streamlit" for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            if node.module == "streamlit":
                return True
    return False


def test_runtime_modules_do_not_import_streamlit():
    runtime_files = [WEB_APP_PATH, MAIN_PATH, MODEL_PATH, DATA_PROCESSING_PATH]
    offenders = [str(path) for path in runtime_files if _imports_streamlit(path)]
    assert not offenders, f"Streamlit imports remain in runtime modules: {offenders}"


def test_core_api_endpoints_use_cached_api():
    # Identify the files where these functions now reside
    route_files = [
        Path("app/routes/core.py"),
        Path("app/routes/analysis.py"),
        Path("app/routes/visualization.py"),
    ]

    decorators = {}
    for path in route_files:
        if path.exists():
            decorators.update(_function_decorators(path))

    cached_endpoints = [
        "get_data_summary",
        "run_clustering",
        "calculate_soft_power",
        "get_network_graph",
        "get_pca_plot",
        "divergence_report",
    ]

    for fn in cached_endpoints:
        assert fn in decorators, f"{fn} is missing from route files"
        assert (
            "cached_api" in decorators[fn]
        ), f"{fn} should be protected by @cached_api to prevent repeated heavy work"


def test_core_modules_do_not_use_streamlit_cache_decorators():
    files = [MAIN_PATH, MODEL_PATH, DATA_PROCESSING_PATH]
    for path in files:
        decorators = _function_decorators(path)
        for fn_name, fn_decorators in decorators.items():
            assert not any(
                d.startswith("st.cache") for d in fn_decorators
            ), f"{path}:{fn_name} still uses Streamlit cache decorator {fn_decorators}"


def test_reloading_data_clears_request_caches(monkeypatch):
    import pandas as pd

    import src.data_processing
    from app import services
    from src.cache_utils import api_cache

    monkeypatch.setattr(services, "df_global", services.df_global)
    frame = pd.DataFrame({"year": [2025], "date": [pd.Timestamp("2025-12-01")]})
    monkeypatch.setattr(src.data_processing, "load_and_preprocess_data", lambda path: (frame, None))
    for cache in (api_cache, services.soft_power_trends_registry, services.network_animation_registry):
        cache.put("stale", {"from": "the previous frame"})

    services.load_data()

    assert services.get_df() is frame
    for cache in (api_cache, services.soft_power_trends_registry, services.network_animation_registry):
        assert cache.get("stale") is None


def test_cached_responses_do_not_replay_the_first_callers_headers():
    """flask-cors writes Access-Control-Allow-Origin onto the Response; caching
    that object pinned every later caller to the first caller's origin."""
    from flask import Flask, jsonify, request
    from flask_cors import CORS

    from src.cache_utils import api_cache, cached_api

    app = Flask(__name__)
    CORS(app)
    calls = []

    @app.get("/api/thing")
    @cached_api
    def thing():
        calls.append(request.headers.get("Origin"))
        return jsonify({"n": len(calls)})

    api_cache.clear()
    with app.test_client() as client:
        first = client.get("/api/thing", headers={"Origin": "https://first.example"})
        second = client.get("/api/thing", headers={"Origin": "https://second.example"})
        bare = client.get("/api/thing")
    api_cache.clear()
    assert len(calls) == 1 and first.json == second.json == bare.json == {"n": 1}
    assert second.mimetype == "application/json"
    assert second.headers.get("Access-Control-Allow-Origin") != "https://first.example"
    assert bare.headers.get("Access-Control-Allow-Origin") != "https://first.example"


def test_a_json_body_that_is_not_an_object_is_a_client_error():
    import pytest

    from app.services import validate_train_test_years, validate_year_range

    for body in ([1], "x", 5):
        with pytest.raises(ValueError):
            validate_year_range(body)
        with pytest.raises(ValueError):
            validate_train_test_years(body)
    with pytest.raises(ValueError):
        validate_year_range({"start_year": float("inf")})


# ── API hardening ────────────────────────────────────────────────────────────


def _limited_app():
    from flask import Flask, jsonify

    from app import middleware

    app = Flask(__name__)
    middleware.register_middleware(app)

    @app.get("/api/thing/<int:n>")
    def thing(n):
        return jsonify({"n": n})

    return app, middleware


def test_a_forged_forwarded_address_does_not_escape_the_rate_limit(monkeypatch):
    app, middleware = _limited_app()
    monkeypatch.setattr(middleware, "MAX_API_REQUESTS_PER_MIN", 5)
    middleware._rate_buckets.clear()
    with app.test_client() as client:
        codes = [client.get("/api/thing/1", headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
                 for i in range(8)]
    assert codes == [200] * 5 + [429] * 3
    middleware._rate_buckets.clear()


def test_forwarded_address_is_used_only_behind_a_declared_proxy(monkeypatch):
    app, middleware = _limited_app()
    monkeypatch.setattr(middleware, "MAX_API_REQUESTS_PER_MIN", 2)
    monkeypatch.setattr(middleware, "TRUSTED_PROXY_COUNT", 1)
    middleware._rate_buckets.clear()
    with app.test_client() as client:
        # The proxy appends the address it saw; anything to its left is the client's own claim.
        def hit(real):
            return client.get("/api/thing/1", headers={"X-Forwarded-For": f"6.6.6.6, {real}"}).status_code
        assert [hit("1.1.1.1"), hit("1.1.1.1"), hit("1.1.1.1"), hit("2.2.2.2")] == [200, 200, 429, 200]
    middleware._rate_buckets.clear()


def test_rate_limit_buckets_follow_the_route_not_the_raw_path(monkeypatch):
    app, middleware = _limited_app()
    monkeypatch.setattr(middleware, "MAX_API_REQUESTS_PER_MIN", 3)
    middleware._rate_buckets.clear()
    with app.test_client() as client:
        codes = [client.get(f"/api/thing/{i}").status_code for i in range(5)]
        for i in range(5):
            client.get(f"/api/unknown/{i}")
    assert codes == [200, 200, 200, 429, 429]
    assert sorted(k.split(":", 1)[1] for k in middleware._rate_buckets) == [
        "GET:/api/thing/<int:n>", "GET:unmatched"]
    middleware._rate_buckets.clear()


def test_library_errors_are_not_shown_to_clients():
    from flask import Flask
    from werkzeug.exceptions import RequestEntityTooLarge

    from app.routes.core import make_server_error, reject_value_error

    def caught(fn):
        try:
            fn()
        except Exception as exc:   # noqa: BLE001 - the handlers under test take any exception
            return exc

    with Flask(__name__).test_request_context():
        ours = caught(lambda: (_ for _ in ()).throw(ValueError("start_year cannot be greater than end_year")))
        body, status = reject_value_error(ours)
        assert status == 400 and body.json == {"error": "start_year cannot be greater than end_year"}

        import pandas as pd
        theirs = caught(lambda: pd.DataFrame({"a": [1, 2]}).assign(b=[1, 2, 3]))
        assert isinstance(theirs, ValueError)
        body, status = reject_value_error(theirs)
        assert status == 500 and body.json == {"error": "Internal server error"}

        body, status = make_server_error("test", RequestEntityTooLarge())
        assert status == 413 and "Too Large" in body.json["error"]
        # A wrong-typed field read by route code is the caller's mistake...
        import app.services as services
        malformed = caught(lambda: services._require_object.__globals__["int_field"]([1], "x", 1))
        body, status = make_server_error("test", malformed)
        assert status == 400 and "wrong type" in body.json["error"]
        # ...but the same exception type from deep inside a library stays a 500.
        deep = caught(lambda: pd.Timestamp("2020-01-01") - object())
        assert isinstance(deep, TypeError)
        assert make_server_error("test", deep)[1] == 500


def test_number_fields_reject_null_lists_fractions_and_non_finite_values():
    import pytest

    from app.services import float_field, int_field

    assert int_field({}, "window", 5) == 5 and int_field({"window": "7"}, "window", 5) == 7
    assert float_field({"m": "0.25"}, "m", 0.15, lo=0.0, hi=1.0) == 0.25
    for bad in (None, [1], {"a": 1}, True, "abc", 2.5, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="window must be a whole number"):
            int_field({"window": bad}, "window", 5)
    with pytest.raises(ValueError, match="between 1 and 10"):
        int_field({"window": 0}, "window", 5, lo=1, hi=10)
    for bad in (None, [1], float("nan"), float("inf"), -1, 2):
        with pytest.raises(ValueError):
            float_field({"m": bad}, "m", 0.15, lo=0.0, hi=1.0)


def test_country_code_set_is_cached_per_frame(monkeypatch):
    import pandas as pd

    from app import services

    first = pd.DataFrame({"country_identifier": ["USA", "usa ", "FRA"]})
    monkeypatch.setattr(services, "df_global", first)
    codes = services.country_codes()
    assert codes == {"USA", "FRA"} and services.country_codes() is codes
    monkeypatch.setattr(services, "df_global", pd.DataFrame({"country_identifier": ["CUB"]}))
    assert services.country_codes() == {"CUB"}
