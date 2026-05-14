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
