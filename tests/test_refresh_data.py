"""The complete-roll-call refresh runs outside the web process."""

from pathlib import Path

import pandas as pd
import yaml

from scripts import refresh_data


def test_prune_keeps_the_newest_archives_and_the_source(tmp_path):
    source = tmp_path / "votes.csv"
    source.write_text("current")
    stamps = ["20260514T144355Z", "20260908T171427Z", "20260930T040000Z"]
    for stamp in stamps:
        (tmp_path / f"votes.archived-{stamp}.csv").write_text(stamp)
    other = tmp_path / "votes.pre-refresh-20260514T142007Z.csv"
    other.write_text("manual backup")

    removed = refresh_data._prune_archives(source, keep=2)

    assert [p.name for p in removed] == [f"votes.archived-{stamps[0]}.csv"]
    left = sorted(p.name for p in tmp_path.glob("votes.archived-*.csv"))
    assert left == [f"votes.archived-{s}.csv" for s in stamps[1:]]
    assert source.exists() and other.exists()


def test_keep_archives_must_leave_a_rollback_copy(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["refresh_data.py", "--promote", "--keep-archives", "0"])
    try:
        refresh_data.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--keep-archives 0 was accepted")
    assert "at least 1" in capsys.readouterr().err


def test_quarantined_roll_calls_are_reported_and_the_rest_still_merge(tmp_path, monkeypatch):
    existing = pd.DataFrame([{"undl_id": 1, "ms_code": "USA", "ms_name": "UNITED STATES",
                              "ms_vote": "Y", "date": "2026-07-01", "resolution": "A/RES/80/1"}])
    fresh = pd.DataFrame([{"undl_id": 2, "ms_code": "USA", "ms_name": "UNITED STATES",
                           "ms_vote": "N", "date": "2026-09-17", "resolution": "A/RES/81/1"}])
    seen = {}

    def fetch(since_date, existing_df, quarantined):
        quarantined.append({"symbol": "A/RES/81/2", "date": "2026-09-18",
                            "reason": "58 A names against a published tally of 59"})
        seen["quarantined"] = quarantined
        return fresh

    monkeypatch.setattr(refresh_data, "UN_VOTES_CSV_PATH", tmp_path / "votes.csv")
    monkeypatch.setattr(refresh_data, "_load_existing_csv", lambda path: existing)
    monkeypatch.setattr(refresh_data, "fetch_recent_votes_github", fetch)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))

    delta = refresh_data._run(90, dry_run=True, output=None)

    assert delta == 1
    assert printed == ["::warning title=Quarantined roll call::A/RES/81/2: "
                       "58 A names against a published tally of 59"]


def test_the_web_process_never_promotes_the_voting_csv():
    for path in ("web_app.py", "src/pulse_worker.py", "app/services.py"):
        assert "--promote" not in Path(path).read_text(), path


def test_compose_runs_the_refresh_as_its_own_service():
    services = yaml.safe_load(Path("docker-compose.yml").read_text())["services"]
    app = services["un-voting-app"]
    refresher = services["un-voting-refresher"]
    assert "command" not in app
    script = " ".join(refresher["command"])
    assert "scripts/refresh_data.py" in script and "--promote" in script
    assert "--keep-archives" in script
    assert "./data:/app/data" in refresher["volumes"]
    # The image's HEALTHCHECK probes the web port, which the refresher lacks.
    assert refresher["healthcheck"] == {"disable": True}
