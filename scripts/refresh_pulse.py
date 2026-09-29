#!/usr/bin/env python3
"""Collect sources, calculate research and export a publication unattended."""

import argparse
import json
from pathlib import Path
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pulse import data_dir  # noqa: E402
from src.pulse_worker import refresh_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-research", action="store_true", help="Fail if voting data cannot produce a research edition")
    args = parser.parse_args()
    from src.config import UN_VOTES_CSV_PATH
    from src.data_processing import load_and_preprocess_data
    df = None
    if UN_VOTES_CSV_PATH.exists():
        df, _ = load_and_preprocess_data(str(UN_VOTES_CSV_PATH))
    snapshot = refresh_all(dataframe=df)
    if snapshot is None:
        raise SystemExit("Another refresh is already running")
    from app import create_app
    app = create_app()
    with app.test_client() as client:
        response = client.get("/briefing")
        if response.status_code != 200:
            raise SystemExit("Briefing rendering failed")
        html = response.get_data(as_text=True)
        root = Path(__file__).resolve().parents[1]
        for css in ("style.css", "briefing.css"):
            html = html.replace(f'<link rel="stylesheet" href="/static/css/{css}">', f'<style>{(root / "static" / "css" / css).read_text()}</style>')
        # A downloadable edition is self-contained. Interactive navigation
        # belongs to the running application, not to a downloaded HTML file.
        document = BeautifulSoup(html, "html.parser")
        for node in document.select("nav, form, link[rel=alternate]"):
            node.decompose()
        for anchor in document.select("a[href]"):
            if anchor["href"].startswith("/"):
                anchor.unwrap()
        html = str(document)
        (data_dir() / "briefing.html").write_text(html)
        (data_dir() / "feed.xml").write_bytes(client.get("/briefing/feed.xml").data)
    research_path = data_dir() / "research.json"
    research = json.loads(research_path.read_text()) if research_path.exists() else {}
    print(json.dumps({"sources_ok": sum(s["status"] == "ok" for s in snapshot["sources"]),
                      "sources_total": len(snapshot["sources"]), "articles": len(snapshot["items"]),
                      "research_status": research.get("status", "missing"),
                      "newsletter": str(data_dir() / "newsletter.html"), "output": str(data_dir())}))
    if args.require_research and research.get("status") != "ready":
        raise SystemExit("Research publication requires sufficient voting data")
    if args.require_research and not (data_dir() / "newsletter.json").exists():
        raise SystemExit("Newsletter publication did not produce a canonical edition")
    if not any(s["status"] == "ok" for s in snapshot["sources"]):
        raise SystemExit("All context sources failed; retained previous articles and published source-health status")


if __name__ == "__main__":
    main()
