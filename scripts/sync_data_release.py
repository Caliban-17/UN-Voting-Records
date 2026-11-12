#!/usr/bin/env python3
"""Bring the voting CSV in step with the newest ``data-*`` GitHub release.

Runs outside the web process (container start, then daily). The release is cut
by refresh-data.yml after validation, so this only downloads and swaps: the web
worker notices the new file and reloads it.
"""

import json
import logging
import os
from pathlib import Path
import sys
import tempfile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import UN_VOTES_CSV_PATH  # noqa: E402

REPO = os.getenv("DATA_RELEASE_REPO", "Caliban-17/UN-Voting-Records")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sync_data_release")


def newest_release(session) -> dict | None:
    releases, page = [], 1
    while True:
        batch = session.get(f"https://api.github.com/repos/{REPO}/releases",
                            params={"per_page": 100, "page": page}, timeout=30)
        batch.raise_for_status()
        items = batch.json()
        releases += [r for r in items if r["tag_name"].startswith("data-")]
        if len(items) < 100:
            break
        page += 1
    # published_at, never created_at: see CLAUDE.md.
    return max(releases, key=lambda r: r["published_at"], default=None)


def main() -> int:
    target = Path(UN_VOTES_CSV_PATH)
    marker = target.with_suffix(".release.json")
    session = requests.Session()
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    release = newest_release(session)
    if release is None:
        logger.warning("No data-* release found in %s", REPO)
        return 1
    try:
        current = json.loads(marker.read_text()).get("tag")
    except (OSError, ValueError):
        current = None
    if current == release["tag_name"] and target.exists():
        logger.info("Already on %s", current)
        return 0
    assets = [a for a in release["assets"] if a["name"].endswith(".csv")]
    if len(assets) != 1:
        logger.error("Expected exactly one CSV in %s, found %d", release["tag_name"], len(assets))
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=target.parent, suffix=".download")
    try:
        with os.fdopen(fd, "wb") as handle, session.get(assets[0]["browser_download_url"], stream=True, timeout=120) as response:
            response.raise_for_status()
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
        if os.path.getsize(temp) != assets[0]["size"]:
            raise ValueError("Download is incomplete")
        if target.exists():
            target.chmod(0o644)
        os.replace(temp, target)
        target.with_suffix(".parquet").unlink(missing_ok=True)
        marker.write_text(json.dumps({"tag": release["tag_name"], "published_at": release["published_at"]}))
        logger.info("Voting data is now %s (%d MB)", release["tag_name"], assets[0]["size"] // 1_000_000)
        return 0
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


if __name__ == "__main__":
    raise SystemExit(main())
