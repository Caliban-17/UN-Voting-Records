#!/bin/sh
# Container entry point: make sure the voting data is present, keep it in step
# with the newest data release in a separate process, then serve.
set -e
python scripts/sync_data_release.py || echo "start: no data release fetched; continuing with what is on disk"
( while sleep 86400; do python scripts/sync_data_release.py || true; done ) &
# One worker: the frame is held in memory once (about 1 GB). Threads serve
# the parallel requests the atlas makes on load.
exec gunicorn --workers 1 --threads 8 --timeout 180 --bind "0.0.0.0:${PORT:-5001}" web_app:app
