#!/usr/bin/env bash
# Local upload UI: http://127.0.0.1:8765
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt; }
. .venv/bin/activate
exec uvicorn glb2pbr.server:app --host 127.0.0.1 --port "${PORT:-8765}"
