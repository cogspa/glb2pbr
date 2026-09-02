@echo off
cd /d %~dp0
if not exist .venv ( python -m venv .venv && .venv\Scripts\pip install -r requirements.txt )
.venv\Scripts\uvicorn glb2pbr.server:app --host 127.0.0.1 --port 8765
