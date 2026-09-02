"""
Local upload server.

    uvicorn glb2pbr.server:app --reload --port 8765
    open http://127.0.0.1:8765

POST /api/extract   multipart: file=<.glb|.gltf>, size, mesh, sss_gain, sss_tint, height, curvature
GET  /api/jobs/{id}                  status + log + manifest
GET  /api/jobs/{id}/files/{path}     any output file (thumbs, maps, uv, sbs helpers)
GET  /api/jobs/{id}/zip              everything zipped
GET  /api/jobs                       list

Jobs live under GLB2PBR_JOBS (default ./jobs). Nothing leaves the machine.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
import zipfile
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .pipeline import run

JOBS_ROOT = os.path.abspath(os.environ.get("GLB2PBR_JOBS", "jobs"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
os.makedirs(JOBS_ROOT, exist_ok=True)

app = FastAPI(title="glb2pbr", version="0.1.0")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _job_dir(job_id: str) -> str:
    d = os.path.join(JOBS_ROOT, job_id)
    if not os.path.isdir(d):
        raise HTTPException(404, "job not found")
    return d


def _persist(job: dict):
    with open(os.path.join(JOBS_ROOT, job["id"], "job.json"), "w", encoding="utf-8") as fh:
        json.dump(job, fh, indent=2)


def _run_job(job_id: str, input_path: str, out_dir: str, params: dict):
    job = _jobs[job_id]
    job["status"] = "running"
    job["started"] = time.time()

    def log(line: str):
        job["log"].append(line)

    try:
        manifest = run(input_path, out_dir, log=log, **params)
        job["manifest"] = manifest
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(e)
        log(f"ERROR: {e}")
    job["finished"] = time.time()
    _persist(job)


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as fh:
        return fh.read()


@app.post("/api/extract")
async def extract(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    size: Optional[int] = Form(None),
    mesh: str = Form("obj"),
    sss_gain: float = Form(1.0),
    sss_tint: float = Form(0.75),
    height: bool = Form(True),
    curvature: bool = Form(False),
    height_highpass: float = Form(1.0 / 24.0),
):
    name = os.path.basename(file.filename or "model.glb")
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ("glb", "gltf"):
        raise HTTPException(400, "upload a .glb or .gltf (for .gltf with external .bin/images, zip them together and upload the .glb instead)")
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    jdir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(os.path.join(jdir, "input"), exist_ok=True)
    input_path = os.path.join(jdir, "input", name)
    with open(input_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    out_dir = os.path.join(jdir, "out")
    params = dict(size=size or None, mesh=mesh, sss_gain=sss_gain, sss_tint=sss_tint,
                  height=height, curvature=curvature, height_highpass=height_highpass)
    job = {"id": job_id, "file": name, "status": "queued", "log": [], "params": params,
           "created": time.time(), "manifest": None, "error": None}
    with _lock:
        _jobs[job_id] = job
    _persist(job)
    background.add_task(_run_job, job_id, input_path, out_dir, params)
    return {"job": job_id}


@app.get("/api/jobs")
def list_jobs():
    return [{k: v for k, v in j.items() if k != "manifest"} for j in sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        p = os.path.join(_job_dir(job_id), "job.json")
        with open(p, "r", encoding="utf-8") as fh:
            job = json.load(fh)
        _jobs[job_id] = job
    return JSONResponse(job)


@app.get("/api/jobs/{job_id}/files/{path:path}")
def get_file(job_id: str, path: str):
    root = os.path.join(_job_dir(job_id), "out")
    full = os.path.abspath(os.path.join(root, path))
    if not full.startswith(root + os.sep) or not os.path.isfile(full):
        raise HTTPException(404, "file not found")
    return FileResponse(full)


@app.get("/api/jobs/{job_id}/zip")
def get_zip(job_id: str, include_mesh: bool = True):
    jdir = _job_dir(job_id)
    root = os.path.join(jdir, "out")
    job = _jobs.get(job_id) or {}
    stem = os.path.splitext(job.get("file", "model"))[0]
    zip_path = os.path.join(jdir, f"{stem}_pbr{'' if include_mesh else '_nomesh'}.zip")
    if not os.path.exists(zip_path):
        tmp = zip_path + ".part"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    full = os.path.join(dp, fn)
                    rel = os.path.relpath(full, root)
                    if not include_mesh and rel.split(os.sep)[0] == "mesh":
                        continue
                    z.write(full, os.path.join(f"{stem}_pbr", rel))
        os.replace(tmp, zip_path)
    return FileResponse(zip_path, media_type="application/zip", filename=os.path.basename(zip_path))
