"""
Adapter for the substance-pipeline tool registry (increment-style stage).

The pipeline's registry expects a tool with a name, a JSON schema for inputs,
and a callable that takes an input dict + a log callback and returns an
outputs dict. Wire it in with something like:

    from glb2pbr.pipeline_stage import TOOL
    registry.register(TOOL)

Stage id:   glb_unwrap
Inputs:     {"glb": "<path>", "out": "<dir>", "size": 2048, "mesh": "obj", "sss_gain": 1.0, ...}
Outputs:    {"manifest": "<out>/manifest.json", "maps": {...}, "uv": {...}, "mesh": {...},
             "contact_sheet": "<out>/thumbs/<mat>_contact_sheet.png"}

Downstream stages that can consume the manifest:
  - substance_build   -> scripts/substance/build_sbs_pysbs.py (SAT) or in-app Designer script
  - blender_material  -> scripts/blender/build_material.py
  - painter_import    -> the ORM / individual maps drop straight into Painter's texture set
"""
from __future__ import annotations

import os

from .pipeline import run

INPUT_SCHEMA = {
    "type": "object",
    "required": ["glb"],
    "properties": {
        "glb": {"type": "string", "description": ".glb or .gltf path"},
        "out": {"type": "string", "description": "output folder (default <glb>_pbr)"},
        "size": {"type": ["integer", "null"], "default": None},
        "mesh": {"type": "string", "enum": ["none", "obj", "copy"], "default": "obj"},
        "uv": {"type": "boolean", "default": True},
        "height": {"type": "boolean", "default": True},
        "curvature": {"type": "boolean", "default": False},
        "subsurface": {"type": "boolean", "default": True},
        "sss_gain": {"type": "number", "default": 1.0},
        "sss_tint": {"type": "number", "default": 0.75},
        "orm": {"type": "boolean", "default": True},
        "height_highpass": {"type": "number", "default": 1.0 / 24.0},
    },
}


def execute(inputs: dict, log=print) -> dict:
    glb = inputs["glb"]
    out = inputs.get("out") or (os.path.splitext(glb)[0] + "_pbr")
    kwargs = {k: inputs[k] for k in ("size", "mesh", "uv", "height", "curvature", "subsurface",
                                     "sss_gain", "sss_tint", "orm", "height_highpass") if k in inputs}
    manifest = run(glb, out, log=log, **kwargs)
    first = manifest["materials"][0] if manifest["materials"] else {}
    return {
        "manifest": os.path.join(out, "manifest.json"),
        "out_dir": out,
        "materials": [m["name"] for m in manifest["materials"]],
        "maps": {m["safe_name"]: {k: os.path.join(out, v["file"]) for k, v in m["maps"].items()} for m in manifest["materials"]},
        "uv": {k: {kk: os.path.join(out, vv) for kk, vv in v["files"].items()} for k, v in manifest["uv_layouts"].items()},
        "mesh": {k: os.path.join(out, v) for k, v in (manifest.get("mesh") or {}).items() if isinstance(v, str)},
        "contact_sheet": os.path.join(out, manifest["thumbnails"]["contact_sheets"].get(first.get("safe_name", ""), ""))
        if manifest.get("thumbnails") else None,
        "timing_s": manifest["timing_s"],
    }


TOOL = {
    "id": "glb_unwrap",
    "label": "Unwrap GLB to PBR maps",
    "description": "Extract Principled BSDF / Substance channel maps, UV layout, and OBJ from a textured glTF/GLB.",
    "input_schema": INPUT_SCHEMA,
    "execute": execute,
    "local_only": True,
    "produces": ["manifest.json", "png maps", "uv layouts", "obj", "contact sheet"],
}
