"""
One call does everything: glb/gltf in -> folder of PBR maps + manifest.json out.

    from glb2pbr.pipeline import run
    manifest = run("model.glb", "out/", size=2048, mesh="obj")

Output folder layout:
    <out>/
      manifest.json
      <Material>_BaseColor.png, _Metallic.png, _Roughness.png, _Normal_GL.png,
      _Normal_DX.png, _AO.png, _Height.png, _Subsurface.png, _SubsurfaceRadius.png, _ORM.png ...
      uv/<Material>_UV_islands.png, _UV_wire.png, _UV_overlay.png
      mesh/<model>.obj                        (optional)
      blender_build_material.py               (copied helper)
      substance_build_sbs_pysbs.py            (copied helper, SAT/pysbs)
      substance_build_sbs_designer.py         (copied helper, run inside Designer)
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict

from . import __version__
from .extract import ExtractOptions, MAP_SPECS, extract_material, write_maps
from .gltf_reader import GLTFDoc
from .uvlayout import island_mask, render_uv_layout, write_obj

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
HELPERS = [
    ("blender/build_material.py", "blender_build_material.py"),
    ("substance/build_sbs_pysbs.py", "substance_build_sbs_pysbs.py"),
    ("substance/build_sbs_designer.py", "substance_build_sbs_designer.py"),
]


def write_thumbnails(out_dir: str, manifest: dict, size: int = 256) -> dict:
    """256px thumbs for every map + UV layout, and one contact sheet per material."""
    from PIL import Image, ImageDraw
    import numpy as np
    thumb_dir = os.path.join(out_dir, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    result = {"per_file": {}, "contact_sheets": {}}

    def load_small(path):
        im = Image.open(path)
        if im.mode == "I;16":
            a = np.asarray(im).astype(np.float32) / 65535.0
            im = Image.fromarray((a * 255).astype(np.uint8), "L")
        return im.convert("RGB").resize((size, size), Image.BOX)

    for mat in manifest["materials"]:
        items = [(mid, rec["file"]) for mid, rec in mat["maps"].items()]
        items += [(f"uv_{k}", v) for k, v in mat.get("uv", {}).get("files", {}).items()]
        tiles = []
        for mid, rel in items:
            src = os.path.join(out_dir, rel)
            if not os.path.exists(src):
                continue
            im = load_small(src)
            tn = os.path.join("thumbs", os.path.basename(rel))
            im.save(os.path.join(out_dir, tn), compress_level=4)
            result["per_file"][rel] = tn
            tiles.append((mid, im))
        if tiles:
            cols = min(6, len(tiles))
            rows = (len(tiles) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * size, rows * (size + 18)), (16, 16, 16))
            d = ImageDraw.Draw(sheet)
            for i, (mid, im) in enumerate(tiles):
                x, yy = (i % cols) * size, (i // cols) * (size + 18)
                sheet.paste(im, (x, yy + 18))
                d.text((x + 4, yy + 3), mid, fill=(230, 230, 230))
            cs = os.path.join("thumbs", f"{mat['safe_name']}_contact_sheet.png")
            sheet.save(os.path.join(out_dir, cs), compress_level=4)
            result["contact_sheets"][mat["safe_name"]] = cs
    return result


def run(input_path: str, out_dir: str, size: int | None = None, mesh: str = "none",
        uv: bool = True, uv_size: int | None = None, fill_missing: bool = True,
        height: bool = True, subsurface: bool = True, curvature: bool = False,
        sss_gain: float = 1.0, sss_tint: float = 0.75, orm: bool = True,
        height_highpass: float = 1.0 / 24.0, copy_helpers: bool = True, log=print) -> dict:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    doc = GLTFDoc(input_path)
    log(f"[glb2pbr] {os.path.basename(input_path)}  generator={doc.asset.get('generator', '?')}  "
        f"materials={len(doc.materials)}  images={len(doc.json.get('images', []))}  "
        f"extensions={doc.extensions_used or '-'}")

    opts = ExtractOptions(size=size, fill_missing=fill_missing, derive_height=height,
                          derive_subsurface=subsurface, derive_curvature=curvature,
                          sss_gain=sss_gain, sss_tint=sss_tint, write_orm=orm,
                          height_highpass=height_highpass)

    manifest = {
        "glb2pbr_version": __version__,
        "source": {
            "file": os.path.basename(input_path),
            "generator": doc.asset.get("generator"),
            "gltf_version": doc.asset.get("version"),
            "extensions_used": doc.extensions_used,
            "images": [{"index": i, "name": doc.image_name(i), "mime": im.get("mimeType")} for i, im in
                       enumerate(doc.json.get("images", []))],
        },
        "conventions": {
            "normal": "normal_gl is as stored in glTF (OpenGL, +Y up) - use in Blender. normal_dx has green inverted - use if Substance is set to DirectX.",
            "colorspace": "basecolor/emissive/specular_color/sheen_color are sRGB; everything else is linear (Non-Color in Blender).",
            "uv": "glTF V goes down (matches image rows). OBJ export flips V so Substance/Blender read it upright.",
            "orm": "R=occlusion G=roughness B=metallic",
            "height": "16-bit, 0.5 = mean plane, integrated from the normal map (no true displacement data in glTF).",
            "subsurface": "Derived heuristic: (1-metallic)*(1-transmission)*sss_gain. glTF has no SSS channel.",
        },
        "map_specs": {k: {"suffix": v[0], "colorspace": v[1], "channels": v[2], "blender_socket": v[3],
                          "substance_usage": v[4]} for k, v in MAP_SPECS.items()},
        "materials": [],
        "uv_layouts": {},
        "mesh": None,
        "helpers": [],
        "timing_s": 0.0,
    }

    prims = doc.primitives(with_geometry=True) if (uv or mesh != "none") else []

    if not doc.materials:
        log("[glb2pbr] WARNING: no materials in file - nothing to extract")

    for mi in range(len(doc.materials)):
        mat_prims = [p for p in prims if p.material == mi]
        # Pass 1: extract without height so we know the map size, then rasterise UV islands
        # at that size and use them to mask the normal->height integration.
        opts.derive_height = False
        opts.uv_mask = None
        res = extract_material(doc, mi, opts)
        pre = None
        if uv and mat_prims:
            pre = island_mask(mat_prims, size=uv_size or res.size)
            if pre[0] is not None and height and (uv_size in (None, res.size)):
                opts.uv_mask = pre[0]
        if height:
            opts.derive_height = True
            res = extract_material(doc, mi, opts) if "normal_gl" in res.maps else res
        records = write_maps(res, out_dir)
        log(f"[glb2pbr]   material {mi} '{res.name}' -> {len(records)} maps @ {res.size}px")
        for w in res.warnings:
            log(f"[glb2pbr]   WARNING: {w}")
        entry = {
            "index": mi,
            "name": res.name,
            "safe_name": res.safe_name,
            "size": res.size,
            "factors": res.factors,
            "flags": res.flags,
            "maps": records,
            "warnings": res.warnings,
            "blender_principled": {rec["blender_socket"]: rec["file"] for rec in records.values() if rec["blender_socket"]},
            "substance_outputs": [
                {"identifier": rec["substance_usage"], "usage": rec["substance_usage"],
                 "components": rec["substance_components"], "file": rec["file"],
                 "color": rec["channels"] in ("RGB", "RGBA"), "colorspace": rec["colorspace"]}
                for rec in records.values() if rec["substance_usage"]
            ],
        }
        if uv and pre is not None and pre[0] is not None:
            layouts, stats = render_uv_layout(mat_prims, size=uv_size or res.size,
                                              basecolor=res.maps.get("basecolor"), precomputed=pre)
            if layouts:
                uv_dir = os.path.join(out_dir, "uv")
                os.makedirs(uv_dir, exist_ok=True)
                files = {}
                for key, im in layouts.items():
                    fn = f"{res.safe_name}_UV_{key}.png"
                    im.save(os.path.join(uv_dir, fn), compress_level=4)
                    files[key] = f"uv/{fn}"
                entry["uv"] = {"files": files, **asdict(stats)}
                manifest["uv_layouts"][res.safe_name] = entry["uv"]
                log(f"[glb2pbr]   UV layout: {stats.triangles:,} tris, coverage {stats.coverage:.1%}")
        manifest["materials"].append(entry)

    thumbs = write_thumbnails(out_dir, manifest)
    manifest["thumbnails"] = thumbs

    if mesh == "obj" and prims:
        mesh_dir = os.path.join(out_dir, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)
        obj_name = os.path.splitext(os.path.basename(input_path))[0] + ".obj"
        st = write_obj(doc, prims, os.path.join(mesh_dir, obj_name))
        manifest["mesh"] = {"obj": f"mesh/{obj_name}", **st}
        log(f"[glb2pbr]   OBJ: {st['vertices']:,} verts / {st['triangles']:,} tris")
    elif mesh == "copy":
        mesh_dir = os.path.join(out_dir, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)
        dst = os.path.join(mesh_dir, os.path.basename(input_path))
        shutil.copy2(input_path, dst)
        manifest["mesh"] = {"glb": f"mesh/{os.path.basename(input_path)}"}

    if copy_helpers:
        for src, dst in HELPERS:
            sp = os.path.join(SCRIPTS_DIR, src)
            if os.path.exists(sp):
                shutil.copy2(sp, os.path.join(out_dir, dst))
                manifest["helpers"].append(dst)

    manifest["timing_s"] = round(time.time() - t0, 2)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"[glb2pbr] done in {manifest['timing_s']}s -> {out_dir}")
    return manifest
