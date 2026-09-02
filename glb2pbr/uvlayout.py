"""
"Unwrap" view: rasterise the mesh's UV layout so you can see how the sphere
was flattened, plus an OBJ export so Substance Designer's 3D view can show
the real mesh with the real UVs (a primitive sphere won't line up with an
auto-unwrapped Meshy mesh).

glTF UV origin is top-left with V going down, which matches image space
directly - no flip needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .gltf_reader import GLTFDoc, Primitive


@dataclass
class UVStats:
    triangles: int
    vertices: int
    coverage: float           # fraction of texel area inside islands
    islands_bbox: list        # [umin, vmin, umax, vmax]
    out_of_range: bool        # any UV outside [0,1] (tiling / atlas)
    texcoord_set: int


def _uv_to_px(uv: np.ndarray, size: int) -> np.ndarray:
    return np.rint(uv * (size - 1)).astype(np.int32)


def _tri_list(prims: list[Primitive], texcoord: int, size: int) -> tuple[list[np.ndarray], UVStats | None]:
    tris = []
    n_tri = n_vert = 0
    uv_min = np.array([np.inf, np.inf])
    uv_max = np.array([-np.inf, -np.inf])
    oor = False
    for p in prims:
        if p.indices is None or texcoord not in p.uvs:
            continue
        uv = p.uvs[texcoord]
        n_vert += len(uv)
        uv_min = np.minimum(uv_min, uv.min(0))
        uv_max = np.maximum(uv_max, uv.max(0))
        if uv.min() < -1e-4 or uv.max() > 1 + 1e-4:
            oor = True
        px = _uv_to_px(np.mod(uv, 1.0) if oor else uv, size)
        t = px[p.indices]  # (T,3,2)
        n_tri += len(t)
        tris.extend(list(t))
    if n_tri == 0:
        return [], None
    return tris, UVStats(n_tri, n_vert, 0.0, [*uv_min.tolist(), *uv_max.tolist()], oor, texcoord)


def island_mask(prims: list[Primitive], size: int = 2048, texcoord: int = 0):
    """Rasterise UV islands. Returns (mask uint8 HxW, triangle list, stats) or (None, [], None)."""
    tris, stats = _tri_list(prims, texcoord, size)
    if not tris:
        return None, [], None
    islands = np.zeros((size, size), np.uint8)
    cv2.fillPoly(islands, tris, 255)
    stats.coverage = float((islands > 0).mean())
    return islands, tris, stats


def render_uv_layout(prims: list[Primitive], size: int = 2048, texcoord: int = 0,
                     basecolor: Image.Image | np.ndarray | None = None,
                     precomputed=None) -> tuple[dict[str, Image.Image], UVStats | None]:
    """Returns {'islands': L, 'wire': L, 'overlay': RGB (if basecolor)} and stats."""
    islands, tris, stats = precomputed if precomputed is not None else island_mask(prims, size, texcoord)
    if islands is None:
        return {}, None

    # Draw wire at 2x then downsample so very dense meshes still read as lines.
    ss = 2 if size <= 2048 else 1
    wire_hi = np.zeros((size * ss, size * ss), np.uint8)
    tris_hi = [t * ss for t in tris] if ss > 1 else tris
    cv2.polylines(wire_hi, tris_hi, True, 255, 1, cv2.LINE_AA)
    wire = cv2.resize(wire_hi, (size, size), interpolation=cv2.INTER_AREA) if ss > 1 else wire_hi

    out = {"islands": Image.fromarray(islands, "L"), "wire": Image.fromarray(wire, "L")}

    if basecolor is not None:
        if isinstance(basecolor, np.ndarray):
            bc = np.asarray(Image.fromarray((np.clip(basecolor[..., :3], 0, 1) * 255).astype(np.uint8), "RGB")
                            .resize((size, size), Image.LANCZOS), dtype=np.float32)
        else:
            bc = np.asarray(basecolor.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32)
        # dim texels outside islands, tint the wire cyan so it reads on pink/metal.
        # Dense meshes (millions of tris) would flood the overlay, so scale the wire alpha by its density.
        density = float((wire > 0).mean())
        alpha = 0.55 * min(1.0, 0.12 / max(density, 1e-6))
        mask = (islands > 0).astype(np.float32)[..., None]
        base = bc * (0.35 + 0.65 * mask)
        w = (wire.astype(np.float32) / 255.0)[..., None] * alpha
        tint = np.array([0.0, 220.0, 255.0], np.float32)
        overlay = base * (1 - w) + tint * w
        out["overlay"] = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), "RGB")
    return out, stats


def write_obj(doc: GLTFDoc, prims: list[Primitive], path: str, texcoord: int = 0, flip_v: bool = True) -> dict:
    """
    Write all triangle primitives to one OBJ, grouped per primitive/material.
    OBJ/Substance expect V up, so V is flipped from glTF's V-down by default.
    glTF meshes are indexed per-vertex, so v/vt/vn share one index.
    """
    base = 1
    lines_out = []
    stats = {"vertices": 0, "triangles": 0, "groups": 0}
    mat_names = [m.get("name") or f"material_{i}" for i, m in enumerate(doc.materials)]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# glb2pbr OBJ export\n")
        for p in prims:
            if p.positions is None or p.indices is None:
                continue
            n = len(p.positions)
            mname = mat_names[p.material] if p.material is not None and p.material < len(mat_names) else "default"
            fh.write(f"g mesh{p.mesh_index}_prim{p.prim_index}\nusemtl {mname}\n")
            np.savetxt(fh, p.positions, fmt="v %.6f %.6f %.6f")
            if texcoord in p.uvs:
                uv = p.uvs[texcoord].copy()
                if flip_v:
                    uv[:, 1] = 1.0 - uv[:, 1]
                np.savetxt(fh, uv, fmt="vt %.6f %.6f")
            else:
                np.savetxt(fh, np.zeros((n, 2), np.float32), fmt="vt %.6f %.6f")
            if p.normals is not None:
                np.savetxt(fh, p.normals, fmt="vn %.6f %.6f %.6f")
            else:
                np.savetxt(fh, np.tile(np.array([[0, 0, 1]], np.float32), (n, 1)), fmt="vn %.6f %.6f %.6f")
            f = p.indices.astype(np.int64) + base
            # faces: write as v/vt/vn triplets with shared index
            faces = np.repeat(f, 3, axis=1)  # (T,9): a a a b b b c c c
            np.savetxt(fh, faces, fmt="f %d/%d/%d %d/%d/%d %d/%d/%d")
            base += n
            stats["vertices"] += n
            stats["triangles"] += len(p.indices)
            stats["groups"] += 1
    return stats
