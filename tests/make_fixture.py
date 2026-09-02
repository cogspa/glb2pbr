"""Build a small textured UV-sphere GLB (plus a .gltf twin with data URIs) that exercises the extractor."""
from __future__ import annotations

import base64
import io
import json
import os
import struct

import numpy as np
from PIL import Image


def sphere(seg=32, rings=16):
    verts, norms, uvs, idx = [], [], [], []
    for r in range(rings + 1):
        v = r / rings
        phi = v * np.pi
        for s in range(seg + 1):
            u = s / seg
            th = u * 2 * np.pi
            x, y, z = np.sin(phi) * np.cos(th), np.cos(phi), np.sin(phi) * np.sin(th)
            verts.append((x, y, z)); norms.append((x, y, z)); uvs.append((u, v))
    for r in range(rings):
        for s in range(seg):
            a = r * (seg + 1) + s
            b = a + seg + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return (np.asarray(verts, np.float32), np.asarray(norms, np.float32),
            np.asarray(uvs, np.float32), np.asarray(idx, np.uint16))


def png_bytes(arr_u8: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr_u8).save(buf, format="PNG")
    return buf.getvalue()


def textures(size=128):
    yy, xx = np.mgrid[0:size, 0:size]
    checker = (((xx // 16) + (yy // 16)) % 2).astype(np.float32)
    bc = np.stack([0.9 * checker + 0.1, 0.4 * (1 - checker) + 0.2, 0.8 * (xx / size) + 0.1, np.clip(xx / size, 0, 1)], -1)
    mr = np.stack([np.ones_like(checker) * 0.8, 0.3 + 0.5 * checker, 1 - checker], -1)  # R=occ G=rough B=metal
    normal = np.stack([0.5 + 0.2 * np.sin(xx / 6), 0.5 + 0.2 * np.cos(yy / 6), np.ones_like(checker)], -1)
    n = normal * 2 - 1
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normal = n * 0.5 + 0.5
    em = np.stack([checker, np.zeros_like(checker), 1 - checker], -1)
    cc = np.stack([checker, 0.5 * checker, np.zeros_like(checker), np.ones_like(checker)], -1)  # R=clearcoat G=cc rough
    tr = np.stack([xx / size, yy / size, np.zeros_like(checker)], -1)  # R=transmission G=thickness
    to8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)  # noqa: E731
    return {
        "base_color": png_bytes(to8(bc)),
        "metallic_roughness": png_bytes(to8(mr)),
        "normal": png_bytes(to8(normal)),
        "emissive": png_bytes(to8(em)),
        "clearcoat": png_bytes(to8(cc)),
        "transmission": png_bytes(to8(tr)),
    }


def build(out_glb: str, out_gltf: str | None = None, base_color_transform: dict | None = None):
    v, n, uv, idx = sphere()
    tex = textures()
    blobs = [v.tobytes(), n.tobytes(), uv.tobytes(), idx.tobytes()] + list(tex.values())
    names = ["POSITION", "NORMAL", "TEXCOORD_0", "INDICES"] + list(tex.keys())
    bin_ = b""
    views = []
    for b in blobs:
        views.append({"buffer": 0, "byteOffset": len(bin_), "byteLength": len(b)})
        bin_ += b + b"\0" * ((4 - len(b) % 4) % 4)
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3",
         "min": v.min(0).tolist(), "max": v.max(0).tolist()},
        {"bufferView": 1, "componentType": 5126, "count": len(n), "type": "VEC3"},
        {"bufferView": 2, "componentType": 5126, "count": len(uv), "type": "VEC2"},
        {"bufferView": 3, "componentType": 5123, "count": len(idx), "type": "SCALAR"},
    ]
    images = [{"bufferView": 4 + i, "mimeType": "image/png", "name": k} for i, k in enumerate(tex.keys())]
    ti = {k: i for i, k in enumerate(tex.keys())}
    ext_used = ["KHR_materials_emissive_strength", "KHR_materials_clearcoat", "KHR_materials_transmission",
                "KHR_materials_volume", "KHR_materials_ior", "KHR_materials_specular", "KHR_materials_sheen"]
    if base_color_transform:
        ext_used.append("KHR_texture_transform")

    bc_tex = {"index": ti["base_color"]}
    if base_color_transform:
        bc_tex["extensions"] = {"KHR_texture_transform": base_color_transform}

    gltf = {
        "asset": {"version": "2.0", "generator": "glb2pbr-fixture"},
        "extensionsUsed": ext_used,
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "sphere"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}],
        "bufferViews": views, "accessors": accessors, "buffers": [{"byteLength": len(bin_)}],
        "images": images, "samplers": [{}],
        "textures": [{"sampler": 0, "source": i} for i in range(len(images))],
        "materials": [{
            "name": "Fixture Mat",
            "pbrMetallicRoughness": {"baseColorTexture": bc_tex, "baseColorFactor": [1, 0.9, 0.8, 1],
                                     "metallicRoughnessTexture": {"index": ti["metallic_roughness"]},
                                     "metallicFactor": 0.75, "roughnessFactor": 0.5},
            "normalTexture": {"index": ti["normal"], "scale": 1.5},
            "occlusionTexture": {"index": ti["metallic_roughness"], "strength": 0.5},
            "emissiveTexture": {"index": ti["emissive"]}, "emissiveFactor": [1, 1, 1],
            "alphaMode": "BLEND", "doubleSided": True,
            "extensions": {
                "KHR_materials_emissive_strength": {"emissiveStrength": 4.0},
                "KHR_materials_clearcoat": {"clearcoatFactor": 1.0, "clearcoatTexture": {"index": ti["clearcoat"]},
                                            "clearcoatRoughnessFactor": 0.8, "clearcoatRoughnessTexture": {"index": ti["clearcoat"]}},
                "KHR_materials_transmission": {"transmissionFactor": 0.9, "transmissionTexture": {"index": ti["transmission"]}},
                "KHR_materials_volume": {"thicknessFactor": 2.0, "thicknessTexture": {"index": ti["transmission"]},
                                         "attenuationColor": [0.9, 0.5, 0.5], "attenuationDistance": 0.1},
                "KHR_materials_ior": {"ior": 1.33},
                "KHR_materials_specular": {"specularFactor": 0.6, "specularColorFactor": [1, 0.8, 0.6]},
                "KHR_materials_sheen": {"sheenColorFactor": [0.2, 0.3, 0.4], "sheenRoughnessFactor": 0.6},
            },
        }],
    }
    js = json.dumps(gltf).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    with open(out_glb, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A)); fh.write(js)
        fh.write(struct.pack("<II", len(bin_), 0x004E4942)); fh.write(bin_)
    if out_gltf:
        g2 = json.loads(js)
        g2["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(bin_).decode()
        with open(out_gltf, "w") as fh:
            json.dump(g2, fh)
    return out_glb


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    build(os.path.join(here, "fixture_sphere.glb"), os.path.join(here, "fixture_sphere.gltf"))
    print("wrote fixture_sphere.glb / .gltf")
