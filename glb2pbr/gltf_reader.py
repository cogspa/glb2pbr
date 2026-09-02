"""
Minimal, dependency-light glTF 2.0 / GLB reader.

Handles:
  * .glb (binary container, JSON + BIN chunks)
  * .gltf with external .bin / image files or data: URIs
  * accessors (byteStride, normalized ints), images (bufferView or uri)

Only what the PBR extractor and UV-layout renderer need. No scene-graph
transforms are applied (UVs are transform-independent).
"""
from __future__ import annotations

import base64
import io
import json
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

GLB_MAGIC = 0x46546C67  # 'glTF'
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _read_uri(uri: str, base_dir: str) -> bytes:
    if uri.startswith("data:"):
        _, b64 = uri.split(",", 1)
        return base64.b64decode(b64)
    path = os.path.join(base_dir, uri)
    with open(path, "rb") as fh:
        return fh.read()


@dataclass
class Primitive:
    mesh_index: int
    prim_index: int
    material: Optional[int]
    positions: Optional[np.ndarray]
    normals: Optional[np.ndarray]
    uvs: dict = field(default_factory=dict)  # texcoord set index -> (N,2) float32
    indices: Optional[np.ndarray] = None    # (T,3) uint32, triangles only
    mode: int = 4


class GLTFDoc:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.base_dir = os.path.dirname(self.path)
        self.json: dict = {}
        self.buffers: list[bytes] = []
        self._glb_bin: Optional[bytes] = None
        self._image_cache: dict[int, Image.Image] = {}
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self):
        with open(self.path, "rb") as fh:
            head = fh.read(12)
            if len(head) == 12 and struct.unpack("<I", head[:4])[0] == GLB_MAGIC:
                fh.seek(0)
                self._load_glb(fh.read())
            else:
                fh.seek(0)
                self.json = json.loads(fh.read().decode("utf-8"))
        # resolve buffers
        for i, buf in enumerate(self.json.get("buffers", [])):
            uri = buf.get("uri")
            if uri is None:
                if self._glb_bin is None:
                    raise ValueError(f"buffer {i} has no uri and no GLB BIN chunk")
                self.buffers.append(self._glb_bin)
            else:
                self.buffers.append(_read_uri(uri, self.base_dir))

    def _load_glb(self, data: bytes):
        magic, version, length = struct.unpack_from("<III", data, 0)
        if magic != GLB_MAGIC:
            raise ValueError("not a GLB file")
        off = 12
        while off < length:
            clen, ctype = struct.unpack_from("<II", data, off)
            off += 8
            chunk = data[off : off + clen]
            off += clen
            if ctype == CHUNK_JSON:
                self.json = json.loads(chunk.decode("utf-8"))
            elif ctype == CHUNK_BIN and self._glb_bin is None:
                self._glb_bin = chunk

    # -------------------------------------------------------------- helpers
    @property
    def asset(self) -> dict:
        return self.json.get("asset", {})

    @property
    def materials(self) -> list[dict]:
        return self.json.get("materials", [])

    @property
    def extensions_used(self) -> list[str]:
        return self.json.get("extensionsUsed", [])

    def buffer_view_bytes(self, bv_index: int) -> bytes:
        bv = self.json["bufferViews"][bv_index]
        buf = self.buffers[bv["buffer"]]
        off = bv.get("byteOffset", 0)
        return buf[off : off + bv["byteLength"]]

    def accessor(self, acc_index: int) -> np.ndarray:
        acc = self.json["accessors"][acc_index]
        dtype = np.dtype(COMPONENT_DTYPES[acc["componentType"]]).newbyteorder("<")
        ncomp = TYPE_COUNTS[acc["type"]]
        count = acc["count"]
        if "bufferView" not in acc:
            arr = np.zeros((count, ncomp), dtype=dtype)
        else:
            bv = self.json["bufferViews"][acc["bufferView"]]
            buf = self.buffers[bv["buffer"]]
            base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            stride = bv.get("byteStride")
            elem_size = dtype.itemsize * ncomp
            if stride and stride != elem_size:
                raw = np.frombuffer(buf, dtype=np.uint8, count=stride * (count - 1) + elem_size, offset=base)
                view = np.lib.stride_tricks.as_strided(raw, shape=(count, elem_size), strides=(stride, 1))
                arr = np.ascontiguousarray(view).view(dtype).reshape(count, ncomp)
            else:
                arr = np.frombuffer(buf, dtype=dtype, count=count * ncomp, offset=base).reshape(count, ncomp)
        if acc.get("sparse"):
            arr = self._apply_sparse(arr.copy(), acc)
        if acc.get("normalized") and np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            arr = arr.astype(np.float32) / float(info.max)
            if info.min < 0:
                arr = np.maximum(arr, -1.0)
        return arr

    def _apply_sparse(self, arr: np.ndarray, acc: dict) -> np.ndarray:
        sp = acc["sparse"]
        idx_info, val_info = sp["indices"], sp["values"]
        idx_dtype = np.dtype(COMPONENT_DTYPES[idx_info["componentType"]]).newbyteorder("<")
        bv = self.json["bufferViews"][idx_info["bufferView"]]
        idx = np.frombuffer(
            self.buffers[bv["buffer"]], dtype=idx_dtype, count=sp["count"],
            offset=bv.get("byteOffset", 0) + idx_info.get("byteOffset", 0),
        )
        bv = self.json["bufferViews"][val_info["bufferView"]]
        ncomp = arr.shape[1]
        vals = np.frombuffer(
            self.buffers[bv["buffer"]], dtype=arr.dtype, count=sp["count"] * ncomp,
            offset=bv.get("byteOffset", 0) + val_info.get("byteOffset", 0),
        ).reshape(sp["count"], ncomp)
        arr[idx] = vals
        return arr

    # --------------------------------------------------------------- images
    def image_bytes(self, image_index: int) -> tuple[bytes, str]:
        im = self.json["images"][image_index]
        mime = im.get("mimeType", "")
        if "bufferView" in im:
            return self.buffer_view_bytes(im["bufferView"]), mime
        return _read_uri(im["uri"], self.base_dir), mime

    def image(self, image_index: int) -> Image.Image:
        if image_index in self._image_cache:
            return self._image_cache[image_index]
        data, mime = self.image_bytes(image_index)
        if mime in ("image/ktx2", "image/basis") or data[:4] == b"\xabKTX":
            raise ValueError(
                f"image {image_index} is KTX2/Basis-compressed (KHR_texture_basisu). "
                "Re-export the model with PNG/JPEG textures."
            )
        img = Image.open(io.BytesIO(data))
        img.load()
        self._image_cache[image_index] = img
        return img

    def texture_image_index(self, texture_index: int) -> int:
        tex = self.json["textures"][texture_index]
        if "source" in tex:
            return tex["source"]
        # EXT_texture_webp / KHR_texture_basisu store the source in extensions
        for ext_name, ext in tex.get("extensions", {}).items():
            if "source" in ext:
                if ext_name == "KHR_texture_basisu":
                    raise ValueError("KHR_texture_basisu textures are not supported; re-export with PNG/JPEG")
                return ext["source"]
        raise ValueError(f"texture {texture_index} has no image source")

    def texture_image(self, texture_index: int) -> Image.Image:
        return self.image(self.texture_image_index(texture_index))

    def image_name(self, image_index: int) -> str:
        im = self.json["images"][image_index]
        return im.get("name") or os.path.splitext(os.path.basename(im.get("uri", f"image_{image_index}")))[0]

    # ---------------------------------------------------------------- meshes
    def primitives(self, with_geometry: bool = True) -> list[Primitive]:
        out: list[Primitive] = []
        for mi, mesh in enumerate(self.json.get("meshes", [])):
            for pi, prim in enumerate(mesh.get("primitives", [])):
                mode = prim.get("mode", 4)
                attrs = prim.get("attributes", {})
                p = Primitive(mesh_index=mi, prim_index=pi, material=prim.get("material"),
                              positions=None, normals=None, mode=mode)
                if with_geometry:
                    if "POSITION" in attrs:
                        p.positions = self.accessor(attrs["POSITION"]).astype(np.float32)
                    if "NORMAL" in attrs:
                        p.normals = self.accessor(attrs["NORMAL"]).astype(np.float32)
                    for k, v in attrs.items():
                        if k.startswith("TEXCOORD_"):
                            p.uvs[int(k.split("_")[1])] = self.accessor(v).astype(np.float32)
                    p.indices = self._triangles(prim)
                out.append(p)
        return out

    def _triangles(self, prim: dict) -> Optional[np.ndarray]:
        mode = prim.get("mode", 4)
        if "indices" in prim:
            idx = self.accessor(prim["indices"]).reshape(-1).astype(np.uint32)
        else:
            n = self.json["accessors"][prim["attributes"]["POSITION"]]["count"]
            idx = np.arange(n, dtype=np.uint32)
        if mode == 4:  # TRIANGLES
            return idx[: len(idx) - len(idx) % 3].reshape(-1, 3)
        if mode == 5:  # TRIANGLE_STRIP
            tris = []
            for i in range(len(idx) - 2):
                a, b, c = idx[i], idx[i + 1], idx[i + 2]
                tris.append((a, b, c) if i % 2 == 0 else (b, a, c))
            return np.asarray(tris, dtype=np.uint32).reshape(-1, 3)
        if mode == 6:  # TRIANGLE_FAN
            tris = [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
            return np.asarray(tris, dtype=np.uint32).reshape(-1, 3)
        return None  # points / lines: nothing to rasterise

    def node_names_for_mesh(self, mesh_index: int) -> list[str]:
        return [n.get("name", f"node_{i}") for i, n in enumerate(self.json.get("nodes", [])) if n.get("mesh") == mesh_index]
