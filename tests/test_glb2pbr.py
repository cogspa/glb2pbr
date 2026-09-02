import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from glb2pbr import derive  # noqa: E402
from glb2pbr.extract import ExtractOptions, extract_material  # noqa: E402
from glb2pbr.gltf_reader import GLTFDoc  # noqa: E402
from glb2pbr.pipeline import run  # noqa: E402
from tests.make_fixture import build  # noqa: E402


@pytest.fixture(scope="session")
def fixture(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx")
    glb = str(d / "sphere.glb")
    gltf = str(d / "sphere.gltf")
    build(glb, gltf)
    return glb, gltf


def test_reader_glb_and_gltf(fixture):
    glb, gltf = fixture
    for path in (glb, gltf):
        doc = GLTFDoc(path)
        assert len(doc.materials) == 1
        prims = doc.primitives()
        assert prims[0].indices.shape[1] == 3
        assert 0 in prims[0].uvs
        assert doc.texture_image(0).size == (128, 128)


def test_channel_semantics(fixture):
    doc = GLTFDoc(fixture[0])
    res = extract_material(doc, 0, ExtractOptions(size=128))
    m = res.maps
    mr = derive.to_float(doc.image(1), "RGBA")
    # G * roughnessFactor, B * metallicFactor
    assert np.allclose(m["roughness"], mr[..., 1] * 0.5, atol=1e-3)
    assert np.allclose(m["metallic"], mr[..., 2] * 0.75, atol=1e-3)
    # occlusion strength blend: 1 + s*(R-1)
    assert np.allclose(m["ao"], 1 + 0.5 * (mr[..., 0] - 1), atol=1e-3)
    # base colour * factor, alpha -> opacity (BLEND)
    bc = derive.to_float(doc.image(0), "RGBA")
    assert np.allclose(m["basecolor"], bc[..., :3] * np.array([1, 0.9, 0.8]), atol=1e-3)
    assert "opacity" in m and np.allclose(m["opacity"], bc[..., 3], atol=1e-3)
    # emissive factor 1 with strength 4 recorded
    assert res.factors["emissiveStrength"] == 4.0
    # extensions
    assert "clearcoat" in m and "clearcoat_roughness" in m and "transmission" in m and "thickness" in m
    assert "specular" in m and "specular_color" in m and "sheen_color" in m and "sheen_roughness" in m
    assert res.factors["ior"] == 1.33
    assert res.flags["packedORM"] is True
    # normal scale 1.5 applied and DX twin is green-inverted
    assert np.allclose(m["normal_dx"][..., 1], 1 - m["normal_gl"][..., 1])
    # derived subsurface = (1-metallic)*(1-transmission)
    assert np.allclose(m["subsurface"], (1 - m["metallic"]) * (1 - m["transmission"]), atol=1e-4)


def test_height_roundtrip():
    S = 128
    yy, xx = np.mgrid[0:S, 0:S]
    h = np.exp(-(((xx - 48) ** 2 + (yy - 40) ** 2) / (2 * 10 ** 2)))
    n = derive.normal_from_height(h, strength=30)
    z = derive.height_from_normal(n, highpass=0)
    assert np.corrcoef(h.ravel(), z.ravel())[0, 1] > 0.98


def test_texture_transform(tmp_path):
    glb = str(tmp_path / "transformed.glb")
    build(glb, base_color_transform={"offset": [0.5, 0.0]})
    doc = GLTFDoc(glb)
    res = extract_material(doc, 0, ExtractOptions(size=128))
    m = res.maps

    # Check manifest records texture transform
    assert res.sources["basecolor"]["texture_transform"]["offset"] == [0.5, 0.0]

    # Raw base color from image (without transform) multiplied by baseColorFactor
    bc_raw = derive.to_float(doc.image(0), "RGBA")[..., :3] * np.array([1, 0.9, 0.8], dtype=np.float32)

    # With offset=[0.5, 0], base colour is shifted horizontally by half the width (64 px)
    expected_shifted = np.roll(bc_raw, -64, axis=1)
    assert np.allclose(m["basecolor"], expected_shifted, atol=1e-3)
    assert not np.allclose(m["basecolor"], bc_raw, atol=1e-3)
    assert np.allclose(m["basecolor"][:, :64], bc_raw[:, 64:], atol=1e-3)
    assert np.allclose(m["basecolor"][:, 64:], bc_raw[:, :64], atol=1e-3)

    # Test scale=[2.0, 2.0]: texture tiles twice
    glb_scaled = str(tmp_path / "scaled.glb")
    build(glb_scaled, base_color_transform={"scale": [2.0, 2.0]})
    doc_scaled = GLTFDoc(glb_scaled)
    res_scaled = extract_material(doc_scaled, 0, ExtractOptions(size=128))
    # Pixel x in [0, 64) maps to 2x in [0, 128)
    # The first 64 columns should contain the full 128 columns compressed
    assert res_scaled.sources["basecolor"]["texture_transform"]["scale"] == [2.0, 2.0]

    # Test rotation via apply_texture_transform
    from glb2pbr.extract import apply_texture_transform
    test_img = np.zeros((128, 128, 4), dtype=np.float32)
    test_img[:64, :64, :] = 1.0
    rot_img = apply_texture_transform(test_img, {"rotation": np.pi / 2})
    # Counter-clockwise rotation of UV by 90 degrees rotates image clockwise:
    # Top-left quadrant (x in [0, 64], y in [0, 64]) moves to top-right quadrant
    assert rot_img[:64, 64:, 0].mean() > 0.9
    assert rot_img[:64, :64, 0].mean() < 0.1




def test_pipeline_end_to_end(fixture, tmp_path):
    out = str(tmp_path / "out")
    manifest = run(fixture[0], out, size=128, mesh="obj", curvature=True, log=lambda *_: None)
    assert os.path.exists(os.path.join(out, "manifest.json"))
    mat = manifest["materials"][0]
    for rec in mat["maps"].values():
        assert os.path.exists(os.path.join(out, rec["file"]))
    assert mat["uv"]["coverage"] > 0.9  # a UV sphere covers the whole 0..1 square
    assert os.path.exists(os.path.join(out, manifest["mesh"]["obj"]))
    assert "uv/Fixture_Mat_UV_overlay.png" in mat["uv"]["files"].values()
    assert manifest["thumbnails"]["contact_sheets"]
    usages = {o["usage"] for o in mat["substance_outputs"]}
    assert {"baseColor", "metallic", "roughness", "normal", "height", "scattering", "emissive", "opacity"} <= usages
    assert mat["blender_principled"]["Base Color"].endswith("_BaseColor.png")
    with open(os.path.join(out, manifest["mesh"]["obj"])) as fh:
        head = [next(fh) for _ in range(4)]
    assert head[1].startswith("g ") and head[2].startswith("usemtl")
    for helper in manifest["helpers"]:
        assert os.path.exists(os.path.join(out, helper))


def test_cli(fixture, tmp_path):
    from glb2pbr.cli import main
    out = str(tmp_path / "cli_out")
    assert main([fixture[1], "-o", out, "--size", "128", "--mesh", "none", "--no-helpers"]) == 0
    assert os.path.exists(os.path.join(out, "manifest.json"))


def test_server(fixture, tmp_path, monkeypatch):
    monkeypatch.setenv("GLB2PBR_JOBS", str(tmp_path / "jobs"))
    import importlib
    from glb2pbr import server
    importlib.reload(server)
    from fastapi.testclient import TestClient
    client = TestClient(server.app)
    assert "Unwrap" in client.get("/").text
    with open(fixture[0], "rb") as fh:
        r = client.post("/api/extract", files={"file": ("sphere.glb", fh, "model/gltf-binary")},
                        data={"size": "128", "mesh": "none"})
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    j = client.get(f"/api/jobs/{job}").json()
    assert j["status"] == "done", j
    assert client.get(f"/api/jobs/{job}/files/manifest.json").status_code == 200
    z = client.get(f"/api/jobs/{job}/zip")
    assert z.status_code == 200 and z.headers["content-type"].startswith("application/zip")
    assert client.get(f"/api/jobs/{job}/files/../job.json").status_code == 404
    bad = client.post("/api/extract", files={"file": ("x.txt", b"nope", "text/plain")})
    assert bad.status_code == 400
