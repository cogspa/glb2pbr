# glb2pbr

Drop in a textured `.glb` / `.gltf` (Meshy, Blender, Painter, anything that writes glTF PBR), get back every
Principled BSDF / Substance channel as its own PNG, the UV unwrap, an OBJ for Designer's 3D view, and the scripts
that rebuild the material as a Substance Designer graph and as a Blender Principled BSDF.

Tested on `Meshy_AI_Fleshbound_Navigator_0902135711_texture.glb` (2M verts, 3 × 2048² JPEG textures): 10 maps +
UV layouts in ~30 s, OBJ export adds ~15 s.

```
Meshy .glb ──▶ glb2pbr ──▶ <model>_pbr/
                            ├─ manifest.json                  factors, sockets, usages, provenance per map
                            ├─ material_BaseColor.png         sRGB
                            ├─ material_Metallic.png          B of metallicRoughness × metallicFactor
                            ├─ material_Roughness.png         G of metallicRoughness × roughnessFactor
                            ├─ material_Normal_GL.png         as stored (OpenGL, Blender)
                            ├─ material_Normal_DX.png         green flipped (Substance default)
                            ├─ material_AO.png                occlusionTexture.R (white if absent)
                            ├─ material_Height.png            16-bit, integrated from the normal map
                            ├─ material_Subsurface.png        derived scattering weight
                            ├─ material_SubsurfaceRadius.png  derived radius colour
                            ├─ material_ORM.png               repacked R=AO G=rough B=metal
                            ├─ (Emissive, Opacity, Transmission, Thickness, Specular, SpecularColor,
                            │   Clearcoat, ClearcoatRoughness, ClearcoatNormal_GL, SheenColor, SheenRoughness,
                            │   Curvature — only when the glTF carries them / when asked)
                            ├─ uv/  material_UV_islands.png  _UV_wire.png  _UV_overlay.png
                            ├─ mesh/<model>.obj               V flipped, per-material groups
                            ├─ thumbs/  256px thumbs + material_contact_sheet.png
                            ├─ blender_build_material.py
                            ├─ substance_build_sbs_pysbs.py
                            └─ substance_build_sbs_designer.py
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. Only numpy, Pillow, opencv-python-headless (UV rasterising) and, for the upload UI, FastAPI + uvicorn.

## Use

CLI — one model, one folder:

```bash
python -m glb2pbr model.glb                       # -> model_pbr/
python -m glb2pbr model.glb -o out/ --size 4096   # upsample everything to 4K
python -m glb2pbr model.glb --mesh none --no-uv   # maps only, fastest
python -m glb2pbr model.glb --sss-gain 0.6 --sss-tint 1.0 --curvature
```

Upload UI — drop files in a browser, results and ZIP come back:

```bash
uvicorn glb2pbr.server:app --port 8765
# http://127.0.0.1:8765     jobs land in ./jobs/<id>/out
```

Python — from your own pipeline:

```python
from glb2pbr.pipeline import run
manifest = run("model.glb", "out/", size=2048, mesh="obj")
```

Pipeline stage — `glb2pbr/pipeline_stage.py` exposes `TOOL` (id `glb_unwrap`) with a JSON input schema and an
`execute(inputs, log)` callable in the shape the substance-pipeline registry uses.

## Rebuild in Substance Designer

Two routes, both read `manifest.json` and link the PNGs (no copies):

**Headless (Substance Automation Toolkit / pysbs)** — writes a `.sbs` without opening Designer:

```bash
pip install "<SAT dir>/Python API/Pysbs-*.whl"
python substance_build_sbs_pysbs.py out/manifest.json --n2h       # -> out/<model>.sbs
```

`--n2h` swaps the baked height bitmap for Designer's own *Normal to Height HQ* node fed by the normal bitmap —
recommended for Meshy exports, whose normal maps are subtle enough that the global FFT integration picks up
streaks at island borders. `--normal dx` wires the DirectX normal instead of OpenGL.

**In-app (Designer's Python editor)** — Windows → Python Editor, open `substance_build_sbs_designer.py`, set
`MANIFEST` (or export `GLB2PBR_MANIFEST` before launching), run. It creates a new package, one graph per material,
bitmap → output for every usage, links the OBJ as a scene resource, and saves `<model>.sbs` next to the manifest.
Then right-click the graph → *View outputs in 3D View* and pick the OBJ as the scene.

Usages written: `baseColor metallic roughness normal ambientOcclusion height emissive opacity scattering
transmissive specularLevel coatOpacity coatRoughness coatNormal sheenColor sheenRoughness` — the names Painter
and the Adobe Standard Material read. From the graph you can publish an `.sbsar` with the outputs exposed, or
keep building procedurally on top of the bitmaps (the curvature and islands masks are meant for exactly that).

Both scripts were written against the documented pysbs / `sd` APIs but I could not run Designer or SAT here —
run them once, and if a call name has drifted in your Designer version the traceback points at the single
line to fix.

## Rebuild in Blender

```bash
blender -b --python out/blender_build_material.py -- --manifest out/manifest.json --glb model.glb --save out/check.blend
# or: --displacement --disp-scale 0.02 --ao-multiply   (no --glb -> UV sphere)
```

Imports the GLB, rebuilds `<material>_pbr` from the extracted maps, and swaps it into the mesh's slots. Socket
names resolve for 4.x/5.x with 3.x fallbacks. Normal goes through a Normal Map node with strength 1 because the
glTF `normalTexture.scale` is already baked into the pixels. `Specular IOR Level` gets the map × 0.5 (glTF
`specularFactor` 1.0 == Blender's 0.5 default).

## What is derived vs. what is in the file

glTF carries base colour, metallic/roughness, normal, occlusion, emissive and the KHR extensions listed in
`extract.py`. It has **no** subsurface and **no** height. Those are starting points, not ground truth:

* `Subsurface` = `(1 − metallic) × (1 − transmission) × sss_gain`. On the Fleshbound sphere that lands SSS on the
  flesh and none on the chrome, which is the right first guess; tune `--sss-gain`.
* `SubsurfaceRadius` = mix of Blender's default `(1, 0.2, 0.1)` and the base colour by `--sss-tint`.
* `Height` = Frankot–Chellappa FFT integration of the normal map, gradients masked to the UV islands, low
  frequencies high-passed (`--height-highpass`, 0 to disable). Good for a Designer *Height* input or Blender bump;
  for hero displacement use Designer's Normal to Height HQ (`--n2h`) or sculpt.
* `AO` is white when the file has no occlusion texture — bake real AO from the mesh in Designer/Painter.
* `ORM` is a convenience repack in glTF/Unreal/Unity-HDRP channel order.

Normals: glTF stores OpenGL (+Y). `Normal_GL` is untouched; `Normal_DX` flips green. Blender wants GL. Designer's
3D view can be told which it's looking at (material → Normal Format), and the export usages are the same either way.

UVs: glTF's V runs down, which is exactly image row order, so the UV renders need no flip. The OBJ export flips V
so Designer, Painter and Blender read it upright.

## Limits

* KTX2/Basis-compressed textures (`KHR_texture_basisu`) are not decoded — re-export with PNG/JPEG.
* `KHR_texture_transform` is recorded in the manifest but not applied to the pixels.
* Multiple texcoord sets: layouts and OBJ use `TEXCOORD_0`; the manifest notes which set each texture referenced.
* Large meshes: the OBJ for a 2M-vert Meshy export is ~375 MB. Use `--mesh copy` or `--mesh none` if you
  already have the GLB in Designer (recent versions import glTF directly via Link → 3D Scene).

## Tests

```bash
python -m pytest tests -q      # builds a synthetic KHR-extension sphere, checks channel math, CLI, server
```
