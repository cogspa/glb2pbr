# IDE prompts

Paste into Claude Code / Cursor / Antigravity from the repo root. Each one is scoped to a single change and
names the files it touches so the agent doesn't wander.

---

## 1. Verify the Designer scripts on this machine

```
Context: glb2pbr/ extracts PBR maps from a GLB into <out>/ with a manifest.json. Two helper scripts rebuild a
Substance Designer graph from that manifest: scripts/substance/build_sbs_pysbs.py (headless, needs pysbs from the
Substance Automation Toolkit) and scripts/substance/build_sbs_designer.py (runs inside Designer's Python editor).
Both were written from the documented APIs but never executed against my Designer install.

Task: run `python -m glb2pbr tests/fixture_sphere.glb -o /tmp/fx --size 512` (generate the fixture first with
`python tests/make_fixture.py`), then run `python scripts/substance/build_sbs_pysbs.py /tmp/fx/manifest.json`.
Fix any pysbs API mismatch (createBitmapNode / createOutputNode / connectNodes / createLinkedResource signatures,
UsageEnum names) so it writes a .sbs that opens in Designer with every output wired. Don't change the manifest
format. Report the exact pysbs version and Designer version you tested against at the top of the script.
```

## 2. Register glb2pbr as a stage in the substance-pipeline

```
Context: the substance-pipeline repo has a tool registry + async job runner (increment 1). glb2pbr/pipeline_stage.py
exposes TOOL = {id: "glb_unwrap", input_schema, execute(inputs, log)} in that shape.

Task: add glb2pbr as a pip-installable dependency (or vendor the package under tools/), register TOOL in the
registry, add a stage entry "glb_unwrap" to the stage manifest between "ingest" and "substance_build", and make the
React console show the stage's contact sheet (outputs["contact_sheet"]) and per-map thumbnails (<out>/thumbs/) when
the job completes. Reuse the existing contact-sheet console component from the sbsrender stage. Keep everything
local; no network calls.
```

## 3. Painter import stage

```
Context: <out>/manifest.json lists maps with substance_usage names and files. Substance 3D Painter's Python API
(substance_painter.project / textureset / resource) can create a project from an OBJ and import bitmaps as resources.

Task: write scripts/substance/painter_import.py that, run from Painter's Python console, creates a new project from
manifest["mesh"]["obj"] (fallback: ask for a mesh path), imports every map in manifest["materials"][0]["maps"] as a
project resource with the right colour space (sRGB for basecolor/emissive, else linear), and builds one fill layer per
channel wired to the imported bitmaps (basecolor, metallic, roughness, normal, height, ambientOcclusion, emissive,
opacity, scattering). Use the Normal_GL map and set the project's normal format to OpenGL.
```

## 4. Apply KHR_texture_transform

```
Context: glb2pbr/extract.py records KHR_texture_transform (offset/rotation/scale, texCoord override) in the
manifest under sources[map].texture_transform but does not apply it.

Task: in extract.py, when a textureInfo has KHR_texture_transform, warp the sampled texture into the untransformed
UV space before it is written, so the output PNGs line up with the mesh's TEXCOORD_0. Use cv2.warpAffine with
BORDER_WRAP (tiling) and derive the 2x3 matrix from offset, rotation and scale per the extension spec (translation *
rotation * scale applied to UV). Add a test in tests/test_glb2pbr.py that builds a fixture with offset=[0.5,0] and
asserts the base colour is shifted by half the width.
```

## 5. Better height: per-island Poisson

```
Context: glb2pbr/derive.py height_from_normal integrates the whole normal map with one global FFT
(Frankot-Chellappa) and masks gradients to the UV islands. Island borders still leave streaks on meshes with many
small islands (Meshy exports).

Task: add height_from_normal_islands(normal, islands_mask) that labels connected components of the mask
(cv2.connectedComponents), solves each island separately on its bounding box with Neumann boundaries (scipy.sparse
Poisson or the existing FFT on a padded, mirrored crop), normalises each island to a shared range, and composites the
result. Wire it behind --height-mode global|islands in cli.py and pipeline.py, default islands. Keep the 16-bit output
and the highpass option. Add a test comparing correlation against the synthetic bump fixture in
tests/test_glb2pbr.py::test_height_roundtrip for both modes.
```

## 6. Batch + watch folder

```
Context: python -m glb2pbr handles one file. Meshy drops several GLBs per session into ~/Downloads/gardena.

Task: add glb2pbr/watch.py: `python -m glb2pbr.watch <folder> [--out <folder>]` that processes every .glb/.gltf
already there, then watches (watchdog, polling fallback) and processes new ones, writing <name>_pbr/ next to each and
appending a row to <out>/index.csv (file, materials, size, timing, contact sheet path). Skip files that already have a
manifest.json newer than the source. Add it to README under Use.
```
