"""
Build a Substance Designer package (.sbs) from a glb2pbr manifest — headless, no Designer UI.

Requires the Substance Automation Toolkit's Python API (pysbs):
    pip install "<SAT>/Python API/Pysbs-*.whl"       (Windows)
    pip install "<SAT>/Python API/Pysbs-*.whl"       (macOS)

Usage:
    python build_sbs_pysbs.py path/to/manifest.json [-o out.sbs] [--normal gl|dx] [--n2h] [--no-height]

What it builds, per material in the manifest:
    one graph  <Material>_pbr
      bitmap nodes (linked PNGs)  ->  output nodes with the standard usages
      baseColor, metallic, roughness, normal, ambientOcclusion, height, emissive,
      opacity, scattering (subsurface), transmissive, specularLevel, coat*, sheen*
    --n2h : instead of the baked height bitmap, drive the height output from
            Substance's own "Normal to Height HQ" node fed by the normal bitmap
            (cleaner than the FFT-integrated height when the normal map is subtle)
    the OBJ from the manifest is linked as a scene resource for the 3D view
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from pysbs import context, sbsenum, sbsgenerator, substance  # noqa: F401
except ImportError:
    print("pysbs not found. Install the wheel from the Substance Automation Toolkit 'Python API' folder.", file=sys.stderr)
    raise

# usage name in manifest -> pysbs UsageEnum attribute name (fallback: raw string)
USAGE_ENUM_NAMES = {
    "baseColor": "BASECOLOR",
    "metallic": "METALLIC",
    "roughness": "ROUGHNESS",
    "normal": "NORMAL",
    "ambientOcclusion": "AMBIENT_OCCLUSION",
    "height": "HEIGHT",
    "emissive": "EMISSIVE",
    "opacity": "OPACITY",
    "scattering": "SCATTERING",
    "transmissive": "TRANSMISSIVE",
    "specularLevel": "SPECULAR_LEVEL",
    "coatOpacity": "COAT_OPACITY",
    "coatRoughness": "COAT_ROUGHNESS",
    "coatNormal": "COAT_NORMAL",
    "sheenColor": "SHEEN_COLOR",
    "sheenRoughness": "SHEEN_ROUGHNESS",
}
LABELS = {
    "baseColor": "Base Color", "metallic": "Metallic", "roughness": "Roughness", "normal": "Normal",
    "ambientOcclusion": "Ambient Occlusion", "height": "Height", "emissive": "Emissive", "opacity": "Opacity",
    "scattering": "Scattering (Subsurface)", "transmissive": "Transmission", "specularLevel": "Specular Level",
    "coatOpacity": "Coat Opacity", "coatRoughness": "Coat Roughness", "coatNormal": "Coat Normal",
    "sheenColor": "Sheen Color", "sheenRoughness": "Sheen Roughness",
}
ORDER = list(LABELS.keys())


def usage_key(name: str):
    return getattr(sbsenum.UsageEnum, USAGE_ENUM_NAMES.get(name, ""), name)


def build(manifest_path: str, out_path: str | None, normal: str, use_n2h: bool, include_height: bool) -> str:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    out_path = out_path or os.path.join(base_dir, os.path.splitext(manifest["source"]["file"])[0] + ".sbs")
    out_path = os.path.abspath(out_path)

    ctx = context.Context()
    first_graph = manifest["materials"][0]["safe_name"] + "_pbr" if manifest["materials"] else "material_pbr"
    doc = sbsgenerator.createSBSDocument(ctx, aFileAbsPath=out_path, aGraphIdentifier=first_graph)

    # link the mesh for the 3D view
    mesh = manifest.get("mesh") or {}
    if mesh.get("obj"):
        try:
            doc.createLinkedResource(aResourcePath=os.path.join(base_dir, mesh["obj"]),
                                     aResourceTypeEnum=sbsenum.ResourceTypeEnum.SCENE)
            print(f"[sbs] linked scene resource {mesh['obj']}")
        except Exception as e:  # noqa: BLE001
            print(f"[sbs] could not link OBJ as scene resource: {e}")

    for mi, mat in enumerate(manifest["materials"]):
        gname = mat["safe_name"] + "_pbr"
        graph = doc.getSBSGraph(aGraphIdentifier=gname) if mi == 0 else doc.createGraph(aGraphIdentifier=gname)
        outputs = {o["usage"]: o for o in mat["substance_outputs"]}
        maps = mat["maps"]

        # choose the normal convention
        if normal == "dx" and "normal_dx" in maps:
            outputs["normal"] = {**outputs["normal"], "file": maps["normal_dx"]["file"]}

        if not include_height:
            outputs.pop("height", None)

        y = 0
        step = 160
        normal_bitmap = None
        for usage in [u for u in ORDER if u in outputs] + [u for u in outputs if u not in ORDER]:
            o = outputs[usage]
            png = os.path.join(base_dir, o["file"])
            if not os.path.exists(png):
                print(f"[sbs] missing {png}, skipping {usage}")
                continue
            if usage == "height" and use_n2h and normal_bitmap is not None:
                continue  # handled below
            color_mode = sbsenum.ColorModeEnum.COLOR if o["color"] else sbsenum.ColorModeEnum.GRAYSCALE
            bmp = graph.createBitmapNode(
                aSBSDocument=doc,
                aResourcePath=png,
                aGUIPos=[0, y, 0],
                aParameters={sbsenum.CompNodeParamEnum.COLOR_MODE: color_mode},
                aAutodetectImageParameters=False,
                aIsLinked=True,
            )
            if usage == "normal":
                normal_bitmap = bmp
            comps = sbsenum.ComponentsEnum.RGBA
            out = graph.createOutputNode(
                aIdentifier=o["identifier"],
                aGUIPos=[400, y, 0],
                aAttributes={sbsenum.AttributesEnum.Label: LABELS.get(usage, usage)},
                aUsages={usage_key(usage): {sbsenum.UsageDataEnum.COMPONENTS: comps}},
            )
            graph.connectNodes(aLeftNode=bmp, aRightNode=out)
            print(f"[sbs] {gname}: {usage} <- {o['file']}")
            y += step

        if use_n2h and normal_bitmap is not None:
            try:
                n2h = graph.createCompInstanceNodeFromPath(
                    aSBSDocument=doc, aPath="sbs://normal_to_height_hq.sbs", aGUIPos=[200, y, 0])
                out = graph.createOutputNode(
                    aIdentifier="height", aGUIPos=[400, y, 0],
                    aAttributes={sbsenum.AttributesEnum.Label: "Height (Normal to Height HQ)"},
                    aUsages={usage_key("height"): {sbsenum.UsageDataEnum.COMPONENTS: sbsenum.ComponentsEnum.RGBA}})
                graph.connectNodes(aLeftNode=normal_bitmap, aRightNode=n2h)
                graph.connectNodes(aLeftNode=n2h, aRightNode=out)
                print(f"[sbs] {gname}: height <- normal_to_height_hq(normal)")
            except Exception as e:  # noqa: BLE001
                print(f"[sbs] Normal to Height HQ failed ({e}); falling back to baked height bitmap")
                o = outputs.get("height")
                if o:
                    bmp = graph.createBitmapNode(aSBSDocument=doc, aResourcePath=os.path.join(base_dir, o["file"]),
                                                 aGUIPos=[0, y, 0],
                                                 aParameters={sbsenum.CompNodeParamEnum.COLOR_MODE: sbsenum.ColorModeEnum.GRAYSCALE})
                    out = graph.createOutputNode(aIdentifier="height", aGUIPos=[400, y, 0],
                                                 aUsages={usage_key("height"): {sbsenum.UsageDataEnum.COMPONENTS: sbsenum.ComponentsEnum.RGBA}})
                    graph.connectNodes(aLeftNode=bmp, aRightNode=out)

        # graph attributes
        try:
            graph.setAttribute(sbsenum.AttributesEnum.Label, mat["name"])
            graph.setAttribute(sbsenum.AttributesEnum.Description,
                               f"Extracted by glb2pbr from {manifest['source']['file']} ({manifest['source'].get('generator')})")
        except Exception:  # noqa: BLE001
            pass

    doc.writeDoc()
    print(f"[sbs] wrote {out_path}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--normal", choices=["gl", "dx"], default="gl",
                   help="which normal convention to wire into the graph (default gl = as stored in glTF, Blender-ready)")
    p.add_argument("--n2h", action="store_true", help="derive height with Normal to Height HQ instead of the baked bitmap")
    p.add_argument("--no-height", action="store_true")
    a = p.parse_args(argv)
    build(a.manifest, a.out, a.normal, a.n2h, not a.no_height)


if __name__ == "__main__":
    main()
