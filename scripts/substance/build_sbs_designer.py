"""
Rebuild the extracted PBR material as a Substance Designer graph — run INSIDE Designer.

How to run
  1. Substance 3D Designer > Windows > Python Editor
  2. Set MANIFEST below (or export GLB2PBR_MANIFEST=/path/to/manifest.json before launching Designer)
  3. Paste this file / open it in the editor and run it

Result
  A new package with one graph per material:
     bitmap (linked PNG) -> output node   for every standard usage
     baseColor, metallic, roughness, normal, ambientOcclusion, height, emissive,
     opacity, scattering, transmissive, specularLevel, coat*, sheen*
  The OBJ (if the manifest has one) is linked as a scene resource so the 3D view
  can show the real mesh: right-click the scene resource > Show in 3D View,
  then right-click the graph > View outputs in 3D View.

Normal format: NORMAL_CONVENTION = "gl" wires the OpenGL (+Y) map, which is what
glTF stores and Blender expects. Set it to "dx" if your Designer 3D view is set to
DirectX and shading looks inverted (Materials > Default > Edit > Normal Format).
"""
import json
import os
import traceback

import sd
from sd.api.sbs.sdsbscompgraph import SDSBSCompGraph
from sd.api.sdbasetypes import float2
from sd.api.sdproperty import SDPropertyCategory
from sd.api.sdresource import EmbedMethod
from sd.api.sdresourcebitmap import SDResourceBitmap
from sd.api.sdvaluebool import SDValueBool
from sd.api.sdvaluestring import SDValueString

# ------------------------------------------------------------------ config
MANIFEST = os.environ.get("GLB2PBR_MANIFEST", "")
NORMAL_CONVENTION = "gl"      # "gl" or "dx"
INCLUDE_HEIGHT = True
SAVE_SBS = True               # save next to the manifest as <model>.sbs
# ---------------------------------------------------------------------------

if not MANIFEST:
    try:
        MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")
    except NameError:
        raise SystemExit("Set MANIFEST to the path of manifest.json (or the GLB2PBR_MANIFEST env var).")

LABELS = {
    "baseColor": "Base Color", "metallic": "Metallic", "roughness": "Roughness", "normal": "Normal",
    "ambientOcclusion": "Ambient Occlusion", "height": "Height", "emissive": "Emissive", "opacity": "Opacity",
    "scattering": "Scattering (Subsurface)", "transmissive": "Transmission", "specularLevel": "Specular Level",
    "coatOpacity": "Coat Opacity", "coatRoughness": "Coat Roughness", "coatNormal": "Coat Normal",
    "sheenColor": "Sheen Color", "sheenRoughness": "Sheen Roughness",
}
ORDER = list(LABELS.keys())


def make_usage_array(usage_name, components="RGBA", colorspace=""):
    """Build the SDValueArray of SDValueUsage that an output node's 'usages' annotation expects."""
    from sd.api.sdtypeusage import SDTypeUsage
    from sd.api.sdusage import SDUsage
    from sd.api.sdvaluearray import SDValueArray
    from sd.api.sdvalueusage import SDValueUsage
    usage = None
    for args in ((usage_name, components, colorspace), (usage_name, components, ""), (usage_name, components)):
        try:
            usage = SDUsage.sNew(*args)      # Designer 11+: (name, components, colorSpace); older: (name, components)
            break
        except TypeError:
            continue
    arr = SDValueArray.sNew(SDTypeUsage.sNew(), 0)
    arr.pushBack(SDValueUsage.sNew(usage))
    return arr


def set_annotation(node, prop_id, value):
    if hasattr(node, "setAnnotationPropertyValueFromId"):
        node.setAnnotationPropertyValueFromId(prop_id, value)
        return
    prop = node.getPropertyFromId(prop_id, SDPropertyCategory.Annotation)
    if prop is None:
        raise RuntimeError(f"annotation property '{prop_id}' not found")
    node.setPropertyValue(prop, value)


def main():
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    base_dir = os.path.dirname(os.path.abspath(MANIFEST))

    ctx = sd.getContext()
    app = ctx.getSDApplication()
    pkg_mgr = app.getPackageMgr()
    pkg = pkg_mgr.newUserPackage()
    print(f"[glb2pbr] new package for {manifest['source']['file']}")

    # link the mesh for the 3D view
    mesh = manifest.get("mesh") or {}
    if mesh.get("obj"):
        try:
            from sd.api.sdresourcescene import SDResourceScene
            SDResourceScene.sNewFromFile(pkg, os.path.join(base_dir, mesh["obj"]), EmbedMethod.Linked)
            print(f"[glb2pbr] linked scene resource {mesh['obj']}")
        except Exception as e:  # noqa: BLE001
            print(f"[glb2pbr] could not link OBJ ({e}); import it manually via Link > 3D Scene")

    for mat in manifest["materials"]:
        graph = SDSBSCompGraph.sNew(pkg)
        graph.setIdentifier(mat["safe_name"] + "_pbr")
        outputs = {o["usage"]: dict(o) for o in mat["substance_outputs"]}
        maps = mat["maps"]
        if NORMAL_CONVENTION == "dx" and "normal_dx" in maps and "normal" in outputs:
            outputs["normal"]["file"] = maps["normal_dx"]["file"]
        if not INCLUDE_HEIGHT:
            outputs.pop("height", None)

        y = 0.0
        for usage in [u for u in ORDER if u in outputs] + [u for u in outputs if u not in ORDER]:
            o = outputs[usage]
            png = os.path.join(base_dir, o["file"])
            if not os.path.exists(png):
                print(f"[glb2pbr] missing {png}; skipping {usage}")
                continue
            try:
                res = SDResourceBitmap.sNewFromFile(pkg, png, EmbedMethod.Linked)
                res.setIdentifier(os.path.splitext(os.path.basename(png))[0])

                bmp = graph.newNode("sbs::compositing::bitmap")
                bmp.setInputPropertyValueFromId("bitmapresourcepath", SDValueString.sNew(res.getUrl()))
                bmp.setInputPropertyValueFromId("colorswitch", SDValueBool.sNew(bool(o["color"])))
                bmp.setPosition(float2(0.0, y))

                out = graph.newNode("sbs::compositing::output")
                set_annotation(out, "identifier", SDValueString.sNew(o["identifier"]))
                set_annotation(out, "label", SDValueString.sNew(LABELS.get(usage, usage)))
                set_annotation(out, "usages", make_usage_array(usage, o.get("components", "RGBA"),
                                                               "sRGB" if o.get("colorspace") == "sRGB" else "Linear"))
                out.setPosition(float2(400.0, y))

                bmp.newPropertyConnectionFromId("unique_filter_output", out, "inputNodeOutput")
                print(f"[glb2pbr] {graph.getIdentifier()}: {usage} <- {o['file']}")
            except Exception:  # noqa: BLE001
                print(f"[glb2pbr] failed on {usage}:\n{traceback.format_exc()}")
            y += 160.0

    if SAVE_SBS:
        sbs_path = os.path.join(base_dir, os.path.splitext(manifest["source"]["file"])[0] + ".sbs")
        pkg_mgr.savePackageAs(pkg, sbs_path)
        print(f"[glb2pbr] saved {sbs_path}")
    print("[glb2pbr] done - open the graph, right-click > View outputs in 3D View")


main()
