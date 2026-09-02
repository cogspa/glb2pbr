"""
Rebuild the extracted PBR maps as a Principled BSDF material in Blender (3.6 - 5.x).

Headless:
    blender -b --python build_material.py -- --manifest out/manifest.json --glb model.glb --save out/check.blend
Interactive (Text Editor > Run Script): set MANIFEST / GLB below, or leave GLB empty for a UV sphere.

Wiring (Blender 4.x socket names, with 3.x fallbacks):
    basecolor          -> Base Color              (sRGB)          [x AO if --ao-multiply]
    opacity            -> Alpha                   (Non-Color)
    metallic           -> Metallic                (Non-Color)
    roughness          -> Roughness               (Non-Color)
    normal_gl          -> Normal Map node -> Normal (Non-Color)   [glTF stores OpenGL; no flip]
    emissive           -> Emission Color, strength = emissiveStrength
    subsurface         -> Subsurface Weight       (Non-Color)
    subsurface_radius  -> Subsurface Radius       (Non-Color)
    transmission       -> Transmission Weight
    specular           -> Specular IOR Level (x0.5: glTF specularFactor 1.0 == Blender 0.5)
    clearcoat*         -> Coat Weight / Coat Roughness / Coat Normal
    sheen_color        -> Sheen Tint  (Sheen Weight = 1)
    height             -> Displacement node (--displacement)
    ior                -> IOR
"""
import argparse
import json
import os
import sys

import bpy

# ---- interactive defaults (ignored when CLI args are given) ---------------
MANIFEST = ""
GLB = ""
# ---------------------------------------------------------------------------

SOCKET_ALIASES = {
    "Base Color": ["Base Color"],
    "Alpha": ["Alpha"],
    "Metallic": ["Metallic"],
    "Roughness": ["Roughness"],
    "Normal": ["Normal"],
    "IOR": ["IOR"],
    "Emission Color": ["Emission Color", "Emission"],
    "Emission Strength": ["Emission Strength"],
    "Subsurface Weight": ["Subsurface Weight", "Subsurface"],
    "Subsurface Radius": ["Subsurface Radius"],
    "Transmission Weight": ["Transmission Weight", "Transmission"],
    "Specular IOR Level": ["Specular IOR Level", "Specular"],
    "Specular Tint": ["Specular Tint"],
    "Coat Weight": ["Coat Weight", "Clearcoat"],
    "Coat Roughness": ["Coat Roughness", "Clearcoat Roughness"],
    "Coat Normal": ["Coat Normal", "Clearcoat Normal"],
    "Sheen Weight": ["Sheen Weight", "Sheen"],
    "Sheen Roughness": ["Sheen Roughness"],
    "Sheen Tint": ["Sheen Tint"],
}


def find_socket(node, logical_name):
    for name in SOCKET_ALIASES.get(logical_name, [logical_name]):
        if name in node.inputs:
            return node.inputs[name]
    return None


def load_image(path, srgb):
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
    return img


def build_material(mat_entry, base_dir, ao_multiply=False, displacement=False, disp_scale=0.02):
    name = mat_entry["name"]
    maps = mat_entry["maps"]
    factors = mat_entry["factors"]
    mat = bpy.data.materials.new(name + "_pbr")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    ior = find_socket(bsdf, "IOR")
    if ior:
        ior.default_value = float(factors.get("ior", 1.5))

    y = 500
    tex_nodes = {}

    def tex(map_id):
        if map_id in tex_nodes:
            return tex_nodes[map_id]
        rec = maps.get(map_id)
        if not rec:
            return None
        nonlocal y
        path = os.path.join(base_dir, rec["file"])
        if not os.path.exists(path):
            print(f"[blender] missing {path}")
            return None
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = load_image(path, rec["colorspace"] == "sRGB")
        n.label = map_id
        n.location = (-700, y)
        y -= 300
        tex_nodes[map_id] = n
        return n

    def link(map_id, socket_name, use_alpha=False):
        n = tex(map_id)
        s = find_socket(bsdf, socket_name)
        if n is None or s is None:
            if n is not None:
                print(f"[blender] no socket '{socket_name}' on this Blender version; skipped {map_id}")
            return None
        nt.links.new(n.outputs["Alpha" if use_alpha else "Color"], s)
        return n

    # base colour (optionally multiplied by AO)
    bc = tex("basecolor")
    if bc is not None:
        if ao_multiply and "ao" in maps:
            ao = tex("ao")
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            mix.location = (-300, 500)
            nt.links.new(bc.outputs["Color"], mix.inputs[6])
            nt.links.new(ao.outputs["Color"], mix.inputs[7])
            nt.links.new(mix.outputs[2], find_socket(bsdf, "Base Color"))
        else:
            nt.links.new(bc.outputs["Color"], find_socket(bsdf, "Base Color"))

    link("opacity", "Alpha")
    if "opacity" in maps:
        blend = factors.get("alphaMode") == "BLEND"
        if hasattr(mat, "surface_render_method"):          # Blender 4.2+ (EEVEE Next)
            mat.surface_render_method = "BLENDED" if blend else "DITHERED"
        else:                                               # Blender 3.x - 4.1
            mat.blend_method = "BLEND" if blend else "CLIP"
    link("metallic", "Metallic")
    link("roughness", "Roughness")

    n = tex("normal_gl")
    if n is not None:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-300, -200)
        nm.inputs["Strength"].default_value = 1.0  # normalScale is already applied to the pixels
        nt.links.new(n.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], find_socket(bsdf, "Normal"))

    if "emissive" in maps:
        link("emissive", "Emission Color")
        es = find_socket(bsdf, "Emission Strength")
        if es:
            es.default_value = float(factors.get("emissiveStrength", 1.0))

    link("subsurface", "Subsurface Weight")
    link("subsurface_radius", "Subsurface Radius")
    link("transmission", "Transmission Weight")

    if "specular" in maps:
        n = tex("specular")
        s = find_socket(bsdf, "Specular IOR Level")
        if n is not None and s is not None:
            m = nt.nodes.new("ShaderNodeMath")
            m.operation = "MULTIPLY"
            m.inputs[1].default_value = 0.5
            m.location = (-300, -700)
            nt.links.new(n.outputs["Color"], m.inputs[0])
            nt.links.new(m.outputs[0], s)
    link("specular_color", "Specular Tint")

    link("clearcoat", "Coat Weight")
    link("clearcoat_roughness", "Coat Roughness")
    n = tex("clearcoat_normal")
    if n is not None and find_socket(bsdf, "Coat Normal") is not None:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-300, -1000)
        nt.links.new(n.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], find_socket(bsdf, "Coat Normal"))

    if "sheen_color" in maps:
        link("sheen_color", "Sheen Tint")
        sw = find_socket(bsdf, "Sheen Weight")
        if sw:
            sw.default_value = 1.0
        link("sheen_roughness", "Sheen Roughness")

    if displacement and "height" in maps:
        n = tex("height")
        d = nt.nodes.new("ShaderNodeDisplacement")
        d.location = (250, -900)
        d.inputs["Midlevel"].default_value = 0.5
        d.inputs["Scale"].default_value = disp_scale
        nt.links.new(n.outputs["Color"], d.inputs["Height"])
        nt.links.new(d.outputs["Displacement"], out.inputs["Displacement"])
        try:
            mat.displacement_method = "BUMP"  # switch to DISPLACEMENT + adaptive subdivision for true displacement
        except AttributeError:
            mat.cycles.displacement_method = "BUMP"

    if mat_entry.get("flags", {}).get("doubleSided"):
        mat.use_backface_culling = False
    return mat


def get_targets(glb_path):
    if glb_path:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=glb_path)
        return [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=1.0)
    obj = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    return [obj]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--glb", default=GLB, help="import this GLB/glTF and assign the rebuilt material(s); omit for a UV sphere")
    p.add_argument("--save", default="", help="save a .blend here")
    p.add_argument("--ao-multiply", action="store_true", help="multiply base colour by AO")
    p.add_argument("--displacement", action="store_true", help="wire the height map into a Displacement node")
    p.add_argument("--disp-scale", type=float, default=0.02)
    a = p.parse_args(argv)
    if not a.manifest:
        raise SystemExit("--manifest is required (or set MANIFEST at the top of the script)")

    with open(a.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    base_dir = os.path.dirname(os.path.abspath(a.manifest))

    targets = get_targets(a.glb)
    mats = [build_material(m, base_dir, a.ao_multiply, a.displacement, a.disp_scale) for m in manifest["materials"]]
    by_index = {m["index"]: mat for m, mat in zip(manifest["materials"], mats)}

    for obj in targets:
        if a.glb and obj.data.materials:
            # replace each imported glTF material slot with the rebuilt one of the same name
            for i, slot_mat in enumerate(obj.data.materials):
                for m, mat in zip(manifest["materials"], mats):
                    if slot_mat is not None and slot_mat.name.split(".")[0] == m["name"]:
                        obj.data.materials[i] = mat
                        break
                else:
                    if i < len(mats):
                        obj.data.materials[i] = mats[i]
        else:
            obj.data.materials.clear()
            obj.data.materials.append(by_index.get(0, mats[0]))
    print(f"[blender] built {len(mats)} material(s), assigned to {len(targets)} object(s)")

    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
        print(f"[blender] saved {a.save}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
