"""
glTF material -> individual PBR channel maps.

Channel semantics (glTF 2.0 spec + KHR extensions):
  baseColorTexture            RGB(A) sRGB   * baseColorFactor
  metallicRoughnessTexture    G = roughness * roughnessFactor
                              B = metallic  * metallicFactor
  occlusionTexture            R, blended by strength
  normalTexture               RGB tangent space, OpenGL (+Y up), scale
  emissiveTexture             RGB sRGB * emissiveFactor * KHR_materials_emissive_strength
  KHR_materials_transmission  transmissionTexture.R * transmissionFactor
  KHR_materials_volume        thicknessTexture.G * thicknessFactor, attenuationColor/Distance
  KHR_materials_specular      specularTexture.A * specularFactor, specularColorTexture.RGB
  KHR_materials_clearcoat     clearcoatTexture.R, clearcoatRoughnessTexture.G, clearcoatNormalTexture
  KHR_materials_sheen         sheenColorTexture.RGB, sheenRoughnessTexture.A
  KHR_materials_ior           ior
  KHR_materials_pbrSpecularGlossiness (legacy) diffuse + specular/glossiness -> roughness = 1 - gloss

Derived (not in glTF):
  subsurface (scattering) weight, subsurface radius, height (from normal), DX normal, curvature
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import derive
from .gltf_reader import GLTFDoc


# Map id -> (file suffix, colorspace, channels, Blender Principled socket, Substance usage, Substance components)
MAP_SPECS = {
    "basecolor":         ("BaseColor",        "sRGB",     "RGB", "Base Color",           "baseColor",          "RGBA"),
    "opacity":           ("Opacity",          "Linear",   "L",   "Alpha",                "opacity",            "RGBA"),
    "metallic":          ("Metallic",         "Linear",   "L",   "Metallic",             "metallic",           "RGBA"),
    "roughness":         ("Roughness",        "Linear",   "L",   "Roughness",            "roughness",          "RGBA"),
    "normal_gl":         ("Normal_GL",        "Linear",   "RGB", "Normal",               "normal",             "RGBA"),
    "normal_dx":         ("Normal_DX",        "Linear",   "RGB", None,                   None,                 "RGBA"),
    "ao":                ("AO",               "Linear",   "L",   None,                   "ambientOcclusion",   "RGBA"),
    "emissive":          ("Emissive",         "sRGB",     "RGB", "Emission Color",       "emissive",           "RGBA"),
    "height":            ("Height",           "Linear",   "L16", None,                   "height",             "RGBA"),
    "curvature":         ("Curvature",        "Linear",   "L",   None,                   None,                 "RGBA"),
    "subsurface":        ("Subsurface",       "Linear",   "L",   "Subsurface Weight",    "scattering",         "RGBA"),
    "subsurface_radius": ("SubsurfaceRadius", "Linear",   "RGB", "Subsurface Radius",    None,                 "RGBA"),
    "transmission":      ("Transmission",     "Linear",   "L",   "Transmission Weight",  "transmissive",       "RGBA"),
    "thickness":         ("Thickness",        "Linear",   "L",   None,                   None,                 "RGBA"),
    "specular":          ("Specular",         "Linear",   "L",   "Specular IOR Level",   "specularLevel",      "RGBA"),
    "specular_color":    ("SpecularColor",    "sRGB",     "RGB", "Specular Tint",        None,                 "RGBA"),
    "clearcoat":         ("Clearcoat",        "Linear",   "L",   "Coat Weight",          "coatOpacity",        "RGBA"),
    "clearcoat_roughness": ("ClearcoatRoughness", "Linear", "L", "Coat Roughness",       "coatRoughness",      "RGBA"),
    "clearcoat_normal":  ("ClearcoatNormal_GL", "Linear",  "RGB", "Coat Normal",         "coatNormal",         "RGBA"),
    "sheen_color":       ("SheenColor",       "sRGB",     "RGB", "Sheen Tint",           "sheenColor",         "RGBA"),
    "sheen_roughness":   ("SheenRoughness",   "Linear",   "L",   "Sheen Roughness",      "sheenRoughness",     "RGBA"),
    "orm":               ("ORM",              "Linear",   "RGB", None,                   None,                 "RGBA"),
}


@dataclass
class ExtractOptions:
    size: int | None = None           # None -> largest source texture (min 512)
    fill_missing: bool = True         # write constant maps for channels defined only by factors
    derive_height: bool = True
    derive_subsurface: bool = True
    derive_curvature: bool = False
    sss_gain: float = 1.0
    sss_tint: float = 0.75
    write_orm: bool = True            # repack Occlusion/Roughness/Metallic for Painter/Unity/Unreal
    apply_normal_scale: bool = True
    height_highpass: float = 1.0 / 24.0   # sigma (fraction of size) to remove integration drift; 0 = off
    uv_mask: object = None            # optional (H,W) uint8/float island mask; zeroes gradients outside islands


@dataclass
class MaterialResult:
    index: int
    name: str
    safe_name: str
    size: int
    maps: dict = field(default_factory=dict)      # map id -> float array
    sources: dict = field(default_factory=dict)   # map id -> description of where it came from
    factors: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")
    return s or "material"


def apply_texture_transform(arr: np.ndarray, transform: dict) -> np.ndarray:
    """Warp a float image into untransformed UV space using KHR_texture_transform."""
    offset = transform.get("offset", [0.0, 0.0])
    rotation = float(transform.get("rotation", 0.0))
    scale = transform.get("scale", [1.0, 1.0])

    tx, ty = float(offset[0]), float(offset[1])
    sx, sy = float(scale[0]), float(scale[1])

    if tx == 0.0 and ty == 0.0 and rotation == 0.0 and sx == 1.0 and sy == 1.0:
        return arr

    cos_r = float(np.cos(rotation))
    sin_r = float(np.sin(rotation))

    h, w = arr.shape[:2]
    # Matrix mapping (x, y) in untransformed UV space to (x', y') in source texture:
    # u' = sx * cos(r) * u - sy * sin(r) * v + tx
    # v' = sx * sin(r) * u + sy * cos(r) * v + ty
    # with x = u * w, y = v * h, x' = u' * w, y' = v' * h:
    m = np.array([
        [sx * cos_r, -sy * sin_r * (w / h), tx * w],
        [sx * sin_r * (h / w), sy * cos_r, ty * h],
    ], dtype=np.float32)

    warped = cv2.warpAffine(arr, m, (w, h), borderMode=cv2.BORDER_WRAP, flags=cv2.INTER_LINEAR)
    if arr.ndim == 3 and warped.ndim == 2:
        warped = warped[..., np.newaxis]
    return np.clip(warped, 0.0, 1.0)


def _tex_info(doc: GLTFDoc, info: dict | None):
    """Return (float RGBA array, description) for a textureInfo dict or (None, None)."""
    if not info or "index" not in info:
        return None, None
    tex_index = info["index"]
    img = doc.texture_image(tex_index)
    arr = derive.to_float(img, "RGBA")
    img_index = doc.texture_image_index(tex_index)
    desc = {
        "texture": tex_index,
        "image": img_index,
        "image_name": doc.image_name(img_index),
        "width": img.width,
        "height": img.height,
        "texcoord": info.get("texCoord", 0),
    }
    if "KHR_texture_transform" in info.get("extensions", {}):
        tt = info["extensions"]["KHR_texture_transform"]
        desc["texture_transform"] = tt
        if "texCoord" in tt:
            desc["texcoord"] = tt["texCoord"]
        arr = apply_texture_transform(arr, tt)
    return arr, desc


def _largest_size(descs: list[dict | None], fallback: int) -> int:
    sizes = [max(d["width"], d["height"]) for d in descs if d]
    return max(sizes) if sizes else fallback


def extract_material(doc: GLTFDoc, mat_index: int, opts: ExtractOptions) -> MaterialResult:
    mat = doc.materials[mat_index]
    name = mat.get("name") or f"material_{mat_index}"
    res = MaterialResult(index=mat_index, name=name, safe_name=safe_name(name), size=0)
    ext = mat.get("extensions", {}) or {}
    pbr = mat.get("pbrMetallicRoughness", {}) or {}

    # ---- gather source textures -----------------------------------------
    bc_arr, bc_desc = _tex_info(doc, pbr.get("baseColorTexture"))
    mr_arr, mr_desc = _tex_info(doc, pbr.get("metallicRoughnessTexture"))
    occ_arr, occ_desc = _tex_info(doc, mat.get("occlusionTexture"))
    nrm_arr, nrm_desc = _tex_info(doc, mat.get("normalTexture"))
    em_arr, em_desc = _tex_info(doc, mat.get("emissiveTexture"))

    sg = ext.get("KHR_materials_pbrSpecularGlossiness")
    sg_diff_arr = sg_diff_desc = sg_spec_arr = sg_spec_desc = None
    if sg:
        sg_diff_arr, sg_diff_desc = _tex_info(doc, sg.get("diffuseTexture"))
        sg_spec_arr, sg_spec_desc = _tex_info(doc, sg.get("specularGlossinessTexture"))
        res.warnings.append("Material uses legacy KHR_materials_pbrSpecularGlossiness; converted to metal/rough (metallic=0, roughness=1-glossiness).")

    tr = ext.get("KHR_materials_transmission")
    tr_arr, tr_desc = _tex_info(doc, tr.get("transmissionTexture")) if tr else (None, None)
    vol = ext.get("KHR_materials_volume")
    th_arr, th_desc = _tex_info(doc, vol.get("thicknessTexture")) if vol else (None, None)
    spec = ext.get("KHR_materials_specular")
    sp_arr, sp_desc = _tex_info(doc, spec.get("specularTexture")) if spec else (None, None)
    spc_arr, spc_desc = _tex_info(doc, spec.get("specularColorTexture")) if spec else (None, None)
    cc = ext.get("KHR_materials_clearcoat")
    cc_arr, cc_desc = _tex_info(doc, cc.get("clearcoatTexture")) if cc else (None, None)
    ccr_arr, ccr_desc = _tex_info(doc, cc.get("clearcoatRoughnessTexture")) if cc else (None, None)
    ccn_arr, ccn_desc = _tex_info(doc, cc.get("clearcoatNormalTexture")) if cc else (None, None)
    sh = ext.get("KHR_materials_sheen")
    shc_arr, shc_desc = _tex_info(doc, sh.get("sheenColorTexture")) if sh else (None, None)
    shr_arr, shr_desc = _tex_info(doc, sh.get("sheenRoughnessTexture")) if sh else (None, None)

    all_descs = [bc_desc, mr_desc, occ_desc, nrm_desc, em_desc, sg_diff_desc, sg_spec_desc, tr_desc, th_desc,
                 sp_desc, spc_desc, cc_desc, ccr_desc, ccn_desc, shc_desc, shr_desc]
    size = opts.size or max(512, _largest_size(all_descs, 1024))
    res.size = size
    R = lambda a: derive.resize(a, size)  # noqa: E731

    # ---- factors ---------------------------------------------------------
    bc_factor = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1]), dtype=np.float32)
    metallic_factor = float(pbr.get("metallicFactor", 1.0))
    roughness_factor = float(pbr.get("roughnessFactor", 1.0))
    emissive_factor = np.asarray(mat.get("emissiveFactor", [0, 0, 0]), dtype=np.float32)
    emissive_strength = float(ext.get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0))
    alpha_mode = mat.get("alphaMode", "OPAQUE")
    alpha_cutoff = float(mat.get("alphaCutoff", 0.5))
    normal_scale = float((mat.get("normalTexture") or {}).get("scale", 1.0))
    occ_strength = float((mat.get("occlusionTexture") or {}).get("strength", 1.0))
    ior = float(ext.get("KHR_materials_ior", {}).get("ior", 1.5))

    res.factors = {
        "baseColorFactor": bc_factor.tolist(),
        "metallicFactor": metallic_factor,
        "roughnessFactor": roughness_factor,
        "emissiveFactor": emissive_factor.tolist(),
        "emissiveStrength": emissive_strength,
        "normalScale": normal_scale,
        "occlusionStrength": occ_strength,
        "ior": ior,
        "alphaMode": alpha_mode,
        "alphaCutoff": alpha_cutoff,
    }
    res.flags = {
        "doubleSided": bool(mat.get("doubleSided", False)),
        "unlit": "KHR_materials_unlit" in ext,
        "extensions": sorted(ext.keys()),
    }

    # ---- base colour / opacity --------------------------------------------
    if sg and sg_diff_arr is not None:
        diff_factor = np.asarray(sg.get("diffuseFactor", [1, 1, 1, 1]), dtype=np.float32)
        bc_rgba = R(sg_diff_arr) * diff_factor
        res.sources["basecolor"] = {**sg_diff_desc, "note": "specGloss diffuseTexture"}
    elif bc_arr is not None:
        bc_rgba = R(bc_arr) * bc_factor
        res.sources["basecolor"] = bc_desc
    else:
        bc_rgba = derive.constant(size, bc_factor, 4)
        res.sources["basecolor"] = {"constant": bc_factor.tolist()}
    res.maps["basecolor"] = bc_rgba[..., :3]
    if alpha_mode != "OPAQUE" or bc_factor[3] < 0.999:
        alpha = bc_rgba[..., 3]
        if alpha_mode == "MASK":
            alpha = (alpha >= alpha_cutoff).astype(np.float32)
        res.maps["opacity"] = alpha
        res.sources["opacity"] = {"from": "baseColor alpha", "alphaMode": alpha_mode}

    # ---- metallic / roughness ---------------------------------------------
    if sg:
        gloss_factor = float(sg.get("glossinessFactor", 1.0))
        if sg_spec_arr is not None:
            gloss = R(sg_spec_arr)[..., 3] * gloss_factor
            res.sources["roughness"] = {**sg_spec_desc, "note": "1 - specGloss alpha"}
        else:
            gloss = derive.constant(size, gloss_factor, 1)
            res.sources["roughness"] = {"constant": 1 - gloss_factor}
        res.maps["roughness"] = 1.0 - gloss
        res.maps["metallic"] = derive.constant(size, 0.0, 1)
        res.sources["metallic"] = {"constant": 0.0, "note": "specGloss workflow"}
    elif mr_arr is not None:
        mr = R(mr_arr)
        res.maps["roughness"] = mr[..., 1] * roughness_factor
        res.maps["metallic"] = mr[..., 2] * metallic_factor
        res.sources["roughness"] = {**mr_desc, "channel": "G"}
        res.sources["metallic"] = {**mr_desc, "channel": "B"}
        # packed ORM: R carries occlusion when occlusionTexture points at the same image
        if occ_desc and occ_desc["image"] == mr_desc["image"]:
            res.flags["packedORM"] = True
    else:
        res.maps["roughness"] = derive.constant(size, roughness_factor, 1)
        res.maps["metallic"] = derive.constant(size, metallic_factor, 1)
        res.sources["roughness"] = {"constant": roughness_factor}
        res.sources["metallic"] = {"constant": metallic_factor}

    # ---- occlusion --------------------------------------------------------
    if occ_arr is not None:
        occ = R(occ_arr)[..., 0]
        res.maps["ao"] = 1.0 + occ_strength * (occ - 1.0)
        res.sources["ao"] = {**occ_desc, "channel": "R", "strength": occ_strength}
    elif opts.fill_missing:
        res.maps["ao"] = derive.constant(size, 1.0, 1)
        res.sources["ao"] = {"constant": 1.0, "note": "no occlusionTexture in glTF"}

    # ---- normal -----------------------------------------------------------
    if nrm_arr is not None:
        n = R(nrm_arr)[..., :3]
        if opts.apply_normal_scale:
            n = derive.apply_normal_scale(n, normal_scale)
        res.maps["normal_gl"] = n
        res.maps["normal_dx"] = derive.normal_gl_to_dx(n)
        res.sources["normal_gl"] = {**nrm_desc, "convention": "OpenGL (+Y), as stored in glTF", "scale": normal_scale}
        res.sources["normal_dx"] = {"from": "normal_gl", "convention": "DirectX (-Y), green inverted"}
    elif opts.fill_missing:
        flat = derive.constant(size, [0.5, 0.5, 1.0], 3)
        res.maps["normal_gl"] = flat
        res.maps["normal_dx"] = flat.copy()
        res.sources["normal_gl"] = {"constant": [0.5, 0.5, 1.0]}
        res.sources["normal_dx"] = {"constant": [0.5, 0.5, 1.0]}

    # ---- emissive ---------------------------------------------------------
    if em_arr is not None:
        res.maps["emissive"] = R(em_arr)[..., :3] * emissive_factor
        res.sources["emissive"] = {**em_desc, "factor": emissive_factor.tolist(), "strength": emissive_strength}
    elif np.any(emissive_factor > 0):
        res.maps["emissive"] = derive.constant(size, emissive_factor, 3)
        res.sources["emissive"] = {"constant": emissive_factor.tolist(), "strength": emissive_strength}

    # ---- transmission / volume -------------------------------------------
    transmission = None
    if tr:
        tf = float(tr.get("transmissionFactor", 0.0))
        transmission = R(tr_arr)[..., 0] * tf if tr_arr is not None else derive.constant(size, tf, 1)
        res.maps["transmission"] = transmission
        res.sources["transmission"] = {**(tr_desc or {}), "channel": "R", "factor": tf}
        res.factors["transmissionFactor"] = tf
    if vol:
        thf = float(vol.get("thicknessFactor", 0.0))
        res.maps["thickness"] = R(th_arr)[..., 1] * thf if th_arr is not None else derive.constant(size, thf, 1)
        res.sources["thickness"] = {**(th_desc or {}), "channel": "G", "factor": thf}
        res.factors["thicknessFactor"] = thf
        res.factors["attenuationColor"] = vol.get("attenuationColor", [1, 1, 1])
        res.factors["attenuationDistance"] = vol.get("attenuationDistance")

    # ---- specular ---------------------------------------------------------
    if spec:
        sf = float(spec.get("specularFactor", 1.0))
        res.maps["specular"] = R(sp_arr)[..., 3] * sf if sp_arr is not None else derive.constant(size, sf, 1)
        res.sources["specular"] = {**(sp_desc or {}), "channel": "A", "factor": sf}
        scf = np.asarray(spec.get("specularColorFactor", [1, 1, 1]), dtype=np.float32)
        res.maps["specular_color"] = R(spc_arr)[..., :3] * scf if spc_arr is not None else derive.constant(size, scf, 3)
        res.sources["specular_color"] = {**(spc_desc or {}), "factor": scf.tolist()}
        res.factors["specularFactor"] = sf
        res.factors["specularColorFactor"] = scf.tolist()

    # ---- clearcoat --------------------------------------------------------
    if cc:
        cf = float(cc.get("clearcoatFactor", 0.0))
        crf = float(cc.get("clearcoatRoughnessFactor", 0.0))
        res.maps["clearcoat"] = R(cc_arr)[..., 0] * cf if cc_arr is not None else derive.constant(size, cf, 1)
        res.maps["clearcoat_roughness"] = R(ccr_arr)[..., 1] * crf if ccr_arr is not None else derive.constant(size, crf, 1)
        res.sources["clearcoat"] = {**(cc_desc or {}), "channel": "R", "factor": cf}
        res.sources["clearcoat_roughness"] = {**(ccr_desc or {}), "channel": "G", "factor": crf}
        if ccn_arr is not None:
            res.maps["clearcoat_normal"] = R(ccn_arr)[..., :3]
            res.sources["clearcoat_normal"] = ccn_desc
        res.factors["clearcoatFactor"] = cf
        res.factors["clearcoatRoughnessFactor"] = crf

    # ---- sheen ------------------------------------------------------------
    if sh:
        scf = np.asarray(sh.get("sheenColorFactor", [0, 0, 0]), dtype=np.float32)
        srf = float(sh.get("sheenRoughnessFactor", 0.0))
        res.maps["sheen_color"] = R(shc_arr)[..., :3] * scf if shc_arr is not None else derive.constant(size, scf, 3)
        res.maps["sheen_roughness"] = R(shr_arr)[..., 3] * srf if shr_arr is not None else derive.constant(size, srf, 1)
        res.sources["sheen_color"] = {**(shc_desc or {}), "factor": scf.tolist()}
        res.sources["sheen_roughness"] = {**(shr_desc or {}), "channel": "A", "factor": srf}
        res.factors["sheenColorFactor"] = scf.tolist()
        res.factors["sheenRoughnessFactor"] = srf

    # ---- derived maps -----------------------------------------------------
    if opts.derive_subsurface:
        res.maps["subsurface"] = derive.subsurface_weight(res.maps["metallic"], transmission, opts.sss_gain)
        res.maps["subsurface_radius"] = derive.subsurface_radius(res.maps["basecolor"], opts.sss_tint)
        res.sources["subsurface"] = {"derived": "(1 - metallic) * (1 - transmission) * sss_gain", "gain": opts.sss_gain,
                                     "note": "glTF carries no subsurface data; heuristic starting point"}
        res.sources["subsurface_radius"] = {"derived": "mix((1,0.2,0.1), baseColor, sss_tint)", "tint": opts.sss_tint}
    if opts.derive_height and "normal_gl" in res.maps and nrm_arr is not None:
        mask = None
        if opts.uv_mask is not None:
            mask = np.asarray(opts.uv_mask, dtype=np.float32)
            if mask.max() > 1.0:
                mask = mask / 255.0
        res.maps["height"] = derive.height_from_normal(res.maps["normal_gl"], mask=mask, highpass=opts.height_highpass)
        res.sources["height"] = {"derived": "Frankot-Chellappa integration of normal_gl", "bits": 16,
                                 "masked_to_uv_islands": mask is not None, "highpass_sigma_frac": opts.height_highpass}
    if opts.derive_curvature and "normal_gl" in res.maps:
        res.maps["curvature"] = derive.curvature_from_normal(res.maps["normal_gl"])
        res.sources["curvature"] = {"derived": "divergence of normal XY"}
    if opts.write_orm:
        ao = res.maps.get("ao", derive.constant(size, 1.0, 1))
        res.maps["orm"] = np.stack([ao, res.maps["roughness"], res.maps["metallic"]], axis=-1)
        res.sources["orm"] = {"packed": "R=occlusion G=roughness B=metallic (glTF/Unreal/Unity HDRP order)"}

    return res


def write_maps(res: MaterialResult, out_dir: str) -> dict:
    """Write every map as PNG. Returns map id -> file record (for the manifest)."""
    os.makedirs(out_dir, exist_ok=True)
    records = {}
    for map_id, arr in res.maps.items():
        suffix, colorspace, channels, blender_socket, sd_usage, sd_components = MAP_SPECS[map_id]
        fname = f"{res.safe_name}_{suffix}.png"
        bits = 16 if channels == "L16" else 8
        img = derive.to_image(arr, bits=bits)
        img.save(os.path.join(out_dir, fname), optimize=False, compress_level=4)
        records[map_id] = {
            "file": fname,
            "colorspace": colorspace,
            "channels": "L" if channels == "L16" else channels,
            "bits": bits,
            "blender_socket": blender_socket,
            "substance_usage": sd_usage,
            "substance_components": sd_components,
            "source": res.sources.get(map_id, {}),
        }
    return records
