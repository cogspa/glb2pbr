"""
CLI:  python -m glb2pbr model.glb -o out/ [--size 2048] [--mesh obj|copy|none] [--no-uv] ...
"""
from __future__ import annotations

import argparse
import sys

from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glb2pbr",
                                description="Unwrap a textured .glb/.gltf into Principled BSDF / Substance Designer PBR maps.")
    p.add_argument("input", help=".glb or .gltf file")
    p.add_argument("-o", "--out", default=None, help="output folder (default: <input>_pbr)")
    p.add_argument("--size", type=int, default=None, help="output map size (default: largest source texture)")
    p.add_argument("--mesh", choices=["none", "obj", "copy"], default="obj",
                   help="also export the mesh as OBJ (V flipped, for Substance 3D view), copy the GLB, or skip")
    p.add_argument("--no-uv", action="store_true", help="skip UV layout renders")
    p.add_argument("--uv-size", type=int, default=None, help="UV layout render size (default: map size)")
    p.add_argument("--no-fill", action="store_true", help="don't write constant maps for factor-only channels")
    p.add_argument("--no-height", action="store_true", help="skip height-from-normal")
    p.add_argument("--no-subsurface", action="store_true", help="skip derived subsurface maps")
    p.add_argument("--curvature", action="store_true", help="also write a curvature map")
    p.add_argument("--no-orm", action="store_true", help="skip packed ORM export")
    p.add_argument("--sss-gain", type=float, default=1.0, help="subsurface weight multiplier (default 1.0)")
    p.add_argument("--sss-tint", type=float, default=0.75, help="0 = Blender default radius, 1 = base colour (default 0.75)")
    p.add_argument("--height-highpass", type=float, default=1.0 / 24.0,
                   help="high-pass sigma as a fraction of map size for the derived height (0 = off, default 1/24)")
    p.add_argument("--no-helpers", action="store_true", help="don't copy the Blender/Substance helper scripts into the output")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out or (args.input.rsplit(".", 1)[0] + "_pbr")
    try:
        run(args.input, out, size=args.size, mesh=args.mesh, uv=not args.no_uv, uv_size=args.uv_size,
            fill_missing=not args.no_fill, height=not args.no_height, subsurface=not args.no_subsurface,
            curvature=args.curvature, sss_gain=args.sss_gain, sss_tint=args.sss_tint, orm=not args.no_orm,
            height_highpass=args.height_highpass, copy_helpers=not args.no_helpers)
    except Exception as e:  # noqa: BLE001
        print(f"[glb2pbr] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
