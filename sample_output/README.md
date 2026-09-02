# Sample output — Fleshbound Navigator (Meshy)

Produced by `python -m glb2pbr Meshy_AI_Fleshbound_Navigator_0902135711_texture.glb --mesh none --curvature`.
The OBJ is omitted here (375 MB); regenerate with `--mesh obj` if you want Designer's 3D view to show the real mesh,
or link the original GLB in Designer via Link > 3D Scene.

Rebuild the material:

    blender -b --python Fleshbound_Navigator_pbr/blender_build_material.py -- \
        --manifest Fleshbound_Navigator_pbr/manifest.json --glb <path to the .glb> --save check.blend

    python Fleshbound_Navigator_pbr/substance_build_sbs_pysbs.py Fleshbound_Navigator_pbr/manifest.json --n2h
