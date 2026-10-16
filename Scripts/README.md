# Map-type data for Create ML

Export one original image of each recognized map type per material folder:

```sh
swift Scripts/export-map-type-dataset.swift --source /opt/ipde/sources_mats --output /opt/ipde/mat_type_ds
```

Use `--dry-run` to print the selection without writing anything. The defaults
are the two paths above. The utility uses saved Loom Factory image tags, not
filename guesses or the category classifier. It reads native dimensions with
ImageIO without decoding full images or installing Python dependencies.

Each material contributes at most one image to each of `diffuse`, `depth`,
`rough` and `normal`. Selection prefers native 1K, then 2K (measured by
the shorter side, including rectangular maps),
then the smallest available larger image; if only smaller images exist, it uses
the largest. DirectX normals, ignored files, unknown types and untagged files
are excluded. Bump maps are excluded as well and are never depth examples.

The output contains symbolic links to the originals. Keep those originals in
place until Create ML finishes. Sources are neither resized nor rewritten.
`map-type-export-report.json` records the selection and counts outside the
class folders. Re-running updates only links owned by this export and removes
its obsolete links. Unrelated output files are preserved. A fresh run with the
same files creates no duplicate examples. This utility is an explicit export;
it does not retrain, install or connect a map-type classifier to the apps.

# Blender material review from a user template

`build-blender-template-review.py` creates self-contained `.blend` comparisons
from a saved example and an existing artistic-review configuration. It preserves
the example's simple quad, UV/Mapping chain, shader displacement, normal-map
basis, subdivision modifier and both render-engine configurations. Four native
1K maps are packed byte-for-byte; no dense mesh, OBJ or resampled map is created.
The world uses strength .5, Cycles uses Blackman-Harris filter width1, and CPU
threads use Auto-Detect. Matching scenes retain the example's denoising; a
separate labeled Cycles scene disables it. Additional scenes isolate normal,
height and roughness contributions and provide a constant .85 roughness control.

```sh
blender --background --factory-startup --python Scripts/build-blender-template-review.py -- \
  --template /path/to/example.blend --review /path/to/artistic-review \
  --output /path/to/new-blender-review
blender --background --factory-startup --python Scripts/verify-blender-template-review.py -- \
  /path/to/new-blender-review
```

Use a new output directory; existing scene files are never overwritten. The
verification reopens every file, checks all scene controls, exact packed map
hashes, topology/UVs/modifiers, mapping, material texture settings, world strength,
Eevee settings and Cycles settings. It does not claim live UI acceptance or
learned material quality. The supplied template must have a single quad, one UV
layer, direct map connections to its Principled shader, tangent Normal Map node
and Displacement node. Original source images and example files stay unchanged.
