# Map-type data for Create ML

Export one original image of each recognized map type per material folder:

```sh
swift Scripts/export-map-type-dataset.swift --source ./source-materials --output ./map-type-dataset
```

Supply your source and export folders explicitly. Use `--dry-run` to print the
selection without writing anything. The utility uses saved Loom Factory image tags, not
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

# Test Model → Export to Blender

Choose any saved model or checkpoint and a diffuse image on **Test Model**, then
**Run Model → Export to Blender…**. Test other map types on the same unchanged
input to make them available together. The export options list each exact tested
version; choose the maps to combine and optionally **Add source material on a
second plane**. Uniquely paired native source maps are used when available;
missing/ambiguous maps are recorded and use flat height/normal or roughness .85.
External images work without paired source maps. Changing the input/category
clears the accumulated test results. Export never runs another model implicitly.

The new export folder contains `Material Review.blend`, unchanged native map
copies, `review.json`, `export-receipt.json`, a background Blender log, and a
retry script. All images and the studio environment are packed. The model names
and exact run/checkpoint identities are recorded on objects/materials and texture
nodes. One or two simple quads use one UVMap each, shared UV/Mapping coordinates,
Non-Color numeric maps, OpenGL generated normals and the explicit convention of
source normals. Shader displacement is .025 with midpoint .5. Missing maps never
silently substitute another model. No dense displaced mesh is baked.

Nine Cycles/Eevee scenes follow the reviewed settings: world .5, CPU Auto-Detect,
Blackman-Harris pixel width1, Simple adaptive subdivision, map-isolation and
matte controls. The no-denoise Cycles scene is explicit. Eevee is a shading
diagnostic; use Cycles for adaptive geometry. Export saves without rendering or
changing model approvals/selections. Blender.app is located in Applications or
chosen by the user. The bundled Python file calls only Blender's API inside
that separate process; native model training/inference have no Python runtime.

If Blender export fails, exact maps and the manifest remain available. Retry:

```sh
blender --background --factory-startup --disable-autoexec --python-exit-code 1 \
  --python /path/to/export/BlenderExport.py -- /path/to/export/review.json
blender --background --factory-startup --disable-autoexec --python-exit-code 1 \
  --python Scripts/verify-blender-export.py -- /path/to/export
```

# Historical Blender material review from a user template

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
