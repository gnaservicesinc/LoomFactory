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
