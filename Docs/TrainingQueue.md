# Bounded material training queue

Build the native runner with `swift build -c release --product LoomTrainingQueue`.
It uses the same source scan, pairing, trainer, checkpoints, model tests and
Core ML export as Loom Factory. The app may be closed while it runs. Do not run
another GPU workload while measuring its throughput.

`LoomTrainingQueue --inventory SOURCE OUTPUT.json` creates a diagnostic receipt
from current files. It includes one complete input and one common whole-material
holdout for depth, normal and roughness, using native 1024 crops, one source
resolution level, 10% validation and validation seed 9587. Training always scans
current source membership again; the receipt is never authoritative membership.

`LoomTrainingQueue --run CONFIGURATION.json` reads a version-one queue:

```json
{
  "schema": "loom-quality-training-queue-v1",
  "sourceFolder": "/path/to/originals",
  "deadline": "2026-10-16T12:00:00Z",
  "jobs": [{
    "id": "bricks-depth",
    "category": "Bricks",
    "datasetID": "current-category-id",
    "map": "depth",
    "settings": { "imageSize": 1024 },
    "seconds": 1200,
    "parentRunID": null,
    "evaluationInput": "/path/to/original-diffuse.png",
    "evaluationIsHeldOut": true
  }],
  "renderer": "/absolute/path/to/Scripts/render-material-comparison.py"
}
```

Provide explicit, validated `TrainingSettings` for a real run. Omitted legacy
fields keep their decoding defaults. `evaluationIsHeldOut` is a receipt label;
set it only for the actual shared split and never for an inherited model that
has already seen that material. All three jobs for a category must use the same
evaluation input and validation split.

Jobs run sequentially. The time limit requests Save & Stop, preserving a real
update and optimizer checkpoint. SIGINT, SIGTERM or an empty `STOP` file beside
the configuration request the same save. The deadline bounds training; an
in-flight save/export/render must still finish. The process lock excludes both
another queue and GUI GPU operations until the owner finishes or exits.

`Progress.html`, atomic `status.json`, and `results.json` are saved beside the
configuration. Checkpoints, Float32 tests, and separate Core ML candidates stay
in the source project's `.loom-factory`. Exporting a checkpoint candidate does
not change explicit production assignments. The queue first pins every existing effective reference, including legacy
newest-model fallbacks, and saves reference-models.json. Categories with no
existing export gain their first review candidate. The runner does not approve quality or publish a package.

Optional joint reviews require Blender at `/Applications/Blender.app`, Python
with NumPy/Pillow, and ffmpeg. They save unchanged map references, native numeric
metrics, and rendered source/candidate materials with the same camera, lights,
normal convention and height scale. Diffuse and neutral views expose all three
maps together. Rendering uses four CPU threads; the trainer owns the GPU.
The fixed displacement scale is an artistic comparison, not measured relief.

`--select-checkpoint SOURCE DATASET MAP RUN_ID` exports and explicitly selects
that exact saved checkpoint; `--select-model` selects an existing export.
Use these only for an intentional reference selection. All earlier weights and
checkpoints remain available for comparison.
