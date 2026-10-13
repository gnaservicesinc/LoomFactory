# Loom Factory runtime

This runtime is a new implementation. It trains every layer of a material-map
network from random initialization. Image classifiers are trained separately
in Create ML and imported into Loom Factory. The runtime imports no
previous Texture Studio training modules or pretrained map weights.

The app provisions a private Python 3.12 environment with `setup_runtime.sh
<source-folder>`. Python, dependencies, training requests, progress and outputs
all live in `<source-folder>/.loom-factory`. Source images may be symlinks; they
are decoded in place and never resized or rewritten for map training.
PNG/TIFF16-bit scalar and RGB data are read at native precision. HEIC/HEIF
decode without forcing HDR sources to 8-bit. EXR flat scalar targets retain
their float values; target values must use the `[0,1]` deployment range. EXR
diffuse linear-light pixels are converted to the sRGB input contract.

## App protocol

Run the returned Python executable with `PYTHONPATH` set to this Runtime folder:

```
python -m loom_factory train-map --request /source/.loom-factory/runs/UUID/request.json
```

Stdout contains JSON lines with `event`: `status`, `started`, `progress`,
`completed`, `cancelled`, or `error`. Diagnostics use stderr. SIGINT/SIGTERM
request cooperative cancellation. Completed runs publish `model.json`,
`model.safetensors` and `model.mlpackage`. A cancelled run never publishes a
completed `model.json`.
Each completed epoch saves `checkpoint.safetensors` and `checkpoint.json`;
cancellation also saves current finite weights. `resume_weights` can point to
those weights in the same source project's runs. Continuation requires the
same project, dataset, map, architecture, crop size and depth-transform setting.
It trains current folder members using a fresh optimizer. Checkpoints are never native
deployment artifacts and cannot be assembled as completed models.

Request schema `loom-factory-training-v1` requires `source_root`, `project_id`,
`dataset_id`, and `output_directory` inside that source's `.loom-factory` folder.
Map `samples` contain `{id,input,target}` and optional `normal_convention`
(`OpenGL` or `DirectX`). `map_kind` is `depth`, `roughness`, or `normal`.
The app constructs each request from the current physical folder and image metadata.
Requests describe a run; they are never a source of dataset membership. The
app sets `require_image_metadata=true`; the runtime rechecks each file's current
dataset tag, review state, map role and pairing metadata before using it and
before exporting. A tag change stops the run and preserves the source images.

Training options: `image_size` (native square crop; default1024), `epochs`
(10), `batch_size` (1), `learning_rate` (0.001), `width` (24),
`blocks` (4), `shuffle_each_epoch` (true), `flip_crops` (true),
`rotate_crops` (false), depth-only `exaggerate_features` (false), `seed` (42),
and `device` (`auto`, `mps`, `cpu`). Crop positions are random whenever a larger
native source permits multiple positions. Disabling shuffle preserves the
sample order each epoch; it does not disable crop augmentation.

Production map training keeps a run-local decoded-pixel cache capped at 256 MiB.
The backend request option `image_cache_megabytes` accepts 0…2048 (0 disables it).
Oversized images are decoded without being retained; this never loads the entire
source collection into memory. Every cache access checks the current link and
physical file identity, and every batch still checks current image tags. Random
crop positions, flips and rotations remain fresh. Original-byte verification,
finite-loss/gradient checks and checkpoints remain enabled. A single small GPU
transfer reports loss and checks both loss and gradient norm before each update;
no anomaly detector or profiler runs in normal training.

## Deployment contract

Core ML map input `rgb`: float32 multiarray `[1,3,H,W]`, RGB sRGB pixels
normalized exactly once to `[0,1]`. Flexible H/W range32…8192. Output
`material_map`: float32 `[1,1,H,W]` depth/roughness or `[1,3,H,W]` OpenGL normal,
all `[0,1]`. Depth/roughness targets preserve original integer code values:
8-bit /255, 16-bit /65535, float targets already `[0,1]`. No per-image
min/max normalization occurs. Exaggeration applies an anchored tanh contrast
curve in memory to depth training targets; originals remain untouched.

Safetensors holds the exact PyTorch state dictionary for future adaptation;
`model.json.architecture` names the architecture and `architecture_config`
records its width, block count and channel count.
Core ML conversion uses Apple's documented PyTorch conversion API:
https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html

Successful export proves execution and compatibility, not unseen-material
quality. Users should inspect model outputs and train for a suitable number of
epochs before bundling a deployment.

## Tests

```
<project-runtime-python> -m unittest discover -s RuntimeTests -v
```

Tests train real small networks, compare native Core ML predictions with the
saved PyTorch weights, verify paired native16-bit crops, fixed/shuffled sample
order, cancellation, and physical source membership. The CLI rejects classifier
training commands.

`RuntimeTests/benchmark_pixel_loading.py` measures repeated 1024-pixel paired
crops from disposable 2048-pixel sources, verifies exact pixels/crop order, and
reports cache memory. Its timing is pixel preparation, not model-training speed.
