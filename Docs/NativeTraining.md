# Native training

Loom Factory and Loom Workshop use the `LoomNative` Swift library. It implements
the registered `loom-factory-material-pyramid-v1` and
`loom-factory-material-residual-v1` networks with Metal/MPSGraph autodiff and
GPU-resident AdamW, Adam, SGD and Lion updates. Full-model training
updates every weight; Workshop freezes the exact installed base and learns only
low-rank convolution deltas. Safetensors names and OIHW layouts remain compatible
with previously saved models.

Fresh training defaults to Pyramid: three feature scales with widths w, 2w and
4w, residual blocks at quarter resolution, and skip-connected nearest-neighbor
decoders. The width-38, four-block scalar model has 1,926,373 parameters. Most
computation runs on smaller feature maps; original image crops remain exact.
The output retains the same sRGB RGB input, scalar/normal output and Float32
precision contract. The network picker also permits the original residual
architecture. Saved identities are checked rather than inferred from width or
parameter count, and unknown or contradictory descriptors are rejected.

Training runs outside the project actor so the Stop action can signal a locked
cancellation token immediately. Pixel decoding, conversion and crop loops check
the token, as do GPU boundaries. An already submitted GPU operation finishes
before cancellation returns; the latest completed finite update is checkpointed.
A cancelled run never publishes a completed model descriptor. Factory also saves
the first optimizer update before the end of the first epoch.

ImageIO reads native integer or floating-point pixels without drawing into an
8-bit bitmap or rewriting original files. Crops stay aligned and retain native
dimensions. Input conversion, normal-coordinate transforms and the optional
depth transform affect only training tensors. A bounded decoded-image cache
avoids repeated work where possible. Current metadata and source identity are
checked per use, and original byte hashes are verified before publishing.
The decoded numeric cache uses one eighth of physical memory, bounded between
512 MiB and 8 GiB, and is released at the end of a run. Cached pixels remain
keyed by source file identity and modification time, with validation retained
only while those unchanged pixels remain cached.

Core ML export serializes the selected graph and float32 weights in Swift.
Apple Core ML compiles the exported `.mlmodel`, and numerical tests compare native
predictions with the Core ML graph, including scalar and normal products.
Pyramid uses dynamic ND slicing to preserve odd and rectangular inference sizes.
Its native resize has fixed graph dimensions and a supported GPU gradient.

Training keeps the established synchronous graph path and explicit owned,
initialized buffers for gradients and optimizer state. An experiment combining
initialization with command-buffer encoding was rejected after the real 1K
gradient regression diverged. Internal NHWC/HWIO convolutions improve measured
residual throughput; Pyramid retains NCHW because it was faster in measurements.
The optional probe exposes execution timings and layout switches without adding
those details to training progress.

Existing `.loom-factory/runtime` folders are unused by the app and preserved in
place. No interpreter or Python worker is shipped, installed or launched.
CPU training is unavailable in this GPU backend; a saved CPU preference receives
a clear error and can be changed to Automatic or Apple GPU in the controls.

`swift test` verifies learning, independent numerical gradients, optimizer
equations, accumulation, true frozen-base LoRA fusion, cancellation recovery,
image precision and native export. `LoomNativeProbe` is an optional Release
benchmark tool. Execution tests and timings do not qualify learned output on
unseen materials. Large-grid performance must be measured at the intended
architecture and image dimensions.
