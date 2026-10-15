# Native training

Loom Factory and Loom Workshop use the `LoomNative` Swift library. It implements
the registered `loom-factory-material-pyramid-v1` and
`loom-factory-material-residual-v1` networks with Metal/MPSGraph autodiff and
GPU-resident AdamW, Adam, SGD and Lion updates. Full-model training
updates every weight; Workshop freezes the exact installed base and learns only
low-rank convolution deltas. Safetensors names and OIHW layouts remain compatible
with previously saved models.

Fresh training defaults to the full-resolution residual network, width 164 and
four blocks (1,943,893 scalar parameters). Every feature layer retains the input
pixel grid. Selecting a fresh run from saved Pyramid settings switches to this
network at approximately the same parameter count; the original weights cannot
be continued under a different architecture and remain preserved.

Pyramid remains an explicit experimental choice for existing versions and A/B
work: three scales with widths w, 2w and 4w, residual blocks at quarter
resolution, and skip-connected nearest-neighbor decoders. Its saved brick models
show phase-dependent depth changes under one-pixel crop shifts, including after
detail-loss fine-tuning. It is not the default for geometry-sensitive maps.
Original image crops are exact under both architectures; this sampling is inside
the learned network. Saved identities and output contracts remain unchanged,
and unknown or contradictory descriptors are rejected.

Scalar training has an explicit, independently versioned loss contract. Fresh
Factory depth and roughness runs default to `logit-cross-entropy-v2`: soft-target binary
cross-entropy computed directly from pre-sigmoid logits. Its derivative is
`sigmoid(logit) - target`, preserving a teaching signal when an incorrect scalar
prediction has saturated to white or black. Its stable expression multiplies
the positive and negative parts of the logit by `(1-target)` and `target`, then
adds `log(1+exp(-abs(z)))`. This avoids subtracting large products near target
endpoints; its exponential cannot overflow and its derivative is correct at zero.
Targets retain their original numeric values. Normal products keep their
existing cosine loss.

Training settings, run requests, checkpoints and exports record `scalar_loss`.
Absent loss identities mean legacy `absolute-v1` (mean absolute error after
sigmoid), and continuing a saved run preserves its recorded objective unless the
user explicitly changes **Training objective**. The loss
version changes training behavior without changing either registered inference
architecture, checkpoint tensor shapes or Core ML output interpretation. V2 loss
values include target entropy and must not be compared numerically with V1 L1
values. Original saved runs and source images remain untouched.

The opt-in **Native pixel detail (v3, experimental)** objective is
`native-detail-v3`: v2's stable logit cross-entropy plus twice the arithmetic
mean of the signed native difference errors at offsets 1, 2 and 4 along both
image axes. Each error is `mean(abs((prediction[b]-prediction[a]) -
(target[b]-target[a])))` over all batches, channels and valid pixel positions.
Offsets that do not fit a dimension are omitted; a 1×1 map has no detail term.
Targets are neither resized nor filtered, and output images are not sharpened.
The level term retains corrective gradients when sigmoid detail gradients
vanish at saturation. Normal-map training is unchanged.

Selecting a different objective on saved weights creates a separate run with a
fresh optimizer and epoch schedule; old optimizer moments, best-validation
selection and partial-epoch cursor are not reused across objectives. Lineage,
cumulative work and earlier metrics are preserved, with each earlier loss
labelled by its actual objective. `trainMap` callers opt in with
`scalarLossOverride`; omitting it retains the existing default/resume behavior.
V3 is a candidate for A/B inspection, not a guarantee of artifact-free depth or
roughness. Evaluate whole unseen materials and the intended displacement use.

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
independently for at most 4,096 unchanged file identities. Evicting decoded
pixels does not require rescanning a successfully validated whole image. Both
caches are released at the end of the run.

Two upcoming batches can prepare concurrently while the serial owner trains
on the GPU. Running and completed preparation jobs share a two-slot limit;
results and errors are consumed in the original submission order. Crop and
augmentation choices are still resolved on the serial owner, and optimizer
updates retain the same ordering. Cancellation joins both preparation workers
before clearing decoded pixels.

New training settings use seed `-1`, resolved once from
`SystemRandomNumberGenerator` before the run is recorded. Requests, run metadata,
checkpoints and model descriptors contain the resolved nonnegative seed. An
explicit seed remains reproducible; missing legacy seeds retain their seed-zero
interpretation. Continuing with `-1` inherits the saved seed. The validation seed
remains explicit so architecture comparisons can keep a common material split.

Crop coordinates are uniform integer native origins over `0...(width-size)` and
`0...(height-size)`, including both edges. Input and target share the same crop;
their refresh follows the selected visit/epoch/run cadence. Source pair selection
still contributes one crop visit per selected material resolution and paired
diffuse variant, without enumerating extra crop positions.

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

`NativeScalarLossTests` verifies analytic corrective gradients for saturated
white and black predictions, zero and extreme logits, unchanged normal training,
and exact optimizer-state continuation. Its saved-failure regression is opt-in:
set `LOOM_NATIVE_ROUGHNESS_RUN_DIRECTORY` to an exported collapsed roughness run,
`LOOM_NATIVE_ROUGHNESS_INPUT` and `LOOM_NATIVE_ROUGHNESS_TARGET` to its untouched
1K teaching pair, then run `swift test --filter NativeScalarLossTests`. The test
uses a 64-pixel diagnostic crop and updates only in-memory copies of the saved
weights. This proves a recoverable teaching signal, not whole-material quality.
