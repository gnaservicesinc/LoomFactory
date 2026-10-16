# Training mathematics and model lifecycle validation

Native training inspection used the saved source project at
`/path/to/workspace/sources_mats/.loom-factory`. Original images, saved requests,
checkpoints, optimizer states and models were read without rewriting them.
Receipts and diagnostic sources are outside the repositories under
`~/Library/Application Support/Loom Factory/Validation/2026-10-15-training-math-audit`.
The four temporary standalone executables were removed after their results were recorded.

## Saved artifacts and the reported flat loss

The inspected checkpoint hashes match their recorded SHA256 values. All saved
weight and optimizer tensors are finite Float32; all second moments are
nonnegative.

| Run | Constructor | Saved updates | Objective | Relevant settings |
|---|---|---:|---|---|
| `2358e442` | Residual v1, width 84, four blocks | 897 | Logit cross entropy v2 | LR 0.001, no target exaggeration |
| `3ef8e93a` | Compact NAF, width 32 | 2,804 | Linear detail v1 | LR 0.0001, no target exaggeration |
| `8c415268` | Compact NAF, width 32 | 5,600 | Linear detail v1 | LR 0.0001, target exaggeration enabled |

Their raw losses are not comparable across objectives. For soft scalar target
`y`, logit cross entropy is `H(y) + KL(y || prediction)`. Even a perfect
prediction has the positive target entropy `H(y)`, except at endpoints. Exact
NativeImages Float32 decoding of the native 1K subset of `2358e442` measured:

| Baseline on the native 1K subset | Cross entropy |
|---|---:|
| Perfect per-pixel prediction | 0.557571 |
| One constant global-mean prediction | 0.629294 |

The subset's global target mean is 0.676752. These baselines are not the exact
random crop/all-resolution training schedule, but establish why a loss near 0.6
cannot be interpreted as 0.6 pixel error or as proof of no updates. The saved
residual run's epoch 16 loss was 0.635462, near this constant baseline. The
receipt is `target-entropy-baselines.json`.

The Bark run deliberately applies the selected per-crop depth exaggeration:
its teaching values are remapped with a monotone tanh curve within that crop's
original minimum/maximum. Those targets differ from the unmodified depth codes;
the original files remain unchanged. The other two inspected runs disable it.

## Hidden teaching paths became inactive

An independent NumPy Float32 forward calculation used exact 64 × 64 crops
exported by NativeImages at native coordinates (480, 480). It implemented the
saved OIHW cross correlations, zero borders, residual sums, ReLU, nearest
replication and sigmoid directly. No GPU session or optimizer update ran during
this audit. Crop boundary fields can differ from whole-image predictions.

For residual `2358e442`, the third block's first hidden activation was zero at
99.918% of values on brick_wall_006, 99.947% on red_brick_03 and 99.955% on
brick_wall_10, including the crop boundaries. Between 72 and 75 of its 84
channels were zero throughout those entire tested fields. Neighboring blocks
also had very sparse hidden activations. The output logits were modest, rather
than at sigmoid saturation.

For Pyramid continuations `395fb421` and `5493b91f`, the final deep residual
activation was exactly zero in all 152 channels on the red_brick_03 crop. The
earlier parent `80bffde0` still had 18 nonzero channels at that point on the same
crop. The saved `5493b91f` body first-moment RMS values were approximately
`1e-18`, while input/output first-moment RMS values were 0.00603 and 0.00448.
The receipt is `cpu-hidden-activations.json`.

ReLU's derivative is zero for a negative hidden preactivation. An inactive
branch therefore loses its teaching signal even when an output bias or a skip
path still produces a healthy global gradient norm. Stable logit cross entropy
corrects an incorrectly saturated scalar head; it cannot restore a hidden path
whose ReLU derivative is zero. The measured branch inactivity is a concrete
failure mechanism on these saved crops. It does not establish that every
material or every reported artifact has this cause.

## Native arithmetic and data-path inspection

The reviewed graph declares Float32 inputs, targets, parameters, accumulators,
optimizer moments and results. No model-side half-float cast was found. Native
source/target crops are exact pixel selections; the reviewed scalar path does
not resize, gamma-correct, normalize or quantize the targets. Linear EXR diffuse
input is converted to the documented sRGB model-input interpretation; this
conversion does not apply to scalar teaching maps.

Accumulation weights each batch mean gradient by its sample count divided by
the actual accumulation-window sample count, including partial final windows.
Global norm clipping precedes optimizer arithmetic. Reviewed Adam/AdamW bias
correction and moments, SGD momentum, Lion, learning-rate scheduling and reset
of accumulated gradients match their implemented contracts. Persistent graph
results retain separate owned buffers, avoid active feed buffers, and preserve
the existing initialization and autorelease boundaries. This source inspection
and finite saved tensors do not prove every GPU execution is correct; focused
numerical regressions below cover the changed path.

## Separate residual v2 constructor

`loom-factory-material-residual-v2` uses a 0.1 negative slope at the original
residual hidden activation sites. It keeps full-resolution convolutions, kernel
padding, Float32 storage, parameter shapes and seeded initialization, and the
existing sigmoid/output interpretation. The constructor has a distinct identity
and records its hidden activation and slope. It does not reinterpret existing
residual v1, Pyramid or Compact saved models.

Six focused native tests passed, covering independently calculated negative
forward values and analytic gradients in both layouts, finite differences
through negative branches, a finite nonzero path at zero crossings, optimizer
continuation, all three products' Core ML parity on odd rectangles, and unchanged
legacy graph bytes. The legacy Core ML network-field hashes were independently
captured from the pre-change `fccff21` constructors; all three old constructors
remain byte-identical under that fixture.

A fresh fixed synthetic pixel relation trained for 120 updates at LR 0.0001,
width eight and two blocks. Residual v2 MAE fell from 0.178983 to 0.119077 and
cross entropy from 0.692054 to 0.646262; hidden gradient RMS remained nonzero at
the measured updates. Fresh residual v1 also learned this small pair. This is
an execution/optimization sanity check, not evidence that v2 is superior or
qualifies real texture quality. Receipt: `fresh-low-rate-learning.json`.

## Workshop compatibility

Workshop recognizes the new immutable constructor, validates its complete
activation descriptor, keeps the installed base weights frozen, trains v2 LoRA
with logit cross entropy, and exports the matching Core ML graph. Old constructors
retain their prior objectives. Checkpoints now record architecture and objective;
new residual v2 continuation requires both. This prevents identical tensor shapes
and seeded weight hashes from silently changing the saved graph or objective.
Existing checkpoints without those new fields retain legacy continuation.

Focused Workshop checks passed for descriptor rejection, graph/objective
checkpoint compatibility, new and legacy frozen-base LoRA export/Core ML parity,
and exact partial-epoch continuation for residual v1 and v2. Final integrated
suite, install and lifecycle evidence can be appended below after completion.


## Training controls and checkpoint lifecycle

The previous checkpoint writer removed tensors belonging to the preceding
successful save. Bark run `8c415268` logged interval saves at epochs 25, 50, 75
and 100, but only its last checkpoint revision remained. Earlier weights cannot
be reconstructed from the remaining logs. New saves retain immutable weight,
optimizer, optional best-weight and descriptor revisions. `checkpoint.json`
remains an atomic latest alias. Test Model shows every retained revision with
epoch/update labels, tests exact selected tensors, and promotes any checkpoint
as a separate model while preserving the original run and export. The previous
and current outputs can be compared on the same unchanged input. Category/map
filters scope both run choices and nonignored diffuse inputs.

Continuation loads saved settings, then honors edited crop size, seed and depth
exaggeration. Network dimensions still match the chosen weights. Crop-size or
target-preparation changes reset optimizer/schedule; original images and earlier
runs remain intact. The user can restore all saved settings explicitly.

The requested seed is recorded separately from the effective seed. Requested
`-1` obtains a system seed on every fresh start and continuation, resets optimizer
moments/update count, and starts fresh shuffle and augmentation state. Explicit
fixed seeds may restore compatible optimizer and exact epoch position; old
fixed-seed checkpoints remain compatible. Neither option promises reproducibility
across changed source sets, frameworks or hardware.

Training exposes a retained-decoded-images toggle and MiB budget. The per-run
cache fills on demand, evicts at its admitted limit, and can be disabled for
streaming. Both modes use bounded background preparation. The cache ceiling is
half of physical RAM, leaving space for GPU training and other applications;
run details record the effective admitted limit. No corpus-wide pixel preload or
source-hash barrier precedes the first image. Early saves omit an exact-position
cursor if complete source proofs are not yet ready, preserving weights and
optimizer immediately. Final verification covers all sources, services Save &
Stop before each file, and validates original bytes again after export.

## Integrated regression evidence

`swift test -c release --no-parallel` passed the complete Factory suite: 88 native
checks (four optional skips), 203 core checks (three optional skips), 67 hosted UI
checks and two material-review checks. This is 353 passed checks and seven skips.
The stronger residual-v2 AdamW continuation test also passed with nonzero moments.
All 14 settings checks passed, including the custom cache-budget descriptor round-trip.

The first broad run exposed two old workflow fixtures that relied on ignoring a
requested crop size. They now supply larger originals and verify that continuation
uses the requested native crop while retaining immutable network dimensions.
A concurrent hosted UI run exceeded the existing 0.25-second heartbeat threshold
once (0.283 seconds); the complete serial run passed the unchanged threshold.
No threshold was weakened.

Hosted renders verify the checkpoint/category/map/A-B controls and editable
training settings. Native computer automation was unavailable with
`Sky Computer Use native pipe startup failed`; hosted renders are not a live
installed gesture test. Original constructor CoreML bytes, Float32 IO, exact
fixed-seed continuation, random-state reset, archive retention, source mutation,
cache concurrency and streaming parity all have focused regression coverage.


## Confirmed temporary buffer retention

The full streaming workflow showed a large resident-memory peak during source
verification. A fresh-process isolated reproduction repeatedly SHA256-hashed the
same unchanged 390,649,018-byte PNG eight times. The original FileHandle loop
increased peak resident memory on every read, reaching 3,143,712,768 bytes.
An autorelease pool around each 1 MiB read/hash chunk kept peak resident memory
at 9,207,808 bytes; all 16 hashes were identical. This is temporary autoreleased
buffer retention on a long-lived worker, rather than a changed hash algorithm.
Receipt: `2026-10-15-cache-throughput/hash-memory-comparison.json`.

A shared bounded native hasher now covers training source proofs, model version
validation, package inventory hashes, diagnostics and Workshop file checks.
Studio package validation and package staging drain the same temporary buffers
per chunk. Native publishing also bounds chunk lifetimes while hashing, framing
base64 data and copying multipart slices. No hash identity, source pixel or
upload content changes. CPU regressions cover empty files, non-multiple chunk
boundaries, repeated reads, original preservation, cancellation and IO errors.
Mock upload tests cover exact regular and multipart payloads without network
publishing. The post-fix streaming workflow is measured separately below.


## Real 1K cache and streaming measurements

Three fresh-process workflows used the current 55 paired Bricks examples, one
native 1024 crop per selected resolution, Compact NAF width 32 / one block
(999,937 parameters), fixed seed 198087, AdamW LR 0.0001, zero weight decay,
and two preparation workers. Each completed two epochs / 110 updates. They wrote
only to external validation folders. All 110 selected source files and all 232
existing project files kept identical SHA256 values and membership in each run.
Preservation audits warmed the file-system cache before timing, so these are not
cold-disk results. Both modes produced exactly the same final weight SHA256:
`198814cb4e344c08938b099908745c084b7dfb9315429d8704515102ef69e887`.

| Mode | Second-epoch mean cycle | Mean forward/backward | Mean optimizer | Mean image wait |
|---|---:|---:|---:|---:|
| Bounded 16 GiB cache | 0.495331 s | 0.477519 s | 0.015750 s | 0.000653 s |
| Streaming, before hash-pool repair | 0.795310 s | 0.505730 s | 0.014600 s | 0.273614 s |

The cached second-epoch maximum was 0.534386 seconds. The first image began
training at 0.214861 seconds and the first real update completed at 3.487351
seconds, including initial graph compilation. The cached first epoch averaged
0.967277 seconds per cycle with large-image decode spikes; a first visit to an
8K original can still exceed one second. Total workflows, including model setup,
checkpoint writes, source verification and CoreML export, took 90.80 and 99.83
seconds. These numbers separate warmed training throughput from setup and IO.

The post-repair streaming workflow completed with the same exact weights and
preserved files. Its peak resident memory was 5,818,744,832 bytes (5.42 GiB),
versus 23,136,403,456 bytes (21.55 GiB) before the hash repair. Its decoded-image
retention budget was zero. A CPU-only mocked-upload test build overlapped this
last run, so its timings are excluded from comparative throughput. The first two
runs had no competing test/build job. GPU allocation and resident-memory counters
measure different resources and must not be added as independent RAM totals.

Receipts: `~/Library/Application Support/Loom Factory/Validation/2026-10-15-cache-throughput/`
contains `cached/measurement.json`, `stream/measurement.json`,
`stream-bounded-hash/measurement.json`, per-run preservation manifests and
`summary.json`. This is throughput, resource and execution parity evidence;
it does not establish artifact-free learned depth or held-out texture quality.

The complete Factory suite passed after the bounded hasher change. The five
mocked publishing checks passed after bounding regular/base64 and multipart
copy chunk lifetimes. Models now labels counts as runs with checkpoints and
labels latest-checkpoint actions explicitly; exact epochs are chosen in Test
Model. This avoids presenting one latest alias per run as the total saved count.
