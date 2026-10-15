# Full-resolution residual throughput investigation

Status: implementation, production-worker measurements and regression tests
complete. Installed Release verification is recorded below.

## Reproduction and measured regression

The reported run uses native 1024 × 1024 crops, batch one, residual width 118,
four blocks, 1,007,839 parameters, AdamW and Stable pixel loss (v2). The selected
saved-run row in the screenshot describes another run; its width 164 is not the
active model's width. Its approximately 2.23-second cycle is dominated by native
forward/backward execution, with negligible preparation wait and optimizer time.

Fresh, serial probe processes on this Apple M2 Max used identical synthetic
Float32 pixels, seed 198087, architecture, objective and optimizer. Eight updates
per layout were measured; the table reports the median of updates two to eight.
The first compilation update is excluded.

| Existing execution layout | Forward/backward | Complete update |
| --- | ---: | ---: |
| NHWC/HWIO, previous residual default | 2.21545 s | 2.22014 s |
| NCHW/OIHW, explicit diagnostic choice | 1.41571 s | 1.42113 s |

NCHW reduces update duration by approximately 36.0% in this fixture. The earlier
NHWC default was based on width-164 measurements and does not generalize to this
width-118 model. A blanket layout policy is therefore a performance regression
for this workload. Timing alone does not establish gradient correctness.

Raw evidence is outside the repository under
`~/Library/Application Support/Loom Factory/Validation/2026-10-14-residual-throughput/`,
including `before-nhwc.json`, `before-nchw.json`, associated logs and the excluded
experimental patch. Reports include graph timing, allocation and memory fields.

## Historical speed comparison

The recorded Loom residual width-116/four-block model had 974,053 parameters and
warmed native 1K updates of 1.39790 and 1.39554 seconds using NCHW and the older
sigmoid/L1 objective. Its forward/backward times were 1.39355 and 1.39111 seconds.
The receipt is `Validation/training-progress-2026-10-14/gpu-before.json` in the
same application-support root. This is a useful near-capacity timing reference,
but the objective and width differ from the current run.

The older Texture Studio compact scalar width-32 experiment did run 200 native
1K updates in 108.315 seconds, approximately 0.542 seconds per update including
its recorded measurements. Its receipt is under
`~/Library/Application Support/Texture Studio/Validation/gray-collapse-2026-10-12/full-native-training/real-bark/report.json`.
That 999,937-parameter architecture used three downsampling levels and NAF-style
blocks. It is not the same full-resolution residual workload. Parameter count
alone does not determine convolution cost. Retired application code was inspected
as historical evidence; it was not imported or called.

## Experiments excluded from production

| Experiment | Observed warm complete-update median or outcome |
| --- | --- |
| Explicit cached executable, NCHW | 1.40785 s; little improvement over NCHW alone |
| Compiler optimization level 0, NCHW | 1.41143 s; no substantial additional gain |
| Compiler automatic layout conversion | 1.40741 s; no substantial additional gain |
| Mixed activation/kernel layouts | 2.25791 s; slower |
| SAME padding, NCHW / NHWC | 1.41112 / 2.21764 s; no substantial gain |
| Width 120 / 128 alignment trials, NHWC | 2.20177 / 2.19225 s; changed capacity and did not solve the slowdown |
| Exact spatial tiling, NCHW / NHWC | 1.47567 / 1.61011 s; slower than plain NCHW |
| Im-to-column plus matrix multiplication | Framework assertion: operation gradient not implemented |
| Slice-built columns plus matrix multiplication | Single-layer warmed calls 1.028–5.570 s versus convolution 0.177–0.184 s |

These exploratory runs have different sample counts and are not a controlled
ranking of small timing differences. The slice-built matrix fixture's weight
gradient maximum absolute difference was 1.04308e-7, but its timing and storage
cost ruled it out. No reduced-precision arithmetic was adopted.

## Implementation and correctness boundary

The implementation calibrates both unchanged Float32 layouts once for eligible
full-resolution residual sessions. Separate temporary sessions use the current
weights, discard their gradients, and perform no optimizer updates. It compares
warmed execution time plus loss and every parameter gradient before choosing
NCHW; NHWC remains the fallback if parity fails or the timing advantage is below
3%. Numerical agreement is checked for each session using its own starting weights
and first large-grid input; no agreement from another checkpoint is reused. Pyramid and explicit
diagnostic layout choices bypass calibration. A memory-admission check limits
calibration, and cancellation is checked throughout.

The architecture, native pixel grid, Float32 values, loss, learned parameter
shapes and checkpoint interfaces remain unchanged. Layout differences can still
produce floating-point differences; the implementation requires explicit agreement
before using the faster path. Calibration must preserve pending accumulation,
optimizer moments and the completed-update counter.

For the identical first synthetic 1K step, NHWC/NCHW losses were exactly
0.6883542537689209. Convolution-weight gradients differed by relative L2
0.00040–0.00053; maximum absolute differences were 3.10e-6–4.48e-5. Bias
relative differences were approximately 1e-7. The first trial's arbitrary 0.1%
maximum-error cutoff rejected some tensors despite this agreement. The final
check bounds both each tensor's maximum error (0.2% of its reference maximum,
with an absolute floor of 2e-6) and relative L2 error (0.1%). Exactly zero
reference tensors must remain exactly zero. Tests reject equal-norm sign and
permutation corruption as well as malformed/nonfinite tensors.

## Final workflow measurements

`automatic-verified.json` ran 20 updates with the unchanged 1,007,839-parameter
1K residual model. It selected NCHW, passed gradient agreement and recorded
7.992 seconds of one-time calibration. Median complete update over updates
3–20 was 1.41923 seconds, about 36% shorter than the prior 2.22-second layout.
Metal allocations were exactly 9,993,404,416 bytes at updates 3, 6, 11 and 20.
This short observation does not exclude every long-run leak.

`production-55-pairs/measurement.json` ran the real Factory training worker on
all 55 current Bricks depth pairs at 1K, batch one, width 118, four blocks, v2,
seed 198087, AdamW, fixed clipping and one epoch. Output stayed in the separate
validation directory. The worker selected NCHW with gradient agreement:
1.40629 seconds versus NHWC 2.21534 seconds in calibration. Over updates 3–55,
median cycle was 1.42621 seconds, mean 1.47611 seconds, median training was
1.42093 seconds and mean preparation wait was 0.04692 seconds. The full workflow,
including source checks, 8.026-second calibration, checkpoints and export,
completed in 98.805 seconds. The first cycle was 9.921 seconds; it includes
calibration, so initial ETA and averages temporarily include that startup cost.

`pipeline-cpu-sample.txt` sampled the worker while processing the real sources.
The training thread was predominantly waiting for Metal command completion;
image-preparation threads were decoding PNG files. This does not show a CPU
fallback for convolutions. The diagnostic process exited normally after export.
All 92 preexisting run/model files in the recorded preservation manifest kept
identical SHA-256 hashes and modification times. No diagnostic model was assigned
or approved in the user's project.

## Verification

`swift test --no-parallel --filter 'LoomNativeTests|LoomCoreTests'` passed:
64 native XCTest cases and 185 core XCTest cases, with three opt-in skips in
each suite and zero failures, plus two Swift Testing cases. The four new
calibration tests also passed separately in Release, including continuation
from nonzero optimizer moments, exact state/gradient agreement with the selected
explicit layout, malformed input, fallback and cancellation.

Both opt-in real 1K regression tests subsequently passed in Release using the
unchanged Bark Willow 02 sources (58.977 seconds total). They check the historical
losses and gradient norms, identical fresh/reloaded final weights, and source
SHA-256 preservation. Remaining opt-in checks cover the saved collapsed roughness
fixture, categorizer benchmark, package integration and existing model-test UI;
these were not required to validate this layout change.

`Scripts/build-app.sh` built the optimized app, verified the native bundle and
strict code signature. Installation at `/Applications/Loom Factory.app` passed
strict/deep signature and native-bundle checks. Its signed executable matches
the packaged app exactly, SHA-256
`d4608ec29de4f10f4367461eeae68c6e697f2c3704603586f1139cccebd21407`.
The installed Release launched as PID 82658. Native UI automation was unavailable
(`Sky Computer Use native pipe startup failed`), so this is launch/process and
bundle evidence, not a rendered training interaction claim.

The user requested no obsolete app copies. The backup created by this install
and the build's previous app copy were removed after verifying their identities.
The installer now retains rollback only during replacement and removes it after
successful verification; the build script also removes its previous bundle after
success. Two controlled destination-root installs preserved the exact signed
executable and left only the current app, with no staging/backup copies.

A subsecond update for this full-resolution model remains unachieved. The older
0.542-second architecture comparison does not establish that target for this
network. These execution checks do not qualify learned material quality or
establish artifact-free output.
