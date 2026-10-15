# Compact NAF native comparison model

Status: implementation, historical parity and real 55-pair timing verified;
complete suites and installed Release verification pending.

## Performance requirement and first measurement

The user requires native 1024 × 1024 training below one second per update, with
usable material detail. Full-resolution residual width 118 was approximately
2.22 seconds with NHWC and 1.42 seconds with NCHW. The latter fixes a layout
regression but still does not satisfy that speed requirement. Compact NAF is a
separate model choice restoring the earlier fast task-specific network design.

On this Apple M2 Max, a fresh serial process trained ten synthetic 1K depth
updates, batch one, width 32, 999,937 parameters, NCHW, Float32, AdamW, learning
rate 0.0001 and the original linear/detail objective. Updates two through ten:

| Measurement | Time |
| --- | ---: |
| Median forward/backward | 0.45849 s |
| Median optimizer update | 0.01271 s |
| Median complete update | 0.47125 s |
| Complete-update range | 0.46732–0.48404 s |
| First update, including compilation | 11.01985 s |

The real production worker then trained all 55 current Bricks depth pairs for
two epochs with the same native 1K grid, width 32, seed 198087, AdamW learning
rate 0.0001, no weight decay and fixed global clipping at one. The first epoch
averaged 0.85144 seconds per cycle including cold preparation waits. Once the
originals were cached, epoch two averaged **0.47436 seconds per cycle**, median
**0.47186**, range **0.46787–0.50198**. Total preparation wait in that second epoch
was 0.03305 seconds. The complete two-epoch run, including source verification,
checkpoints and Core ML export, finished in 86.332 seconds. Individual cold
compile/decode cycles can still exceed one second.

A separate 1K normal-map probe (1,000,515 parameters) had warm median complete
update 0.47373 seconds, range 0.47100–0.48997. Scalar depth and roughness use the
same one-channel graph. These measurements satisfy the warmed-update requirement
on this machine; they do not promise the same speed at larger sizes or widths.

Receipts are retained outside the repository at
`~/Library/Application Support/Loom Factory/Validation/2026-10-14-residual-throughput/`:
`compact-1k-first.json`, `compact-normal-1k.json`, and
`compact-production-55x2/measurement.json` with its request and training log. The initial synthetic receipt used the previous
8 GiB decoded-cache cap; its pixels were synthetic, so that cap does not account
for its measured training throughput.

## Architecture and exact historical agreement

The immutable architecture is `loom-factory-material-compact-naf-v1`, with one
block in each of seven stages and 142 named Float32 tensors. Width 32 has
999,937 scalar parameters or 1,000,515 normal parameters. Original residual and
Pyramid architecture identities and saved weights retain their own meanings.

The compact network has encoder widths w, 2w, 4w and a bottleneck width 8w.
Three unpadded 2 × 2 stride-two convolutions downsample learned features; decoder
1 × 1 convolutions reduce channels before exact nearest-neighbor repetition.
Each decoder adds its corresponding encoder skip, including the full-resolution
skip. NAF blocks use channel normalization with epsilon 1e-6, depthwise 3 × 3
convolution, multiplicative gates, spatial-mean channel attention and learned
residual gains initialized to 0.1. Stride-one 3 × 3 borders use reflection.
These operations differ from Pyramid's ReLU residual blocks and sampling path.

The output is linear for both scalar and normal components: no sigmoid, clamp,
per-image normalization or normal-vector renormalization. Its fixed objective
is original-pixel L1 plus four times the sum of signed-difference L1 errors at
offsets 1, 2, 4 and 8 along both axes, weighted by inverse offset. This objective
has the separate ID `linear-detail-v1`; logit objectives cannot reinterpret it.
Initialization preserves the historical Xavier, gain and SplitMix64 rules.
Dense convolution LoRA has 43 eligible layers; depthwise and normalization
parameters remain part of the frozen base. The NAFNet MIT notice is retained.

The retained historical full-1K bark checkpoint was loaded as a controlled
numeric fixture without training or altering its bytes. Its 142 tensors match
the new constructor exactly. Against every saved historical Float32 pixel:

- Maximum prediction difference: **0**.
- Mean prediction difference: **0**.
- Recomputed objective: **0.15394018590450287**, exactly the historical value.

The checkpoint SHA-256 is
`5bc66a3742eb8eef26174f3fb4283ae9d5b7acab2b73ad1d0d2d4cac74cf45c7`.
The fixture is under Texture Studio's retained
`Validation/gray-collapse-2026-10-12/full-native-training/real-bark` directory.
It consumes `srgb-code-v1` diffuse, matching the new model's input codes. This
is a compatibility experiment on a known training material, not automatic import
or reinterpretation of historical application checkpoints.

## Original pixels, whole-field execution and memory

Integer diffuse and material samples retain their original numeric codes divided
by the code maximum; paired crops preserve native coordinates and precision.
No source image is resized, denoised, recompressed or rewritten. Internal feature
downsampling does not replace or resample supervision. Canonical EXR stores the
raw Float32 output; only the separate PNG preview clamps to fixed [0,1].

Native compact grids must be at least 32 pixels and multiples of eight in each
dimension. Unsupported originals fail explicitly. Spatial-mean attention uses
the complete input field, so tiling would change the network's math even with
overlap. Compact model tests therefore use one whole native field; their memory
cost can grow substantially at larger resolutions. No resized or tiled fallback
is represented as equivalent whole-field inference.

The measured 55-pair Bricks request references 110 physical training files. Its
decoded original numeric storage is **7.916 GiB**, compressed files total
5.289 GiB, and one prepared 1K RGB/depth crop per pair totals 0.859 GiB. The
prior 8 GiB cache could narrowly hold this exact corpus. The new decoded-cache
budget is one quarter of physical RAM, capped at **16 GiB** on this 64 GiB Mac,
providing room for other selected maps. It is bounded storage, not preallocation
or a total process-memory cap. Crop refresh still needs the original full field.
Raising the cache alone cannot remove GPU convolution time. The read-only audit
verified source identities and is retained under `memory-audit/` with per-file
dimensions, precision, provider storage lengths and aggregate receipts.

## Validation boundary

Three native Core ML parity tests and four compact project/workflow contract
tests passed. Math checks cover exact initialization, linear extended-range
outputs, reflection borders, all-parameter teaching signals and dense LoRA
fusion. Independent CPU equations agree with the linear/detail objective and
every prediction gradient. A first finite-difference fixture crossed nonsmooth
L1 behavior; it was replaced with a smooth MSE network-derivative probe using a
Double CPU numerical objective. That stricter gate/depthwise/normalization/
attention/bottleneck/gain check passed in 3.866 seconds.

Factory validation passed 76 native XCTest cases (four opt-in skips),
190 core XCTest cases (three opt-in skips), 66 serialized UI tests and two core
Swift Testing cases. The historical 1K fixture ran separately and matched every
output bit. A comparison compatibility regression found by the full suite was
fixed by retaining old residual/Pyramid descriptors' existing acceptance rules;
strict new metadata checks apply to compact models.

Studio passed six compact contract tests, plus the Float32 input-filter test;
two unrelated opt-in deployment tests were skipped. The model input now transfers
its newly allocated tensor safely across actor isolation. Native inference was
also measured in fresh 1K and 2K processes: approximately 5.06 GB and 20.17 GB of
Metal allocations. Whole-field preflight uses a conservative memory forecast and
remaining memory budgets; it rejects oversized requests before full input/model
allocation without changing image dimensions. This guard is conservative and
can reject large repeated requests when framework caches remain resident.

Workshop passed all 38 core XCTest cases (two opt-in skips) and 13 serialized UI
tests, including compact frozen-base adaptation and Core ML fusion parity.
Installed Release verification remains pending. These execution,
compatibility and export checks do not establish artifact-free learned output,
held-out detail quality, or larger-grid training capacity. Existing runs and
original checkpoints remain preserved for visual whole-material comparisons.

The preservation audit found all 91 original run/model data files unchanged in
hash and modification time after the compact measurements. A Finder `.DS_Store`
entry changed independently; it is not a model or training record. No diagnostic
model was assigned or approved. Twenty-four obsolete Factory app backup copies
were removed after checking their bundle identities; installers now retain
rollback copies only during replacement and remove them after success.
