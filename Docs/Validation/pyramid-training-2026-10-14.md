# Pyramid training and run cleanup validation

Measured on an Apple M2 Max with 64 GiB unified memory. Receipts, source checks,
benchmark models and UI renders are outside the repository in the user's
`Library/Application Support/Loom Factory/Validation/2026-10-14-pyramid-speed`.

The stopped Bricks depth run used 1K crops, batch 1, accumulation 2, width 164,
four residual blocks and 44 teaching pairs from eleven materials across native
1K/2K/4K/8K tiers. Its completed cycles spent 97.5% in training and 2.35% waiting
for image preparation. Its first epoch took approximately 120.86 seconds.

The fresh Pyramid depth model uses width 38 and four blocks, with 1,926,373
parameters versus 1,943,893 in the previous network. An isolated project linked
to those same 88 original files trained two epochs with the same crop size,
batch/accumulation, seed, flips, learning rate and clipping. The final synchronous
execution path measured:

| Epoch | Optimizer steps | Total cycle time | Preparation wait |
|---|---:|---:|---:|
| First, decoding originals | 22 | 47.81 s | 32.15 s |
| Second, decoded pixels retained | 22 | 14.93 s | 0.032 s |

The warmed epoch is approximately 8.1 times faster than the stopped run's first
epoch. Extrapolating these cycles to 500 epochs gives about 2.08 hours, excluding
source checks, final verification and export. This is a projection from two
epochs, not evidence of a completed 500-epoch run or learned texture quality.
The original files passed the trainer's before/after identity and hash checks.

Actual ImageIO storage for the eleven 8K pairs was 5.875 GiB; all four tiers
require approximately 7.80 GiB. The previous 512 MiB cache could not retain the
corpus and often could not retain one full 8K pair. The new bounded cache uses
physical memory / 8, capped at 8 GiB, and releases its contents after a run.

Nine warmed synthetic 1K updates measured residual NCHW at 2.741 s/image and
NHWC/HWIO at 2.517 s/image (8.2% lower time). Pyramid NCHW was approximately
0.338 s/image; NHWC was slower, so Pyramid retains NCHW. Results use Float32
throughout. The full-resolution real-source regression also exposed an
allocation-sensitive large NCHW residual gradient path; residual sessions now
select NHWC by default, including direct native callers.

A command-buffer encoding experiment passed small checks but diverged in the
real 1K gradient regression. It was removed. Production retains the synchronous
graph path and explicitly initialized, owned result buffers.

Checks cover all three map products, odd/rectangular native and Core ML parity,
sampled finite-difference gradients through strided encoders and nearest decoders,
four optimizers, accumulation, normal normalization, frozen-base LoRA fusion,
real 1K gradients after allocation churn, checkpoint continuation, unknown and
contradictory architecture rejection, and whole-folder run deletion boundaries.
The clean staged-source suite passed; optional asset-dependent checks were skipped.

The seven old map-training runs were moved to macOS Trash through the checked
project API. The project now has no training runs and starts fresh with Pyramid
width 38. Originals, classifier outputs and package exports were retained.
Deletion controls were rendered offscreen; this does not establish interactive
desktop UI validation or qualify the exported benchmark model for deployment.
