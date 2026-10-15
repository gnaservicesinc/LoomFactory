# Depth fitting and Compact memory advisory

The source project was `/opt/ipde/sources_mats/.loom-factory`. Inspection and
diagnostic outputs are outside the repository in
`~/Library/Application Support/Loom Factory/Validation/2026-10-15-cliff-500k-investigation`.
Original images and saved project weights were not rewritten.

## Loss interpretation and current fitting

Rock Cliff Marble run `be39b1c3` uses residual v2, width 84, four blocks,
511,813 parameters and native-detail-v3. It stopped with a verified weights and
optimizer checkpoint at update 482. Its nine completed epoch losses moved from
0.675092 to 0.668241, with a minimum of 0.660846 at epoch eight. The run has no
validation materials. All 53 requested diffuse/displacement pairs have matching
native dimensions, material folders and pairing metadata.

V3 is soft-target cross entropy plus twice the mean native signed-difference
error at offsets 1, 2 and 4 in both axes. Its positive grayscale target-entropy
floor makes the displayed objective unsuitable for direct comparison with
Compact's linear-detail-v1 objective or interpretation as a pixel error rate.

Exact native 1K inference on Cliff Side, Marble Cliff 03 and Tiger Rock found
flat relief in both this checkpoint and residual-v2 run `333fbd6b` (1,943,893
parameters, 585 updates, cross-entropy v2). Five of six predictions had worse
pixel MAE than a constant image set to the corresponding target's mean. These
are early checkpoints on known training materials with different seeds,
objectives and update counts, not a controlled capacity or generalization test.

Two diagnostic branches then started from identical update-482 weights and
identical native 1K Cliff Side pixels. Both used fresh AdamW, LR 0.0001, no
weight decay, clipping at one and no augmentation. After 50 same-pair updates,
MAE fell from 0.089045 to 0.073258 with v2 and 0.075077 with v3. Correlation rose
from 0.127 to 0.543 and 0.557. V2 increased incorrect fine variation; both still
missed broad depth structure. This establishes useful optimization on this pair,
not a qualified model. The original weights, optimizer, checkpoint, request and
both source images retained their SHA256 values.

## Longer Compact Bricks run

`3ef8e93a` has 999,937 parameters and stopped after 2,804 updates: 50 complete
epochs plus 54 of 55 images in the next epoch. It trained 55 pairs across 15
materials with no validation, no target exaggeration, native 1024 crops, AdamW
at 0.0001 and clipping at one. Successive ten-epoch mean objectives were 0.3654,
0.2787, 0.2592, 0.2485 and 0.2398. All logged updates were finite with nonzero
global gradients; 97.29% exceeded the clipping threshold. This is continued
fitting, without evidence that additional epochs will recover useful relief.

Existing whole-field tests matched the saved weights SHA256 and known training
inputs. Brick Wall 005 at 2K had MAE 0.189899 versus a target-mean constant
baseline of 0.094866, with positive mean bias 0.180007. Brick Wall 006 at 1K had
MAE 0.133027 versus 0.037509, with mean bias -0.128952. Output standard deviations
were 48.0% and 41.5% of the targets. These examples show partial structure with
large mean/contrast errors. Both used one whole field and zero overlap, so tile
stitching does not explain their softness.

The RGB16 Brick Wall 006 target was decoded through FFmpeg RGB48; Pillow's
ordinary RGB decode silently reduces precision. The corrected metrics and
fixed-range preview are in `compact-bricks-metrics.json` and
`compact-bricks-fixed-range.png`. The target means differ substantially between
these materials, but their native 1K/2K target grids agree to half a 16-bit code
under corresponding area comparison. No pairing or resolution-scale defect was
found in these two materials.

## Advisory memory estimate

The previous Factory preflight rejected Compact inference whenever its
conservative forecast exceeded the estimated budget. The forecast is now
advisory: compatible images proceed to unchanged whole-field Float32 inference,
and progress plus `test.json` record when the estimate exceeds the budget.
Invalid grids, arithmetic overflow, source/weight identity checks and output
validation remain enforced. No resizing or tiled approximation was introduced.

Four focused CPU contract tests passed serially: forecast/budget arithmetic,
over-budget prediction attempts, exact raw whole-field output and incompatible
grid rejection. The over-budget regression injects a tiny budget and checks that
the full image reaches the predictor and the canonical EXR remains exact.

A fresh-process prediction-only diagnostic used the actual Compact Bricks
weights and unchanged Brick Wall 005 originals. At 2K it completed in 1.07 seconds
of prediction, with 14,246,166,528 bytes of Metal allocations and
13,433,046,144 bytes of process physical footprint at the final measurement.
It performed zero optimizer updates and preserved image/weight hashes. The 4K
attempt was terminated by signal 9 before producing an output or final memory
measurement. This does not verify 4K capacity or identify the termination cause.
The changed policy permits an attempt; it is not a claim that every image fits.

Texture Studio has a separate Compact processing preflight; this change concerns
Factory model testing only.
