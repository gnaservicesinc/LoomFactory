# Saved depth checkpoint diagnostics

The Release native probe compared the 13-epoch Bricks checkpoint `80bffde0`
with its 103-epoch continuation `395fb421`. Both use Pyramid width 38, eleven
blocks and `logit-cross-entropy-v2`. The two inputs were original 1024 × 1024
`red_brick_03` and `Bricks059` diffuse maps, with their paired original depth
targets. Both materials occur in the training request; this is training-material
inspection, not held-out generalization evidence.
Epoch counts are the local completed counts recorded by each saved run. Their
recorded seeds differ (5,927,806 and 198,087), so this is a comparison of the
user's saved candidates, not a controlled experiment isolating training duration.

Reports, logs and exact Float32 prediction/target EXRs are outside the project at
`~/Library/Application Support/Loom Factory/Validation/2026-10-14-checkpoint-quality-audit`.
Each report verifies unchanged input, target and checkpoint hashes, zero optimizer
updates and unchanged loaded model weights.

| Material | Measurement | Parent, 13 epochs | Continuation, 103 epochs |
|---|---|---:|---:|
| red_brick_03 | Prediction mean (target 0.549271) | 0.652862 | 0.598711 |
| red_brick_03 | Pixel MAE | 0.130090 | 0.098137 |
| red_brick_03 | Horizontal one-pixel difference MAE | 0.007025 | 0.008482 |
| red_brick_03 | Vertical one-pixel difference MAE | 0.010437 | 0.012077 |
| Bricks059 | Prediction mean (target 0.683622) | 0.708548 | 0.667444 |
| Bricks059 | Pixel MAE | 0.087705 | 0.088447 |
| Bricks059 | Horizontal one-pixel difference MAE | 0.006509 | 0.007174 |
| Bricks059 | Vertical one-pixel difference MAE | 0.013480 | 0.014038 |

Both tested outputs became darker on average. None had any output pixels within
0.00001 of white or black. The saved later checkpoint's first 6,800 updates had
maximum gradient norm 5.8762, while its recent epoch means remained nonzero.
These particular outputs therefore do not reproduce the previous endpoint
saturation/zero-gradient failure. They do show worse local pixel-difference
errors despite improved overall pixel MAE on red_brick_03. The probe does not
establish a specific cause or a general quality improvement from those metrics.

Training image preparation reads exact native source/target crops into Float32
without spatial resampling or target normalization. Pyramid-v1 does reduce
learned feature grids to half and quarter size, then uses nearest upsampling
and skip features. That is an architectural operation with a possible detail
tradeoff, distinct from source-file compression. No new loss or architecture was
substituted on the basis of this small comparison.

The separate original-residual candidate `979304a5` is not a sound quality
baseline yet: its saved training log contains thirteen updates with gradient
norm above 100, peaking at 3,271,820,288. Two existing test records on different
brick inputs contain identical nearly constant output statistics. A controlled
architecture A/B requires stable fresh candidates with the same source split,
seed, training budget and native-size whole-material inspection.

There is also a measured preparation bottleneck in the current larger corpus.
Across the last 330 saved update cycles, `395fb421` spent 101.57 of 303.21 seconds
(33.5%) waiting for preparation, and `5493b91f` spent 70.60 of 269.84 seconds
(26.2%). These are logged wall times, not an estimate from Activity Monitor.

Probe verification used a small deterministic synthetic training update followed
by predict-only reload: pixel diagnostics and objective loss were identical,
loaded weights stayed unchanged, and a requested output over the supplied weight
file or its symbolic/hard-link aliases was rejected before execution. Release
compilation passed.

A separate serial synthetic run matched the screenshot architecture: 1024-pixel
input, Pyramid width 38, eleven blocks, AdamW, v2 loss and 120 updates. Between
updates 20 and 120, Metal allocated storage stayed exactly 4,580,229,120 bytes.
Physical footprint changed from 4,769,793,152 to 5,167,531,136 bytes, with a
nonmonotonic warm range of 4,529,292,416–6,011,847,808 bytes. Resident CPU memory
changed from 156.6 to 157.7 MB. The last 100 updates took 60.03 seconds with a
0.59879-second median. These timings describe this isolated synthetic workload;
they do not measure real-source preparation or establish a speedup. No linear
Metal allocation growth was reproduced, but this short run does not exclude a
slower leak. Receipts are `memory-120.json` and `memory-120-summary.json` in the
same external validation directory.
