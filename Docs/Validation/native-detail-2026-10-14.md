# Native detail and Pyramid spatial artifacts — 2026-10-14

The experimental `native-detail-v3` loss reduces measured false variation and
native edge errors, but it does **not** fix Pyramid for artifact-sensitive depth
maps. A matched six-epoch comparison worsened pixel MAE on 13 of 17 known
training materials, and a separate translation check reproduced changes of up
to 0.064739 in the same interior output pixels when the input origin moved by
one pixel. This is not an artifact-free model or a production qualification.

Fresh training settings now use the original full-resolution residual network;
Pyramid remains an explicit experimental choice for comparison. This avoids
making an unresolved Pyramid defect the default. It does not qualify the
full-resolution network's learned output. Existing saved networks keep their
architecture and weights.

## Matched experiment

Raw receipts are in the user storage directory:

`/Users/andrewsmith/Library/Application Support/Loom Factory/Validation/2026-10-15-native-detail`

The UTC directory date is October 15; the local session date is October 14.
Only `control-v2` and `detail-v3`, produced by the current-object harness, are
used below. Both completed 396 updates over six epochs with:

- The exact same saved Pyramid checkpoint: width 38, 11 blocks, depth output,
  `loom-factory-material-pyramid-v1`.
- All 66 paired native-resolution examples from the same saved request,
  one 1024-pixel crop per selected pair per epoch.
- Seed 198087, matching shuffled order, arbitrary native crop origins, paired
  flips, and per-epoch refresh; no rotation.
- Learning rate 0.0001 and a fresh optimizer, with the same native optimizer
  defaults and fixed gradient clipping.
- Whole native 1024 × 1024 evaluations of all 17 eligible 1K input/target pairs
  at epochs 0, 1, 3, and 6. These are known training materials, not held-outs.

The sole intentional experiment variable is the scalar objective:
`logit-cross-entropy-v2` versus `native-detail-v3`. V3 adds twice the mean
absolute signed difference residual at original-pixel offsets 1, 2, and 4,
in both axes, to v2's stable logit cross-entropy. This loss does not resize or
smooth either image or change the inference graph.

The starting checkpoint is:

`/opt/ipde/sources_mats/.loom-factory/runs/395fb421-7868-4f51-affe-018e47cbf83f/checkpoint-6e34cead-75ec-44f9-a68b-132cb6f48612.safetensors`

Its SHA-256 is
`10a1506d96e135e5b2ddbfee6294985c3af0a510b5751e5126bb0ae13e0a03d1`.
The request SHA-256 is
`a1b43587502039c0c07866300dcabf4996bbebbe2801150a871146ae2b355942`.
The two `transforms.json` files are byte-identical, SHA-256
`d78f8257002edc3685a667686f6550c0c497533d6a8137d909c92c2048ab4f52`.
All 17 starting prediction EXRs and all 17 target EXRs are also byte-identical
between the two experiments.

Both result files report `sources_unchanged: true`. The harness hashes all 132
input/target paths plus the saved request and starting weights before training,
and checks those 134 identities again at completion. Training and evaluation
leave the source images and saved run untouched. Outputs are single-channel,
uncompressed Float32 EXRs at native dimensions; the checked EXR channel type is
FLOAT (2), compression is NONE (0). PNG contact sheets use one common [0, 1]
display conversion for inspection; measurements use the Float32 EXRs.

Final weights:

| Experiment | SHA-256 | Elapsed including evaluations/hash checks |
| --- | --- | --- |
| V2 control | `9c80deb7085b424de01d442516a7a686d2bb265d5a2c677fbbe1ff3af02cc10e` | 461.242 s |
| Detail V3 | `f042f8d55faf6bff3c46962c5ea0f9d243771fbff5f0d8707b939d2161e8eeef` | 464.967 s |

## Six-epoch result and limits

`comparison-summary.json` averages each metric equally across the 17 whole
materials. Difference errors compare signed prediction differences to signed
target differences, averaged across horizontal and vertical axes. Flat-target
variation measures absolute prediction differences only where the target's
absolute difference is below 0.001; this is a noise proxy, not an artifact
detector or a measure of recovered detail. Curvature errors are absolute second
differences of the prediction residual along the two image axes.

| Metric | V2, epoch 6 | Detail V3, epoch 6 | Change versus V2 |
| --- | ---: | ---: | ---: |
| Pixel MAE | 0.116658 | 0.119986 | 2.85% worse |
| Pixel RMSE | 0.136383 | 0.138555 | 1.59% worse |
| Signed 1-pixel difference error | 0.009639 | 0.008757 | 9.15% lower |
| Signed 2-pixel difference error | 0.017602 | 0.015981 | 9.21% lower |
| Signed 3-pixel difference error | 0.024291 | 0.022123 | 8.93% lower |
| Signed 4-pixel difference error | 0.030009 | 0.027489 | 8.40% lower |
| Signed 5-pixel difference error | 0.034893 | 0.032154 | 7.85% lower |
| 1-pixel variation on flat target regions | 0.004133 | 0.003145 | 23.91% lower |
| Vertical residual curvature error | 0.010722 | 0.010073 | 6.05% lower |
| Horizontal residual curvature error | 0.006502 | 0.006058 | 6.84% lower |
| Prediction 1-pixel variation | 0.006073 | 0.004603 | 24.21% lower |
| Prediction standard deviation | 0.062049 | 0.049478 | 20.26% lower |

All 17 materials improved the 1-pixel difference error and flat-target variation.
Sixteen improved the 3-pixel difference error. Only four improved MAE. Reduced
noise is therefore accompanied by reduced overall output variation; the result
cannot be described as recovered native detail across the dataset.

Per-material values below are from `comparison.json`. Smaller errors are
better. The last column is the change in the 1-pixel difference error; it does
not imply that the material is free from visible or downstream artifacts.

| Material input stem | V2 MAE | V3 MAE | 1-pixel difference error, V2 → V3 |
| --- | ---: | ---: | ---: |
| Bricks059_1K-PNG_Color | 0.086794 | 0.090830 | 0.010755 → 0.010553 |
| BricksSubstance007_COMPILED_XL_baseColor_1k | 0.059757 | 0.057870 | 0.010547 → 0.010167 |
| brick_wall_005_diff_1k | 0.238799 | 0.243706 | 0.012441 → 0.010279 |
| brick_wall_006_diff_1k | 0.094702 | 0.097604 | 0.008197 → 0.007081 |
| brick_wall_10_diff_1k | 0.080213 | 0.091908 | 0.016724 → 0.015990 |
| brick_wall_11_diff_1k | 0.016603 | 0.012853 | 0.004752 → 0.003847 |
| broken_brick_wall_diff_1k | 0.104414 | 0.108695 | 0.006452 → 0.005799 |
| painted_worn_brick_diff_1k | 0.044074 | 0.053479 | 0.009901 → 0.009188 |
| red_brick_03_diff_1k | 0.096416 | 0.100260 | 0.010871 → 0.009280 |
| red_brick_diff_1k | 0.303012 | 0.310725 | 0.008918 → 0.007773 |
| rock_wall_10_diff_1k | 0.066171 | 0.064683 | 0.006542 → 0.005281 |
| seaworn_sandstone_brick_diff_1k | 0.139739 | 0.149262 | 0.009663 → 0.008928 |
| stacked_brick_wall_diff_1k | 0.118801 | 0.135207 | 0.006206 → 0.005683 |
| stacked_stone_wall_diff_1k | 0.077058 | 0.083299 | 0.009810 → 0.009150 |
| stone_wall_diff_1k | 0.241734 | 0.213910 | 0.011393 → 0.010097 |
| whitewashed_brick_diff_1k | 0.092837 | 0.101274 | 0.014293 → 0.014069 |
| wood_inlaid_stone_wall_diff_1k | 0.122063 | 0.124189 | 0.006397 → 0.005711 |

The aggregate prediction means are 0.673806 (V2) and 0.675034 (V3), with target
mean 0.676113. The near-endpoint fraction is zero for every evaluated candidate
and material at the comparison script's 1e-5 tolerance. These measurements do
not reproduce white saturation. They show a separate detail/spatial problem;
absence of saturation is not evidence of acceptable depth quality.

## Translation exposes an unresolved Pyramid phase dependency

`shift-probe.swift` predicts five 512-pixel native crops from each 1K source,
starting at `(256 + offset, 256)` for offsets 0 through 4. It aligns the outputs
back to the same source pixels and compares only their interior, trimming 160
pixels from each side. That margin exceeds the network's receptive-field radius
and excludes boundary padding as the explanation for the measured difference.

| Saved candidate | Material | One-pixel shift mean absolute change | One-pixel shift maximum absolute change |
| --- | --- | ---: | ---: |
| Starting checkpoint | red_brick_03 | 0.001387406 | 0.062748730 |
| Detail V3, epoch 6 | red_brick_03 | 0.001277983 | 0.064738512 |
| Starting checkpoint | Bricks059 | 0.000420146 | 0.022979856 |
| Detail V3, epoch 6 | Bricks059 | 0.000441730 | 0.021569371 |

Three-pixel shifts reproduce the same maxima, with slightly different means
from the overlap extent. Two- and four-pixel shifts are exactly equal in this
check. This odd/even pattern is consistent with Pyramid's stride-two sampling
phase. The same scene point can receive a materially different depth because
the crop origin changes by one pixel, even with matched original context.
The detail loss leaves this defect in place and slightly worsens the largest
red-brick change. The check covers two materials and one shift axis; it is not
an exhaustive artifact audit.

## Full-resolution fallback: execution and spatial check only

`full-resolution-check.swift` completed 20 updates on the same whole 1K
`red_brick_03` pair using fresh full-resolution residual weights, width 164,
four blocks, 1,943,893 parameters, seed 198087, v2 loss, AdamW, learning rate
0.001 and warmup gradient clipping. All recorded losses and gradient norms
were finite. The largest pre-clipping norm was 54.444153 at update 9. Final
evaluation loss was 0.6662801504 and prediction range was
0.1710483432–0.6795309186. Metal allocated bytes were 13,876,264,960 from update
2 through update 20, after 13,852,573,696 at the first update. Twenty updates
cannot rule out a slower leak or establish long-run optimizer stability.

Using those saved weights in the same interior red-brick translation check,
`translation-full-resolution.json` reports maximum change
`1.1920928955078125e-7` for one- and three-pixel shifts, with mean changes
`8.5444376522749915e-9` and `8.5424618693676371e-9`. Two- and four-pixel shifts
are exactly equal. This removes the measured odd/even shift defect in this
specific fallback check to Float32 rounding scale. It is a spatial consistency
result for one material, not evidence that its learned depth has acceptable
detail. Repeated training on one pair for 20 updates is execution evidence;
there is no whole-material held-out quality claim for this fallback.

## Raw receipts and reproduction

The receipt directory contains:

- `control-v2/configuration.json`, `detail-v3/configuration.json`: exact input,
  checkpoint and request hashes and experiment settings.
- `control-v2/result.json`, `detail-v3/result.json`: completion, update count,
  unchanged-source checks, elapsed time and final weight hashes.
- `control-v2-current.log`, `detail-v3-current.log`: actual update/evaluation
  records from the valid harness.
- Both `transforms.json` files and `evaluation-{0,1,3,6}.json` files.
- Both experiments' `epoch-{0,1,3,6}-*.exr` and `target-*.exr` files, together
  with their final `model.safetensors` files.
- `comparison.json`, `comparison-summary.json`, `compare.py`: full per-material
  metrics, aggregate metrics and the Float32 EXR analysis implementation.
- `shift-probe.swift`, `translation-check.json`: interior translation check
  implementation and combined initial/detail results.
- `full-resolution-check.swift`, `full-resolution-1k.log`,
  `full-resolution-1k/result.json`, `full-resolution-1k/model.safetensors`,
  `full-resolution-1k/prediction.exr`, `translation-full-resolution.json`:
  the separate 20-update fallback execution and spatial check.
- `Bricks059_1K-PNG_Color-comparison-six-epochs.png` and
  `red_brick_03_diff_1k-comparison-six-epochs.png`: common-scale contact sheets.

`invalid-stale-library-v3` is explicitly excluded. Its diagnostic harness linked
a pre-v3 cached static library and was stopped when the first loss contradicted
the additive objective. Its `EXCLUDED.json` records this. It contributes no
metrics or comparisons here. `control-v2-old-build` is also not the six-epoch
control reported above. The valid `*-current` harnesses link the current
`LoomNative.o` directly; a cached `libLoomNative.a` must not be used to reproduce
this experiment.

From the LoomFactory repository, build and link a new harness to the just-built
object. Use fresh output directory names; the harness rejects existing ones.
These commands perform new GPU training and should run sequentially:

```sh
swift build --configuration release --product LoomNativeProbe
task_products="$(swift build --configuration release --show-bin-path)"
task_receipts='/Users/andrewsmith/Library/Application Support/Loom Factory/Validation/2026-10-15-native-detail'
task_request='/opt/ipde/sources_mats/.loom-factory/runs/395fb421-7868-4f51-affe-018e47cbf83f/request.json'
task_weights='/opt/ipde/sources_mats/.loom-factory/runs/395fb421-7868-4f51-affe-018e47cbf83f/checkpoint-6e34cead-75ec-44f9-a68b-132cb6f48612.safetensors'
swiftc -O -I "$task_products" "$task_receipts/detail-replay.swift" \
  "$task_products/LoomNative.o" -o "$task_receipts/detail-replay-reproduced"
"$task_receipts/detail-replay-reproduced" "$task_request" "$task_weights" \
  logit-cross-entropy-v2 "$task_receipts/control-v2-reproduced" 6 0.0001
"$task_receipts/detail-replay-reproduced" "$task_request" "$task_weights" \
  native-detail-v3 "$task_receipts/detail-v3-reproduced" 6 0.0001
swiftc -O -I "$task_products" "$task_receipts/shift-probe.swift" \
  "$task_products/LoomNative.o" -o "$task_receipts/shift-probe-reproduced"
"$task_receipts/shift-probe-reproduced" "$task_weights" \
  /opt/ipde/sources_mats/red_brick_03/red_brick_03_diff_1k.png \
  /opt/ipde/sources_mats/red_brick_03/red_brick_03_disp_1k.png
"$task_receipts/shift-probe-reproduced" "$task_receipts/detail-v3-reproduced/model.safetensors" \
  /opt/ipde/sources_mats/red_brick_03/red_brick_03_diff_1k.png \
  /opt/ipde/sources_mats/red_brick_03/red_brick_03_disp_1k.png
swiftc -O -I "$task_products" "$task_receipts/full-resolution-check.swift" \
  "$task_products/LoomNative.o" -o "$task_receipts/full-resolution-check-reproduced"
"$task_receipts/full-resolution-check-reproduced" "$task_receipts/full-resolution-1k-reproduced"
"$task_receipts/shift-probe-reproduced" "$task_receipts/full-resolution-1k-reproduced/model.safetensors" \
  /opt/ipde/sources_mats/red_brick_03/red_brick_03_diff_1k.png \
  /opt/ipde/sources_mats/red_brick_03/red_brick_03_disp_1k.png \
  164 4 loom-factory-material-residual-v1
```

`compare.py` reads the original `control-v2` and `detail-v3` directories. Running
it again recomputes the existing receipts; point a copied script at reproduced
directories to compare a new run without replacing these receipts. It requires
NumPy and Pillow. The Swift harness source fixes the architecture, seed and
augmentation settings specified above.

## Integration validation and handoff

- `swift test --no-parallel --filter 'LoomNativeTests|LoomCoreTests'` passed
  60 native XCTest cases and 185 core XCTest cases, with three opt-in fixture
  skips per target, plus two core Swift Testing cases. Receipts:
  `native-core-tests.log`.
- The extended stopped-mid-epoch regression passed separately after review.
  It branches the partial checkpoint under v3, compares its final weights exactly
  against a branch from the same promoted checkpoint weights with a fresh
  optimizer, checks nine new optimizer updates and inherited one update/image,
  and verifies preservation of the original checkpoint. Receipt:
  `mid-epoch-objective-test.log`.
- All 65 UI Swift Testing cases passed with `--skip-build --no-parallel` in
  28.013 seconds. The objective selector and saved-version controls were rendered
  and inspected; renders are under `/opt/ipde/ipde/out/native-detail-ui`.
  Receipt: `all-serialized-ui-tests.log`. This is rendered/state verification,
  not a claim of live installed-button interaction.
- `Scripts/build-app.sh` produced the optimized Release app;
  `Scripts/install-app.sh --skip-build` installed it into `/Applications`.
  Native bundle verification and strict signing checks passed. Built and installed
  executable SHA-256 values match:
  `4ace61e7435bdc85a29a939afc0f455475d431d51543067ab0dbbc0108fc1782`.
  The previous app was retained at
  `/Users/andrewsmith/Library/Application Support/Loom Factory/Backups/Loom Factory.backup-20261015-005631-2B1ED310.app`.
  Receipts: `release-build.log`, `installation.log`,
  `installed-verification.log`, `integration.json`.

The source project received no new trained candidates or changed model
assignments from these experiments. Experimental outputs stay in the separate
validation directory. Fresh full-resolution training is required to replace a
Pyramid architecture; old Pyramid weights cannot be converted by changing a
setting.

The short fallback check establishes execution and spatial consistency only.
Qualifying learned material quality still requires a stable, matched
full-resolution versus Pyramid comparison on whole held-out materials,
native-scale inspection, and downstream depth use. No candidate in this report
has met an artifact-free acceptance criterion.
