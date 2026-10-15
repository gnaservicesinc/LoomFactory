# Native training diagnostics

`LoomNativeProbe` measures native execution and compares a saved model's raw
Float32 prediction with an unchanged teaching map. Build the Release executable
with `swift build -c release --product LoomNativeProbe`. Obtain its actual output
directory with `swift build -c release --show-bin-path`.

For a saved checkpoint, use the exact architecture, width, blocks, map kind and
scalar objective in its `checkpoint.json`. For example:

```sh
"/build/output/directory/LoomNativeProbe" --predict-only \
  --input /path/to/diffuse.png --target /path/to/depth.png \
  --weights-in /path/to/checkpoint.safetensors \
  --architecture loom-factory-material-pyramid-v1 --width 38 --blocks 11 \
  --map depth --size 1024 --scalar-loss logit-cross-entropy-v2 \
  --output /path/to/validation/report.json \
  --prediction-out /path/to/validation/prediction.exr \
  --target-out /path/to/validation/target.exr
```

`--predict-only` makes no optimizer updates and requires supplied weights. The
probe reads the centered exact native crop of `--size`; it does not resize the
source or target. The optional EXRs contain the exact Float32 prediction and
target bits without normalization or quantization. Output paths must differ
from each other and from input images or supplied weights, including symbolic
and hard-link aliases.
Source and weight hashes are checked again before completing the report.

The report records the selected objective and evaluation loss. The default
objective remains `absolute-v1` for compatibility with old probe invocations;
always pass `--scalar-loss` when comparing recorded training losses. Both
objectives share the same inference interpretation.

`pixel_diagnostics` includes the raw target and prediction range, mean, standard
deviation, endpoint fractions, MAE, RMSE, mean signed error and correlation.
`native_pixel_differences` compares unfiltered horizontal and vertical pixel
differences at distances one, two and four. A smaller prediction difference than
the target can describe lost local variation; it does not by itself identify
its cause. Mean bias or local error can exist without endpoint saturation.

These measurements describe the exact tested source/target crop. A training
image result cannot establish generalization, and aggregate metrics cannot
replace native-size visual inspection of whole held-out materials. Run GPU
probes serially so competing processes do not contaminate timings or memory
measurements.
