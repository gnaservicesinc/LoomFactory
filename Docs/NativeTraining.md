# Native training

Loom Factory and Loom Workshop use the `LoomNative` Swift library. It implements
the existing `loom-factory-material-residual-v1` network with Metal/MPSGraph
autodiff and GPU-resident AdamW, Adam, SGD and Lion updates. Full-model training
updates every weight; Workshop freezes the exact installed base and learns only
low-rank convolution deltas. Safetensors names and OIHW layouts remain compatible
with previously saved models.

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

Core ML export serializes the actual residual graph and float32 weights in Swift.
Apple Core ML compiles the exported `.mlmodel`, and numerical tests compare native
predictions with the Core ML graph, including scalar and normal products.

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
