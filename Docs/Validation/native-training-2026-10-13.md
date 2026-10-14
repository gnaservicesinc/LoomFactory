# Native migration verification — 2026-10-13

The new Loom applications had reintroduced private Python workers after Texture
Studio's earlier native migration. This change replaces their training, image
preparation, model export and publishing paths with native Swift. The app build
rejects interpreters and worker resources; installers and their tests use Swift.
Existing source images, settings and saved weights remain in their source project.

Native training keeps model parameters, accumulated gradients and optimizer
moments on Metal. GPU results use explicitly owned contiguous buffers at graph
boundaries. Repeated full-size real-source tests exposed incorrect gradients
from implicit graph result storage; owned output buffers produce identical
losses, norms and final weights across fresh and Safetensors-loaded sessions.
Prefilling those buffers with NaN bit patterns still produced the same results,
showing that initialization was not concealing unwritten output values.
The exact internal MPSGraph cause remains unproven; the storage change is
validated by the repeated results and independent gradient checks.

The cancellation flag is independent of the project actor. Stop checks occur
throughout pixel preparation, file hashing and at GPU boundaries. Completed
updates are checkpointed, including the first update before the first epoch ends.
Cancellation never publishes a completed model or adapter descriptor. Core ML
export preserves the existing residual-v1 network, Safetensors keys and exact-base
LoRA contract. CPU training is unavailable; the Apple GPU backend is explicit.

## Image preparation on this M2 Max

Measured first-use decoding and exact 1024-pixel crop preparation from real
16-bit PNG pairs; originals' byte hashes were unchanged.

| Original dimensions | Previous Python | Native Swift | First-use speedup |
| --- | ---: | ---: | ---: |
| 1024 × 1024 | 0.689 s | 0.069 s | 10.0× |
| 2048 × 2048 | 2.785 s | 0.190 s | 14.7× |
| 4096 × 4096 | 11.629 s | 0.642 s | 18.1× |

The native path caches original numeric codes within 512 MiB and converts only
active crop pixels to floats. Python's cached NumPy crop alone was faster
(approximately 2 ms versus native 31 ms) while both complete decoded arrays stayed
in its 256 MiB cache. Cold decoding matters when visiting uncached sources or
when the full multi-resolution corpus exceeds the cache. These timings measure
image preparation, not total training speed or prediction quality.

Full 1024-pixel GPU updates for the 1,943,893-parameter width-164 model remain
approximately 2.7 seconds after compilation in both implementations. The
native path removes the slow interpreter-based preprocessing; it does not claim
an order-of-magnitude improvement in convolution computation. The raw JSON
contains individual measurements, source hashes and cancellation evidence.

## Validation boundaries

The native tests cover independent finite-difference gradients, analytic masked
large-grid gradients, all optimizer equations, accumulation, actual training
updates, normal vectors, frozen-base LoRA fusion, whole-material validation,
best/latest weight selection, source membership, Stop/checkpoint continuation,
precision and dynamic-size Core ML parity. Publishing tests mock HTTP and do not
upload a model. Optional external classifier, source and installed-document
integration fixtures remain explicitly skipped.

Final suites passed: Factory 99 checks (2 optional fixture skips), Workshop 41
checks (2 optional fixture skips), with no failures. The Factory run enabled the
real full-resolution Bark fixture. Native build and installer checks also passed.

The computer-use service failed to start. Core and offscreen UI tests establish
execution behavior; they do not establish that a live button was clicked.
App installation is verified separately by bundle metadata, signatures,
executable identity, process launch and absence of interpreter resources.
