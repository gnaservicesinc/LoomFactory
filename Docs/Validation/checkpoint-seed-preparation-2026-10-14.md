# Checkpoint choice, seed replay and native preparation

`Use checkpoint as model` exports a checkpoint's verified, byte-identical
Safetensors into a new selected deployment version. It preserves the source
run, optimizer/cursor state, existing export, human metadata and lineage. Human
approval remains separate. Tests cover stopped/checkpoint-only sources,
corruption rejection without changing selection, reopening, model testing,
package weight equality and continuation counters inherited through export.

New settings default to seed `-1`. System randomness resolves it once before
writing run metadata. An integration test checks the same resolved nonnegative
seed in the request, run metadata, checkpoint and model descriptor. Explicit
seeds remain deterministic, absent legacy seeds retain zero, and continuation
with `-1` inherits the recorded source seed.

The native crop implementation already selected arbitrary unbiased integer x/y
positions throughout the fitting source rectangle. New regressions prove edge
reachability, the full small-source origin set, large rectangular bounds,
independent refresh streams and exact paired pixels under flips/rotations.
Selected material/resolution/diffuse-variant counts and sample visits remain
unchanged. No fixed crop grid was replaced or extra crop visits introduced.

Whole-image validation records now survive decoded-pixel eviction. They are
bounded independently to 4,096 file identities; changed file identities are
validated again. The decoded-pixel budget remains one eighth of physical memory,
capped at 8 GiB. A two-epoch preparation-only comparison over the actual 66 Bricks
pairs produced exactly identical input/target crop hashes. Warm preparation time
changed from 51.64 to 50.92 seconds, only 1.4%; this cache change alone does not
resolve the measured preparation bottleneck.

Factory now prepares up to two upcoming batches concurrently. Running and
finished jobs share the two-slot bound, while results/errors, crop RNG and
optimizer updates retain their original order. Each worker operation drains an
autorelease pool. Cancellation/close joins both workers before clearing caches.
Tests prove overlap and bounds, FIFO errors, cancellation and close, and exact
serial/two-worker losses, gradient norms, weights and optimizer states across
uneven accumulation windows. Interrupted checkpoint continuation also passes.

Receipts, logs and the optimized benchmark helper are outside the repository at
`~/Library/Application Support/Loom Factory/Validation/2026-10-14-checkpoint-random-crops`.
The helper compiles the current core sources with `-O -whole-module-optimization`
and links the Release native library. Measurements run in separate serial
processes, using the actual 66 source pairs, 1024 crops, Pyramid width 38/eleven
blocks, AdamW and explicit seed 198087. Each process performs one epoch, original
source hash verification, checkpoint writing and Core ML export into the
validation directory, without adding runs to the user's source project.

| Measured workload | One preparation slot | Two preparation slots |
|---|---:|---:|
| Complete run, seconds | 80.0140 | 68.5556 |
| Training-cycle wall time, seconds | 68.5943 | 57.3409 |
| Waiting for preparation, seconds | 27.9635 | 17.2054 |
| Native forward/backward time, seconds | 39.6577 | 39.1840 |

This matched run used 14.3% less total time and 16.4% less training-cycle time.
Both final weight files have SHA256
`34f656141428ad9de815cf9228daf4d6620d4b00cda519a27454cd5fdfe7fa5c`.
The result establishes execution equivalence and this workload's improvement;
it does not establish universal utilization, speed or learned map quality.

Saved-candidate diagnostics and the matching 120-update memory probe are
documented in [checkpoint quality audit](checkpoint-quality-audit-2026-10-14.md).
The tested later maps had worse fine-edge errors without endpoint saturation.
This work does not qualify learned quality or claim to have cured those artifacts.

The final source passed 55 native XCTest checks (three optional fixture skips),
183 core XCTest checks (three optional fixture skips), two core Swift Testing
checks and all 65 UI checks with explicit serialized execution:
`swift test --skip-build --no-parallel --filter LoomFactoryUITests`.
Default UI suite overlap twice exceeded the existing 250 ms heartbeat limit in
the 1,000-map browser responsiveness test. That test passed alone, and the full
UI suite passed when overlap was disabled. No timing threshold was relaxed.
Checkpoint state/layout evidence is at
`/opt/ipde/ipde/out/checkpoint-selection-ui/coverage-select-checkpoint-dark.png`.

Desktop automation failed with `Sky Computer Use native pipe startup failed`.
Hosted workflow tests and rendered checkpoint controls provide state/layout
evidence; they are not live installed-app click-through evidence.

The optimized app was built with `Scripts/build-app.sh`, installed with
`Scripts/install-app.sh --skip-build`, and launched from
`/Applications/Loom Factory.app`. The previous installed bundle was retained by
the installer. Installed native-bundle and strict code-signature checks passed;
the build configuration is Release. Built and installed executable SHA256 values
match:
`03d4cc9a03f1460d2fa3ee54c6991fffa5a5929b3f270de87b04714e99f6355b`.
