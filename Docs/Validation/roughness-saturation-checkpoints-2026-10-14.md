# Brick roughness saturation and saved checkpoint visibility

Loom Factory 0.2.2 build 4 exposes checkpoint-only training versions in Models
and uses `logit-cross-entropy-v2` for fresh scalar training. Existing saved runs
retain their recorded objective; missing objective IDs mean `absolute-v1`.
Inference architecture IDs, image values and normal-vector training are unchanged.

## Observed failure

The source project originally contained four Bricks runs: completed normal
`93d7aabc`, completed roughness `efd24701`, cancelled depth `26b366b3` with a
102-epoch checkpoint, and inactive depth `6c09932c` with only an epoch-zero
checkpoint despite 28 epochs in its progress history. Models counted completed
exports and hid unfinished checkpoints. The grid now lists saved checkpoint
counts separately, with visible Test, Continue and Show folder actions.
Checkpoints still cannot satisfy deployment approval or package coverage.

The roughness run's saved Float32 predictions were within 0.00001 of white on
three tested originals. Teaching maps contained real variation: red brick 03
ranged 0.431357–0.752071 and brick wall 10 ranged 0.735668–0.908171.
The original sigmoid/L1 objective lost its gradient at saturated predictions:
mean gradient norm fell to 6.99e-12 at epoch 5 and remained nearly zero.
Direct inference from saved native weights ruled out Core ML export and image
preview precision as the cause.

## Correction and evidence

Stable scalar loss uses pre-sigmoid logits and soft targets without transforming
the targets. Its corrective gradient survives white/black saturation. Objective
IDs are stored in settings, requests, checkpoints and model descriptors; legacy
continuation retains its loss and optimizer/history compatibility checks include
the objective. Test output records now include raw numeric statistics and warn
when scalar output is almost entirely at an endpoint, without changing pixels.

Focused native/core integration checks passed: 31 tests, 2 opt-in fixture skips.
The saved failure opt-in ran separately and passed, along with analytic gradients
at zero/extreme logits, near-endpoint targets, normal parity and optimizer restore.
Core ML export and checkpoint continuation passed in the isolated task patch.
Coverage, settings, output diagnostics and UI checks passed. A pre-existing UI
fixture omitted native input/output names; supplying those names repaired the
fixture while preserving its package-restoration assertion.

A copy of the failed weights recovered variation after 100 small diagnostic
updates: range 0.601966–0.833444; MAE 0.021929 versus 0.208239 initially.
A fresh 1024-pixel replay used all 44 original pairs, width 55, four blocks,
seed 407897, AdamW at 0.001, weight decay 0.01, clipping 1, and the original
shuffle/crop/flip schedule. Six epochs (264 updates) completed in 313.90 seconds.
No update had a zero gradient norm, and all 11 full-size training material
diagnostics varied (combined range 0.612298–0.918172). All 88 source hashes matched
the failed run's recorded proofs; all 99 source/failed-run files stayed unchanged.
This verifies collapse prevention through the old failure point, not final
texture quality or generalization on held-out materials.

The actual-source Models view was rendered offscreen and showed both depth
checkpoints. Before/after checks found no changes to 29 project JSON records or
1,895 image metadata fingerprints. Native computer-use startup failed, so no live
button clicks are claimed.

## User-requested cleanup and installation

After validation, the user explicitly requested deleting the failed roughness
model. The project lifecycle API moved only `efd24701` to macOS Trash. Its 11
files remained intact there; the other 24 run files were unchanged. The project
now contains the normal model and two depth checkpoints. A fresh roughness run
uses the corrected loss automatically. Historical test outputs remain available.

All three apps were built and installed as Release through `make install`;
the final Factory patch was rebuilt and installed afterward. Installed Factory
matches its built executable byte-for-byte, passes strict deep signing and native
bundle checks, and is version 0.2.2 build 4. Installed Studio smoke passed.

Raw replay and deletion receipts are in user storage under
`~/Library/Application Support/LoomFactory/Validation/roughness-v2-20261014`.
The actual-source render is `out/model-coverage-2026-10-14` in the suite checkout.
