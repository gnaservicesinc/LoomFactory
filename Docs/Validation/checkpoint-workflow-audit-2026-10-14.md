# Training checkpoint and workflow audit

This audit covers the native training lifecycle in Loom Factory and Loom
Workshop: saving during a run, safe stopping, continuation, source switching,
quit, failure recovery, saved-run discovery and final result publication.
It does not qualify learned model quality or certify every application feature.

## Findings addressed

- Factory exposed only scheduled saves and a generic Stop control. Manual
  Save Checkpoint and Save & Stop now queue work on the serial training owner
  and report the completed update, success or failure to the user.
- Factory checkpoints omitted optimizer and partial-epoch continuation state.
  New saves include optimizer moments, completed updates, sample order, crops,
  augmentation/random state and the current epoch metrics. Exact continuation
  requires compatible settings and the same source identities and pixel hashes.
  Older saves remain usable as weights with an explicitly fresh optimizer.
- Replacing fixed checkpoint weights before replacing the descriptor could
  invalidate the previous good save. New checkpoints publish immutable tensor
  revisions through an atomic descriptor replacement. Failed publication leaves
  the old checkpoint usable. Previous owned revisions are reclaimed only after
  successful publication.
- Opening another source folder could silently cancel a training run. The menu
  and model now prevent source replacement during active work.
- Quitting during training silently refused the request. The app now offers
  Keep Training or Save & Quit, waits for a successful save and stops only then.
  Failed saves keep the app and training alive.
- The original cancellation message claimed progress had been saved even when
  checkpoint writing failed. Save status now reflects the actual outcome.
- Workshop lacked manual save/stop controls and checkpoint continuation. Its
  corresponding controls, saved-run discovery and safe continuation are part of
  the companion Workshop change.
- Workshop could remove a completed adapter manifest after a later cancellation
  or failure. Completed validated results are preserved.

## Running job preserved

The existing installed Factory process was left running throughout development.
Its Bricks depth run uses Pyramid width 55, four blocks, 1024-pixel crops,
batch size one and 44 pairs. Its epoch-100 checkpoint contains 4,400 updates.
The weights SHA256 was verified as
`626c37d3a81354ddfaa56d42736d24fc09755b21c13198e7de941b379fb5eed3`.
The user subsequently stopped and closed the app. Its final checkpoint contains
4,497 updates after 102 completed epochs, with verified SHA256
`49df7e7b272c70f88916a532413dcee789efcaf4c0b71f003fc8dd9a62860bf3`.
That existing run was backed up before installation. It is a legacy weights-only
checkpoint; its missing optimizer moments cannot be reconstructed by the update.

Before test contention, 60 actual update cycles averaged 0.65503 seconds:
0.64544 seconds training, 0.00781 seconds optimization and 0.00053 seconds
preparation wait. The prior memory benchmark used width 47, so it is not a
matched speed comparison with this larger model.

## Validation boundary

Validation evidence and production bundles are retained outside the repository
under `~/Library/Application Support/Loom Factory/Validation/2026-10-14-checkpoint-audit`.
Desktop control failed with `Sky Computer Use native pipe startup failed`.
Automated state tests and offscreen rendering are not interactive click-through
evidence. A replacement app cannot change the code already executing in the
running training process; installation must preserve that process until it is
stopped and closed.

Factory checkpoint tests verify a live save continues training, Save & Stop
acknowledges only a durable completed update, and an identical partial-epoch
resume produces the same optimizer, crop order and weights as uninterrupted
training. Publication tests inject failure after tensor writes and verify the
previous checkpoint remains valid, failed revisions are removed, and successful
revisions keep disk usage bounded. Quit tests cover retaining active work,
waiting for exactly one save, and refusing termination after a failed save.

Workshop's full suite passed 36 core tests (two environment-only checks skipped)
and 13 UI tests. Its exact-resume integration compares shuffled order,
per-sample augmentation, accumulation, AdamW, warmup and cosine scheduling across
a mid-epoch stop and resume. This also exposed and fixed canonical-path errors
between macOS `/var` and `/private/var` in source signatures and package paths.
