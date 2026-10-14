# Preparation overlap and useful run progress

The serial training owner now prepares one upcoming minibatch on a bounded CPU
worker while the current one trains. Image/crop order, weights, accumulation
boundaries, source verification and original pixels are unchanged. The run panel
shows material/image, stage, epoch/image/step positions, steps left, cycle speed,
elapsed time and qualified remaining-time estimates. Measurements and raw events
are collapsed by default.

Performance evidence on this Apple M2 Max:

- Synthetic 1024 × 1024, width 116, four blocks: warmed forward/backward plus
  optimizer took 1.398 and 1.396 seconds per image; first compilation cycle
  took 1.873 seconds.
- Real `red_brick` 8192 × 8192 diffuse/depth originals, cropped exactly to 1024:
  cold preparation took 2.425 seconds; cached preparation averaged 0.032 seconds.
- Eight different real 8K Bricks pairs, 1K crops, width 116, four blocks, batch
  one, accumulation two, one epoch: serial wall time 41.631 seconds; lookahead
  wall time 30.198 seconds, a 27.46% reduction. Preparation work stayed about
  23.5 seconds; exposed preparation wait fell from 23.541 to 12.585 seconds.
- Every measured loss and gradient norm matched exactly. Both executions saved
  weights with SHA-256
  `e17b25aa2cb63959d4201226f68fae503e027e799c443dfbf19ef1485cf631f8`.
  All 16 original source hash receipts matched. This is a cold, 8K-heavy execution
  fixture; it does not establish a universal speedup or learned texture quality.

Validation:

- The isolated staged change, compiled without unrelated checkout changes,
  passed 18 core tests and six native UI tests with zero failures. Coverage
  includes optimizer parity, uneven accumulation, real workflow telemetry,
  cancellation, source changes, errors, rolling speed, time estimates, stage
  overruns and a 290-point native sidebar render.
- The optimized Release bundle built, passed native bundle/signature checks,
  installed and launched. Built and installed executable SHA-256 matched:
  `b0e7f41d2ed09ccca1150ae46c47d83ff177d471c0dbfabfd6d2f6683fb67e57`.
- Live interactive UI inspection was unavailable because the computer-use tool
  returned `Sky Computer Use native pipe startup failed`. Native offscreen
  rendering provides the UI evidence. AppKit logs confirmed launch followed by
  approved normal termination; a persistent interactive session was not verified.

Receipts and the rendered panel are under
`~/Library/Application Support/Loom Factory/Validation/training-progress-2026-10-14`.
