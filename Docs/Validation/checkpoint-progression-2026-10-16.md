# Completed-run checkpoint testing and progression

The completed run's checkpoints were already retained on disk. Test Model hid
their picker while exported weights were selected, and its A/B flow required
manually testing each candidate. The page now lists exported weights and all
checkpoint revisions together and explicitly changes the weight source when a
checkpoint is chosen.

Run All Checkpoints tests the same original image in cumulative update order,
holds the project operation/GPU lease, and releases each native network before
loading the next. Each successful test retains native-size Float32 EXR output,
fixed-scale 16-bit PNG preview and its source/weights checksum record. Numbered
folders and progression.json preserve order and individual errors. Cancellation
keeps completed results. Input changes stop the sweep rather than mixing source
versions. The gallery provides A/B selectors, full-size inspection, Finder
access, and restores the latest progression for the same run and unchanged input
after reopening.

## Verification

- `swift test --filter 'CheckpointArchiveTests|ModelTestControlsTests|ModelTestingTests|ModelVersionLifecycleTests'`:
  29 XCTest cases, three opt-in skips, no failures; ten Swift Testing cases,
  one opt-in skip, no failures. Includes completed-run native inference,
  chronology, saved-result reopening, cancellation, failed checkpoints,
  input mutation, original-byte preservation and UI workflow/renders.
- Production sweep: completed Rock Cliff Marble depth Compact NAF run
  `8cebdd80-8ce5-4807-b8a7-eb3ca2eac643`, all 23 immutable checkpoints,
  existing `Rock Face 03/rock_face_03_diff_1k.png`. All 23 native 1024 x 1024
  outputs succeeded in 111.041 seconds in the debug test process. This is
  execution evidence, not a throughput benchmark or a quality endorsement.
- SHA256 checks across 80 existing run/source files found zero changed files.
  Both 1K and 2K source-image hashes were included. The production sweep used
  the existing 1K image; it does not establish 2K Compact inference behavior.
- Opt-in production reopening tests loaded all 23 saved results without
  inference and rendered the actual comparison/gallery. Repair/Lighting and
  synthetic-data sheet renders also passed after the surface-data guidance edit.
- Release app built with `Scripts/build-app.sh`; native-bundle and strict
  code-signature checks passed. Installed-app executable is left unchanged.
- Native computer-use connection failed with
  `Sky Computer Use native pipe startup failed`. Hosted renders and model
  workflow tests passed; live gesture/click-through acceptance is unverified.

Production output:
`/opt/ipde/sources_mats/.loom-factory/model-tests/progression-c29f441e-f12c-4b6c-8599-b772966423e5/`.
Evidence and rendered gallery:
`~/Library/Application Support/Loom Factory/Validation/checkpoint-progression-2026-10-16/`.

## Surface-data suitability

The synthetic generator already selects diffuse material images from the source
project. The earlier real-camera trainer diagnostic used six SIDD capture pairs
from three physical scenes, including one carpet surface. It remains diagnostic
data, not broad surface coverage. Photo Training and its guide now explicitly
request aligned close-up surface pairs, clean-surface identity controls, neutral
same-surface Lighting references, and splits by whole physical surface/session.
Existing datasets and models were preserved; no new data or trained-quality
claim was introduced by this guidance change.
