# Source recovery and crop audit — 2026-11-02

The Tiles And Pavers normal plan showed 18 pairs at 1K, 15 at 2K and 18 at
4K. The difference was real: `anti_skid_tiles_nor_gl_2k.png`,
`square_pavers_nor_gl_2k.png` and `stone_tile_wall_nor_gl_2k.png` were absent.
Their diffuse counterparts and other resolution levels were present and tagged.

Git history confirms that the old setup workflow recovered missing source tiers
(`ipde` commits `dec6021` and `a814146`). The fresh Factory started by selecting
existing physical pairs (`1a5fa45`); it never acquired an equivalent recovery
step. The retired Texture Studio implementation was removed in `ipde` commit
`8920063`. The repair uses a new native provider client, with no dependency on
retired trainers, manifests, recovery receipts or executables.

## Changes

- Prepare missing Poly Haven diffuse and selected target originals before the
  final training snapshot and pair selection. Expose the same operation through
  **Check & Download** so users can inspect the counts without training. Refresh
  the visible dataset immediately after preparation during a training start,
  rather than leaving the old counts on screen until that run finishes.
- Use the provider's live file URLs, published sizes and MD5 hashes, then check
  PNG format and native dimensions. Keep existing bytes, metadata choices and
  ignored files intact. Preserve alternate diffuse variants and existing paired
  files even when their extension/name differs from the provider filename.
- Keep the operation lease throughout preparation and training. Show download
  events, support task cancellation and append an audit log to the project.
  Before training has started, Stop cancels preparation rather than attempting
  to save a nonexistent checkpoint.
- Use a seeded permutation of every valid native crop origin. Every epoch gets
  a different position until the finite origin domain is exhausted. Input and
  target share exact coordinates and flips. No resizing/interpolation is added.
- Enforce per-epoch crops and disable rotation at both training entry points;
  remove the rotation and crop-refresh controls for new runs. Retain historical
  settings and low-level transforms for compatibility. Record sampler v2 so an
  old cursor cannot silently resume with a different sampling algorithm.

At a 1024-pixel training grid, 2K has 1,050,625 valid origins, 8K has 51,394,561,
and 16K has 235,960,321. A 1K original has just one. The old sampler already drew
random in-bounds coordinates at its selected cadence, but drew with replacement.

## Live evidence

The production-corpus integration check downloaded precisely the three absent
files and verified their provider checksums, byte counts and native dimensions.
Every pre-existing Tiles image retained its original SHA256. A fresh scan reports
18 training normal pairs at each of 1024, 2048 and 4096. A second preparation
downloaded zero files. No real model training or model replacement was performed.

The saved Tiles depth request `21df3392-fb85-493f-a41c-40a5d027b5dd` used
18/18/18 pairs at 1K/2K/4K. The saved normal request
`e12ed927-f9ad-4ee2-bd47-80ccf5d1c729` used 18/15/18. Both selected per-epoch
crops. These facts do not establish that all models need retraining or that
existing model quality is invalid. The affected normal model can be compared
against a continuation or fresh run using the repaired data.

Focused checks cover 1,000 unique bounded origins at 2K, 8K and 16K, rectangular
sources, complete small-domain coverage and exhaustion, aligned numeric pixels,
independent transform streams, sampler serialization, seeded training, lookahead
parity, exact checkpoint resumption, source-level planning and rendered controls.
The first broader run exposed different crop policies in the public entry point
and direct worker; central enforcement at both entry points fixed the resulting
resume mismatch. The subsequent focused run passed. UI snapshots were inspected.

Evidence is retained in user storage at
`~/Library/Application Support/Loom Factory/Validation/2026-11-02-source-crops/`,
including test logs, live download logs, saved-request counts and UI renders.
Release 0.2.3 (build 5) was built, signed and installed. Its installed executable
matches the packaged Release executable byte for byte, and model assignments
retain their pre-repair SHA256. Installation did not launch the app.

## Scope and limits

Recovery recognizes standard Poly Haven filenames, including numbered diffuse
variants. Custom names and other providers are reported explicitly and continue
to use their current local files. No provider identity is fabricated. Published
tiers or PNG maps that do not exist are reported rather than synthesized.
All available fills intermediate gaps through each material's highest local
tier; it does not download every 8K/16K remote tier automatically. Levels 1–5
request through 1×/2×/4×/8×/16× the base native size, capped at published 16K.
With a 1K base, levels 4 and 5 explicitly request 8K and 16K source originals.
Native 1K remains the minimum
published tier for smaller training grids. Training dimensions remain separate
from source resolution: large originals still supply 1K crops.

Uniqueness is per source pair within the run/sequence, not a claim that arbitrary
overlapping crops or differently named duplicate source files contain different
pixel content. Sources with fewer origins than epochs eventually exhaust their
finite domain. Validation retains its fixed center crop and whole-folder split.
Existing cursors lacking the new sampler contract start a new sampling segment
from saved weights; compatible current checkpoints resume the exact sequence.
