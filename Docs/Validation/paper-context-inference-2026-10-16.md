# Paper inference borders, 2026-10-16

The reported Paper side strips and additional 2K interior bands reproduce with
saved run `ec471005-e4a6-49b7-8bd2-13920027ad6f`: Pyramid v1, width 55,
four blocks, 4,033,591 parameters, native 1024 training crops. This run finished
500 epochs / 5,500 updates in 3,559.54 seconds. Its saved model SHA256 is
`7bafaca2765db31592640790cb2493f34df0c03cda25b37be084771763bb6468`.
No training or source pixel changes were needed to reproduce or correct the bands.

## Cause and correction

The old 2K planner uses 1024 tiles starting at 0, 896 and 1024 in each axis.
Although it feathers overlaps, it assigns positive weight to predictions near
each tile's zero-padded graph border. Those unreliable predictions contaminate
the stitched interior. A 1K image has no interior joins but retains the graph's
response at its outer edge. Paper003's source color does not contain matching
column jumps; rerunning legacy inference reproduces the original 1K EXR pixels
bit for bit.

`context-reflect-valid-v1` keeps the saved native graph unchanged and gives each
output pixel one valid tile interior. Surrounding context comes from adjacent
native input pixels. Coordinates outside the actual image reflect without
repeating the endpoint, in temporary inference memory only. No resampling,
blur, target adjustment, source rewrite, contrast normalization or output
feathering is performed. This is an explicit new inference boundary policy,
not a different learned model. The old policy remains available for comparison.

For this Pyramid, a conservative 44-pixel halo covers the finite convolution
path. Origins stay aligned to the two encoder strides (four-pixel phase), and
936-pixel cores are copied from 1024-pixel model inputs. Other Pyramid block
counts use `8 * blocks + 12`; full-resolution residual models use
`2 * blocks + 2`. Compact NAF has global attention and never uses this tiled
path: its complete-field inference and raw output contract remain unchanged.
An inference tile too small for a model's halo fails explicitly.

## Verification

Evidence is saved outside `/opt` in
`~/Library/Application Support/Loom Factory/Paper-border-diagnosis-2026-10-16`.

- The original 1K and 2K Paper003 files and exact saved model were tested.
  Input and weight hashes were checked; selection bytes were unchanged.
- Legacy 1K replay has maximum absolute error **0** against the reported EXR.
- The corrected 2K output has no corresponding central or outer border bands
  in matched fixed-contrast inspection. Its strongest interior column-mean
  jump is about **0.0000856**, versus **0.0023868** before. These measurements
  diagnose the bands; they are not a material-quality score.
- At native 2048 resolution, nine context tiles were compared against one
  complete 2136-square reflected input using the same saved 4M model and graph.
  All 4,194,304 output pixels match exactly: maximum absolute difference **0**.
  The test performed **zero** optimizer updates.
- CPU tests cover odd rectangular images, exact pixel ownership, stride phase,
  reflection, and rejection of a synthetic tile-border bias that the old
  feather admits. Existing saved-model/Float32, Compact whole-field and Blender
  export tests pass. Hosted UI tests exercise same-checkpoint policy A/B.
- Two explicit GPU tests require `LOOM_PAPER_CONTEXT_INFERENCE=1` and
  `LOOM_PAPER_CONTEXT_PARITY=1`, respectively, plus an evidence output directory
  in `LOOM_PAPER_CONTEXT_OUTPUT`. They use the named project/run above, acquire
  its operation lease, and never train. Ordinary tests skip them.

## Cost and limits

GPU input scratch remains bounded by the original inference tile dimensions.
This 1K image now uses four tiles instead of one; 2K uses nine, the same count
as before. Extra context can increase inference time, especially for a source
exactly equal to the old tile size. Training cost and checkpoints are unchanged.

The corrected Paper prediction remains muted and lacks some source target
microdetail. Removing seams does not establish that the learned model has the
desired artistic detail, and it does not qualify Pyramid or Compact for every
material. Existing weights can be retested before spending another training run.
This change is scoped to Factory's native model-test inference; deployment
consumers are not claimed to use this policy automatically.

The replacement Release app is supplied for manual installation. It is not
installed or launched as part of this diagnosis. UI evidence is a hosted render
and control test, not an installed-app click-through claim.
