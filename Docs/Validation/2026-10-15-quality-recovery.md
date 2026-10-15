# Material quality recovery and bounded training

The source project is `/opt/ipde/sources_mats`. Diagnostic receipts, predictions,
matched three-map renders and failed experiments are in user storage under
`~/Library/Application Support/Loom Factory/Validation/2026-10-15-quality-recovery`.
Original source pixels and pre-existing model/checkpoint tensors were preserved.

## Confirmed regression and reference recovery

Fresh-run preparation previously silently changed an explicitly selected
Pyramid recipe to a parameter-matched full-resolution residual network and
changed its scalar objective to v2. Starting fresh now preserves the selected
architecture, width and objective. Legacy missing objective IDs still mean v1.
Compact retains its own recorded linear-detail contract.

Both detailed brick images supplied by the user came from checkpoint
`26b366b3-d02c-4700-9fc1-305e54fd137b`: Pyramid v1, width 55, four blocks,
4,033,591 parameters, native 1024 crops, LR 0.001, AdamW, and original absolute
error. Current inference reproduces the saved 1K reference PNG within one 16-bit
rounding code. That checkpoint was exported as a separate selectable Core ML
version, preserving its exact tensor SHA256:
`49df7e7b272c70f88916a532413dcee789efcaf4c0b71f003fc8dd9a62860bf3`.
The earlier normal and roughness versions remain the other brick references.

The original reference has stronger visible relief than the short new pilots,
but also contains false fine relief and poor absolute height calibration versus
the artist target. Sharpness is not a sufficient quality measure. The same
input, all three predicted maps, all three artist targets, fixed geometry,
displacement scale and two light directions are used in the comparison renders.
Both neutral and diffuse views are retained.

## Experiments rejected before the overnight run

A short continuation of the original Pyramid checkpoint became flatter. An
experimental offset-invariant reconstruction loss drifted too bright; a
calibrated variant also failed to improve the combined material. Neither loss
is shipped as an application option. Their complete replay binary, source patch,
new run directories and outputs remain recoverable in the diagnostic folder;
no pre-existing run was removed. Two-minute fitting probes do not establish
long-run learned quality or generalization.

The saved-run arithmetic audit found completely inactive deep ReLU features.
Pyramid v2 keeps Pyramid's larger spatial context and efficient feature scales,
but assigns every hidden negative activation a 0.1 gradient. The initial real
pilot exposed another failure: the legacy sigmoid/L1 depth head became exactly
white with zero gradients, and the sigmoid normal head became a constant
diagonal direction. Stable scalar logit objectives and a normalized raw normal
vector avoid these head failures. Large wrong normal logits retain a finite
corrective gradient in the numerical regression. This is a new,
explicit constructor, not a reinterpretation of the earlier weights. Its
negative encoder/body/decoder forward values and gradients are independently
checked, optimizer resume is exact, and all three Core ML exports match native
inference on even and odd rectangular grids. Legacy Core ML network hashes
remain unchanged. Workshop validates the same new constructor and frozen-base
adapter export. These checks prove the implemented arithmetic, not texture
quality.

## Training and review policy

The bounded queue trains the three products serially for every current category.
It reads actual native source pixels, keeps full-resolution skip features, uses
10% whole-material validation with a shared split, saves immutable interval and
final optimizer checkpoints, and exports separate Core ML candidates. The
candidate export API preserves explicit reference assignments; the queue also
pins pre-existing implicit reference choices before creating any candidates. A process lease
prevents concurrent GUI and queue GPU operations in either start order.

The initial recipes use native 1024 crops, one source-resolution level, batch
one, AdamW at 0.0002 with two warmup epochs, fixed global clipping at one, and
no target exaggeration. Bricks uses width 55 to retain the successful model's
capacity; other categories use width 38. Depth candidates use native-detail v3 to retain corrective scalar gradients
and compare signed native detail. Roughness uses stable
logit cross entropy v2; normals use the original angular objective. The training
budget gives Bricks, Bark, Rock/Cliff/Marble and Wood more time than the smaller
categories. These are explicit recovery candidates, not certified per-category
best settings. The eight-hour run and its combined held-out renders are the next
quality evidence.

Progress, stop control, per-job settings, results, unchanged map references,
Float32 numeric comparisons and combined material views are saved outside the
repositories. SIGTERM or STOP requests Save & Stop. The training deadline is
fixed; the final in-flight checkpoint/export/render can finish afterward.

## First completed set and bounded retry

At 11:23 UTC the first complete Bricks set failed its combined material review.
The Stone Wall holdout was too smooth and shallow, with false roughness
variation. Depth correlation was -0.028 and roughness standard deviation was
0.067 versus the artist target's 0.011. The queue acknowledged Save & Stop;
all five completed or interrupted runs and their exported candidates remain.
The restored detailed brick reference remains explicitly selected.

A separate read-only evaluation of the known brick_wall_005 training material
and the brick_wall_10 whole-material holdout confirmed missing surface detail
on both. On brick_wall_10, depth standard deviation was 0.016 versus 0.154 in
the artist target; diffuse color partly hid the missing relief. The review CLI
ran all three saved checkpoint inferences, native metrics and eight fixed-light
renders per material successfully without training or changing selection.

The original successful brick request used all four available source levels,
44 pairs across 11 materials. The initial overnight recipe instead selected
only the smallest native originals: 13 training pairs and two holdouts. Most
training inputs exactly equaled the 1024 crop grid and could supply no changing
crop positions. That recipe restriction is confirmed; it is not proof that
restoring crop variety alone will solve the learned-quality failure.

The bounded retry uses all available native source levels, the same 1024 grid,
material holdouts and versioned constructors. It increases the initial learning
rate to 0.0005 with two warmup epochs to test faster adaptation while retaining
the stable scalar-logit and raw-vector heads. Prior runs are parent versions,
not overwritten. Separate retry settings, lineage and quality receipts remain
in user storage. The original 17:21:18 UTC deadline is unchanged. These remain
unapproved candidates until their three-map material renders pass review.

At 12:22 UTC the native-crop retry also failed. The completed normal prediction
on Stone Wall had per-channel standard deviations 0.000459, 0.001343 and
0.000030, against target values 0.162, 0.176 and 0.097. The neutral combined
surface was nearly flat. Separate saved-checkpoint evaluations confirmed the
same normal collapse on brick_wall_10 and the known brick_wall_005 training
material. Known-brick depth also drifted bright (MAE 0.544, correlation 0.016).
Roughness improved on that known brick (correlation 0.837), but this does not
make the three-map material usable. Native metrics and all matched renders
remain in `native-crop-retry/diagnostics` under the bounded queue directory.
Save & Stop was acknowledged at 12:22:34 UTC; the active Bark checkpoint and
every completed Bricks child version were preserved. Restoring source levels
did not solve the failure, and Pyramid v2 arithmetic tests do not establish a
successful learning recipe.

A separate three-job quality gate now compares the existing Compact NAF v1
constructor, width 32, raw linear component/detail objective, native 1024 grid,
two available native source levels, LR 0.001 and two warmup epochs. It uses the
shared brick_wall_10 whole-material holdout; a known-brick diagnostic must also
be reviewed. The 39 remaining jobs are explicit but unstarted. They may only
start after all three maps pass a combined visual review, and must fit the
unchanged 17:21:18 UTC deadline. No new constructor, objective or production
selection is introduced by this gate; its quality is not yet established.

The 18-minute Compact gate also failed joint visual review on brick_wall_10,
and a separate known-brick evaluation remained too flat. Its held-out depth
standard deviation was 0.011 versus target 0.154, with correlation -0.538.
The remaining 39 jobs were kept unstarted.

Two direct diagnostics narrow the problem. An eight-update, real 1024-grid,
width-32 Compact replay alternated three original material pairs. The persistent
session did not materialize gradients, weights or moments between updates; its
losses, gradient norms, final weights and optimizer moments matched exactly a
reference that recreated the session from CPU-restored state at every update.
This rules out the suspected changing-feed buffer error for that workload.
It does not prove long-run training quality. Separately, 100 extra updates on
one known brick recovered fitted normal structure and accurate roughness:
roughness MAE 0.007 and correlation 0.995; depth MAE 0.082 and correlation 0.513.
The combined surface still lacked much artist microrelief. That is a fitting
diagnostic, not a usable category or unseen-material result. A read-only
cross-map directional audit found consistent height/normal signs for 14 of 15
current brick materials; brick_wall_11 needs separate source review. No source
pixel or tag was modified or excluded on that diagnostic alone.

A concrete queue policy bug was found: every timed Save & Stop reviewed and
exported the last checkpoint tensors despite `exportBestWeights: true`.
The queue now exports the immutable saved best-validation tensors when present,
then tests and renders that exact model version. Checkpoints now record the
best epoch and validation score explicitly. Older checkpoints derive the epoch
from completed validation history. Corrupt best tensors fail rather than
silently exporting latest; pre-validation checkpoints use latest. Manual exact
checkpoint selection retains its original behavior. Focused lifecycle tests
check exact best bytes, lineage, epoch/loss receipts, no automatic approval,
reference/source preservation, latest fallback before validation, and corruption
rejection. A real three-map re-export from the saved Compact gate confirms
the lower validation score still does not provide usable held-out relief.

The next bounded gate continues the three saved Compact category checkpoints
with a fresh optimizer/sampler, LR 0.0002, no weight decay, and four-image
gradient accumulation (physical batch one). This tests whether steadier updates
and 20 minutes per product can retain the structure recovered by the
single-material diagnostic across the category. It keeps native pixels,
the same two source levels and whole-material split. The remaining categories
are still gated on visible joint improvement. Best-validation candidates are
reviewed by the corrected queue; the original deadline is not extended.

At 13:22 UTC a live start-state audit found that this intended continuation
actually had no parent lineage or resume path. Save & Stop preserved it. A wider
audit then found the same mismatch in all configured continuations from the
first queue and native-crop retry. Earlier descriptions of those attempts as
continued training were incorrect; their outputs remain evidence only for the
fresh runs that actually occurred. The fresh Compact gate did start as intended.

The frozen balanced runner reproduced the missing parent on three separate
one-update map jobs. A clean build retained the exact checkpoint parent and
checksum, both with the former argument expression and an explicitly typed
start. The precise artifact/compiler cause is not yet isolated; the evidence
does not establish that the argument expression itself was the defect.
Failed runners and every run remain preserved. The queue now records its
requested start and fails if saved child lineage disagrees, and the core checks
that an explicitly requested parent survives preparation. A new native training
regression checks that changing optimizer/sampler settings resets their state
while preserving checkpoint starting weights, source bytes and selection.
All 14 focused lifecycle/training tests pass.

The exact newly frozen runner passed one-update preflights for depth, normal
and roughness: each real request referenced its intended parent checkpoint
file, with the matching SHA256. These are execution proofs, not quality results.
The next gate uses those original category parents, never the mistaken fresh
run or single-material fits. It retains LR 0.0002, four-image accumulation and
the original cutoff, allocating 16 minutes to depth and normals and eight to
roughness so the combined set can be reviewed before remaining category work.

That verified Compact gate completed at 14:23 UTC with all three actual parents
intact. Both diffuse and neutral combined views still fail the quality gate:
brick_wall_10 has shallow relief, depth correlation -0.228 and standard deviation
0.0206 versus the target's 0.1535. Normals and roughness improved, but the joint
material remains unusably flat. On known brick_wall_005, depth correlation is
0.465 but relief is still weak, and roughness has a +0.090 level bias. The saved
best exports were used for every review. No remaining category job was started.

A separate read-only comparison evaluates the three selected legacy references
together. The liked old depth map has detailed but sometimes incorrect relief;
it does not establish that the legacy normal and roughness maps are good. On
Bricks072, absent from every legacy parent request, the old depth preserves
useful structure, while normals are nearly flat and roughness has a +0.381
bias. The unchanged source and candidate maps, both lighting directions,
diffuse/neutral renders and native metrics are preserved under
`verified-compact-gate/diagnostics/selected-reference-072` in the recovery queue.

The next bounded comparison continues the three original legacy category
checkpoints with the existing Pyramid v1 width-55, four-block constructor and
their saved objectives: absolute-v1 depth, angular normals and stable logit
cross-entropy-v2 roughness. It changes no constructor. Native 1024 crops use all
current original resolution levels; learning rates are 0.0001 for depth/normals
and 0.0002 for roughness, with no weight decay and four-image accumulation.
Seed 9589 resets optimizer/sampler state. Validation seed 9783 reserves
Bricks072 and brick_wall_10 across all three maps. Bricks072 is new to all three
legacy parents; brick_wall_10 was seen by those parents and is only a diagnostic.
The exact frozen runner passed real parent-path/checksum preflights for all
three legacy checkpoints before launch. Budgets are 12/12/6 minutes, retaining
the original 17:21:18 UTC cutoff. These are unapproved recovery experiments.
The failed Compact recipes for 39 unfinished jobs remain preserved and cannot
be rolled out merely because a different constructor's gate completes.

The legacy reference gate completed at 15:02 UTC, with correct parent paths and
checksums for all three maps. Best depth and normal exports came from epoch two;
later training worsened their validation scores. Native combined reviews of
both best exports and latest optimizer checkpoints show no sufficient joint
improvement. Bricks072 retains depth geometry but has nearly flat normals and
roughness bias +0.291 (best) / +0.305 (latest). Known brick_wall_005 retains the
old depth bias and loses detail at latest. The inherited brick_wall_10 diagnostic
is better structured than Compact but still lacks target relief. Neither the
best score nor the liked old depth image qualifies the three-map recipe.

No generic category rollout follows. A separate bounded Bark pilot uses the
settings of its pre-existing Compact depth model: width 32, linear-detail-v1,
LR 0.0001, no weight decay, physical batch one with accumulation one, and native
1024 crops from all current source levels. Normals start fresh for 15 minutes,
roughness starts fresh for ten, and depth continues the original Bark checkpoint
for five. Depth retains its original explicit feature-exaggeration teaching
setting; source files remain unchanged. The selected model and same-run
checkpoint have identical parameter tensor bytes despite different safetensors
file/header hashes. The exact runner's one-update Bark continuation preflight
checks its actual parent, path, checksum and settings before launch.

The shared current split uses validation seed 9587. Fresh normal/roughness maps
exclude those materials, but inherited Bark depth has already seen the entire
current Bark corpus, so the joint material is explicitly not an independent
unseen-material test. Review Bark Willow 02 plus saved-model diagnostics on
Bark Brown 01 and Pine Bark in diffuse and neutral views before judging it.
All new candidates remain unapproved; the detailed brick reference and other
prior effective assignments stay selected. Thirty-six jobs in the other twelve
categories remain unstarted. The original 17:21:18 UTC training cutoff remains.

The Bark pilot completed at 16:03 UTC and failed joint appearance on Willow 02,
Bark Brown 01 and Pine Bark. Depth standard deviation on Willow was 0.0054
versus target 0.0932; normal angular error was 28.2 degrees. Pine depth was
weakly negatively correlated with its target. Roughness followed texture on
Willow/Pine but had material-dependent level errors. None is a qualified
three-map replacement, and no category-wide rollout follows.

An explicit, opt-in Compact objective, `squared-structure-v1`, tests squared
pixel residuals plus four times the mean squared signed-residual differences
over native offsets 1, 4, 16 and 64 in both directions. Small diagnostic grids
omit offsets that do not fit. It uses unchanged target pixels, no resampling,
normalization, variance reward or sharpening. It does not change a constructor,
prediction head or inference. `linear-detail-v1` and every old default retain
their prior behavior. Changing the objective branches from exact saved weights
with a fresh optimizer; original versions and selections remain intact.
Factory exposes it as experimental for all Compact map components. Workshop
inherits the recorded Compact objective and binds resume compatibility to it.

An independent 128-grid CPU pixel calculation checks the objective including
the 64-pixel terms; a finite-difference check validates its GPU gradient.
Separate tests check exact zero error for a matching valid target, bit-identical
inference across objectives for all three map types, and a real checkpoint
branch retaining starting weights/source bytes/selection. These are execution
and numerical checks, not proof that a squared objective improves materials.
The final recovery experiment must retain the original training cutoff and
must pass joint visual review before any quality claim or model selection.
