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
