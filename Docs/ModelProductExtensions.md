# Model product extension design

Loom Factory currently trains depth, roughness and normal products. Their
versioned architecture IDs, explicit native input/output contracts, and package
catalog are the extension boundary. The current package format is deliberately
limited to those implemented products. Photo-to-diffuse generation and additional
processing stages are future work.

Before introducing another product, consolidate the existing product rules into
one small product catalog and an architecture-constructor registry. This keeps a
new product from requiring separate, potentially inconsistent conditionals in
the trainer, export, Studio inference and Workshop adaptation code. Do not add a
generic plugin or graph execution framework ahead of a concrete requirement.

A product is distinct from an image's map role. The product definition declares:

- Stable product ID, version and human display name.
- Input and target image roles and the required paired teaching data.
- Native tensor/image names, layout, channels, color space, value range and size
  constraints, including where normalization takes place.
- Architecture ID, construction settings, training loss and permitted transforms.
- Output interpretation and whether that architecture supports LoRA adaptation.

Modifying a product's settings belongs in that definition and its implementation.
A change to its tensor, training or interpretation contract requires a new
version. Architecture constructors should be registered by immutable IDs. Unknown
IDs must be rejected, rather than interpreted as the current residual network.
Continuation and adapters must still match the actual architecture and weights.

Scalar training objectives have independent immutable IDs: `absolute-v1`
preserves the original sigmoid/L1 objective and `logit-cross-entropy-v2`
trains from pre-sigmoid logits to retain gradients at saturated endpoints.
The objective ID is recorded in run settings, requests, checkpoints and exported
model descriptors. Missing IDs identify v1. Fresh scalar runs default to v2;
continuation retains the saved objective. The explicit experimental
`native-detail-v3` choice adds signed original-pixel difference matching at
1, 2 and 4 pixels to stable v2 cross-entropy. Objective changes start a new loss
history, optimizer state and epoch schedule. Prior metrics retain their original
objective labels. These objectives share the exact same inference graph,
tensor interface and architecture IDs, so changing the objective does not
reinterpret existing weights or require another inference constructor.

For a product that needs several stages, introduce an ordered, versioned stage
plan with the product's real package contract. Each stage declares its inputs,
outputs, model artifact, required or optional status, and when it should run. An
optional stage can be bypassed only when the preceding output satisfies the next
stage's input contract. Every possible stage's native model and weights are
packaged and validated at build time; stages cannot silently refer to missing
models or obtain replacement models after installation.

The plan must state which stages run before category classification and which
run after routing to a category. A photo-to-diffuse stage may need to precede a
classifier trained on diffuse maps; it must not create a circular dependency on
that classifier. The classifier, product plans and category models remain one
fixed Texture Loom deployment. Studio and Workshop cannot swap the classifier
independently.

Photo-to-diffuse is an RGB prediction product with its own teaching data, loss
and color contract. It is not another scalar map or a normal-vector prediction.
Potential learned cleanup/redrawing stages should remain optional until tests
show that they improve usable material detail. Learned redrawing can invent or
remove detail; preserving every detail is a quality goal to evaluate, not a
guarantee. Keep original images untouched and compare each stage and its bypass
on held-out materials before deployment.

Extension checks should cover native export/inference parity, paired input
membership, stage input/output compatibility, optional-stage bypass, complete
category routing, and exact base/adapter identity. Change the package schema when
its meaning expands; older consumers should reject an unsupported product or
stage contract clearly.


The optional `loom-factory-material-compact-naf-v1` constructor preserves the
earlier seven-stage NAF design with reflection padding and a raw linear output.
Its scalar and normal component training uses `linear-detail-v1`, with no
sigmoid or normal renormalization. It retains the existing RGB sRGB NCHW input,
native pixel grid and material output name, but declares its raw Float32 output
contract explicitly. Consumers validate this architecture and its metadata;
legacy bounded models keep their existing interpretation. Whole-field spatial
attention precludes silently substituting tiled inference. Widths are editable,
blocks per stage are fixed at one, and native dimensions must be multiples of
eight and at least 32. Only dense convolutions are adapted by LoRA.


`loom-factory-material-residual-v2` keeps the full-resolution residual tensor
shapes and bounded scalar/normal output contract, but uses leaky ReLU with a
negative slope of 0.1 at every hidden activation. Its descriptor must explicitly
record `hidden_activation: leaky-relu` and `negative_slope: 0.1`. This provides a
nonzero gradient through negative hidden features; the original residual and
pyramid IDs retain their exact ReLU graphs. Selecting v2 starts a new model;
old checkpoints are never silently reinterpreted. CoreML export and Workshop
adaptation validate the new identity and activation metadata. Workshop checkpoints
also bind the architecture and training objective, because identical parameter
shapes and weight hashes alone cannot distinguish these two graphs.

The random seed preference is distinct from its resolved run receipt. A requested
`-1` draws a system seed on every start, including continuation, and resets the
optimizer and sampler state. A fixed seed admits compatible saved state. Changing
seed, crop size, target preparation or optimizer contract starts new optimizer
state; changes to image-cache mode and budget do not change training math.

Decoded source pixels are retained only when the user enables a bounded per-run
RAM cache. Images enter on first use, with bounded worker prefetch in both cached
and streaming modes. File identities are captured cheaply at admission; full
source hashes are computed lazily. Early checkpoints preserve weights and
optimizer without waiting for unvisited images. Exact image-position cursors are
included once every source proof is complete; earlier saves continue with a new
epoch schedule. Final export verifies every registered source before and after
export. Each successful save keeps immutable tensors and a matching checkpoint
descriptor, alongside the latest-checkpoint alias for older consumers.

`loom-factory-material-pyramid-v2` retains Pyramid v1's native tensor shapes,
three feature scales, nearest replication/trimming, full-resolution skip,
zero padding and bounded output. Its hidden activations use leaky ReLU with a
0.1 negative slope at every encoder, residual and decoder activation. Both
activation fields, Pyramid scale fields and output-head identity are mandatory
in its descriptor. Normal output uses a normalized raw vector encoded into
[0,1], avoiding the legacy sigmoid head's zero gradient at saturation; scalar
heads retain sigmoid and use the explicitly recorded stable logit objective.
It is a separate constructor; choosing it creates fresh weights, and legacy
Pyramid checkpoints retain their original ReLU graph. Its purpose is to keep
Pyramid's larger spatial context and lower-resolution computation without a
completely inactive hidden branch. That arithmetic property is tested separately
from whole-material learned quality. Core ML export and Workshop validate the
same versioned graph. New source projects retain existing defaults; an explicit
saved architecture/objective is never silently migrated when starting fresh.
