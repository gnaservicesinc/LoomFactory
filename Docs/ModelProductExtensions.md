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
model descriptors. Missing IDs identify v1. Fresh scalar runs use v2; continuation
retains the saved objective. Objective changes must start a separate loss history
and optimizer state. These objectives share the exact same inference graph,
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
