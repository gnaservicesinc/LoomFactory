# Managing Loom models

Open **Models** in Loom Room to see the shared category classifier and every
current dataset's depth, roughness and normal products. The matrix is built from
the current source-folder categories and saved model artifacts. Running, stopped
and failed training runs remain visible in history but do not count as completed
models.

Any saved checkpoint with completed training updates can become the selected
model. In its row, choose **Use checkpoint as model**. Factory copies those exact
weights into a separate immutable version, exports its Core ML model, and selects
that version for testing and package export. The original run, its existing
model, and its continuation checkpoint remain unchanged. The new version records
the source checkpoint and its checksum. It starts without approval so you can
inspect its output and compare it with other saved versions before approving it.

Select a cell to review its saved candidates. **Use This Model** saves the model
assignment for that group and product. **Approve** records your own quality
decision separately: approving a candidate does not select it, and selecting a
candidate does not approve it. Test outputs and compare held-out materials before
approving. Approval is not a conclusion drawn from training loss.

The status distinguishes missing models, available candidates, approved
candidates, the current assignment and an approved current assignment. A latest
model used by default is labelled separately from an explicit choice. New
training runs and classifier imports preserve explicit choices. A selected model
that is removed remains unavailable until you choose a replacement.

All three map products are initially needed for each dataset. Mark a product
optional when this system intentionally omits it. Optional status changes the
maintenance plan, not the model's native interface or whether a selected model
is included in an export. The classifier is shared and always needed. Coverage
and approval summaries describe readiness; the existing package format can
still export partial map coverage and the package panel identifies that state.

Assignments live in `.loom-factory/model-selection.json`; classifier choices,
requirements and approvals live in `.loom-factory/model-management.json`. These
project records move with the source folder and survive reopening. Approvals bind
the descriptor, native model and trainable weights identity. Replaced files need
review again. Existing projects retain their latest completed model defaults
until you choose explicitly; no existing model is automatically approved.

## Extension boundary

Management uses a stable group and product slot, separate from image map roles.
The current product catalog declares the implemented classifier and map products.
Future products can extend that catalog and, when needed, identify individual
steps in a concrete versioned stage plan. This screen is a management view; it
does not add an execution graph or expand `texture-loom.package.v1`. Follow
`ModelProductExtensions.md` when implementing a new product or a chain, including
stage contracts, routing order, bypass compatibility and complete native package
validation.
