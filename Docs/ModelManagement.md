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

## Closing a finished category

**Close Category…** appears beside the category name only when its current depth,
normal and roughness models are all approved and still available. Optional
products still need approval before closing. Nothing closes or deletes sources
automatically. The button first verifies provider sources and previews the file
count, material count, disk savings and number of files that will stay. Canceling
the confirmation leaves every source file in place.

Confirming saves `.loom-factory/closed-categories/<category-id>.json` before
removing any source image. The readable list records Poly Haven / ambientCG asset
links, filenames, local paths, native dimensions and review/pairing metadata.
Only current provider originals with matching published bytes are removed.
Generated or custom files, edited images, unrecognized names, external imports
and symbolic links stay on disk. A missing or unpublished provider file stays,
too. Poly Haven verification uses its live file catalog; ambientCG verification
uses the published ZIP directory, avoiding a duplicate full dataset download.
Removal is permanent rather than moving files to Trash, so space is reclaimed.
Models, approvals, checkpoints, history and saved outputs are preserved.

Closed categories stay visible even if they have no physical source images.
Their classifier labels and package/model assignments remain intact. The recovery
list never supplies virtual training pairs. Reopen from **Models** before
preparing sources or training again.

**Reopen Category…** asks before downloading. It resolves current provider
releases, restores only the saved files and tiers, checks current download
integrity and image dimensions, and restores image tags. Historical source
hashes are not persisted or enforced, so provider fixes can be downloaded.
Existing matching files are kept; conflicting files are never overwritten.
Downloads show progress and support cancellation. One archive is handled at a
time, with a free-space check and a reserve. Completed files and the recovery
list survive cancellation, network failures and partial restores; press Reopen
again to resume. The category becomes open only when all listed files exist
with their saved tags. Internet access and continued provider availability are
required. Reopening requests no additional tiers, including no automatic 16K.

Closing and reopening require the exclusive project lease, so neither can run
during training, testing or an overnight queue. The final close rechecks approvals
and source changes after the preview. Keep the entire `.loom-factory` directory
with its source folder; it owns the recovery list as well as the trained models.

## Extension boundary

Management uses a stable group and product slot, separate from image map roles.
The current product catalog declares the implemented classifier and map products.
Future products can extend that catalog and, when needed, identify individual
steps in a concrete versioned stage plan. This screen is a management view; it
does not add an execution graph or expand `texture-loom.package.v1`. Follow
`ModelProductExtensions.md` when implementing a new product or a chain, including
stage contracts, routing order, bypass compatibility and complete native package
validation.
