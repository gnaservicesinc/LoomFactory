# Model management validation — October 14, 2026

Loom Room now has a Models workspace with per-group/product assignments,
independent human approvals, optional product requirements, the shared classifier,
coverage counts, and actionable filters. Current source categories determine the
matrix. Saved model files determine availability; checkpoints and unfinished runs
do not masquerade as completed models. Stable product/group slots provide the
management boundary for later products without adding a graph executor or
changing `texture-loom.package.v1`.

New training preserves explicit assignments. Deleting a chosen run retains an
unavailable assignment until another model is chosen. Approvals bind the saved
descriptor, native model and weights, with file revision checks and cached hashes.
Modified or missing files invalidate approval. Classifier choices are also
persistent. A classifier may categorize a new folder before all deployment routes
are present; coverage and package approval still report that mismatch.

The package panel reports available required products and chosen approved models,
and identifies partial exports. The existing format still permits partial map
coverage. Assembly validates the exact chosen native files; approval remains a
human review decision and does not establish generalization or aesthetic quality.

## Evidence

- The task-only Git index was exported to `/tmp/loom-management-review` to verify
  independence from the checkout's pre-existing classifier and training edits.
- Final isolated Release validation passed **42 tests**, zero failures: 28
  XCTest core/lifecycle/export tests and 14 Swift Testing native UI/workspace
  tests. Command: `swift test --configuration release --filter
  'ModelCoverage|ModelManagementExport|ModelVersionTraining|ModelVersionLifecycle|RunDeletion|LoomRoom'`.
- Release tests cover coverage and persistence, approval versus assignment,
  changed or damaged files, classifier route/descriptor mismatch, exact selected
  export contents, explicit choice after later training, and deletion without
  promotion. Native model replacement/continuation runs use small real models.
- Native AppKit/SwiftUI renders cover light/dark appearances, candidate controls,
  minimum/full coverage panels, and the full Models workspace at 1120 × 760 and
  1380 × 980. The rendered matrix and controls were inspected for clipping.
- Interactive computer use was unavailable: `Sky Computer Use native pipe startup
  failed`. Offscreen renders are layout evidence, not a live click-through test.
- `Scripts/build-app.sh` completed the optimized Release app, compiled the
  classifier assets, verified its native bundle, and passed strict code-signing
  verification. A normal quit request did not close the existing installed app;
  it was not force-quit or replaced during active work.

The full working-tree Release run also exposed three failures outside this change:
the native masked-gradient assertion, a pre-existing map-type classifier expected
count, and a model-test fixture which declares `weights.safetensors` as its
checkpoint but deletes `checkpoint.safetensors`. The latter reproduces in the
focused run and its fixture/lookup match Git HEAD. Native gradient implementation
and the untracked classifier implementation were not changed by this task.
