# Model versions, A/B comparison and clipping validation

Validated on this Mac on October 13, 2026.

The full Loom Factory suite executed 150 cases: 147 passed, three optional
integration checks skipped, zero failures. The skipped checks require external
real 1K paired maps, a categorization benchmark folder/classifier, or an external
classifier integration fixture.

The new tests exercise:

- Actual native training of an initial model, a fresh replacement, and a
  continuation from the explicitly selected older exported model. Earlier run
  files and original image bytes remain unchanged; active selection persists.
- Continuation from checkpoints versus exported weights, project/architecture
  binding, parent hashes, legacy metadata and selection after reopening.
- Package assembly using a selected historical model's weights and native
  model, including rejection of changed checksums.
- Global clipping of an accumulated multi-tensor gradient, a common
  direction-preserving scale, successful-update warmup, fixed/off/zero behavior,
  old-settings migration and restart semantics.
- Two actual Core ML exports with known constant predictions on the same
  native center crop. Error measurements, shared 16-bit preview scale, ancestry
  labels and source byte preservation match expectations.
- Native SwiftUI controls and an A/B result render. A manually selected model
  remains active when the other model has lower error. Missing selected weights
  block continuation; stale package and comparison state is invalidated.

An isolated source snapshot containing only this task's changes was compiled and
its version, comparison, clipping and UI cases were run separately. This verifies
the committed change does not depend on the unrelated uncommitted classifier
and source-resolution/browser edits in the shared checkout. A numerical test
assertion was corrected to use a tolerance for JSON Double roundtrip precision;
the actual training and preservation checks passed.

`make install` completed all three Release builds, signature checks,
cooperative closure, installation and launch. The installed Loom Factory binary
matches the Release bundle SHA-256:

`75c9bba84938ef4e46a1193da6507f465cdaed2093fafb119d57cba8a8dcaf16`

Its installed `LoomBuildConfiguration` is `Release` and deep strict signature
verification passed. Native UI automation could not start:
`Sky Computer Use native pipe startup failed`. Therefore UI evidence comes from
the real FactoryModel workflow tests and AppKit/SwiftUI renders, not a claimed
interactive click-through of the installed app.

After the user's launch preference changed, the installer Makefile default was
changed to `INSTALL_LAUNCH=0`. Factory was left running and untouched; unused
Workshop was closed cooperatively. No further app launches or installations were
performed after that preference was received.

The small native runs and synthetic comparison outputs verify execution and
preservation. They do not establish learned texture quality, generalization,
or that clipping warmup improves this material-map trainer. Choose models by
visual comparison on multiple held-out materials, alongside numerical signals.
