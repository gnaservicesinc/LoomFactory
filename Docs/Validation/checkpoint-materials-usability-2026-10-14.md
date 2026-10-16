# Checkpoint and Materials usability validation

Validated source and optimized Release bundle on 2026-10-14. After the user
closed Loom Factory, the validated build was installed at
`/Applications/Loom Factory.app` without launching it. Its executable hash,
complete bundle contents, native resources and strict signature match the
validated build. Live interaction qualification remains pending: computer-use
inspection failed with `Sky Computer Use native pipe startup failed`.

## User-visible changes

| Request | Result |
| --- | --- |
| Manage and edit checkpoints | Saved-run chooser at the top of Training; direct name/notes fields, Test, Load checkpoint, Show folder and recoverable deletion. Names/notes live separately from immutable artifacts. |
| Materials layout and responsiveness | Full-height split workspace; background image decoding and filesystem presentation, cached hierarchy/search/selection, two decoders, bounded 128 MiB/256-image cache, cancelled obsolete requests and background thumbnail priority. Unchanged polls retain the mounted snapshot. |
| Mark without expanding | Review button on collapsed folder rows and a fixed Mark Reviewed action above the selected material. Bulk review covers its whole physical folder and preserves every other metadata field and original bytes. |
| Folder/resolution previews | Selecting the row immediately displays labeled maps and review counts; disclosure is a separate control. Losing one representative file preserves the selected material while other maps remain. |
| Obvious model approval | Fixed saved-version chooser, Approve & Use and Test output above independently scrolling groups/details. Approval and selection complete in one action. |
| Remove artificial test limits | Testing permits any source category, ignored source diffuse, or external input. Approval and missing deployment CoreML exports do not gate usable native weights. Actual active-operation serialization remains. |
| Restore checkpoint settings | Load restores the complete recorded form. Explicit later edits remain. Implicit checkpoint changes after dataset/map selection also restore that run's settings. |
| Preserve continuation information | Saved prior history, cumulative epochs/updates/image visits/measured time, exact compatible optimizer/cursor resumption, and bounded recovery of measured timing from older native logs. Sidebar measurements survive thousands of verification events independently of the bounded event log. |
| Edit names and notes | Direct editable saved model/checkpoint fields in Training and Models; separate display metadata leaves weights, descriptors and recorded settings unchanged. |
| Original source maps | Paired native target shown before and after testing. Resolution/pairing identity is explicit; missing or ambiguous targets explain the issue while allowing inference. Cached original-map selection updates when an implicit model changes. |

## Checks

- Final selected native/core/UI functional run: 130 tests passed, one opt-in
  existing-project checkpoint test skipped. Native training tests exercised real
  save/stop, exact next-image continuation, exported-weight continuation, retained
  historical weights, cumulative history, original-byte preservation and native
  model testing. No existing user checkpoint was trained or edited by these tests.
- Responsiveness is measured separately from synchronous hosted-render tests.
  The 1,000-map probe performs 12 searches and two folder polls while sampling a
  main-executor heartbeat every 10 ms. The isolated final probe and JSON receipt
  passed with a maximum gap of 43.3 ms. Together with the functional run, 131
  selected tests passed and one opt-in test was skipped. A combined initial timing run was confounded
  by other suites rendering native windows simultaneously; functional tests and
  the isolated timing probe are separate final gates.
- 6,000-map index correctness and 60,000 cached selection lookups passed. Unchanged
  polling does not publish another snapshot. Preview tests cover coalescing,
  cancellation, cache byte/count bounds and source replacement.
- Hosted renders were inspected for readable folder previews, minimum-size Models
  approval controls, checkpoint settings and original-target comparisons. Hosted
  renders and heartbeat measurements do not prove live scrolling is free of lag
  or that all hangs have been eliminated.
- Release build, bundled-native verification and strict code-signature validation
  passed. Release executable SHA-256:
  `392335fb0896fc3f8e91ec690749c421617c14adc7d39c19c337173f231d239a`.
- Existing native map-role/source-resolution/Float32 output support required by
  the revised viewer was retained and its focused checks passed. A stale map-role
  fixture was corrected to explicitly recognize a numbered alternate diffuse
  instead of counting it as an ignored unknown map.

## Installation and evidence

Evidence is stored outside the app source at:
`/path/to/workspace/ipde/out/loom-factory-materials-usability-2026-10-14`.

The validated bundle is:
`/path/to/workspace/ipde/LoomFactory/.build/application/Loom Factory.app`.

`Scripts/install-app.sh --skip-build` completed after confirming Factory was
closed. The installer retained the prior app at:
`/path/to/user/Library/Application Support/Loom Factory/Backups/Loom Factory.backup-20261014-220740-AF67246D.app`.

`installation-receipt.json` and `installation-completed.log` record the completed
installation. The installed executable SHA-256 is the validated hash above;
every installed file and symbolic link matches the built bundle. Strict
code-signature verification and `Scripts/verify-native-bundle.sh` passed against
the installed application. Factory remains closed.

Exercise folder selection, review, approval, testing, checkpoint loading and
continuation in the live app before calling the user's no-hang acceptance
requirement met.

Execution tests do not establish learned model quality or generalization.
