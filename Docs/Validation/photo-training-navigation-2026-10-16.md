# Photo training navigation — 2026-10-16

Photo Training is a main Loom Room workspace, with toolbar and Edit menu entry
points. Repair and Lighting each expose a trainer and stage-specific synthetic
pair generation. The trainer can import the latest generated manifest directly.
Changing source projects clears generated-photo and enhancer result locations.
Existing material training settings and membership remain independent.

Verification:

- Twelve focused tests passed across Loom Room, photo trainer and synthetic
  data UI suites. These cover both supported workspace sizes, both trainer
  stages, actual CPU pair generation, and clearing photo state on source change.
- Native workspace renders at 1120 × 760 and 1380 × 980 were inspected. Both
  stage actions and pair generation are visible at the minimum size.
- The optimized Release build, native bundle checks and code signature passed.
- The installed application held the source project's model-operation lease.
  A signed update and installer were staged in user storage; the running app
  and its active operation were left intact. Installation remains deferred.
- The native computer-use pipe could not start. Rendered checks do not establish
  live installed-app click-through acceptance or learned restoration quality.

Public documentation links to the repositories and the photo training workflow.
Machine-specific paths in published Markdown have been replaced with portable
examples or placeholders. Historical measurements and hashes are unchanged.
