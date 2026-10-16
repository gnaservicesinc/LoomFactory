# Test Model Blender export — 2026-10-15

The Test Model workspace now exposes **Export to Blender…**. Run any selected
exported model or immutable checkpoint on a diffuse image first. Results of other
map types tested on the same unchanged input can be combined explicitly. New
input/category selections clear that collection. The optional source comparison
uses a second plane in the same project and uniquely paired native source maps.
Object/material names and texture-node labels identify the tested models;
custom properties and the manifest retain full run/checkpoint provenance.

The exporter copies exact native map bytes, checks input/provenance consistency,
and starts only a new background Blender child. It never renders, starts model
inference, changes effective references or replaces existing outputs. Export
folders must be outside the active source corpus. Blender is discovered under
Applications or selected by the user. Cancellation targets only that child.
Failure retains maps, manifest, retry script and log for diagnosis/retry.

A bundled Blender API script builds simple quads with one explicit UVMap and
shared mapping, shader displacement, normal and roughness inputs, packed maps and
studio lighting. This is the direct Blender workflow, not a baked dense OBJ.
Nine scenes preserve the reviewed Cycles/Eevee settings and map-isolation controls.
World .5, displacement .025 / midpoint .5, CPU Auto-Detect, Blackman-Harris width1,
Simple adaptive subdivision and an explicit no-denoise scene are retained. The
reviewed Eevee shadow, tracing and fast-GI settings are explicit. Source DirectX
normals are identified separately; generated normals use OpenGL. Missing maps
remain documented defaults. Native sizes, including rectangular inputs, are kept.

Verification completed without replacing or launching the installed app:

- Eleven focused Swift tests passed: eight Test Model controls/output tests,
  including the opt-in real Blender subprocess test, plus three export tests.
  Checks cover exact checkpoint versus exported-model identity, changed/mixed
  inputs, output byte preservation, retained map selection and source bypass.
- The actual Swift subprocess path exported a rectangular CPU fixture to nine
  Blender scenes with an optional source plane; pre-start cancellation passed.
- Native 1024-pixel existing material outputs were exported in one-plane and
  two-plane forms. Both final files were reopened by a separate Blender process.
  The verifier checked all nine scenes, exact packed image SHA256, object and
  material identities, one UV per quad, no dense mesh, relevant engine settings,
  and shared image data rather than nine copies of every texture.
- NSHostingView rendered the options and Test Model controls. This is not a claim
  of live user click-through acceptance.
- Release build and signed application assembly passed. The installed app stays
  unchanged; the user will install after their current training job.
- Bundle-boundary checks allow only the single named Blender export script and
  still reject retired workers/interpreters and a symlinked export resource.
  Model training/inference remain native Swift/Metal; no Python runtime is shipped.

The build, test logs, CPU fixtures, reopen receipts and uninstalled application are
preserved under the user's Application Support in `Loom Factory/BlenderExport-2026-10-15`.
Finished `.build` trees are removed after preserving the handoff. The separate
`Docs/PhotoToDiffusePlan.md` records researched candidates and a staged RGB cleanup
proposal; it does not introduce or activate another model product.
