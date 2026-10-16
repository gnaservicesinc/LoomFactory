# Native photo enhancer training

Factory's **Photo Training** workspace exposes two optional RGB products. Open
it from the main workspace tabs, toolbar or **Edit → Photo Training**
(Shift-Command-P), then choose **Train Repair…** or **Train Lighting…**. Both
train natively with Swift and MPSGraph.
These are experimental training paths, not included pretrained weights or a
claim that camera cleanup is solved.

Training data should match close-up surface photography: stone, wood, brick,
fabric, metal and the other material categories used by Studio. The synthetic
generator already derives its clean targets from the current project's diffuse
material maps. Ordinary room/object photographs can diagnose a camera/noise
pipeline, but they do not establish that a model preserves surface grain, fibers,
pores or color. The six-pair SIDD staging sample used for lifecycle checks is
three physical scenes, including one carpet surface; it is not a suitable broad
surface-training corpus. Preserve it as diagnostic evidence.

Build Repair pairs from aligned noisy/clean and mildly blurred/sharp captures of
the same surface, plus unchanged clean-surface controls. Build Lighting pairs
separately from unevenly lit surface captures and aligned neutral references of
that same surface. A low-noise reference may still contain baked-in lighting.
Hold out entire physical surfaces and capture sessions, across every crop,
resolution and degradation variant. Synthetic surface degradations are a useful
starting experiment; real held-out surface captures are still needed to judge
whether they match camera noise, blur, shadows and glare.

| Stage | Product | Teaching pairs | First settings |
| --- | --- | --- | --- |
| Repair | `photo-repair-v1` | aligned noisy/clean or mildly blurred/sharp RGB; clean identity examples | Photo Compact NAF v1, width32 (1,000,515 parameters), native1K, AdamW LR0.0001, WD0, batch1, 30 minutes |
| Lighting | `photo-lighting-v1` | uneven/exposed RGB → artist-neutral RGB; identity controls | same initial network, separate weights/data, same bounded first settings |

This is a distinct `loom-factory-photo-compact-naf-v1` constructor. Input is
`rgb`, Float32 NCHW numeric sRGB [0,1]; output is `enhanced_rgb`, input plus a
learned raw RGB residual. Its head starts at zero, so initialization is identity.
There is no sigmoid, normal-vector conversion, target normalization or source
pixel rewriting. The explicit `rgb-linear-detail-v1` objective reuses the native
linear pixel/detail equation, independently of material product identity.
Existing material constructors, initializers and output contracts are unchanged.
Photo LoRA is explicitly unsupported in v1; material packages remain v1.

## First run

1. Choose a source project and open **Photo Training**. Finish any current
   operation before generating data or starting another GPU job.
2. Under **Repair**, choose **Generate Training Pairs…**. The Repair preset is
   selected automatically. **Materials → Synthetic Photos…** also remains available. Choose all materials
   or a seeded percentage within each category; keep higher-resolution native1K
   crops enabled. Start with two variants plus identity, disk budget4GiB. Review
   the generated examples; increase selection if whole-material validation is empty.
3. Choose **Train Repair…** → **Import Last Generated Pairs**, or use
   **Import Pair Manifest…** to select a generated `manifest.json` or real-pair manifest.
   Imports make new derivative copies under the selected source
   project's `.loom-factory/photo-datasets`; target bytes are deduplicated.
   Import has an8GiB bound and interrupted imports are marked incomplete.
   The old all-effects mixed dataset cannot silently become a single-purpose stage.
4. Start width32, LR0.0001, 30 minutes. Every run first checks its actual starting
   weights at1K using native GPU versus Core ML CPU, tolerance0.0002. Failure
   prevents teaching. This check is numerical compatibility, not learned quality.
5. Use **Checkpoint** or **Save & Stop** at any time. Checkpoints occur at complete
   optimizer updates, every minute and each validation epoch. Save & Stop also
   evaluates the partial epoch before exporting the exact best reviewed-by-score
   weights. This is a candidate, not a human-approved model.
6. **Test Saved Enhancer…** → `photo-runs/<id>/export/enhancer.json` → native1K
   PNG crop. CPU-only A/B review preserves raw Float32 numeric sRGB EXR and a
   clamped16-bit PNG preview. Inspect several unfamiliar real surfaces and good
   clean photos. **Test Checkpoint…** also exports/tests latest or best weights
   explicitly, so a metric cannot prevent artistic A/B selection. A lower RGB
   error alone does not approve the result.
7. Use **Continue Checkpoint…** to branch from `checkpoint.json`. This restores
   latest weights, optimizer moments, epoch and pair cursor in a new child run.
   Kind/width/LR/seed and current dataset fingerprint must match. Total epochs
   and this session's time budget may change. Original runs stay intact. Automatic saves retain the latest resumable state
   and best weights; explicit manual checkpoint revisions are retained.
8. Generate and train **Lighting** separately from the Photo Training workspace,
   once Repair gives useful results. Its generation button selects the Lighting preset.
   Synthetic exposure/illumination teaches a starting correction; moving a light
   around a room does not provide neutral albedo targets. Review/edit lighting
   references before expecting hard-shadow removal.

Real data import also accepts `photo-capture-diagnostic-split-v1`, including the
bounded SIDD sample in user storage, for Repair only. Its six pairs are three
physical scenes, diagnostic-only. Do not call a pretrained-SIDD comparison on
those scenes independent generalization. To import other aligned native1K data,
use this transfer format (membership subsequently lives in image metadata):

```json
{"schema":"loom.photo-import.v1","pairs":[
  {"input":"noisy.png","target":"clean.png","group":"physical-surface-session-1","split":"train"}
]}
```

Use `train`, `validation`, `test`, with each physical surface/session in one split.
No review flag changes membership. Live per-file photo metadata stores product,
target, physical group, category, split, provenance and ignore state. Training
category names are carried into checkpoints/exports as training scope, never
as an automatic claim that those categories are qualified. Scan current files,
not a saved membership manifest. Duplicate bytes across splits are rejected.
Changed pixels/metadata halt the run with completed updates preserved. Originals
and existing material tags/selections are never edited.

## Output and Studio integration

Runs: `.loom-factory/photo-runs/<id>`. `checkpoint.json` atomically points to an
immutable revision containing latest weights, optimizer and best weights, with
SHA256 checks. Best model exports contain `enhancer.json`, `Enhancer.mlmodel`,
`weights.safetensors`; descriptor records exact best update and checksums. A
zero-update identity export is not eligible for Studio inclusion. Inference
retains Compact's whole-field attention: never silently tile a larger photo.

Texture Studio uses optional fixed **build-time** products, separate from its
material package. After A/B review, include either or both exported folders:

```sh
TEXTURE_LOOM_PACKAGE="$PWD/exports/Texture-Loom.textureloom" \
PHOTO_REPAIR_MODEL="$PWD/exports/repair" \
PHOTO_LIGHTING_MODEL="$PWD/exports/lighting" make build
```

The material package supplies the classifier and depth/normal/roughness models.
Without it Studio remains a shell or an attached-map workflow; photo enhancers
alone do not supply material-map generation.

The stager validates product/interface/architecture/width/checksums, compiles
Core ML, and signs the finished app. No source photos enter the app. Missing
variables remove stale products from a previous build. Installing a build is an
explicit user action; training or exporting does not replace a running app.

Studio's **Photo enhancers · after framing** controls run Repair then Lighting
before classification/map prediction. Each has an independent toggle/strength;
defaults have all learned stages and the broad manual lighting balance off.
Old explicit balance settings persist. Stage1 supports opaque SDR1K only and
fails clearly for HDR/alpha/larger grids instead of silently resizing/tiling.
Strength blends numeric sRGB corrections, then clamps the finished color to
[0,1]. Sources stay untouched; Factory A/B tests separately retain raw output. Recipes
pin the exact enabled model hashes. Without included weights the UI says so.

## Remaining work and honest boundaries

- The opt-in1K update/resume/Core ML parity test passed at width16 after the
  material GPU lease became free; run it again when changing the graph:
  `LOOM_PHOTO_GPU_1K=1 swift test --filter PhotoEnhancerNetworkTests`.
  The test refuses to compete with a locked live material project. CPU tests and
  UI rendering alone do not establish native update stability or learned quality.
  The separate real1K lifecycle probe also passed Save & Stop, child continuation
  from update2 to4, native/CPU starting-weight checks, and exact latest/best
  exports. Three SIDD scenes and four updates establish execution only. The
  retained best in that short diagnostic was still identity at update0.
- Qualify Repair on multiple real surfaces per claimed category, including
  clean-input identity. Expand beyond the three-scene diagnostic sample first.
- Qualify Lighting with neutral references and inspect tint, clipped hotspots,
  dark mortar, low-frequency shading, and downstream material appearance.
- Add separately controlled **generated microdetail** after those stages. Small
  residual generation may invent plausible pores/fibers; preserve cracks, weave
  direction and defining shapes. Current deterministic repair is not a tested
  stochastic detail generator. No automatic classifier/router is required first.
- Carpet needs independent pile/pattern tests. Alpha rope/nets need a separate
  opacity/foreground model and loss masks. Geometry stays programmatic initially.
- Sampling currently uses each imported pair once per epoch, not an implemented
  60/30/10 real/synthetic/identity mixer. That ratio remains an experiment.
