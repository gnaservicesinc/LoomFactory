# Conservative photo-to-diffuse recovery — proposal, 2026-10-15

This is the original research proposal. Experimental native Repair and Lighting
trainers are now available in **Photo Training**; see the current
[training guide](PhotoEnhancerTraining.md) for data preparation, checkpoints,
evaluation and Texture Studio integration. The research candidates below are
separate from the implemented trainer and its quality evidence.

The intended input is a reasonably focused, reasonably lit photograph of a
material surface. The output is a useful color/diffuse texture of that same
surface: the same grain, marks, fibers and layout, with camera and lighting
defects reduced. This is not scene-to-material synthesis, tile generation,
invented detail, or physical reflectance measurement. Every original stays
immutable. All stages have a bypass and an independent strength control.

This document is the groundwork requested by the user. No new cleanup model,
download, training run or product/package behavior is activated by this change.
Hugging Face model cards, repository metadata and primary project pages were
checked; no pretrained weights were downloaded and no model was benchmarked.

## What already exists

| Task | Candidate and primary evidence | Practical implication |
| --- | --- | --- |
| Learned denoising and mild deblurring | [Official NAFNet](https://github.com/megvii-research/NAFNet), [HF Qualcomm denoising](https://huggingface.co/qualcomm/NAFNet-DeNoise), [HF OpenCV deblurring](https://huggingface.co/opencv/deblurring_nafnet) | Strong first baseline. Upstream MIT; hardware-specific distributions still need their own artifact/notice checks. These full RGB restoration checkpoints are not compatible with our 1M material-map Compact weights. |
| Apple Silicon restoration prototype | [HF NAFNet SIDD MLX](https://huggingface.co/mlx-community/NAFNet-SIDD-width64), [REDS MLX](https://huggingface.co/mlx-community/NAFNet-REDS-width64) | Community ports provide an on-device evaluation path. Port parity and speed claims are card claims until verified locally. Their model sizes differ from our Compact material networks. |
| Separating albedo, colored illumination and highlights | [HF Marigold IID lighting](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1) | Primary lighting candidate for comparison or teacher guidance. Outputs linear-space albedo, diffuse shading and non-diffuse residual from sRGB input. Effective resolution is about 768px; using its albedo wholesale may lose native fine texture. Open RAIL++ terms need review before redistribution. |
| High-resolution intrinsic decomposition | [HF author demo](https://huggingface.co/spaces/ccareaga/Intrinsic), [author code](https://github.com/compphoto/Intrinsic) | Relevant shading/chromaticity/specularity decomposition. The HF demo declares CC BY-NC-SA 4.0: evaluation reference, not a cleared shipping dependency. Code and weights have to be checked separately. |
| Lens and camera calibration | [HF AnyCalib](https://huggingface.co/javrtg/AnyCalib), [GeoCalib author demo](https://huggingface.co/spaces/veichta/GeoCalib), [author code](https://github.com/cvg/GeoCalib) | Predict camera parameters/rays, then apply a deterministic warp. AnyCalib's HF card declares Apache 2.0. Calibration does not itself choose a desirable material crop or reconstruct hidden texture. |
| Training denoisers without perfectly clean captures | [Noise2Noise paper](https://proceedings.mlr.press/v80/lehtinen18a.html) | Repeated aligned captures with independent, suitable noise can provide supervision. This does not remove the assumptions about noise or solve arbitrary shadow removal. |

For comparison, [Albedo LBM](https://huggingface.co/davidserra9/albedo-lbm)
also exists, but its two large SDXL-based stages and noncommercial license make
it a lower priority for this machine and product. This is a research shortlist,
not evidence that any candidate preserves our material details reliably.

## Proposed processing order

1. **Inspect, without changing pixels.** A small shared quality estimator returns
   separate scores/masks for mild blur, noise/JPEG artifacts, uneven illumination,
   glare/clipping, and geometric distortion. Use a multi-label output with
   confidence rather than a forced choice of one defect. Initially manual toggles
   and simple diagnostics can provide the same controls; train the router only
   after review labels exist. Material category classification must not be a
   prerequisite for producing the diffuse image it expects.
2. **Correct geometry only when needed.** Prefer recorded lens calibration or
   camera metadata; use a learned estimate as a proposal when absent. Let the
   user adjust four corners/crop. Compose undistortion, perspective rectification,
   and crop into one resampling of a derivative. Show the valid footprint and
   discard invalid borders rather than inventing pixels. Natural grain is not a
   straight-line constraint; a paper photograph may not support reliable automatic
   calibration. A small pixel-change budget does not apply to a geometric warp.
3. **Apply conservative restoration.** Begin with pretrained NAFNet baselines,
   then evaluate a distinct small RGB residual network. Use identity initialization
   and predict a correction rather than redraw the image. Enable denoising and
   deblurring independently, with spatial confidence and user strength. Mild
   recoverable blur is in scope; severe defocus and clipped texture are not
   guaranteed recoverable. Deblurring can invent ringing or plausible detail even
   when learned, so it needs its own inspection.
4. **Remove lighting while retaining texture.** Estimate a smooth, colored
   illumination/gain field and, separately, localized shadow/glare residuals.
   Apply the field to the full-resolution derivative. A lower-resolution analysis
   branch is permissible, but the original high-frequency image remains the
   carrier; do not simply replace it with a 768px generated albedo. Intrinsic
   decomposition supplies hypotheses, not ground truth: dark paint and a shadow
   can be ambiguous. Preserve genuinely colored grain and stains.
5. **Review and export.** Show original, each stage, final, difference overlay,
   uncertain regions, and strength/bypass controls. Then use the cleaned color in
   the same Blender material with fixed height/normal/roughness and move the light.
   Judge whether lighting has been baked out without erasing the surface identity.
   Save the approved derivative plus an explicit processing receipt; never replace
   the original photograph or silently promote it into training.

## What “only change a small percentage” should mean

Prefer a small correction amplitude and preserved structure, with localized
changes where the evidence supports them. A hard limit on the *number* of changed
pixels would prevent useful denoising or whole-field lighting correction: slight
noise or a light gradient can cover the entire image.

A proposed restoration head is `output = input + strength * confidence * residual`.
Initialize its correction to zero. Penalize unnecessary residual magnitude and
changes outside known degraded regions. Train clean inputs to remain unchanged.
Use a separate smooth multiplicative lighting field in a declared linear color
space, because global shading correction has different behavior from local noise
removal. A user “preserve texture” setting can control correction strength; an
arbitrary numerical cap is not evidence of faithfulness. Corrected coordinates
must be distinguished from the unwarped identity baseline.

Neither neural restoration nor conventional restoration can guarantee that every
grain remains real. Fully clipped highlights contain no recoverable observed
texture; flag them for retaking/manual review rather than quietly hallucinating.
Keep uncertainty and an untouched fallback available.

## Data and training experiments

Start with a deliberately small evaluation collection: roughly 30–50 good photos
of at least 15 independent surfaces, including paper, cloth, wood, stone, brick,
painted surfaces and near-uniform patches. Include clean photographs that should
pass through unchanged. Hold out whole physical surfaces and capture sessions;
neighboring crops and alternate resolutions must stay in the same split.

Collect tripod-aligned repeated exposures where possible: low ISO/neutral diffuse
lighting reference, mild noise, small focus/motion changes, soft shadows and mild
hotspots. Cross-polarized diffuse captures are useful when available. Artist edits
can supply practical targets but require alignment and a record of what was
intentionally changed. Our existing authored diffuse maps are useful clean seeds
for controlled degradations, not automatically paired targets for unrelated raw
photos. Generate mild noise, JPEG artifacts, blur, lens/perspective changes and
rendered lighting as derivative training examples. Mix in real capture pairs;
synthetic camera noise and shadows do not cover every real failure mode.

First compare frozen pretrained candidates with bypass on the evaluation set.
Only then pilot a shared small RGB NAF-style residual network, starting near
1–4M parameters as a hypothesis. Camera defects can be shared across materials;
we need not train another complete category family immediately. Use native 1K
patches for the detail branch and explicit whole-image context for lighting.
Measure step time and memory before choosing training size. A 30-minute learning
probe and a bounded hour-long pilot are useful budget gates, not promises of a
qualified model trained from scratch in an hour. Do not train while the user's
current material job is active. Test native export equivalence before moving a
research prototype into the shipped Swift/Metal/Core ML path.

Suggested objectives: paired RGB reconstruction, confidence-weighted gradient
agreement, residual sparsity/magnitude, clean-input identity, and smoothness of
the illumination field. Start without adversarial/perceptual texture invention.
Do not reward sharpness alone: oversharpening and invented microtexture can score
well on superficial detail metrics. Pseudo-targets from large decomposition
models require human review and must not contaminate the independent test set.

## Acceptance and integration gates

Human whole-image/native-crop review and Blender appearance decide usefulness.
Use quantitative diagnostics to find failure modes: change area and amplitude,
color drift, retained fiber/grain edges, halos, clipping, fixed-pattern residue,
straight-line geometry when appropriate, and time/memory at 1K and larger sizes.
Evaluate the identity/bypass path and the composition of stages, not only isolated
losses. Natural material shading and defects can be artistically intentional;
maintain user control and document ambiguity.

Implementation should follow `ModelProductExtensions.md`: a distinct versioned
RGB cleanup product, declared input/output transfer functions and ICC/orientation
handling, RGB target loader and native Float32 output contract. Do not repurpose
the scalar height or normal-vector head because it happens to have three channels.
Consolidate the product catalog/constructor registry when implementing the first
real product. Add a versioned stage contract only with a concrete working stage;
do not introduce an unused generic pipeline now. Keep the main image unpadded and
unresized on disk; any derived model input, composed warp and color conversion
must be explicit and recorded.

Every deployed stage must have fixed compatible weights, checksums, license
clearance and a complete package contract. Preserve Studio/Workshop base identity,
continuation settings and old material models. Show all evaluated candidates and
approved derivatives separately. The first useful deliverable is a stage-by-stage
review on real photos, not a claim that every kind of photograph is solved.
