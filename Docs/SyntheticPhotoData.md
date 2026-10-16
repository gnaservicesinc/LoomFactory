# Synthetic photo data

Open **Materials → Synthetic Photos…**. This native, CPU-only generator makes
paired degraded-input / clean-color datasets. It does not run a model, change
source tags, resize originals, select models, or begin training.

## Selection and repeatability

Choose all categories or one category, then set **100%** for every eligible
material or a seeded random percentage **within each category**. Selection is
by physical material folder, not by the number of map files. Small categories
round up to one selected material. Only current, non-ignored, tagged color/diffuse
images are eligible; depth, normals and roughness are excluded. Conflicting
material categories are excluded.

By default, use all available native resolution levels at least 1024 pixels in
both dimensions. Choose one deterministic color variant at each dimension level.
A native 1024×1024 image contributes one crop; larger images contribute the
configured number of **exact 1024×1024 crops**. Origins cover all valid native
positions, including edges. There is no source resampling. Crops, tiers and
variants of a material share one group/split. Identical file content coalesces
material groups before split assignment. Factory uses project-relative material
identity so moving the source project does not change the selection rank.

The seed, generator version and settings are saved once in `recipe.json`.
**Load Recipe…** restores a previous export's settings. The same version, seed
and unchanged source set reproduce the same crops/effects and output pixels.
Changing available source files, tags or categories changes eligibility; a seed
cannot recreate a missing original. The compact manifest retains source identity
once and the pair references/group/split needed to consume the generated data.
It does not log every sampled effect value or transform matrix.

## Effects in v1

* Exposure: linear-light gain from a randomly selected −0.6 to +1.2 EV range,
  multiplied by strength. This includes deliberate clipping.
* Mild defocus: separable normalized Gaussian blur, sigma 0.35–2 pixels times
  strength, in linear light with clamped border sampling.
* Noise: additive approximate Gaussian noise in linear RGB, sigma 0.001–0.012
  times strength. This is a generic approximation, not calibrated phone RAW noise.
* Illumination: a smooth directional gain gradient and soft hotspot, amplitude
  0.2–0.6 times strength. It does not simulate arbitrary 3D cast shadows or glare.
* Reframing: a pinhole projection of a plane tilted in 3D, pitch/yaw within ±15°
  times strength. Only the synthetic input is bilinearly resampled. The target
  remains the original exact crop. Input-valid and target-visible masks identify
  background and unobserved pixels. No arbitrary background is taught as surface.

Each enabled restoration effect is independently sampled with probability 0.65,
with at least one retained. Unchanged clean examples can be included separately.
Restoration (`photo-restore-rgb`), identity (`photo-identity`) and reframing
(`photo-reframe`) are explicit tasks. A pixel-aligned restoration trainer must
not silently consume a tilted input against an unwarped target.

Camera effects follow: decode numeric sRGB → linear illumination/exposure →
linear blur → noise → clip → sRGB encode. Current sources must be opaque sRGB or
gray up to 16-bit. HDR, other color profiles, alpha, severe lens distortion,
JPEG artifacts and motion blur need separately versioned support. Originals
retain their original bytes/precision; generated derivatives are 16-bit PNGs.

## Output and operation

The default output parent is `.loom-factory/synthetic-photo-data` in the source
project. An alternate parent can be chosen outside visible source material
folders. Each operation creates a new uniquely named dataset. The last recipe is
also retained as project configuration; source membership is still enumerated
from current files and their metadata.

The preview shows selected materials, native crops, total pairs and a conservative
uncompressed disk forecast. Generation rejects recipes above the explicit disk
budget before producing images, and checks actual output size too. One clean
target is shared by its variants; identity pairs reuse that same file. Higher
resolution decoding uses a streaming cache and allocates only the selected crop's
Float32 RGB array, rather than retaining the corpus.

`Review.html` shows the first 60 input/target pairs with links to native pixels.
It is a data review, not a model-result comparison. Stop preserves completed
pairs, the recipe and a cancelled manifest. Failed or incomplete exports are
never labeled completed. Existing export folders are never overwritten. Sources
are revalidated against current tags, dimensions and hashes while generating.

RGB photo cleanup has distinct native products in **Photo Enhancers…**. Use the
Repair or Lighting preset before importing pairs into that trainer; the mixed
default recipe is not a single-purpose stage. See [PhotoEnhancerTraining.md](PhotoEnhancerTraining.md).
Exports are not added to depth/normal/roughness membership. Their manifest marks
them for review/import rather than claiming a trained or approved RGB product.
A fixed independent split seed assigns material groups approximately 80/10/10;
small exports may not populate every split and need coverage review.

## Data mixing experiment

A starting training sampler hypothesis is **60% real paired captures, 30%
synthetic examples, 10% clean identity examples**. Compare that with real-only
and a higher synthetic share on unfamiliar real photographs. This is a future
sampler ratio, separate from the UI's percentage of source materials. Avoid
counting several synthetic variants as independent held-out surfaces.

Synthetic degradation is established in restoration research, including
[BSRGAN](https://github.com/cszn/BSRGAN) and
[RSBlur](https://github.com/rimchang/RSBlur). Our v1 is a small independent CPU
implementation, not an implementation or quality reproduction of those models.
Synthetic data complements the real repeated-capture collection; it does not
prove real-phone restoration or neutral diffuse quality.
