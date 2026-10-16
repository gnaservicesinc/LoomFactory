# Blender template material review

The user found useful generated brick relief after reducing object Z scale and
requested their own Blender setup replace the custom dense-mesh/OBJ review.
This change delivers packed `.blend` files from that saved example. Numerical
reconstruction metrics are diagnostics, not artistic acceptance criteria.

The example has a four-vertex plane, one UVMap, shared UV/Mapping inputs to all
four images, OpenGL tangent normals on the Displaced Base, shader displacement
scale .025/midpoint .5, and Simple adaptive subdivision. The builder preserves
those settings and both engine configurations, lowers world strength to .05,
uses Blackman-Harris filter width1 and CPU Auto-Detect, and creates explicitly
labeled combined, no-denoise and map-isolation scenes. Matching scenes retain
the example's denoising. Native1K image bytes are packed unchanged.

Reopening all seven main material files verified63 scenes: original base
geometry/UVs/modifier settings, mapping, material/texture settings, exact packed
image hashes, engine settings, world brightness and map-isolation links. Actual
hybrid Eevee128-sample and Cycles96-sample no-denoise renders completed in about
5.3s and33.9s; CPU Auto-Detect reports12 threads. These are execution/render
checks, not a claim of live UI acceptance or universal learned quality.

A separate saved Pyramid v2 comparison uses the same template and original
native outputs, without new training. Its earlier queue mistakenly started
fresh despite requested parents, so its short training is not comparable to
the useful older Pyramid v1 height checkpoint's cumulative training.

The shared older renderer also gets explicit single-UV bindings, Auto-Detect
threads, width1 filtering and .05 environment defaults for future renders. A
regression creates a duplicate UV layer and verifies color/normal/displacement
all bind the one named generated basis. Historical scenes/renders remain intact.

Review assets and the exact user template are private data outside this source
repository. The builder refuses existing output files. No models, source pixels
or effective model selections are changed. Scene copies are retained explicitly
so map-isolation scenes survive save/reopen.

## Human verdict and world-strength correction

The user inspected the hybrid in Blender and judged it usable, with possible
local feature inversions still to investigate. The saved Pyramid v2 comparison
was rejected for missing key brick shape and a rolled perimeter. This is a human
artistic verdict on those particular saved versions and material, not an
automatic assignment change or qualification of other categories.

The intended world strength is .5, correcting the preceding .05 review setting.
The generator, verifier and older renderer defaults now use .5. The user's saved
edited Blender files already have .5 and are preserved rather than overwritten.

The edge diagnosis reads the original numeric outputs unchanged. On this v2
height map, the central field averages about .978 while the outer edge averages
.648-.699, returning close to the center level around16 pixels inward. The
reviewed mesh still uses Simple subdivision. Thus a perimeter height fall-off
exists before Blender renders it. Pyramid's zero-padded convolutions are a
plausible mechanism, not an isolated root-cause proof. Existing weights and
inference contracts are not silently changed. Height-only versus combined
normal shading is the next human check for suspected local inversions.
