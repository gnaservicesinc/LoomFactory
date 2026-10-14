# Shared targets for diffuse variants

Each color variant now contributes its own diffuse/target sample, reusing the
same original depth, roughness or OpenGL normal at matching dimensions. Manual
pairing accepts multiple diffuse inputs, retains unique target roles, and keeps
shared variants in one material folder for whole-material validation.

Validation on 2026-10-14:

- The exact staged change, exported independently of preexisting checkout
  edits, passed 34 core tests with zero failures. These include 13 new variant
  tests, native training with two distinct diffuse inputs and one 16-bit depth
  target, manual pairing, source membership and saved-model comparison.
- The working checkout passed 36 focused core tests and two native offscreen
  pairing UI tests with zero failures before the final cross-folder manual
  pairing test was added; that final test passed in the isolated staged run.
- Live `Leather Red 03` produced five pairs for each target: one at 1024,
  two at 2048 and two at 4096. Both alternate-color inferred metadata records
  were repaired; image IDs, categories, reviews, notes and ignore states stayed
  unchanged. SHA-256 comparisons confirmed all 14 original map files unchanged.
- The optimized Release app built and passed bundle/signature verification,
  installed in `/Applications`, and launched. Its executable hash matched the
  built app: `1f162c4f917c398547fbeff8eeb26b76919bbbc03908ba4c3aa2782fe895bda0`.

Receipts are under
`~/Library/Application Support/Loom Factory/Validation/diffuse-variants-2026-10-14`.
Native interactive UI verification was unavailable because the computer-use
tool returned `Sky Computer Use native pipe startup failed`; offscreen SwiftUI
rendering, real native training, source proofs and installed-process checks
provide the validation above. The training fixture verifies sample execution,
not learned texture quality. Existing ignored variants remain ignored.
