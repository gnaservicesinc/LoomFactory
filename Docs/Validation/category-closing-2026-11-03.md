# Category closing and 8K source parity — 2026-11-03

The Tiles And Pavers screenshot reflects actual physical files: Patterned
Terracotta Tiling and Stone Tile Wall have diffuse, displacement, normal and
roughness originals at 8K; the other sixteen material folders do not. All eighteen
have the lower tiers. There are seventeen distinct Poly Haven asset IDs because
Terracotta Floor Tiles is represented in two folders. A read-only live catalog
audit found 8K PNG diffuse, displacement, roughness and OpenGL normal downloads
for every asset used by the eighteen folders.

Previously `sourceResolutionLevels == 0` filled gaps only through each individual
material's highest local tier. It now matches the highest existing native tier
across the category, with an automatic 8K ceiling. The UI calls this **Match
category (up to 8K)**. Explicit level selections remain available for 16K; the
automatic mode never requests 16K, including when another local original is 16K.
No missing production-corpus 8K images were downloaded during this task, given
the user's limited free space and active training run.

## Manual close/reopen

The Models matrix shows **Close Category…** only when the effective depth, normal
and roughness models are all currently approved and available. The action checks
the provider catalogs and previews file/material counts, reclaimable bytes and
preserved files before asking for manual confirmation. Confirmation rechecks
approvals and source file stamps/tags under the exclusive project lease.

A durable JSON recovery recipe is saved before the first source file is removed.
It includes provider identities, asset URLs, filenames, resolutions and image
metadata. Poly Haven originals must match current published size/MD5; ambientCG
originals must match current ZIP entry size/CRC32. ambientCG verification fetches
only the ZIP directory via HTTP ranges. Generated, edited, unrecognized,
unpublished, linked and external source files are preserved. No folders or
unlisted ancillary files are deleted. No production category was closed here.

Closed categories remain in the model matrix, classifier catalog and package
exports even with zero source images. Recovery lists do not create training
pairs. Training and source preparation require explicit reopening first.

Reopening resolves the provider's current catalog instead of enforcing historical
hashes or stale download URLs. Updated provider bytes and sizes are accepted.
Current download integrity, image dimensions and saved tags are validated. It
never overwrites conflicts. Partial restores retain completed files and their
recipe for retry. Downloads have progress/cancellation, use one archive at a time
and check free space before proceeding. The category opens only after every
listed file is present. No additional resolution tiers are requested.

## Verification

The final focused run executed 43 XCTest cases (one opt-in production-download
test skipped), plus 18 Swift Testing UI cases in two suites, with no failures.
Coverage includes all-three-approval gating, stale approvals, changed source
bytes/tags after preview, exclusive training leases, generated/edited/link
preservation, unpublished assets, provider updates at changed URLs,
ambientCG directory parsing and extraction, failed and partial reopening with
retry, conflict preservation, empty closed-category persistence, and package
export retaining classifier labels and all three models. Native UI state tests
exercise closing/reopening and training gating. Cached native-view render
artifacts are not a substitute for live installed-app click-through acceptance.

The opt-in read-only provider test passed against the development corpus:
Stone Tile Wall's 1K diffuse matched Poly Haven's current MD5, and Bricks059's 1K
color matched the current ambientCG ZIP directory CRC32. This check changed no
source files and downloaded no complete texture archives.

Evidence is retained in user storage under
`~/Library/Application Support/Loom Factory/Validation/2026-11-03-category-closing/`.
Release 0.2.4 (build 6) is staged with a manual installer in
`~/Library/Application Support/Loom Factory/Releases/0.2.4-build-6/`.
Installation and live installed-app acceptance are deferred while Loom Factory
is running its existing training operation. The installer refuses to replace a
running app and does not launch the app after installation.

The staged bundle passed strict code-signature verification. Its manual installer
also installed into a temporary root, verified the installed signature and
matched the packaged executable byte for byte. The staged executable SHA256 is
`3d11f07ec2d417d7ab571085b9683523d47e809bb317d01419c47ba711e385c3`.
