# Loom Factory

This is a fresh application. Do not copy, import, or call the retired Texture Studio trainer, workbench stores, dataset manifests, or material-tool executables. Trained starter-classifier assets and standard third-party libraries are separate from application code.

The selected source folder owns all project settings, runs, models, and exports under `.loom-factory`. Replacing it replaces all active project state. Reopening it restores that folder's project. Never persist an image-membership inventory: enumerate current files. Categories, map roles, pairing identity, review decisions and notes belong in the images' metadata. Imports may use symbolic links to originals.

Loom Room is the only starting workspace. Datasets are the categories discovered from current metadata; choosing a dataset never changes the source folder. The viewer and training controls share one current folder snapshot and the same training-pair selector.

Preserve original image bytes and precision. Training must check current file/tag membership, report real progress, support cancellation, and export native CoreML plus architecture/weights for compatible LoRA training. Never represent a smoke test or an untrained model as a qualified production model.

Manual review flags and notes are informational; they never determine training
membership. Current physical files, dataset tags, map roles, pairing, ignore
state and native dimensions determine usable pairs.

Texture Loom packages carry one classifier, category models, identities, versions and checksums. They contain no source images, credentials, virtual dataset membership, or references to another project. Texture Studio consumes the package at build time; Loom Workshop trains adapters for that exact installed base.

Before adding or changing a model product, read `Docs/ModelProductExtensions.md`.
Keep product identities separate from image map roles and preserve versioned
native interfaces. Future optional stages require an explicit validated package
contract; do not repurpose scalar-map or normal-vector behavior for RGB diffuse
generation, or add unused pipeline machinery ahead of that feature.

Before finishing completed work, commit and push task-related changes to this
repository's configured origin and verify the remote branch matches the local
commit. Local checkouts may not survive between runs. Preserve unrelated work,
never force-push shared history, and report any push failure. The user has given
standing authorization for these task-scoped commits and pushes.
