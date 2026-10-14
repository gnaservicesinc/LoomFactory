# Diffuse color variants

Each diffuse image is a separate training input. Diffuse variants in the same
material pairing group reuse the matching depth, roughness or OpenGL normal
target at their original pixel dimensions. Target files stay shared; no source
images are copied, resized or rewritten.

Numbered `colN`, `collN`, `col_N` and `coll_N` filenames share their material's
pairing identity. Refreshing repairs older filename-derived alternate-color
pairing keys while preserving file identity, reviews, notes, manual map roles,
manual pairings and ignore decisions. An ignored variant must be explicitly
unignored before it can enter training.

For other names, set the map roles in Image Review and use **Pair Images** to
select all diffuse variants plus one map of each target type at the same
resolution. Keep variants and their shared maps in one material folder. Several
competing targets of the same type and dimensions still require explicit
pairing; the trainer does not choose one arbitrarily.

Pair counts include every diffuse/target example. Variants have distinct sample
identities for crops and comparison. Validation holds out entire material
folders, keeping all color variants and resolutions on the same side of the
training/validation split.
