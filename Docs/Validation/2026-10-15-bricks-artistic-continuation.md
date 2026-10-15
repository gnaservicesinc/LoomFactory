# Bricks artistic continuation

This is a separate user-authorized run, launched 2026-10-15 21:08:42 UTC with
an absolute deadline of 22:08:42 UTC. The preceding overnight queue is closed.

Queue jobs now accept optional `parentWeightSource` (`model` or `checkpoint`).
Omission preserves the existing checkpoint default. Explicit `model` starts
from the exact exported model shown in review, rather than the later optimizer
checkpoint. The post-save lineage guard verifies both parent ID and source.

The exact Release queue binary SHA256
`8b536061ba37321fdeeb6f6ab08454019cee1dfaa71688a7371e16c55a7ba772`
passed three real native 1024-grid one-update starts, checking actual request
lineage, resume weight path and parent file checksum. All three saved and
exported successfully. These checks establish execution, not artistic quality.

The bounded Bricks experiment continues the reviewed Compact normal export
`checkpoint-model-7e8ca467-473f-4eab-b31c-59dbd675f8a9` for 15 minutes,
Compact roughness export `checkpoint-model-2f131d22-8058-4b43-8c67-d5b2684ef5a1`
for 10 minutes, and the user's preferred Pyramid depth export
`checkpoint-model-990f9148-10d0-45d1-a3f6-61bf62a0b1d0` for 10 minutes.
Normal/roughness explicitly use squared-structure-v1 at LR 0.00005; depth keeps
absolute-v1 at LR 0.00001. Physical batch and accumulation are one, weight decay
zero, seed 9592, warmup two epochs. Current tagged native sources are cropped
at 1024 without resampling; no original pixels or reference selections change.

Review uses only existing native 1K color/maps, displaced geometry, environment
and grazing lights, with separate labeled height/normal strength controls.
Depth/normal need not encode the same scales, and roughness need not correlate
with either. Authored-map error is diagnostic, not an artistic acceptance gate.
Stone Wall, brick_wall_005 and Bricks059 can be fitting/appearance diagnostics;
inherited category weights prevent claims of independent unseen-material testing.

User-storage queue folder: `2026-10-15-bricks-artistic-continuation` under Loom
Factory/TrainingQueues. Candidates remain unapproved. Private model/review data
backup belongs in gnaservicesinc/LoomMaterialRecoveryData, never public source
repositories. Remote payload verification is required before claiming it saved.
