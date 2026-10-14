# Training work and timing

The main run panel shows the current stage and material/image, epoch and image
position, completed steps and steps left, cycle speed, elapsed time, and
estimated epoch/run time remaining. One step is one optimizer update, including
its accumulated image visits. Internal minibatch numbers are not the main
progress display. Losses, optimizer measurements and phase timings are available
under **Training measurements**; the event log is under **Run details**.

The trainer prepares one upcoming batch while the current batch trains. Crop
selection, augmentation, image order and optimizer boundaries stay on the serial
training owner. Preparation is bounded to one pending result; source metadata
and file proofs are checked again before that result reaches training. Stop
drains preparation before clearing decoded pixels and preserves completed
updates. Original images are never resized or rewritten.

Cycle timing measures preparation wait, training and the optimizer update.
Separate preparation-work timing records CPU work even when it overlaps GPU
work. This distinguishes a decoder bottleneck from model computation. The saved
training log includes timestamps and full progress/timing snapshots.

Remaining-time estimates start after two completed cycles and use a recent
rolling average, excluding the first compilation cycle. Validation and saving
costs enter the estimate after they have been measured. Estimates label their
scope while those costs are unknown; final verification/export has its own
stage timer. A slow current stage cannot consume estimates for unstarted work.
Stopped runs freeze their timers, and completed runs have no remaining work.
