# Saved models and A/B comparisons

Choose the material dataset and map in Loom Room's training workspace. Give
each run a label that describes the experiment, such as `AdamW, LR 0.0003`.

- **Start a fresh model** starts from new weights and allows changing the
  architecture and crop dimensions. Every previous run stays in the source
  folder's `.loom-factory/runs` directory. A completed run becomes a candidate;
  training preserves your explicit choice of model to use. Groups without an
  explicit choice continue to use their latest completed version.
- **Continue exported model** starts from a selected completed version's deployed
  weights. These can come from the lowest-validation-loss epoch when that export
  option was enabled.
- **Continue checkpoint** starts from a selected run's final saved update,
  including stopped runs. Continuation retains its architecture, dimensions and
  depth transform, while applying the newly chosen optimizer settings with a
  fresh optimizer. Each continuation is a separate run with recorded parent
  identity and weight checksum.
- **Use for export** selects any completed version for the next Texture Loom
  package. The choice is saved in this source project and survives reopening.
  Older projects initially use their latest completed version.

The history lists every run for the selected dataset and map, its settings,
epoch measurements and parent. Folder access includes checkpoints, exported
Core ML, trainable weights and run metadata. There is no automatic loss ranking.

**Delete run…** moves that run's whole folder, including its model, checkpoints
and history, to macOS Trash. **Delete all runs…** clears map-training history
across every dataset and map in the current source folder. Original images,
classifier outputs and existing package exports are preserved. Active project
operations block deletion. Deleting a selected version leaves that assignment
unavailable until you choose another model, so another run cannot silently take
its place. Stopped and failed runs can be deleted even without usable weights.

**A/B Compare** takes two completed versions and a current diffuse/target pair.
It runs both exported Core ML models on the same exact center crop, using the
smaller of their recorded training dimensions. It neither resizes nor rewrites
the originals. Architecture sizes may differ because the native Core ML
interface accepts flexible spatial dimensions.

Compare outputs visually for detail, useful relief, artifacts and texture
character. MAE measures agreement with the same original numeric target; it
does not decide which model looks better. Training or validation loss also
cannot decide aesthetic quality. Whole-material membership labels account for
recorded ancestor runs; unknown or incomplete history is labelled unknown.
Use several held-out materials before judging generalization. Depth models
trained with exaggerated teaching targets have that difference called out.

Comparison folders under `.loom-factory/comparisons` retain the exact input and
target crop, both predictions, numeric measurements and source/model checksums.
Preview PNGs use 16-bit values and a shared fixed [0,1] scale, without separately
normalizing each output. Measurements use the Float32 predictions before preview
encoding.

# Gradient clipping

The native trainer already used global L2 clipping: it computes one norm across
all accumulated trainable gradients and applies one common scale before the
optimizer updates its moments or weights. Smaller final batches contribute
their correct sample-weighted mean. This remains the clipping mechanism.

New projects offer three policies:

- **Warmup** starts at a tighter norm cap, linearly increasing to the regular cap
  over a chosen number of successful optimizer updates. Defaults are 0.25 to 1
  over 100 updates. Learning-rate warmup is a separate setting.
- **Fixed** uses the same cap throughout the run.
- **Off** disables clipping. A zero norm cap also preserves the earlier disabled
  behavior. Previously saved settings without a policy decode as Fixed.

The warmup schedule is a tunable heuristic for this supervised residual map
trainer. It is not a claim that a diffusion-specific method improves these
models. Gradient caps act before Adam's adaptive moments; they do not directly
bound the final parameter update, and uniform scaling can largely cancel in
Adam's first step. Continued weights start a new optimizer and a new clipping warmup.
Progress reports the raw gradient norm and current cap. Epoch history records
mean/peak norms and how many updates exceeded the cap, so experiments can
distinguish clipping activity from their visual results.

PyTorch documents [one global norm across parameter gradients](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad.clip_grad_norm_.html)
and [clipping only after the full accumulated effective batch](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation).
Loom uses Float32 native Metal arithmetic without AMP loss scaling, so there is
no gradient-unscaling stage. The quoted [delta-GClip paper](https://optimization-online.org/wp-content/uploads/2024/07/deltaGClip_Anirbit_Matteo.pdf)
defines a specialized lower-bound update method; it is not interchangeable with
this global norm cap and is not implemented here.
