# Loom Factory

Backend extensions use versioned product and architecture contracts. See
[model product extension design](Docs/ModelProductExtensions.md) before adding
new products or optional processing stages. Photo-to-diffuse generation remains
future work.

Loom Factory creates custom Texture Loom deployments for Texture Studio. It is a fresh macOS app, separate from the texture editor and from Loom Workshop's LoRA trainer.

Open Loom Room and choose a source folder. The folder owns the entire project: settings, training runs, models and exports live in `.loom-factory`. Choosing another folder opens a different project; reopening the first restores its settings and runs. Imports link to original images rather than copying them.

Loom Room reads current image files. Categories, pairing, map roles, review decisions and notes live on the files as metadata, so renaming a file keeps its identity. There is no saved dataset-membership list. The **Dataset** menu shows categories found in the folder, and the viewer and trainer use the same paired-image selection.

The Materials workspace gives the browser and preview the available window height. Select a material or resolution to see its maps immediately; the separate disclosure arrows are optional. **Mark Reviewed** works directly on a collapsed material, records every map in that physical folder, and preserves its category, notes, map roles, ignore decisions and original bytes. **Not reviewed** finds unchecked materials. Previews use a bounded background cache; unchanged folder polls retain the current view and selection. Known diffuse/color, displacement/height, roughness and normal filename conventions are recognized automatically. Numbered Poly Haven colour variants (`col1`, `coll1`, `col_1`, `col2`, and later variants) pair with their matching maps. Bump maps have their own inspection-only type and never substitute for depth targets. **Needs map type** finds unrecognized files; **Ignore this file** keeps a file available for inspection while excluding it from categorization, Create ML export and training. DirectX normals are ignored automatically; normal training uses OpenGL maps.

Filename recognition comes before the bundled native map-type classifier. Bump and known unsupported maps such as AO, metallic, gloss, opacity, emission and packed maps never reach its four-class prediction. Only unresolved names receive a classifier suggestion. Guesses are marked in the browser and Image Review; uncertain predictions retain **Choose map type**. Depth and roughness use stricter acceptance thresholds because the supplied model can confuse them. Confidence is a guardrail, not proof that an image belongs to these classes. Manual choices take priority. Suggestions are cached against the source and classifier identity, and original image bytes are preserved.

1. Categorize each material folder once using one diffuse image. Inspect and correct dataset or map types when needed; manual review is optional. Resuming reuses a folder's consistent existing category for its remaining untagged images. Existing tags and reviewed images are preserved; folders with conflicting categories are listed for correction.
2. Train the depth, roughness or normal models needed for each dataset. Training preserves native map precision and uses aligned crops. Choose the training dimensions, image order and crop augmentation; **Exaggerate features** adjusts depth targets without editing the originals.

3. Export Create ML data: category folders containing one native 1,024 × 1,024 diffuse image per material, linked to the originals. Train the image classifier in Create ML and import its Core ML model into Loom Factory.
4. Export one `.textureloom` package containing the classifier, category models, trainable weights, identities and checksums. The package contains no source images.
5. Include the package when building Texture Studio. The installed app's base model is fixed by that build. Loom Workshop trains compatible LoRAs for it.

Open **Training** inside Loom Room to configure and monitor a run. The network and model-size pickers show exact parameter counts, through roughly four million parameters and custom width/block settings. Fresh training defaults to Original residual at 1,024 pixels and 1,943,893 scalar parameters, keeping every feature layer at native resolution. Pyramid remains an experimental option because saved brick models show sampling-grid artifacts. Saved runs retain their recorded architecture; current project settings can switch fresh training while keeping a similar parameter budget. Individual or all map-training runs can be moved to Trash from the training workspace. Choose AdamW, Adam, SGD momentum or Lion, batch size, gradient accumulation, clipping and weight decay. Learning-rate controls include a constant or cosine schedule and optional warmup. Advanced optimizer coefficients, device, seeds and checkpoint frequency are editable.

For the earlier fast design, start fresh and choose **Network → Compact NAF (comparison)**, then the approximately **1M parameters** model size: 32 channels, seven stages, 999,937 scalar parameters. Choosing this network sets learning rate **0.0001**, weight decay **0** and **Fixed global norm** clipping. Use AdamW and a maximum norm of 1 for the recorded comparison recipe. Its pixel/detail objective is selected automatically; there is no additional objective choice to configure. Original residual and Pyramid checkpoints remain available for A/B comparisons. The first warmed 1K benchmark measured approximately **0.47 s/update**; startup and image loading take additional time. See the [compact comparison evidence](Docs/Validation/compact-naf-2026-10-14.md).

Compact NAF uses native grids that are multiples of eight, with a minimum of 32 pixels. Its model test processes the complete original field because spatial attention changes with the field being tested. The canonical Float32 EXR preserves raw predictions, including values outside [0,1]; the PNG preview uses a fixed clamped range. Compare full-material detail against the target before approving the model.

The builder shows source material folders and image files for the selected dataset, followed by the pairs that will train or validate at each original resolution. **Run → Source resolution levels** selects one, two, three or all available levels. Each material starts with its smallest paired original large enough for the training crop; two levels include sources up to twice that resolution, and three include up to four times. Only originals already in the source folder are selected. New projects start with one level; older saved projects retain all available resolutions until changed. The run summary counts training and validation pair uses, batches and optimizer updates across the requested epochs. Browser search and file filters do not change these dataset counts.

Crop positions and flip/rotation choices have independent timing: every use, once per epoch or once per run. Flip choices apply to each image pair, with a 50% chance for each enabled horizontal/vertical flip. Image-order randomization remains separate. Held-out validation separates whole material folders, including every resolution, and uses fixed native center crops. It can save/export the best validation weights and stop after a selected number of epochs without improvement. A single-material dataset trains without held-out validation; early stopping requires another tagged material folder. Manual review is optional: current files, dataset tags and matching map roles determine training membership.

Crop origins can use every native pixel position that fits the selected size. Seed **-1**, the new default, chooses a system random seed once for a fresh run and records it for replay; explicit seeds remain deterministic. Continuing with -1 keeps the saved seed. Two bounded preparation workers overlap image decoding while consuming crops in the original training order.

**Saved checkpoints and models** appears at the top of Training. Select a run to edit its name and notes, test it, reveal its folder or move it to Trash. **Load checkpoint** restores all recorded training settings, including optimizer, crops, augmentation, validation and schedules; the form remains editable afterward. Compatible interrupted checkpoints restore their optimizer state and next image position. Changed settings or completed runs start a new epoch schedule while preserving prior history, cumulative epochs, updates, image visits and measured time. Names and notes are stored separately from immutable run artifacts. Validation loss and completion describe execution; inspect predictions on unseen materials before calling a model production quality.

**Training objective → Native pixel detail (v3, experimental)** adds matching of signed pixel differences at native resolution for depth and roughness. It can branch from a saved checkpoint or exported model. Changing objectives keeps those weights, starts a fresh optimizer, and labels the prior losses separately. Compare its whole-material output with the preserved original before approving it; the default and ordinary continuation keep their existing objective.

**Models** keeps the chosen saved version and **Approve & Use** visible above independently scrolling groups and details. Approval records a human review and chooses that exact version for export. **Test output** opens any saved model or checkpoint with usable native weights, independently of approval or deployment-package availability. Choose any source diffuse or an external image; paired original maps appear beside the input and result when present. Testing inputs do not inherit training membership or ignore filters.

In **Test Model**, residual and Pyramid inference now uses surrounding native
context and discards tile margins affected by the graph's zero padding. This
removes tile-border contamination without retraining or changing saved weights.
Beyond the actual image boundary, only the temporary model input uses reflected
context; original files, output dimensions and pixel scale stay unchanged.
**Edge handling → Legacy overlap blend (comparison)** retains the earlier result
for A/B review. Compact NAF continues to use one complete field. See the
[Paper border diagnosis](Docs/Validation/paper-context-inference-2026-10-16.md)
for saved-weight evidence, cost and quality limits.

**Use checkpoint as model** in Models or saved training versions exports that checkpoint's exact weights into a separate selected model version. The original run, its continuation state and its existing exported model remain intact. Review and approve the new candidate when ready; package export uses the selected version. With validation disabled, ordinary training export uses final weights, so an earlier checkpoint can be a better visual choice.

See [checkpoint and Materials validation](Docs/Validation/checkpoint-materials-usability-2026-10-14.md) for the measured checks and remaining installed/live verification boundary.

Loom Factory can publish the export to one Hugging Face model repository using the account's existing Hugging Face authentication. Publishing happens when you choose **Publish**.

A separate [map-type export utility](Scripts/README.md) creates one image per type per material folder for Create ML, using saved map-type tags. It does not train or integrate a map-type classifier.

## Build and verify

Requires Apple Silicon, macOS 26 and a Swift 6.2 toolchain.

```sh
swift test
Scripts/build-app.sh
Scripts/install-app.sh
```

Training runs in native Swift with Metal and MPSGraph. Parameters, gradients and optimizer moments remain on the Apple GPU between updates. ImageIO reads original pixels at native precision, and the app exports Core ML models and compatible Safetensors weights without an interpreter or conversion worker. Stop signals the worker independently, saves completed finite updates and returns after the current GPU operation. Existing project settings and saved weights remain compatible. See [the package contract](Docs/TextureLoomPackage.md) and [native training notes](Docs/NativeTraining.md).

Execution tests establish file handling, training and export behavior. They do not establish prediction quality. Review trained models on unseen materials before shipping a deployment.
