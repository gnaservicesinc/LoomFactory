# Loom Factory

Backend extensions use versioned product and architecture contracts. See
[model product extension design](Docs/ModelProductExtensions.md) before adding
new products or optional processing stages. Photo-to-diffuse generation remains
future work.

Loom Factory creates custom Texture Loom deployments for Texture Studio. It is a fresh macOS app, separate from the texture editor and from Loom Workshop's LoRA trainer.

Open Loom Room and choose a source folder. The folder owns the entire project: settings, training runs, models and exports live in `.loom-factory`. Choosing another folder opens a different project; reopening the first restores its settings and runs. Imports link to original images rather than copying them.

Loom Room reads current image files. Categories, pairing, map roles, review decisions and notes live on the files as metadata, so renaming a file keeps its identity. There is no saved dataset-membership list. The **Dataset** menu shows categories found in the folder, and the viewer and trainer use the same paired-image selection.

The browser groups each material folder's native resolutions and map files in expandable rows. Known diffuse/color, displacement/height, roughness and normal filename conventions are recognized automatically. Primary Poly Haven colour variants (`col1`, `coll1`, `col_1`) also pair with their matching maps; alternate colour variants remain available for explicit map-type selection. Bump maps have their own inspection-only type and never substitute for depth targets. **Needs map type** finds unrecognized files; **Ignore this file** keeps a file available for inspection while excluding it from categorization, Create ML export and training. DirectX normals are ignored automatically; normal training uses OpenGL maps.

1. Categorize each material folder once using one diffuse image. Inspect and correct dataset or map types when needed; manual review is optional. Resuming reuses a folder's consistent existing category for its remaining untagged images. Existing tags and reviewed images are preserved; folders with conflicting categories are listed for correction.
2. Train the depth, roughness or normal models needed for each dataset. Training preserves native map precision and uses aligned crops. Choose the training dimensions, image order and crop augmentation; **Exaggerate features** adjusts depth targets without editing the originals.

3. Export Create ML data: category folders containing one native 1,024 × 1,024 diffuse image per material, linked to the originals. Train the image classifier in Create ML and import its Core ML model into Loom Factory.
4. Export one `.textureloom` package containing the classifier, category models, trainable weights, identities and checksums. The package contains no source images.
5. Include the package when building Texture Studio. The installed app's base model is fixed by that build. Loom Workshop trains compatible LoRAs for it.

Open **Training** inside Loom Room to configure and monitor a run. The network and model-size pickers show exact parameter counts, through roughly four million parameters and custom width/block settings. Fresh training defaults to the efficient Pyramid model at 1,024 pixels and 1,926,373 scalar parameters. Saved runs retain their recorded architecture; current project settings can switch fresh training while keeping a similar parameter budget. Individual or all map-training runs can be moved to Trash from the training workspace. Choose AdamW, Adam, SGD momentum or Lion, batch size, gradient accumulation, clipping and weight decay. Learning-rate controls include a constant or cosine schedule and optional warmup. Advanced optimizer coefficients, device, seeds and checkpoint frequency are editable.

Crop positions and flip/rotation choices have independent timing: every use, once per epoch or once per run. Flip choices apply to each image pair, with a 50% chance for each enabled horizontal/vertical flip. Image-order randomization remains separate. Held-out validation separates whole material folders, including every resolution, and uses fixed native center crops. It can save/export the best validation weights and stop after a selected number of epochs without improvement. A single-material dataset trains without held-out validation; early stopping requires another tagged material folder. Manual review is optional: current files, dataset tags and matching map roles determine training membership.

**Continue latest weights** finds completed, stopped and cancelled checkpoints in this source project. It retains their actual architecture, crop dimensions and depth transform, then applies the selected optimizer and learning-rate settings with a fresh optimizer. It is weight continuation, not exact optimizer-state resumption. Validation loss and run completion help assess training; inspect predictions on unseen materials before calling a model production quality.

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
