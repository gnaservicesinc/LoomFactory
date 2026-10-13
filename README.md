# Loom Factory

Backend extensions use versioned product and architecture contracts. See
[model product extension design](Docs/ModelProductExtensions.md) before adding
new products or optional processing stages. Photo-to-diffuse generation remains
future work.

Loom Factory creates custom Texture Loom deployments for Texture Studio. It is a fresh macOS app, separate from the texture editor and from Loom Workshop's LoRA trainer.

Open Loom Room and choose a source folder. The folder owns the entire project: settings, training runs, models and exports live in `.loom-factory`. Choosing another folder opens a different project; reopening the first restores its settings and runs. Imports link to original images rather than copying them.

Loom Room reads current image files. Categories, pairing, map roles, review decisions and notes live on the files as metadata, so renaming a file keeps its identity. There is no saved dataset-membership list. The **Dataset** menu shows categories found in the folder, and the viewer and trainer use the same paired-image selection.

The browser groups each material folder's native resolutions and map files in expandable rows. Known diffuse/color, displacement/height, roughness and normal filename conventions are recognized automatically. Bump maps have their own inspection-only type and never substitute for depth targets. **Needs map type** finds unrecognized files; **Ignore this file** keeps a file available for inspection while excluding it from categorization, Create ML export and training. DirectX normals are ignored automatically; normal training uses OpenGL maps.

1. Categorize each material folder once using one diffuse image, then review its dataset and matching map roles. Resuming reuses a folder's consistent existing category for its remaining untagged images. Existing tags and reviewed images are preserved; folders with conflicting categories are listed for review.
2. Train the depth, roughness or normal models needed for each dataset. Training preserves native map precision and uses aligned crops. Choose the training dimensions, image order and crop augmentation; **Exaggerate features** adjusts depth targets without editing the originals.
3. Export Create ML data: category folders containing one native 1,024 × 1,024 diffuse image per material, linked to the originals. Train the image classifier in Create ML and import its Core ML model into Loom Factory.
4. Export one `.textureloom` package containing the classifier, category models, trainable weights, identities and checksums. The package contains no source images.
5. Include the package when building Texture Studio. The installed app's base model is fixed by that build. Loom Workshop trains compatible LoRAs for it.

Loom Factory can publish the export to one Hugging Face model repository using the account's existing Hugging Face authentication. Publishing happens when you choose **Publish**.

## Build and verify

Requires Apple Silicon, macOS 26 and a Swift 6.2 toolchain.

```sh
swift test
Scripts/build-app.sh
Scripts/install-app.sh
```

The app installs its pinned Python training runtime in the selected source project's `.loom-factory` folder on first use. Training uses PyTorch with Metal acceleration and exports native Core ML models plus Safetensors weights. See [the package contract](Docs/TextureLoomPackage.md) and [runtime notes](Runtime/README.md).

Execution tests establish file handling, training and export behavior. They do not establish prediction quality. Review trained models on unseen materials before shipping a deployment.
