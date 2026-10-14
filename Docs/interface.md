# Loom Factory interface

Loom Factory opens directly into Loom Room. The first step is choosing a source
folder. The folder contains the project settings, run outputs, and exported
packages in `.loom-factory`. Image metadata contains dataset labels, map types,
pairing identities, review notes, and verification state. The application retains
only the most recently opened folder path outside the project.

The **Datasets** panel combines the category picker, image browser, original
image preview, and review editor. Category labels are displayed as datasets.
**All images** and **Uncategorized** are browsing filters; training requires a
specific dataset. A source-folder change has a separate, explicit action and
replaces the whole project state.

Loom Room polls the physical source folder every three seconds while idle.
Imports, review changes, pairing, training, and export also refresh the inventory
after they finish. Generation checks prevent a response from an old source
folder from being displayed in the newly opened project.

The **Models** workspace shows a dataset-by-product coverage matrix and the
shared classifier. Choose the version to use for each group, record approval
independently, and mark intentionally omitted products optional. Missing models,
approved candidates and approved assignments are distinct statuses. See
[Model management](ModelManagement.md) for persistence and extension boundaries.

The **Train a dataset** panel shows the selected dataset and target map, the
current pair count, native training dimensions, epochs, crop augmentation, and
image-order options. The **Exaggerate features** option appears for depth maps
and is passed only to depth training. Finished models are identified by dataset,
map type, and training time. Run details show actual engine events.

Dataset training has a **Continue latest run** option that continues the
matching saved weights with a fresh optimizer.
Missing weights are shown before training. Manual normal-map review includes
an explicit OpenGL or DirectX format selection.

The **Create ML** section exports category folders with one native
1,024 × 1,024 diffuse image per material. The exported images are symbolic links
to the originals. Train the image classifier in Create ML, then use
**Import Classifier** to add its Core ML model to this source project. Folders
that need a clear category or a native 1K diffuse image are listed in Run details.
Loom Factory focuses on training the texture models. Its bundled categorizer
assigns initial categories before a custom classifier is imported.

**Categorize** predicts once per material folder using one diffuse image and
applies that category to its untagged images. Resuming reuses a consistent
category already present in the folder. Existing tags and reviewed images are
preserved. Folders containing different existing categories are listed in Run
details for review. The summary reports the folder count, classifier prediction
count, and number of reused folder categories.

The **Texture Loom package** panel names and versions the unified package,
exports it into the current source project, and publishes an explicitly exported
package to the chosen Hugging Face repository. A failure displays its actual
error. The app does not manufacture a trained model, successful export, or
published URL when an operation fails.

Build the native app with `Scripts/build-app.sh`, or build and launch with
`Scripts/run-app.sh`. The app bundle lives in `.build/application/Loom Factory.app`.
Install into `/Applications` with `Scripts/install-app.sh`; close the running
app first. An existing installed app is retained as a backup during installation.
The build bundles this repository’s fresh runtime and the categorizer asset.
`LOOM_FACTORY_BOOTSTRAP_MODEL` can select a different compatible trained
categorizer asset during a custom build.

The executable accepts `--source-folder /absolute/path` to open a project
directly. The folder follows the same scan and metadata rules as the native
source-folder picker.
