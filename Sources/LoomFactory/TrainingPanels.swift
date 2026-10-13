import AppKit
import LoomCore
import SwiftUI

struct TrainingPanel: View {
    @Bindable var model: FactoryModel
    @State private var advanced = false

    var body: some View {
        FactoryPanel("Train a dataset", subtitle: "Create a custom base model from paired diffuse and material maps.") {
            HStack(alignment: .top, spacing: 20) {
                VStack(alignment: .leading, spacing: 7) {
                    Text("Dataset").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                    Picker("Dataset", selection: Binding(get: { model.selectedDataset?.id ?? "" }, set: { model.selectedDatasetID = $0 })) {
                        Text("Choose a dataset").tag("")
                        ForEach(model.snapshot?.datasets ?? [], id: \.id) { Text($0.name).tag($0.id) }
                    }.labelsHidden().frame(maxWidth: .infinity)
                }
                VStack(alignment: .leading, spacing: 7) {
                    Text("Map to train").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                    Picker("Map to train", selection: $model.selectedMap) {
                        ForEach(MapRole.targets) { Text($0.displayName).tag($0) }
                    }.labelsHidden().frame(maxWidth: .infinity)
                }
            }
            HStack(spacing: 9) {
                Image(systemName: "square.stack.3d.up").foregroundStyle(FactoryStyle.accent)
                if let dataset = model.selectedDataset {
                    Text("\(model.eligiblePairCount.formatted()) reviewed \(model.selectedMap.displayName.lowercased()) pairs ready in \(dataset.name)")
                        .font(.callout)
                } else {
                    Text("Choose the dataset to train.").font(.callout).foregroundStyle(.secondary)
                }
            }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
                .background(FactoryStyle.accent.opacity(0.06), in: RoundedRectangle(cornerRadius: 9))
            Grid(alignment: .leading, horizontalSpacing: 22, verticalSpacing: 14) {
                GridRow {
                    Text("Training dimensions")
                    Picker("Training dimensions", selection: $model.settings.training.imageSize) {
                        ForEach([256, 512, 1024, 2048, 4096], id: \.self) { value in
                            Text("\(value.formatted()) × \(value.formatted())").tag(value)
                        }
                    }.labelsHidden().frame(width: 165)
                    Spacer()
                }
                GridRow {
                    Text("Epochs")
                    TextField("Epochs", value: $model.settings.training.epochs, format: .number)
                        .textFieldStyle(.roundedBorder).frame(width: 165)
                    Text("One pass through the dataset.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("Training uses verified pairs that fit this pixel size. Matching maps share each random crop position, and original pixels are preserved.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Divider()
            VStack(alignment: .leading, spacing: 11) {
                Toggle("Continue latest run", isOn: Binding(
                    get: { model.settings.training.continueLatestWeights ?? false },
                    set: { model.settings.training.continueLatestWeights = $0 }
                ))
                .disabled(!model.canContinueMap && !(model.settings.training.continueLatestWeights ?? false))
                .help("Continue saved weights with a fresh optimizer.")
                if model.settings.training.continueLatestWeights == true && !model.canContinueMap {
                    Text("There are no saved weights for this dataset and map. Turn off Continue latest run to train a new model.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Toggle("Randomize image order each epoch", isOn: $model.settings.training.shuffleEachEpoch)
                    .help("Off keeps the image order the same for every epoch.")
                HStack(spacing: 28) {
                    Toggle("Randomly flip crops", isOn: $model.settings.training.flipCrops)
                    Toggle("Randomly rotate crops", isOn: $model.settings.training.rotateCrops)
                }
                if model.selectedMap == .depth {
                    Toggle("Exaggerate features", isOn: $model.settings.training.exaggerateFeatures)
                        .help("Apply nonlinear contrast to depth values during training. Original images are unchanged.")
                    if model.settings.training.exaggerateFeatures {
                        Text("Nonlinear contrast emphasizes depth features while preserving the map’s range.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }.toggleStyle(.checkbox)
            DisclosureGroup("Advanced training settings", isExpanded: $advanced) {
                Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 12) {
                    settingRow("Learning rate", value: $model.settings.training.learningRate)
                    settingRow("Batch size", value: $model.settings.training.batchSize)
                    settingRow("Model width", value: $model.settings.training.modelWidth)
                    settingRow("Random seed", value: $model.settings.training.seed)
                }.padding(.top, 12)
            }
            FactoryActionButton(title: "Train \(model.selectedMap.displayName) Model", symbol: "play.fill", prominent: true) {
                model.trainMap()
            }.disabled(!model.canTrain)
            if !model.selectedArtifacts.isEmpty {
                Divider()
                Text("Trained models").font(.subheadline.weight(.semibold))
                ForEach(model.selectedArtifacts.prefix(3), id: \.id) { artifact in
                    HStack(spacing: 12) {
                        Image(systemName: "cube").foregroundStyle(FactoryStyle.accent)
                        VStack(alignment: .leading, spacing: 3) {
                            Text("\(artifact.datasetName ?? "Dataset") · \(artifact.map?.displayName ?? "Model")")
                                .font(.callout.weight(.medium))
                            Text("\(artifact.epochs) epochs · \(artifact.createdAt.formatted(date: .abbreviated, time: .shortened))")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Show") { model.reveal(artifact.directory) }
                    }
                }
            }
        }
        .disabled(model.busy)
    }

    private func settingRow(_ title: String, value: Binding<Int>) -> some View {
        GridRow {
            Text(title).font(.callout)
            TextField(title, value: value, format: .number).textFieldStyle(.roundedBorder).frame(width: 165)
        }
    }

    private func settingRow(_ title: String, value: Binding<Double>) -> some View {
        GridRow {
            Text(title).font(.callout)
            TextField(title, value: value, format: .number).textFieldStyle(.roundedBorder).frame(width: 165)
        }
    }
}

struct CreateMLPanel: View {
    @Bindable var model: FactoryModel

    var body: some View {
        FactoryPanel("Create ML", subtitle: "Prepare the images for your dataset classifier.") {
            Text("Export category folders with one native 1K diffuse image per material. Train the image classifier in Create ML, then import its Core ML model.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            FactoryActionButton(title: "Export Create ML Data…", symbol: "square.and.arrow.up") {
                model.exportClassifierData()
            }.disabled(model.busy || model.snapshot?.images.isEmpty != false)
            FactoryActionButton(title: "Import Classifier…", symbol: "square.and.arrow.down") {
                model.importClassifier()
            }.disabled(model.busy || model.snapshot == nil)
            if let classifier = model.classifier {
                HStack {
                    FactoryStatus(text: "Classifier ready", symbol: "checkmark.circle.fill", color: .green)
                    Spacer()
                    Button("Show Model") { model.reveal(classifier.directory) }
                }
                Text("Updated \(classifier.createdAt.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }.disabled(model.busy)
    }
}

struct PackagePanel: View {
    @Bindable var model: FactoryModel

    var body: some View {
        FactoryPanel("Texture Loom package", subtitle: "One package for a custom Texture Studio build.") {
            VStack(alignment: .leading, spacing: 12) {
                labeledField("Model name", text: $model.settings.modelName, prompt: "Texture Loom")
                labeledField("Version", text: $model.settings.modelVersion, prompt: "1.0.0")
            }
            HStack {
                Text("Included models").foregroundStyle(.secondary)
                Spacer()
                Text("\(model.snapshot?.artifacts.filter { $0.kind == "map" }.count ?? 0)").fontWeight(.semibold)
            }.font(.callout)
            FactoryActionButton(title: "Export Texture Loom", symbol: "shippingbox", prominent: true) { model.exportPackage() }
                .disabled(model.busy || model.classifier == nil || (model.snapshot?.artifacts.filter { $0.kind == "map" }.isEmpty ?? true) ||
                          model.settings.modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.settings.modelVersion.isEmpty)
            if let package = model.exportedPackage {
                Button { model.reveal(package) } label: {
                    Label(package.lastPathComponent, systemImage: "checkmark.circle").font(.caption).lineLimit(2)
                }.buttonStyle(.plain).foregroundStyle(.green)
            }
            Divider()
            labeledField("Hugging Face repository", text: $model.settings.huggingFaceRepository, prompt: "Account/Texture-Loom")
            Button("Publish Exported Package", systemImage: "arrow.up.circle") { model.publishPackage() }
                .frame(maxWidth: .infinity).buttonStyle(.bordered).controlSize(.large)
                .disabled(model.busy || model.exportedPackage == nil || model.settings.huggingFaceRepository.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            if let url = model.publishedURL { Link("Open on Hugging Face", destination: url).font(.caption) }
        }.disabled(model.busy)
    }

    private func labeledField(_ title: String, text: Binding<String>, prompt: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption.weight(.medium)).foregroundStyle(.secondary)
            TextField(prompt, text: text).textFieldStyle(.roundedBorder)
        }
    }
}

struct RunPanel: View {
    @Bindable var model: FactoryModel
    @State private var showLog = false

    var body: some View {
        FactoryPanel("Run progress") {
            HStack(spacing: 14) {
                if model.busy { ProgressView().controlSize(.small) }
                Text(model.activityTitle).font(.callout).textSelection(.enabled)
                Spacer()
                if model.busy { Button("Stop") { model.cancel() } }
            }
            if let progress = model.progress {
                ProgressView(value: progress).progressViewStyle(.linear)
            }
            if let latest = model.events.last, let loss = latest.loss {
                HStack {
                    if let epoch = latest.epoch { Text("Epoch \(epoch)") }
                    Text("Loss \(loss.formatted(.number.precision(.significantDigits(5))))")
                }.font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            if !model.events.isEmpty {
                DisclosureGroup("Run details", isExpanded: $showLog) {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 7) {
                            ForEach(Array(model.events.enumerated()), id: \.offset) { _, event in
                                Text(event.message).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }.padding(.vertical, 8)
                    }.frame(height: 180)
                }
            }
        }
    }
}
