import LoomCore
import SwiftUI

struct ModelTestView: View {
    @Bindable var model: FactoryModel
    @State private var imageZoom: ComparisonImageZoom = .fit

    var body: some View {
        VStack(alignment: .leading, spacing: FactoryStyle.spacing) {
            FactoryPanel("Test a model", subtitle: "Run a saved model on a diffuse image and inspect its output at the original dimensions.") {
                modelSelection
                inputSelection
                HStack(spacing: 12) {
                    Button("Run Model", systemImage: "play.fill") { model.testModel() }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .disabled(!model.canTestModel)
                        .accessibilityIdentifier("modelTest.run")
                    if model.busy {
                        ProgressView(value: model.progress).frame(width: 170)
                        Text(model.events.last?.message ?? model.activityTitle).font(.callout).foregroundStyle(.secondary)
                            .lineLimit(2).frame(maxWidth: .infinity, alignment: .leading)
                        Button("Stop") { model.cancel() }.accessibilityIdentifier("modelTest.stop")
                    }
                }
                if let issue = model.modelTestIssue {
                    Text(issue).font(.callout).foregroundStyle(.secondary)
                        .accessibilityIdentifier("modelTest.issue")
                }
            }
            if let result = model.modelTestResult {
                ModelTestResultView(result: result, imageZoom: $imageZoom,
                                    saveOutput: { model.saveModelTestOutput() },
                                    showOutput: { model.reveal(result.outputURL) },
                                    showFolder: { model.reveal(result.directory) })
            } else {
                FactoryPanel("Preview") {
                    HStack(alignment: .top, spacing: 18) {
                        if let input = model.testInputURL {
                            VStack(alignment: .leading, spacing: 7) {
                                Text("Diffuse input").font(.subheadline.weight(.medium))
                                FactoryImage(url: input, revision: input.path, applyOrientationTransform: false).frame(height: 350)
                                    .clipShape(RoundedRectangle(cornerRadius: 7))
                            }.frame(maxWidth: .infinity)
                        }
                        ContentUnavailableView("Your model output appears here", systemImage: "photo.on.rectangle",
                                               description: Text("Choose one saved run and a diffuse image, then run the model."))
                            .frame(maxWidth: .infinity, minHeight: 350)
                    }
                }
            }
        }.accessibilityIdentifier("modelTest.workspace")
    }

    private var modelSelection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 20) {
            VStack(alignment: .leading, spacing: 7) {
                Text("Saved run").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                Picker("Saved run", selection: Binding(get: { model.selectedTestVersionID.isEmpty ? model.selectedTestVersion?.id ?? "" : model.selectedTestVersionID },
                                                        set: { model.selectedTestVersionID = $0 })) {
                    if model.testableModelVersions.isEmpty { Text("No saved models or checkpoints").tag("") }
                    if !model.selectedTestVersionID.isEmpty, model.selectedTestVersion == nil {
                        Text("Selected run is unavailable").tag(model.selectedTestVersionID)
                    }
                    ForEach(model.testableModelVersions) { version in
                        Text("\(version.datasetName) · \(version.map.displayName) · \(ModelVersionPresentation.title(version))")
                            .tag(version.id)
                    }
                }.labelsHidden().accessibilityIdentifier("modelTest.version")
            }.frame(maxWidth: .infinity, alignment: .leading)
            VStack(alignment: .leading, spacing: 7) {
                Text("Saved weights").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                Picker("Saved weights", selection: Binding(get: { model.effectiveTestWeightSource ?? .model },
                                                           set: { model.testWeightSource = $0 })) {
                    if model.availableTestWeightSources.isEmpty { Text("No saved weights").tag(TrainingWeightSource.model) }
                    ForEach(model.availableTestWeightSources, id: \.rawValue) { source in
                        Text(source == .model ? "Exported model" : "Saved checkpoint").tag(source)
                    }
                }.labelsHidden().accessibilityIdentifier("modelTest.weights")
            }.frame(width: 210, alignment: .leading)
            }
            if let version = model.selectedTestVersion {
                HStack(spacing: 16) {
                    if let artifact = version.artifact {
                        Text("Exported model: \(artifact.epochs.formatted()) completed epochs")
                    }
                    if let checkpoint = version.checkpoint {
                        Text("Checkpoint: \(checkpoint.epochsCompleted.formatted()) completed epochs")
                    }
                }.font(.caption).foregroundStyle(.secondary)
                    .accessibilityIdentifier("modelTest.savedEpochs")
            }
        }.disabled(model.busy)
    }

    private var inputSelection: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text("Diffuse input").font(.caption.weight(.medium)).foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Button("Choose Image…", systemImage: "photo.badge.plus") { model.chooseModelTestInput() }
                    .accessibilityIdentifier("modelTest.chooseImage")
                if !model.testDiffuseImages.isEmpty {
                    Picker("Use an image from this dataset", selection: Binding(
                        get: { model.testDiffuseImages.first { $0.url == model.testInputURL }?.id ?? "" },
                        set: { id in
                            if let image = model.testDiffuseImages.first(where: { $0.id == id }) { model.setModelTestInput(image.url) }
                        }
                    )) {
                        Text("Or use a source diffuse…").tag("")
                        ForEach(model.testDiffuseImages) { image in
                            Text("\(image.url.deletingLastPathComponent().lastPathComponent) / \(image.url.lastPathComponent)")
                                .tag(image.id)
                        }
                    }.labelsHidden().frame(maxWidth: 640).accessibilityIdentifier("modelTest.sourceImage")
                }
            }.disabled(model.busy)
            if let input = model.testInputURL {
                Text(input.path).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    .textSelection(.enabled).accessibilityIdentifier("modelTest.inputPath")
            }
            Text("You can test any material, including an image outside this project. Original files remain untouched; no paired target is needed.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}

enum ModelTestOutputExport {
    static func save(_ result: ModelTestResult, to destination: URL, sourceImageURLs: [URL]) throws {
        let resolved = destination.standardizedFileURL.resolvingSymlinksInPath()
        let originals = [result.inputURL] + sourceImageURLs
        guard !originals.contains(where: { $0.standardizedFileURL.resolvingSymlinksInPath() == resolved }) else {
            throw LoomError("Choose another output filename. Saving here would overwrite an original source image.")
        }
        guard resolved != result.outputURL.standardizedFileURL.resolvingSymlinksInPath() else { return }
        try Data(contentsOf: result.outputURL, options: .mappedIfSafe).write(to: destination, options: .atomic)
    }
}

struct ModelTestResultView: View {
    let result: ModelTestResult
    @Binding var imageZoom: ComparisonImageZoom
    let saveOutput: () -> Void
    let showOutput: () -> Void
    let showFolder: () -> Void

    var body: some View {
        FactoryPanel("\(result.map.displayName) output", subtitle: "\(result.width.formatted()) × \(result.height.formatted()) pixels · 16-bit PNG") {
            HStack {
                Picker("Image zoom", selection: $imageZoom) {
                    ForEach(ComparisonImageZoom.allCases) { Text($0.displayName).tag($0) }
                }.pickerStyle(.segmented).frame(width: 260).accessibilityIdentifier("modelTest.zoom")
                Spacer()
                Button("Save Output…", systemImage: "square.and.arrow.up", action: saveOutput)
                    .accessibilityIdentifier("modelTest.saveOutput")
                Button("Show Output", action: showOutput)
                Button("Show Test Folder", action: showFolder)
            }
            HStack(alignment: .top, spacing: 18) {
                imagePanel("Diffuse input", url: result.inputURL)
                imagePanel("Generated \(result.map.displayName.lowercased())", url: result.outputURL)
            }
            Text("Run \(result.runID.prefix(8)) · \(result.source == .model ? "Exported model" : "Saved checkpoint") · \(result.tileCount.formatted()) native \(result.tileSize) px tiles")
                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            Text(result.note).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }.accessibilityIdentifier("modelTest.result")
    }

    private func imagePanel(_ title: String, url: URL) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(.subheadline.weight(.medium))
            if let scale = imageZoom.scale {
                ScrollView([.horizontal, .vertical]) {
                    FactoryImage(url: url, revision: result.directory.path,
                                 maximumPixels: max(result.width, result.height), interpolation: .none,
                                 applyOrientationTransform: false)
                        .frame(width: CGFloat(result.width) * scale, height: CGFloat(result.height) * scale)
                }.frame(height: 480).clipShape(RoundedRectangle(cornerRadius: 7))
            } else {
                FactoryImage(url: url, revision: result.directory.path, maximumPixels: 1600, applyOrientationTransform: false)
                    .frame(height: 480).clipShape(RoundedRectangle(cornerRadius: 7))
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}
