import LoomCore
import SwiftUI
import UniformTypeIdentifiers

struct ModelTestView: View {
    @Bindable var model: FactoryModel
    @State private var imageZoom: ComparisonImageZoom = .fit
    @State private var choosingSource = false

    private var originalMap: ModelTestOriginalMap {
        model.testOriginalMap
    }

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
                if let previous = model.previousTestResult {
                    ModelTestCheckpointComparison(first: previous, second: result)
                }
                ModelTestResultView(result: result, imageZoom: $imageZoom,
                                    saveOutput: { model.saveModelTestOutput(format: .float32EXR) },
                                    savePNGPreview: { model.saveModelTestOutput(format: .png16) },
                                    showOutput: { model.reveal(result.outputURL) },
                                    showFolder: { model.reveal(result.directory) },
                                    originalMap: originalMap,
                                    originalMapRevision: originalMap.image.map(model.imagePreviewRevision))
            } else {
                FactoryPanel("Preview") {
                    HStack(alignment: .top, spacing: 18) {
                        if let input = model.testInputURL {
                            VStack(alignment: .leading, spacing: 7) {
                                Text("Diffuse input").font(.subheadline.weight(.medium))
                                FactoryImage(url: input, revision: inputRevision(input), applyOrientationTransform: false).frame(height: 350)
                                    .clipShape(RoundedRectangle(cornerRadius: 7))
                            }.frame(maxWidth: .infinity)
                        }
                        if let target = originalMap.image {
                            VStack(alignment: .leading, spacing: 7) {
                                Text("Original \(target.role.displayName.lowercased()) map").font(.subheadline.weight(.medium))
                                FactoryImage(url: target.url, revision: model.imagePreviewRevision(target), applyOrientationTransform: false).frame(height: 350)
                                    .clipShape(RoundedRectangle(cornerRadius: 7))
                                Text(target.url.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                            }.frame(maxWidth: .infinity).accessibilityIdentifier("modelTest.originalMapPreview")
                        } else {
                            ContentUnavailableView("Your model output appears here", systemImage: "photo.on.rectangle",
                                               description: Text("Choose one saved run and a diffuse image, then run the model."))
                            .frame(maxWidth: .infinity, minHeight: 350)
                        }
                    }
                    if let issue = originalMap.issue { Text(issue).font(.caption).foregroundStyle(.secondary) }
                }
            }
        }.accessibilityIdentifier("modelTest.workspace")
            .sheet(isPresented: $choosingSource) {
                ModelTestSourcePicker(images: model.testDiffuseImages) { model.setModelTestInput($0.url) }
            }
    }

    private func inputRevision(_ url: URL) -> String {
        model.snapshot?.images.first { $0.url == url }.map(model.imagePreviewRevision) ?? url.path
    }

    private var modelSelection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 7) {
                    Text("Category").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                    Picker("Category", selection: Binding(get: { model.testDatasetID }, set: { model.setModelTestDataset($0) })) {
                        ForEach(model.snapshot?.datasets ?? []) { dataset in Text(dataset.name).tag(dataset.id) }
                    }.labelsHidden().accessibilityIdentifier("modelTest.category")
                }.frame(maxWidth: .infinity, alignment: .leading)
                VStack(alignment: .leading, spacing: 7) {
                    Text("Map").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                    Picker("Map", selection: Binding(get: { model.testMap }, set: { model.setModelTestMap($0) })) {
                        ForEach(MapRole.targets, id: \.self) { Text($0.displayName).tag($0) }
                    }.labelsHidden().accessibilityIdentifier("modelTest.map")
                }.frame(width: 210, alignment: .leading)
            }
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
            if model.effectiveTestWeightSource == .checkpoint {
                VStack(alignment: .leading, spacing: 7) {
                    Text("Checkpoint").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                    Picker("Checkpoint", selection: Binding(get: { model.testCheckpointID ?? "" }, set: { model.setModelTestCheckpoint($0.isEmpty ? nil : $0) })) {
                        Text("Latest saved checkpoint").tag("")
                        ForEach(model.testCheckpointVersions.filter { $0.revisionID != nil }) { checkpoint in
                            Text("Epoch \(checkpoint.totalEpochsCompleted) · update \(checkpoint.totalUpdatesCompleted) · \(checkpoint.modifiedAt.formatted(date: .abbreviated, time: .shortened))")
                                .tag(checkpoint.revisionID ?? "")
                        }
                    }.labelsHidden().accessibilityIdentifier("modelTest.checkpoint")
                    HStack(spacing: 12) {
                        Text("\(model.testCheckpointVersions.count.formatted()) saved checkpoints in this run")
                            .font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        if let version = model.selectedTestVersion {
                            Button("Use selected checkpoint as model", systemImage: "checkmark.circle") {
                                model.useCheckpointAsModel(version, checkpointID: model.testCheckpointID)
                            }.disabled(model.busy).controlSize(.small)
                                .accessibilityIdentifier("modelTest.useCheckpoint")
                        }
                    }
                }
            }
            if let version = model.selectedTestVersion {
                HStack(spacing: 16) {
                    if version.artifact != nil {
                        Text("Exported model: \(version.exportedEpochsCompleted.formatted()) completed epochs")
                    }
                    if let checkpoint = model.selectedTestCheckpoint {
                        Text("Checkpoint: \(checkpoint.totalEpochsCompleted.formatted()) completed epochs")
                    }
                }.font(.caption).foregroundStyle(.secondary)
                    .accessibilityIdentifier("modelTest.savedEpochs")
            }
        }
    }

    private var inputSelection: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text("Diffuse input").font(.caption.weight(.medium)).foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Button("Choose Image…", systemImage: "photo.badge.plus") { model.chooseModelTestInput() }
                    .accessibilityIdentifier("modelTest.chooseImage")
                if !model.testDiffuseImages.isEmpty {
                    Button("Use Source Image…", systemImage: "folder") { choosingSource = true }
                        .accessibilityIdentifier("modelTest.sourceImage")
                }
            }
            if let input = model.testInputURL {
                Text(input.path).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    .textSelection(.enabled).accessibilityIdentifier("modelTest.inputPath")
            }
            Text("You can test any material, including an image outside this project. Original files remain untouched; no paired target is needed.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct ModelTestCheckpointComparison: View {
    let first: ModelTestResult
    let second: ModelTestResult

    var body: some View {
        FactoryPanel("Checkpoint A/B", subtitle: "The previous and current saved weights on the same unchanged input, shown at the same scale.") {
            HStack(alignment: .top, spacing: 18) {
                output(first, label: "A · Previous")
                output(second, label: "B · Current")
            }
        }.accessibilityIdentifier("modelTest.checkpointComparison")
    }

    private func output(_ result: ModelTestResult, label: String) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(label).font(.subheadline.weight(.medium))
            Text(result.source == .model ? "Exported model" : "Checkpoint \(result.checkpointID?.prefix(12) ?? "latest")")
                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            FactoryImage(url: result.previewURL, revision: result.directory.path, maximumPixels: max(result.width, result.height),
                interpolation: .none, applyOrientationTransform: false)
                .frame(height: 400).clipShape(RoundedRectangle(cornerRadius: 7))
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}

enum ModelTestExportFormat: Equatable {
    case float32EXR, png16
    var fileExtension: String { self == .float32EXR ? "exr" : "png" }
    var displayName: String { self == .float32EXR ? "32-bit float EXR" : "16-bit PNG" }
    var contentType: UTType {
        self == .float32EXR ? (UTType(filenameExtension: "exr") ?? UTType(importedAs: "com.ilm.openexr-image")) : .png
    }
}

enum ModelTestOutputExport {
    static func save(_ result: ModelTestResult, to destination: URL, sourceImageURLs: [URL],
                     format: ModelTestExportFormat = .float32EXR) throws {
        let resolved = destination.standardizedFileURL.resolvingSymlinksInPath()
        let originals = [result.inputURL] + sourceImageURLs
        guard !originals.contains(where: { $0.standardizedFileURL.resolvingSymlinksInPath() == resolved }) else {
            throw LoomError("Choose another output filename. Saving here would overwrite an original source image.")
        }
        let source = format == .float32EXR ? result.outputURL : result.previewURL
        guard source.pathExtension.lowercased() == format.fileExtension else {
            throw LoomError(format == .float32EXR
                ? "This saved result has no 32-bit float EXR. Run the model again to create one."
                : "This saved result has no PNG preview to export.")
        }
        guard destination.pathExtension.lowercased() == format.fileExtension else {
            throw LoomError("Choose an .\(format.fileExtension) filename for \(format.displayName).")
        }
        let resolvedSource = source.standardizedFileURL.resolvingSymlinksInPath()
        guard resolved != resolvedSource else { return }
        let resultFiles = [result.outputURL, result.previewURL].map { $0.standardizedFileURL.resolvingSymlinksInPath() }
        guard !resultFiles.contains(resolved) else {
            throw LoomError("Choose another output filename. Saving here would replace another saved result format.")
        }
        try Data(contentsOf: source, options: .mappedIfSafe).write(to: destination, options: .atomic)
    }
}

struct ModelTestResultView: View {
    let result: ModelTestResult
    @Binding var imageZoom: ComparisonImageZoom
    let saveOutput: () -> Void
    let savePNGPreview: (() -> Void)?
    let showOutput: () -> Void
    let showFolder: () -> Void
    let originalMap: ModelTestOriginalMap
    let originalMapRevision: String?

    init(result: ModelTestResult, imageZoom: Binding<ComparisonImageZoom>, saveOutput: @escaping () -> Void,
         savePNGPreview: (() -> Void)? = nil, showOutput: @escaping () -> Void, showFolder: @escaping () -> Void,
         originalMap: ModelTestOriginalMap = .resolve(inputURL: nil, map: nil, images: []),
         originalMapRevision: String? = nil) {
        self.result = result; self._imageZoom = imageZoom
        self.saveOutput = saveOutput; self.savePNGPreview = savePNGPreview
        self.showOutput = showOutput; self.showFolder = showFolder
        self.originalMap = originalMap
        self.originalMapRevision = originalMapRevision
    }

    var body: some View {
        FactoryPanel("\(result.map.displayName) output", subtitle: "\(result.width.formatted()) × \(result.height.formatted()) pixels · \(result.outputURL.pathExtension.lowercased() == "exr" ? "32-bit float EXR" : "PNG output")") {
            HStack {
                Picker("Image zoom", selection: $imageZoom) {
                    ForEach(ComparisonImageZoom.allCases) { Text($0.displayName).tag($0) }
                }.pickerStyle(.segmented).frame(width: 260).accessibilityIdentifier("modelTest.zoom")
                Spacer()
                Button("Save 32-bit EXR…", systemImage: "square.and.arrow.up", action: saveOutput)
                    .accessibilityIdentifier("modelTest.saveOutput")
                if let savePNGPreview {
                    Button("Export 16-bit PNG…", action: savePNGPreview)
                        .accessibilityIdentifier("modelTest.savePNGPreview")
                }
                Button("Show Output", action: showOutput)
                Button("Show Test Folder", action: showFolder)
            }
            HStack(alignment: .top, spacing: 18) {
                if let target = originalMap.image {
                    imagePanel("Original \(result.map.displayName.lowercased()) map", url: target.url, revision: originalMapRevision)
                        .accessibilityIdentifier("modelTest.originalMap")
                } else { imagePanel("Diffuse input", url: result.inputURL) }
                imagePanel("Generated \(result.map.displayName.lowercased())", url: result.previewURL)
            }
            if let target = originalMap.image {
                Text("Original: \(target.url.path)\(target.normalConvention.map { " · " + $0 + " normals" } ?? "")")
                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                DisclosureGroup("Diffuse input") { imagePanel("Original diffuse input", url: result.inputURL) }
            }
            if let issue = originalMap.issue { Text(issue).font(.caption).foregroundStyle(.secondary) }
            Text("The display uses a PNG preview. Save the EXR to keep the model's 32-bit floating-point output; exporting PNG reduces it to 16-bit integer values.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text("Run \(result.runID.prefix(8)) · \(result.source == .model ? "Exported model" : "Saved checkpoint") · \(result.tileCount.formatted()) native \(result.tileSize) px tiles")
                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            if let checkpointID = result.checkpointID {
                Text("Checkpoint \(checkpointID)").font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            Text(result.note).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }.accessibilityIdentifier("modelTest.result")
    }

    private func imagePanel(_ title: String, url: URL, revision: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title).font(.subheadline.weight(.medium))
            if let scale = imageZoom.scale {
                ScrollView([.horizontal, .vertical]) {
                    FactoryImage(url: url, revision: revision ?? result.directory.path,
                                 maximumPixels: max(result.width, result.height), interpolation: .none,
                                 applyOrientationTransform: false)
                        .frame(width: CGFloat(result.width) * scale, height: CGFloat(result.height) * scale)
                }.frame(height: 480).clipShape(RoundedRectangle(cornerRadius: 7))
            } else {
                FactoryImage(url: url, revision: revision ?? result.directory.path, maximumPixels: 1600, applyOrientationTransform: false)
                    .frame(height: 480).clipShape(RoundedRectangle(cornerRadius: 7))
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}
