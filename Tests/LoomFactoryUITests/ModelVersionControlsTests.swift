import AppKit
import ImageIO
import LoomCore
import LoomNative
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

@Suite(.serialized)
struct ModelVersionControlsTests {
    @Test @MainActor
    func everyVersionRemainsAvailableAndHigherErrorVersionCanBeUsedForExport() async throws {
        let fixture = try await VersionControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, source: fixture.source)
        model.selectedDatasetID = fixture.datasetID
        #expect(model.selectedModelVersions.count == 6)
        #expect(model.exportedModelVersions.count == 6)
        #expect(model.activeMapVersions.count == 1)
        #expect(model.exportedPackage != nil)
        let oldest = try #require(model.selectedModelVersions.last)
        let latest = try #require(model.selectedModelVersions.first)
        #expect(try #require(oldest.artifact?.loss) > #require(latest.artifact?.loss))
        model.useModelVersionForExport(oldest)
        try await wait(model)
        #expect(model.selectedModelVersions.first(where: \.isActive)?.id == oldest.id)
        #expect(model.selectedModelVersions.count == 6)
        #expect(model.exportedPackage == nil)
        #expect(model.publishedURL == nil)
        let reopened = FactoryModel(rememberSourceFolder: false)
        try await load(reopened, source: fixture.source)
        reopened.selectedDatasetID = fixture.datasetID
        #expect(reopened.selectedModelVersions.first(where: \.isActive)?.id == oldest.id)
        #expect(reopened.exportedPackage == nil)
        #expect(reopened.firstComparisonVersion?.id == oldest.id)
        #expect(reopened.secondComparisonVersion?.id != oldest.id)
        try await render(TrainingPanel(model: model, showTuning: true), size: NSSize(width: 760, height: 3800), name: "training-version-history.png")
    }

    @Test @MainActor
    func selectedContinuationUsesItsArchitectureAndKeepsNewOptimizerSettings() async throws {
        let fixture = try await VersionControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, source: fixture.source)
        model.selectedDatasetID = fixture.datasetID
        let oldest = try #require(model.selectedModelVersions.last)
        model.settings.training.learningRate = 0.02
        model.settings.training.optimizer = .sgd
        model.settings.training.modelWidth = 164
        model.settings.training.imageSize = 2048
        model.settings.training.validationFraction = 0
        model.continueModelVersion(oldest, source: .exportedModel)
        #expect(model.trainingStart == .model(runID: oldest.id))
        #expect(model.effectiveTrainingSettings.modelWidth == oldest.settings?.modelWidth)
        #expect(model.effectiveTrainingSettings.imageSize == oldest.settings?.imageSize)
        #expect(model.effectiveTrainingSettings.learningRate == 0.02)
        #expect(model.effectiveTrainingSettings.optimizer == .sgd)
        #expect(model.canTrain)
        model.continueModelVersion(oldest, source: .checkpoint)
        #expect(model.trainingStart == .checkpoint(runID: oldest.id))
        #expect(model.selectedCheckpoint?.directory == oldest.directory)
        #expect(model.effectiveTrainingSettings.modelWidth == oldest.checkpoint?.modelWidth)
        model.setTrainingStartMode(.fresh)
        #expect(model.trainingStart == .fresh)
        #expect(model.effectiveTrainingSettings.modelWidth == 164)
        #expect(model.effectiveTrainingSettings.imageSize == 2048)
        #expect(model.selectedModelVersions.count == 6)
        model.selectedMap = .normal
        #expect(model.selectedModelVersions.isEmpty)
        #expect(model.exportedModelVersions.isEmpty)
    }

    @Test @MainActor
    func missingSelectedWeightsBlockContinuationWithoutChoosingAnotherRun() async throws {
        let fixture = try await VersionControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, source: fixture.source)
        model.selectedDatasetID = fixture.datasetID
        let oldest = try #require(model.selectedModelVersions.last)
        model.settings.training.imageSize = 256
        model.continueModelVersion(oldest, source: .checkpoint)
        #expect(model.canTrain)
        try FileManager.default.removeItem(at: oldest.directory)
        await model.pollFolder()
        #expect(model.selectedTrainingVersion == nil)
        #expect(model.selectedCheckpoint == nil)
        #expect(model.trainingStart == nil)
        #expect(!model.canTrain)
        #expect(model.trainingOptionsIssue?.contains("no longer available") == true)
        #expect(model.selectedModelVersions.count == 5)
        model.selectedMap = .normal
        #expect(model.selectedTrainingVersionID.isEmpty)
        model.selectedMap = .depth
        #expect(model.canTrain)
        #expect(model.trainingStart != nil)
    }

    @Test @MainActor
    func comparisonOptionsUseDifferentSavedVersionsAndExposeWholeMaterialMembership() async throws {
        let fixture = try await VersionControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, source: fixture.source)
        model.selectedDatasetID = fixture.datasetID
        await model.loadComparisonPairs()
        #expect(model.comparisonIssue == nil)
        #expect(model.comparisonPairs.count == 2)
        #expect(model.canCompareModels)
        let heldOut = try #require(model.comparisonPairs.first { $0.seenByFirst == .heldOut && $0.seenBySecond == .heldOut })
        let used = try #require(model.comparisonPairs.first { $0.seenByFirst == .training && $0.seenBySecond == .training })
        #expect(heldOut.id != used.id)
        model.comparisonPairID = heldOut.id
        #expect(model.selectedComparisonPair?.id == heldOut.id)
        let oldest = try #require(model.selectedModelVersions.last)
        model.comparisonFirstVersionID = oldest.id
        #expect(!model.canCompareModels)
        await model.loadComparisonPairs()
        #expect(model.canCompareModels)
        #expect(model.firstComparisonVersion?.id == oldest.id)
        #expect(model.secondComparisonVersion?.id != oldest.id)
        let revision = model.comparisonSelectionKey
        let input = try #require(model.selectedComparisonPair?.inputURL)
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(5)], ofItemAtPath: input.path)
        await model.pollFolder()
        #expect(model.comparisonSelectionKey != revision)
        #expect(!model.canCompareModels)
        await model.loadComparisonPairs()
        #expect(model.canCompareModels)
        try await render(ModelComparisonView(model: model), size: NSSize(width: 1140, height: 850), name: "model-comparison-settings.png")
        model.selectedMap = .normal
        #expect(model.comparisonPairs.isEmpty)
        #expect(!model.canCompareModels)
    }

    @Test
    func gradientMeasurementsRemainVisibleAfterCheckpointAndAllClippingSettingsAreShown() {
        let metrics = TrainingRunMetrics(events: [
            .init(type: "progress", message: "Training", gradientNorm: 2.25, gradientClipThreshold: 0.4),
            .init(type: "checkpoint", message: "Saved")
        ])
        #expect(metrics.gradientNorm == 2.25)
        #expect(metrics.gradientClipThreshold == 0.4)
        var settings = TrainingSettings()
        settings.gradientClipPolicy = .warmup
        settings.gradientClipWarmupStartNorm = 0.2
        settings.gradientClipWarmupUpdates = 40
        let values = TrainingSettingValue.values(settings)
        #expect(values.first { $0.id == "gradientClipPolicy" }?.value == TrainingGradientClipping.warmup.displayName)
        #expect(values.first { $0.id == "gradientClipWarmupStartNorm" }?.value == "0.2")
        #expect(values.first { $0.id == "gradientClipWarmupUpdates" }?.value == "40")
        #expect(values.first { $0.id == "exportBestWeights" }?.label == "Export lowest validation loss")
    }

    @Test @MainActor
    func realComparisonRendersBothOutputsAndKeepsManualExportChoice() async throws {
        let fixture = try await VersionControlsFixture(realModels: true)
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, source: fixture.source)
        model.selectedDatasetID = fixture.datasetID
        let activeID = try #require(model.selectedModelVersions.first(where: \.isActive)?.id)
        let originals = try #require(model.snapshot).images.map { ($0.url, try Data(contentsOf: $0.url)) }
        await model.loadComparisonPairs()
        #expect(model.canCompareModels)
        model.compareModels()
        try await wait(model)
        let result = try #require(model.comparisonResult)
        #expect(result.cropSize == 32)
        #expect(result.firstMAE > result.secondMAE)
        #expect(model.selectedModelVersions.first(where: \.isActive)?.id == activeID)
        for (url, original) in originals { #expect(try Data(contentsOf: url) == original) }
        #expect(FileManager.default.fileExists(atPath: result.firstOutputURL.path))
        #expect(FileManager.default.fileExists(atPath: result.secondOutputURL.path))
        try await render(ModelComparisonView(model: model), size: NSSize(width: 1140, height: 1450), name: "model-comparison-results.png")
    }

    @MainActor private func load(_ model: FactoryModel, source: URL) async throws {
        model.openFolder(source)
        try await wait(model)
        _ = try #require(model.snapshot)
    }

    @MainActor private func wait(_ model: FactoryModel) async throws {
        for _ in 0..<1500 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!model.busy)
        #expect(model.errorMessage == nil)
    }

    @MainActor private func render<Content: View>(_ content: Content, size: NSSize, name: String) async throws {
        _ = NSApplication.shared
        let view = NSHostingView(rootView: content.frame(width: size.width))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        view.frame = NSRect(origin: .zero, size: size)
        window.orderFrontRegardless()
        try await Task.sleep(for: .milliseconds(200))
        view.layoutSubtreeIfNeeded()
        let bounds = NSRect(origin: .zero, size: NSSize(width: size.width, height: max(size.height, view.fittingSize.height)))
        window.setContentSize(bounds.size)
        view.frame = bounds
        view.layoutSubtreeIfNeeded()
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1000)
        if let path = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let directory = URL(fileURLWithPath: path, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try png.write(to: directory.appendingPathComponent(name))
        }
        window.close()
    }
}

private struct VersionControlsFixture {
    let source: URL
    let datasetID: String

    init(realModels: Bool = false) async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("loom-model-controls-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        source = folder.resolvingSymlinksInPath().standardizedFileURL
        for material in ["material_a", "material_b"] {
            let directory = source.appendingPathComponent(material, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            for (suffix, role) in [("diff", MapRole.diffuse), ("disp", MapRole.depth)] {
                let url = directory.appendingPathComponent("\(material)_\(suffix)_256.png")
                let pixels = Data(repeating: 140, count: 256 * 256 * 4)
                let provider = try #require(CGDataProvider(data: pixels as CFData))
                let image = try #require(CGImage(width: 256, height: 256, bitsPerComponent: 8, bitsPerPixel: 32,
                                                bytesPerRow: 256 * 4, space: CGColorSpaceCreateDeviceRGB(),
                                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
                let writer = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
                CGImageDestinationAddImage(writer, image, nil)
                #expect(CGImageDestinationFinalize(writer))
                try ImageMetadataStore.write(.init(id: UUID().uuidString, pairingKey: material, role: role,
                                                   category: "Bricks", verified: false, note: ""), to: url)
            }
        }
        let project = try LoomProject(sourceFolder: source)
        let snapshot = try await project.snapshot()
        datasetID = try #require(snapshot.datasets.first?.id)
        for index in 0..<(realModels ? 2 : 6) {
            let runID = "version-\(index)"
            let directory = project.projectFolder.appendingPathComponent("runs/\(runID)", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let width = realModels ? 8 : 24 + index * 8
            let blocks = realModels ? 1 : 2
            let imageSize = realModels ? 32 : 256
            let nativeSettings: [String: Any] = ["image_size": imageSize, "exaggerate_features": false,
                                               "optimizer": "adamw", "learning_rate": 0.001 / Double(index + 1)]
            let architecture: [String: Any] = ["id": "loom-factory-material-residual-v1", "width": width, "blocks": blocks]
            let descriptor: [String: Any] = [
                "schema": "loom-factory-model-v1", "project_id": snapshot.projectID, "kind": "map",
                "dataset_id": datasetID, "dataset_name": "Bricks", "map": "depth", "status": "completed",
                "architecture": "loom-factory-material-residual-v1", "architecture_config": architecture,
                "image_size": imageSize, "weights_path": "weights.safetensors", "coreml_path": "model.mlmodel",
                "training": nativeSettings
            ]
            try Self.write(descriptor, to: directory.appendingPathComponent("model.json"))
            let checkpoint: [String: Any] = [
                "schema": "loom-factory-checkpoint-v1", "project_id": snapshot.projectID, "kind": "map",
                "dataset_id": datasetID, "dataset_name": "Bricks", "map": "depth", "status": "completed",
                "architecture": "loom-factory-material-residual-v1", "architecture_config": architecture,
                "settings": nativeSettings, "epochs_completed": 4
            ]
            try Self.write(checkpoint, to: directory.appendingPathComponent("checkpoint.json"))
            let samples: [[String: Any]] = snapshot.images.filter { $0.role == .diffuse }.map { image in
                ["input": image.url.path, "split": image.url.deletingLastPathComponent().lastPathComponent == "material_a" ? "training" : "validation"]
            }
            var request = nativeSettings
            request["schema"] = "loom-factory-training-v1"
            request["project_id"] = snapshot.projectID; request["dataset_id"] = datasetID
            request["width"] = width; request["blocks"] = blocks; request["run_label"] = "Experiment \(index + 1)"
            request["samples"] = samples
            try Self.write(request, to: directory.appendingPathComponent("request.json"))
            let artifact = ModelArtifact(id: runID, kind: "map", datasetID: datasetID, datasetName: "Bricks", map: .depth,
                                         directory: directory, createdAt: Date(timeIntervalSince1970: Double(1_700_000_000 + index)),
                                         epochs: 4, loss: 0.7 - Double(index) * 0.1,
                                         coreMLPath: "model.mlmodel", weightsPath: "weights.safetensors",
                                         architecture: "loom-factory-material-residual-v1")
            try JSONEncoder().encode(artifact).write(to: directory.appendingPathComponent("artifact.json"))
            if realModels {
                let nativeArchitecture = try NativeArchitecture(mapKind: "depth", width: width, blocks: blocks)
                var weights = Dictionary(uniqueKeysWithValues: nativeArchitecture.parameterShapes.map { name, shape in
                    (name, NativeTensor(shape: shape, values: [Float](repeating: 0, count: shape.reduce(1, *))))
                })
                let value: Float = index == 0 ? 0.75 : 0.25
                weights["output.bias"]!.values[0] = log(value / (1 - value))
                let compiled = try NativeCoreML.export(weights: weights, architecture: nativeArchitecture, imageSize: imageSize,
                                                       to: directory.appendingPathComponent("model.mlmodel"))
                try FileManager.default.removeItem(at: compiled)
                try NativeWeights.write(weights, to: directory.appendingPathComponent("weights.safetensors"))
            } else {
                // Opaque metadata fixtures for saved-version controls; inference uses only realModels above.
                for name in ["weights.safetensors", "model.mlmodel"] {
                    try Data([0]).write(to: directory.appendingPathComponent(name))
                }
            }
            for name in ["checkpoint.safetensors"] {
                try Data([0]).write(to: directory.appendingPathComponent(name))
            }
        }
        let exports = project.projectFolder.appendingPathComponent("exports", isDirectory: true)
        try FileManager.default.createDirectory(at: exports, withIntermediateDirectories: true)
        let archive = exports.appendingPathComponent("previous.textureloom")
        try Data([0]).write(to: archive)
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(-60)], ofItemAtPath: archive.path)
    }

    private static func write(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: url)
    }
    func remove() { try? FileManager.default.removeItem(at: source) }
}
