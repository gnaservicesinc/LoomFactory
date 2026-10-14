import AppKit
import ImageIO
import LoomCore
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

@Suite(.serialized)
struct TrainingPanelTests {
    @Test @MainActor
    func allActiveTrainingControlsRenderWithARealValidationPlan() async throws {
        let fixture = try TrainingPanelFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, source: fixture.source)
        model.selectedDatasetID = try #require(model.snapshot?.datasets.first?.id)
        configureFullTrainingOptions(model)
        #expect(model.trainingOptionsIssue == nil)
        #expect(model.canTrain)
        let plan = try #require(model.trainingPlan)
        #expect(plan.trainingMaterialCount == 1)
        #expect(plan.validationMaterialCount == 1)
        #expect(plan.trainingPairCount == 1)
        #expect(plan.validationPairCount == 1)
        #expect(model.effectiveTrainingSettings.parameterCount(map: .depth) > 500_000)
        #expect(model.settings.training.effectiveBatchSize == 6)
        try await render(model, name: "training-controls-depth.png")
        model.selectedMap = .normal
        model.settings.training.optimizer = .sgd
        model.settings.training.momentum = 0.9
        #expect(model.trainingOptionsIssue == nil)
        try await render(model, name: "training-controls-normal-sgd.png")
    }

    @Test @MainActor
    func taggedPairsAreReadyWithoutManualReview() async throws {
        let fixture = try TrainingPanelFixture(reviewed: false)
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, source: fixture.source)
        model.selectedDatasetID = try #require(model.snapshot?.datasets.first?.id)
        model.settings.training.imageSize = 256
        #expect(model.snapshot?.images.allSatisfy { !$0.verified } == true)
        #expect(model.eligiblePairCount == 2)
        #expect(model.canTrain)
        #expect(model.trainingOptionsIssue == nil)
        let plan = try #require(model.trainingPlan)
        #expect(plan.trainingPairCount == 1)
        #expect(plan.validationPairCount == 1)
        try await render(model, name: "training-controls-unreviewed-tagged-pairs.png")
    }

    @Test @MainActor
    func turningOffValidationKeepsEarlyStoppingRecoverable() async throws {
        let fixture = try TrainingPanelFixture(reviewed: false)
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, source: fixture.source)
        model.selectedDatasetID = try #require(model.snapshot?.datasets.first?.id)
        model.settings.training.imageSize = 256
        model.settings.training.validationFraction = 0.5
        model.settings.training.earlyStoppingPatience = 2
        #expect(model.canTrain)
        model.settings.training.validationFraction = 0
        #expect(model.settings.training.earlyStoppingPatience == 2)
        #expect(!model.canTrain)
        #expect(model.trainingOptionsIssue == "Enable held-out validation to use early stopping.")
        try await render(model, name: "training-controls-recover-early-stopping.png")
        model.settings.training.earlyStoppingPatience = 0
        #expect(model.trainingOptionsIssue == nil)
        #expect(model.canTrain)
        #expect(model.trainingPlan?.trainingPairCount == 2)
        #expect(model.trainingPlan?.validationPairCount == 0)
    }

    @Test @MainActor
    func editedTrainingOptionsRestoreWithTheSourceFolder() async throws {
        let fixture = try TrainingPanelFixture()
        defer { fixture.remove() }
        let edited = FactoryModel(rememberSourceFolder: false)
        try await load(edited, source: fixture.source)
        configureFullTrainingOptions(edited)
        let expected = edited.settings.training
        #expect(await edited.prepareForTermination())
        let restored = FactoryModel(rememberSourceFolder: false)
        try await load(restored, source: fixture.source)
        #expect(restored.settings.training == expected)
        #expect(restored.settings.training.optimizer == .lion)
        #expect(restored.settings.training.gradientAccumulationSteps == 3)
        #expect(restored.settings.training.cropRefresh == .perRun)
        #expect(restored.settings.training.augmentationRefresh == .perEpoch)
        #expect(restored.settings.training.learningRateSchedule == .cosine)
        #expect(restored.settings.training.exportBestWeights)
    }

    @Test @MainActor
    func continuingCancelledWeightsShowsSavedArchitectureAndKeepsNewOptimizerOptions() async throws {
        let fixture = try TrainingPanelFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, source: fixture.source)
        let snapshot = try #require(model.snapshot)
        let dataset = try #require(snapshot.datasets.first)
        model.selectedDatasetID = dataset.id
        let directory = fixture.source.appendingPathComponent(".loom-factory/runs/cancelled-depth", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let checkpoint: [String: Any] = [
            "schema": "loom-factory-checkpoint-v1", "project_id": snapshot.projectID,
            "kind": "map", "dataset_id": dataset.id, "map": "depth",
            "architecture": "loom-factory-material-residual-v1",
            "architecture_config": ["id": "loom-factory-material-residual-v1", "width": 24, "blocks": 2],
            "settings": ["image_size": 256, "exaggerate_features": true],
            "epochs_completed": 3, "status": "cancelled"
        ]
        try JSONSerialization.data(withJSONObject: checkpoint).write(to: directory.appendingPathComponent("checkpoint.json"))
        try Data([0]).write(to: directory.appendingPathComponent("checkpoint.safetensors"))
        await model.pollFolder()
        model.settings.training.imageSize = 2048
        model.settings.training.modelWidth = 164
        model.settings.training.modelBlocks = 4
        model.settings.training.exaggerateFeatures = false
        model.settings.training.optimizer = .sgd
        model.settings.training.learningRate = 0.03
        model.settings.training.gradientAccumulationSteps = 4
        model.settings.training.continueLatestWeights = true
        #expect(model.canContinueMap)
        #expect(model.canTrain)
        #expect(model.effectiveTrainingSettings.imageSize == 256)
        #expect(model.effectiveTrainingSettings.modelWidth == 24)
        #expect(model.effectiveTrainingSettings.modelBlocks == 2)
        #expect(model.effectiveTrainingSettings.exaggerateFeatures)
        #expect(model.effectiveTrainingSettings.optimizer == .sgd)
        #expect(model.effectiveTrainingSettings.learningRate == 0.03)
        #expect(model.effectiveTrainingSettings.gradientAccumulationSteps == 4)
        #expect(model.selectedCheckpoint?.epochsCompleted == 3)
        try await render(model, name: "training-controls-continue-cancelled.png")
    }

    @Test
    func progressMeasurementsSurviveStatusAndCheckpointEvents() {
        let metrics = TrainingRunMetrics(events: [
            .init(type: "started", message: "Starting", parameterCount: 511_813, effectiveBatchSize: 6),
            .init(type: "progress", message: "Training", epoch: 2, loss: 0.2, learningRate: 0.0003, update: 8, totalUpdates: 20),
            .init(type: "epoch_completed", message: "Validated", epoch: 2, loss: 0.15, validationLoss: 0.18),
            .init(type: "checkpoint", message: "Weights saved"),
            .init(type: "status", message: "Exporting")
        ])
        #expect(metrics.epoch == 2)
        #expect(metrics.trainingLoss == 0.15)
        #expect(metrics.validationLoss == 0.18)
        #expect(metrics.learningRate == 0.0003)
        #expect(metrics.update == 8)
        #expect(metrics.totalUpdates == 20)
        #expect(metrics.parameterCount == 511_813)
        #expect(metrics.effectiveBatchSize == 6)
    }

    @MainActor private func configureFullTrainingOptions(_ model: FactoryModel) {
        model.settings.training.imageSize = 256
        model.settings.training.epochs = 4
        model.settings.training.modelWidth = 84
        model.settings.training.modelBlocks = 4
        model.settings.training.optimizer = .lion
        model.settings.training.learningRate = 0.0001
        model.settings.training.beta2 = 0.99
        model.settings.training.batchSize = 2
        model.settings.training.gradientAccumulationSteps = 3
        model.settings.training.gradientClipNorm = 0.75
        model.settings.training.cropRefresh = .perRun
        model.settings.training.augmentationRefresh = .perEpoch
        model.settings.training.flipCrops = true
        model.settings.training.rotateCrops = true
        model.settings.training.shuffleEachEpoch = false
        model.settings.training.exaggerateFeatures = true
        model.settings.training.learningRateSchedule = .cosine
        model.settings.training.warmupEpochs = 1
        model.settings.training.minimumLearningRateRatio = 0.1
        model.settings.training.validationFraction = 0.5
        model.settings.training.validationSeed = 17
        model.settings.training.earlyStoppingPatience = 2
        model.settings.training.exportBestWeights = true
        model.settings.training.checkpointEveryEpochs = 2
        model.settings.training.device = .cpu
    }

    @MainActor private func load(_ model: FactoryModel, source: URL) async throws {
        model.openFolder(source)
        for _ in 0..<100 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.errorMessage == nil)
        _ = try #require(model.snapshot)
        #expect(!model.busy)
    }

    @MainActor private func render(_ model: FactoryModel, name: String) async throws {
        _ = NSApplication.shared
        let size = NSSize(width: 740, height: 2750)
        let view = NSHostingView(rootView: TrainingPanel(model: model, showTuning: true).frame(width: size.width))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        view.frame = NSRect(origin: .zero, size: size)
        window.orderFrontRegardless()
        try await Task.sleep(for: .milliseconds(200))
        view.layoutSubtreeIfNeeded()
        let fittedHeight = max(1, view.fittingSize.height)
        #expect(fittedHeight > 1000)
        window.setContentSize(NSSize(width: size.width, height: fittedHeight))
        view.frame = NSRect(x: 0, y: 0, width: size.width, height: fittedHeight)
        view.layoutSubtreeIfNeeded()
        guard let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
            Issue.record("Training controls could not be rendered")
            window.close()
            return
        }
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1000)
        if let destination = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let directory = URL(fileURLWithPath: destination, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try png.write(to: directory.appendingPathComponent(name))
        }
        window.close()
    }
}

private struct TrainingPanelFixture {
    let source: URL
    init(reviewed: Bool = true) throws {
        source = FileManager.default.temporaryDirectory.appendingPathComponent("loom-training-ui-\(UUID().uuidString)", isDirectory: true)
        for material in ["brick_wall_a", "brick_wall_b"] {
            let folder = source.appendingPathComponent(material, isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for (suffix, role) in [("diff", MapRole.diffuse), ("disp", MapRole.depth), ("nor_gl", MapRole.normal)] {
                let url = folder.appendingPathComponent("\(material)_\(suffix)_1k.png")
                let width = 256
                let pixels = Data(repeating: role == .normal ? 127 : 160, count: width * width * 4)
                let provider = try #require(CGDataProvider(data: pixels as CFData))
                let image = try #require(CGImage(width: width, height: width, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
                let writer = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
                CGImageDestinationAddImage(writer, image, nil)
                #expect(CGImageDestinationFinalize(writer))
                try ImageMetadataStore.write(.init(id: UUID().uuidString, pairingKey: material, role: role, category: "Bricks", verified: reviewed, note: "",
                                                   normalConvention: role == .normal ? "opengl" : nil), to: url)
            }
        }
    }
    func remove() { try? FileManager.default.removeItem(at: source) }
}
