import AppKit
import CryptoKit
import ImageIO
import LoomCore
import LoomNative
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

@Suite(.serialized)
struct ModelTestControlsTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["LOOM_CHECKPOINT_SWEEP_UI"] == "1")) @MainActor
    func completedProductionRunReopensSavedProgressionWithoutInference() async throws {
        let environment = ProcessInfo.processInfo.environment
        let source = try #require(environment["LOOM_CHECKPOINT_SWEEP_SOURCE"])
        let run = try #require(environment["LOOM_CHECKPOINT_SWEEP_RUN"])
        let input = try #require(environment["LOOM_CHECKPOINT_SWEEP_INPUT"])
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(URL(fileURLWithPath: source))
        for _ in 0..<1000 { if !model.busy { break }; try await Task.sleep(for: .milliseconds(20)) }
        #expect(model.errorMessage == nil)
        let version = try #require(model.snapshot?.modelVersions.first { $0.id == run })
        model.showModelTesting(version)
        model.setModelTestInput(URL(fileURLWithPath: input))
        for _ in 0..<500 { if model.checkpointProgression != nil { break }; try await Task.sleep(for: .milliseconds(20)) }
        let progression = try #require(model.checkpointProgression)
        #expect(progression.results.count == version.checkpointVersions.count)
        #expect(progression.results.count > 1)
        try await render(CheckpointProgressionView(progression: progression, busy: false,
            viewOutput: model.viewCheckpointOutput, reveal: { _ in }, version: version).preferredColorScheme(.dark),
            size: NSSize(width: 1400, height: 2500), name: "rock-cliff-marble-checkpoint-progression.png")
        try await render(ModelTestView(model: model).preferredColorScheme(.dark),
            size: NSSize(width: 1400, height: 2200), name: "production-run-identity.png")
        model.prepareModelComparison(from: version, inputURL: URL(fileURLWithPath: input))
        await model.loadComparisonPairs()
        #expect(model.firstComparisonVersion?.id == run)
        #expect(model.selectedComparisonPair?.inputURL == URL(fileURLWithPath: input))
        try await render(ModelComparisonView(model: model).preferredColorScheme(.dark),
            size: NSSize(width: 1200, height: 1100), name: "production-two-run-comparison.png")
        model.selectedDatasetID = version.datasetID
        try await render(HStack(alignment: .top, spacing: 20) {
            ModelVersionRow(model: model, version: version).frame(width: 820)
            TrainingOverviewPanel(model: model).frame(width: 320)
        }.preferredColorScheme(.dark), size: NSSize(width: 1200, height: 820),
            name: "saved-run-versus-next-settings.png")
    }

    @Test @MainActor
    func completedRunCheckpointSweepRunsFromExportedSelectionAndShowsChronologicalComparison() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let run = fixture.source.appendingPathComponent(".loom-factory/runs/one-saved-run")
        let latest = run.appendingPathComponent("checkpoint.json")
        var descriptor = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: latest)) as? [String: Any])
        let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 1)
        for epoch in [10, 20] {
            let weights = architecture.parameterShapes.mapValues { shape in
                NativeTensor(shape: shape, values: [Float](repeating: Float(epoch) / 10000, count: shape.reduce(1, *)))
            }
            let path = run.appendingPathComponent("checkpoint-epoch-\(epoch).safetensors")
            try NativeWeights.write(weights, to: path)
            descriptor["checkpoint_id"] = "epoch-\(epoch)"
            descriptor["epochs_completed"] = epoch; descriptor["updates"] = epoch * 10
            descriptor["weights_path"] = path.lastPathComponent
            descriptor["weights_sha256"] = SHA256.hash(data: try Data(contentsOf: path)).map { String(format: "%02x", $0) }.joined()
            let data = try JSONSerialization.data(withJSONObject: descriptor)
            try data.write(to: run.appendingPathComponent("checkpoint-epoch-\(epoch).json"))
            try data.write(to: latest)
        }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        let version = try #require(model.testableModelVersions.first)
        #expect(version.status == "completed")
        model.showModelTesting(version)
        model.setModelTestInput(fixture.externalInput)
        #expect(model.effectiveTestWeightSource == .model)
        #expect(model.canTestAllCheckpoints)
        try await render(ModelTestView(model: model), size: NSSize(width: 1140, height: 1050), name: "completed-checkpoint-controls.png")
        model.testAllCheckpoints()
        try await wait(model)
        let progression = try #require(model.checkpointProgression)
        #expect(progression.results.count == 2)
        #expect(progression.outputs.map { $0.checkpoint.epochsCompleted } == [10, 20])
        try await render(CheckpointProgressionView(progression: progression, busy: false,
            viewOutput: model.viewCheckpointOutput, reveal: { _ in }),
            size: NSSize(width: 1140, height: 1100), name: "completed-checkpoint-progression.png")
        model.viewCheckpointOutput(try #require(progression.results.first))
        #expect(model.effectiveTestWeightSource == .checkpoint)
        #expect(model.testCheckpointID == "epoch-10")
        model.viewCheckpointOutput(try #require(progression.results.last))
        #expect(model.modelTestResult?.checkpointID == "epoch-20")
        #expect(model.previousTestResult?.checkpointID == "epoch-10")
        model.testWeightSource = .model
        model.setModelTestCheckpoint("epoch-10")
        #expect(model.effectiveTestWeightSource == .checkpoint)
        #expect(model.canTestModel)
        model.testModel()
        try await wait(model)
        #expect(model.modelTestResult?.checkpointID == "epoch-10")
        #expect(model.modelTestResult?.epochsCompleted == 10)
        #expect(model.snapshot?.modelVersions.count == 1)
        model.setModelTestInput(fixture.sourceInput)
        #expect(model.checkpointProgression == nil)
    }

    @Test @MainActor
    func boundaryPolicyComparisonRetainsTheSameSavedCheckpointAndInput() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        model.showModelTesting(try #require(model.testableModelVersions.first))
        model.setModelTestInput(fixture.externalInput)
        model.testBoundaryHandling = .legacy
        model.recordModelTestResult(fixture.result)
        model.testBoundaryHandling = .context
        #expect(model.modelTestResult == nil)
        #expect(model.previousTestResult?.boundaryHandling == ModelTestBoundaryHandling.legacy.rawValue)
        let r = fixture.result
        model.recordModelTestResult(ModelTestResult(directory: r.directory, inputURL: r.inputURL, outputURL: r.outputURL,
            runID: r.runID, source: r.source, map: r.map, width: r.width, height: r.height,
            tileSize: r.tileSize, tileCount: r.tileCount, note: "Context policy fixture", previewURL: r.previewURL,
            boundaryHandling: ModelTestBoundaryHandling.context.rawValue))
        #expect(model.previousTestResult != nil)
        #expect(model.modelTestResult?.boundaryHandling == ModelTestBoundaryHandling.context.rawValue)
        #expect(model.blenderTestMaps.count == 1)
        try await render(ModelTestView(model: model), size: NSSize(width: 1140, height: 1700), name: "model-test-context-policy.png")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["LOOM_TEST_BLENDER_EXPORT"] == "1"))
    @MainActor func backgroundBlenderProcessExportsExactCheckpointWithoutInference() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let blender = try #require(BlenderExportProcess.installedBlender)
        let directory = fixture.root.appendingPathComponent("real-blender-export")
        let manifest = try BlenderMaterialExport.prepare([
            .init(result: fixture.result, modelName: "Checkpoint branch · epoch 17")
        ], sourceImages: [], includeSource: true, directory: directory)
        let process = BlenderExportProcess()
        let blend = try await Task.detached { try process.run(blender: blender, manifest: manifest) }.value
        #expect(FileManager.default.fileExists(atPath: blend.path))
        let receipt = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("export-receipt.json"))) as? [String: Any])
        #expect((receipt["scenes"] as? [String])?.count == 9)
        #expect(receipt["rendered"] as? Bool == false)
        let stopped = BlenderExportProcess()
        stopped.cancel()
        #expect(throws: CancellationError.self) { try stopped.run(blender: blender, manifest: manifest) }
    }

    @Test @MainActor
    func blenderExportRetainsExactTestedMapsAcrossMapChoicesAndClearsOnNewInput() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        model.showModelTesting(try #require(model.testableModelVersions.first))
        model.setModelTestInput(fixture.externalInput)
        let r = fixture.result
        model.recordModelTestResult(r)
        #expect(model.blenderTestMaps.count == 1)
        #expect(model.blenderTestMaps[0].result.source == .checkpoint)
        model.setModelTestMap(.normal)
        #expect(model.modelTestResult == nil)
        #expect(model.blenderTestMaps[0].result.runID == r.runID)
        let normal = ModelTestResult(directory: r.directory, inputURL: r.inputURL, outputURL: r.outputURL,
            runID: "another-normal-model", source: .model, map: .normal, width: r.width, height: r.height,
            tileSize: r.tileSize, tileCount: 1, note: "CPU selection fixture")
        model.recordModelTestResult(normal)
        #expect(model.blenderTestMaps.count == 2)
        #expect(model.blenderTestMaps.last?.result.source == .model)
        #expect(model.blenderTestMaps.first?.modelName.contains("one-save") == true)
        try await render(BlenderExportOptions(maps: model.blenderTestMaps) { _, _ in },
            size: NSSize(width: 680, height: 500), name: "blender-export-options.png")
        model.setModelTestInput(fixture.sourceInput)
        #expect(model.blenderTestMaps.isEmpty)
    }

    @Test @MainActor
    func defaultTestVersionRefreshesOriginalMapWhenTrainingMapChangesWithoutAnExplicitRun() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let depthTarget = fixture.source.appendingPathComponent("brick_depth.exr")
        let normalTarget = fixture.source.appendingPathComponent("brick_normal.png")
        try FileManager.default.copyItem(at: fixture.output, to: depthTarget)
        try FileManager.default.copyItem(at: fixture.sourceInput, to: normalTarget)
        try ImageMetadataStore.write(.init(pairingKey: "brick", role: .depth, category: "Bricks"), to: depthTarget)
        try ImageMetadataStore.write(.init(pairingKey: "brick", role: .normal, category: "Bricks"), to: normalTarget)

        let runs = fixture.source.appendingPathComponent(".loom-factory/runs")
        let depthRun = runs.appendingPathComponent("one-saved-run")
        let normalRun = runs.appendingPathComponent("one-saved-normal-run")
        try FileManager.default.copyItem(at: depthRun, to: normalRun)
        for name in ["model.json", "checkpoint.json"] {
            let file = normalRun.appendingPathComponent(name)
            let data = try Data(contentsOf: file)
            let object = try JSONSerialization.jsonObject(with: data)
            var descriptor = try #require(object as? [String: Any])
            descriptor["map"] = "normal"
            let settingsKey = name == "checkpoint.json" ? "settings" : "training"
            var savedSettings = try #require(descriptor[settingsKey] as? [String: Any])
            savedSettings["optimizer"] = "lion"
            savedSettings["learning_rate"] = 0.00037
            savedSettings["crop_refresh"] = "per_run"
            savedSettings["seed"] = 734
            descriptor[settingsKey] = savedSettings
            try JSONSerialization.data(withJSONObject: descriptor).write(to: file)
        }
        let artifactData = try Data(contentsOf: depthRun.appendingPathComponent("artifact.json"))
        let depthArtifact = try JSONDecoder().decode(ModelArtifact.self, from: artifactData)
        let normalArtifact = ModelArtifact(id: normalRun.lastPathComponent, kind: "map",
            datasetID: depthArtifact.datasetID, datasetName: depthArtifact.datasetName, map: .normal,
            directory: normalRun, createdAt: depthArtifact.createdAt, epochs: depthArtifact.epochs,
            loss: nil, coreMLPath: depthArtifact.coreMLPath, weightsPath: depthArtifact.weightsPath,
            architecture: depthArtifact.architecture)
        try JSONEncoder().encode(normalArtifact).write(to: normalRun.appendingPathComponent("artifact.json"))

        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        model.selectedDatasetID = try #require(depthArtifact.datasetID)
        model.setTrainingStartMode(.checkpoint)
        #expect(model.selectedTrainingVersionID.isEmpty)
        // Workspace tab navigation leaves the saved-run selection at its default.
        model.workspaceSection = .testing
        #expect(model.selectedTestVersionID.isEmpty)
        #expect(model.selectedTestVersion?.map == .depth)
        model.setModelTestInput(fixture.sourceInput)
        for _ in 0..<250 {
            if model.testOriginalMap.image?.url == depthTarget { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.testOriginalMap.image?.url == depthTarget)

        model.workspaceSection = .training
        model.selectedMap = .normal
        #expect(model.selectedTrainingVersionID.isEmpty)
        #expect(model.selectedTrainingVersion?.id == normalArtifact.id)
        var expectedSettings = try #require(model.selectedTrainingVersion?.savedSettings)
        #expect(expectedSettings.optimizer == .lion)
        #expect(expectedSettings.learningRate == 0.00037)
        #expect(expectedSettings.cropRefresh == .perRun)
        #expect(expectedSettings.seed == 734)
        expectedSettings.continueLatestWeights = true
        #expect(model.settings.training == expectedSettings)
        model.workspaceSection = .testing
        #expect(model.selectedTestVersionID.isEmpty)
        #expect(model.selectedTestVersion?.map == .normal)
        #expect(model.testInputURL == fixture.sourceInput)
        for _ in 0..<250 {
            if model.testOriginalMap.image?.url == normalTarget { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.testOriginalMap.image?.url == normalTarget)
        #expect(model.testOriginalMap.image?.role == .normal)
    }

    @Test @MainActor
    func sourceTestingFiltersCategoryAndIgnoreStateAndDisplaysOriginalPairedMap() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let target = fixture.source.appendingPathComponent("brick_depth.exr")
        try FileManager.default.copyItem(at: fixture.output, to: target)
        try ImageMetadataStore.write(.init(pairingKey: "brick", role: .depth, category: "Bricks"), to: target)
        let other = fixture.source.appendingPathComponent("metal_diffuse.png")
        try FileManager.default.copyItem(at: fixture.externalInput, to: other)
        try ImageMetadataStore.write(.init(pairingKey: "metal", role: .diffuse, category: "Metal", ignored: true), to: other)
        let originalTarget = try Data(contentsOf: target)
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        let version = try #require(model.testableModelVersions.first)
        model.showModelTesting(version)
        model.setModelTestInput(fixture.sourceInput)
        #expect(!model.testDiffuseImages.contains { $0.url == other })
        #expect(model.testDiffuseImages.allSatisfy { $0.category == "Bricks" && !$0.ignored })
        #expect(model.canTestModel)
        let paired = ModelTestOriginalMap.resolve(inputURL: model.testInputURL, map: version.map,
                                                 images: model.snapshot?.images ?? [])
        #expect(paired.image?.url == target)
        #expect(paired.issue == nil)
        try await render(ModelTestView(model: model), size: NSSize(width: 1140, height: 930), name: "model-test-original-target-preview.png")
        try await render(ModelTestResultView(result: fixture.result, imageZoom: .constant(.fit), saveOutput: {},
            showOutput: {}, showFolder: {}, originalMap: paired), size: NSSize(width: 1140, height: 780), name: "model-test-original-target-result.png")
        #expect(try Data(contentsOf: target) == originalTarget)
    }

    @Test @MainActor
    func categoryMapAndCheckpointControlsStayScopedAndRetainPreviousCheckpointForAB() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let run = fixture.source.appendingPathComponent(".loom-factory/runs/one-saved-run")
        let latest = run.appendingPathComponent("checkpoint.json")
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: latest))
        var descriptor = try #require(object as? [String: Any])
        descriptor["checkpoint_id"] = "epoch-10"; descriptor["epochs_completed"] = 10; descriptor["updates"] = 100
        try JSONSerialization.data(withJSONObject: descriptor).write(to: run.appendingPathComponent("checkpoint-epoch-10.json"))
        descriptor["checkpoint_id"] = "epoch-20"; descriptor["epochs_completed"] = 20; descriptor["updates"] = 200
        try JSONSerialization.data(withJSONObject: descriptor).write(to: latest)
        try JSONSerialization.data(withJSONObject: descriptor).write(to: run.appendingPathComponent("checkpoint-epoch-20.json"))
        let metal = fixture.source.appendingPathComponent("metal_diff.png")
        try FileManager.default.copyItem(at: fixture.sourceInput, to: metal)
        try ImageMetadataStore.write(.init(pairingKey: "metal", role: .diffuse, category: "Metal"), to: metal)
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        let bricksID = ImageMetadataStore.categoryID("Bricks")
        model.setModelTestDataset(bricksID)
        #expect(model.testableModelVersions.count == 1)
        #expect(model.testDiffuseImages.map(\.url) == [fixture.sourceInput])
        model.setModelTestMap(.roughness)
        #expect(model.testableModelVersions.isEmpty)
        model.setModelTestMap(.depth)
        model.showModelTesting(try #require(model.testableModelVersions.first))
        model.testWeightSource = .checkpoint
        model.setModelTestInput(fixture.externalInput)
        model.setModelTestCheckpoint("epoch-10")
        #expect(model.testCheckpointVersions.count == 2)
        #expect(model.selectedTestCheckpoint?.epochsCompleted == 10)
        #expect(model.canTestModel)
        func result(_ id: String) -> ModelTestResult {
            .init(directory: fixture.result.directory, inputURL: fixture.externalInput, outputURL: fixture.output,
                runID: "one-saved-run", source: .checkpoint, map: .depth, width: 384, height: 256,
                tileSize: 256, tileCount: 2, note: "UI fixture", previewURL: fixture.preview, checkpointID: id)
        }
        model.recordModelTestResult(result("epoch-10"))
        model.setModelTestCheckpoint("epoch-20")
        #expect(model.modelTestResult == nil)
        #expect(model.previousTestResult?.checkpointID == "epoch-10")
        model.recordModelTestResult(result("epoch-20"))
        #expect(model.previousTestResult?.checkpointID == "epoch-10")
        #expect(model.modelTestResult?.checkpointID == "epoch-20")
        try await render(ModelTestView(model: model).preferredColorScheme(.dark),
            size: NSSize(width: 1140, height: 1700), name: "model-test-checkpoint-selection-ab.png")
        model.setModelTestDataset(ImageMetadataStore.categoryID("Metal"))
        #expect(model.testableModelVersions.isEmpty)
        #expect(model.testDiffuseImages.map(\.url) == [metal])
        #expect(model.testCheckpointID == nil)
        #expect(model.modelTestResult == nil)
        #expect(model.previousTestResult == nil)
    }

    @Test @MainActor
    func oneSavedRunAndAnExternalDiffuseNeedNoTargetOrSelectedTrainingDataset() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        #expect(model.selectedDatasetID.isEmpty)
        #expect(model.snapshot?.images.count == 1)
        #expect(model.snapshot?.images.allSatisfy { $0.role == .diffuse } == true)
        #expect(model.testableModelVersions.count == 1)
        #expect(!model.canCompareModels)
        let version = try #require(model.testableModelVersions.first)
        model.selectedMap = .normal
        #expect(!model.canTestModel)
        model.showModelTesting(version)
        #expect(model.workspaceSection == .testing)
        #expect(model.selectedMap == .normal)
        #expect(model.selectedTestVersion?.map == .depth)
        #expect(model.availableTestWeightSources == [.model, .checkpoint])
        #expect(model.testInputURL == fixture.sourceInput)
        #expect(model.canTestModel)
        let original = try Data(contentsOf: fixture.externalInput)
        model.setModelTestInput(fixture.externalInput)
        #expect(model.canTestModel)
        #expect(model.modelTestIssue == nil)
        #expect(model.snapshot?.images.contains { $0.url == fixture.externalInput } == false)
        #expect(model.selectedDatasetID.isEmpty)
        #expect(try Data(contentsOf: fixture.externalInput) == original)
        try await render(ModelTestView(model: model), size: NSSize(width: 1140, height: 930), name: "model-test-controls.png")

        model.testWeightSource = .checkpoint
        try FileManager.default.removeItem(at: version.directory.appendingPathComponent("checkpoint.safetensors"))
        await model.pollFolder()
        #expect(!model.canTestModel)
        #expect(model.modelTestIssue?.contains("no longer available") == true)
        model.testWeightSource = .model
        #expect(model.canTestModel)
        try FileManager.default.removeItem(at: version.directory)
        await model.pollFolder()
        #expect(model.selectedTestVersion == nil)
        #expect(!model.canTestModel)
        #expect(model.modelTestIssue?.contains("selected run") == true)

        let empty = fixture.root.appendingPathComponent("Another Source")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        model.openFolder(empty)
        try await wait(model)
        #expect(model.testInputURL == nil)
        #expect(model.modelTestResult == nil)
        #expect(model.selectedTestVersionID.isEmpty)
        #expect(model.workspaceSection == .materials)
    }

    @Test @MainActor
    func resultRendersInputAndOutputAtFitAndNativeZoom() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let result = fixture.result
        for zoom in ComparisonImageZoom.allCases {
            let content = ModelTestResultView(result: result, imageZoom: .constant(zoom),
                                             saveOutput: {}, savePNGPreview: {}, showOutput: {}, showFolder: {})
            try await render(content, size: NSSize(width: 1140, height: 740), name: "model-test-result-\(zoom.rawValue).png")
        }
        let image = try #require(CGImageSourceCreateWithURL(result.outputURL as CFURL, nil))
        let properties = try #require(CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [CFString: Any])
        #expect(properties[kCGImagePropertyPixelWidth] as? Int == result.width)
        #expect(properties[kCGImagePropertyPixelHeight] as? Int == result.height)
        #expect(properties[kCGImagePropertyDepth] as? Int == 32)
        let preview = try #require(CGImageSourceCreateWithURL(result.previewURL as CFURL, nil))
        let previewProperties = try #require(CGImageSourceCopyPropertiesAtIndex(preview, 0, nil) as? [CFString: Any])
        #expect(previewProperties[kCGImagePropertyDepth] as? Int == 16)
    }

    @Test
    func savingCopiesOutputBytesAndProtectsOriginalImagesThroughAliases() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let result = fixture.result
        let originals = [fixture.externalInput, fixture.sourceInput]
        let before = try originals.map { try Data(contentsOf: $0) }
        let alias = fixture.root.appendingPathComponent("original-alias.png")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: fixture.externalInput)
        for path in originals + [alias] {
            #expect(throws: (any Error).self) {
                try ModelTestOutputExport.save(result, to: path, sourceImageURLs: [fixture.sourceInput])
            }
        }
        for (index, url) in originals.enumerated() { #expect(try Data(contentsOf: url) == before[index]) }
        let saved = fixture.root.appendingPathComponent("saved-depth.exr")
        try ModelTestOutputExport.save(result, to: saved, sourceImageURLs: [fixture.sourceInput])
        #expect(try Data(contentsOf: saved) == Data(contentsOf: result.outputURL))
        let decoded = try NativeImages.readTarget(from: saved, mapKind: "depth")
        #expect(decoded.values == fixture.outputValues)
        try ModelTestOutputExport.save(result, to: result.outputURL, sourceImageURLs: [fixture.sourceInput])
        let png = fixture.root.appendingPathComponent("saved-depth.png")
        #expect(throws: (any Error).self) {
            try ModelTestOutputExport.save(result, to: png, sourceImageURLs: [fixture.sourceInput])
        }
        try ModelTestOutputExport.save(result, to: png, sourceImageURLs: [fixture.sourceInput], format: .png16)
        #expect(try Data(contentsOf: png) == Data(contentsOf: result.previewURL))
        let pngImage = try #require(CGImageSourceCreateWithURL(png as CFURL, nil))
        let pngProperties = try #require(CGImageSourceCopyPropertiesAtIndex(pngImage, 0, nil) as? [CFString: Any])
        #expect(pngProperties[kCGImagePropertyDepth] as? Int == 16)
        let legacyResult = ModelTestResult(directory: result.directory, inputURL: result.inputURL, outputURL: result.previewURL,
            runID: result.runID, source: result.source, map: result.map, width: result.width, height: result.height,
            tileSize: result.tileSize, tileCount: result.tileCount, note: "Legacy preview-only fixture")
        #expect(throws: (any Error).self) {
            try ModelTestOutputExport.save(legacyResult, to: fixture.root.appendingPathComponent("legacy.exr"), sourceImageURLs: [])
        }
        try ModelTestOutputExport.save(legacyResult, to: fixture.root.appendingPathComponent("legacy.png"), sourceImageURLs: [], format: .png16)
    }

    @MainActor private func wait(_ model: FactoryModel) async throws {
        for _ in 0..<250 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!model.busy)
        #expect(model.errorMessage == nil)
        _ = try #require(model.snapshot)
    }

    @MainActor private func render<Content: View>(_ content: Content, size: NSSize, name: String) async throws {
        _ = NSApplication.shared
        let view = NSHostingView(rootView: content.padding(20).frame(width: size.width))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        view.frame = NSRect(origin: .zero, size: size)
        window.orderFrontRegardless()
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(800))
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
    }
}

private struct ModelTestControlsFixture {
    let root: URL
    let source: URL
    let sourceInput: URL
    let externalInput: URL
    let output: URL
    let preview: URL
    let outputValues: [Float]

    var result: ModelTestResult {
        .init(directory: output.deletingLastPathComponent(), inputURL: externalInput, outputURL: output,
              runID: "one-saved-run", source: .checkpoint, map: .depth, width: 384, height: 256,
              tileSize: 256, tileCount: 2,
              note: "Preview layout fixture. The original diffuse and generated map retain their native dimensions.",
              previewURL: preview)
    }

    init() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-model-test-ui-" + UUID().uuidString)
            .resolvingSymlinksInPath().standardizedFileURL
        source = root.appendingPathComponent("Source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        sourceInput = source.appendingPathComponent("brick_diffuse.png")
        externalInput = root.appendingPathComponent("unseen_diffuse.png")
        output = root.appendingPathComponent("depth.exr")
        preview = root.appendingPathComponent("depth.png")
        var values = (0..<(384 * 256)).map { Float(($0 % 1024) + 1) / Float(1027) }
        values[0] = 0.12345679; values[1] = Float(1).nextDown; values[2] = .leastNormalMagnitude
        outputValues = values
        try Self.writeImage(to: sourceInput, color: true)
        try Self.writeImage(to: externalInput, color: true)
        try NativeFloatEXR.write(values, channels: 1, width: 384, height: 256, to: output)
        try Self.writePreview(values, to: preview)
        try ImageMetadataStore.write(.init(pairingKey: "brick", role: .diffuse, category: "Bricks"), to: sourceInput)
        let project = try LoomProject(sourceFolder: source)
        let snapshot = try await project.snapshot()
        let dataset = try #require(snapshot.datasets.first)
        let run = project.projectFolder.appendingPathComponent("runs/one-saved-run")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let architecture: [String: Any] = ["id": "loom-factory-material-residual-v1", "width": 8, "blocks": 1]
        let saved: [String: Any] = ["image_size": 256, "exaggerate_features": false]
        let descriptor: [String: Any] = ["schema": "loom-factory-model-v1", "project_id": snapshot.projectID,
            "kind": "map", "dataset_id": dataset.id, "dataset_name": dataset.name, "map": "depth",
            "architecture": "loom-factory-material-residual-v1", "architecture_config": architecture,
            "image_size": 256, "training": saved, "weights_path": "weights.safetensors"]
        try JSONSerialization.data(withJSONObject: descriptor).write(to: run.appendingPathComponent("model.json"))
        var checkpoint = descriptor
        checkpoint["schema"] = "loom-factory-checkpoint-v1"
        checkpoint["settings"] = saved
        checkpoint["epochs_completed"] = 20
        checkpoint["weights_path"] = "checkpoint.safetensors"
        try JSONSerialization.data(withJSONObject: checkpoint).write(to: run.appendingPathComponent("checkpoint.json"))
        let artifact = ModelArtifact(id: "one-saved-run", kind: "map", datasetID: dataset.id, datasetName: dataset.name,
                                     map: .depth, directory: run, epochs: 20, loss: nil, coreMLPath: "model.mlmodel",
                                     weightsPath: "weights.safetensors", architecture: "loom-factory-material-residual-v1")
        try JSONEncoder().encode(artifact).write(to: run.appendingPathComponent("artifact.json"))
        // Selection-only fixtures do not execute these opaque saved files.
        for name in ["checkpoint.safetensors", "weights.safetensors", "model.mlmodel"] {
            try Data([0]).write(to: run.appendingPathComponent(name))
        }
    }

    private static func writeImage(to url: URL, color: Bool) throws {
        let width = 384, height = 256
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let mortar = y % 44 < 5 || (x + (y / 44 % 2) * 45) % 90 < 5
                let i = (y * width + x) * 4
                let level: UInt8 = mortar ? 60 : UInt8(175 + (x + y) % 45)
                pixels[i] = color ? (mortar ? 80 : 170) : level
                pixels[i + 1] = color ? (mortar ? 80 : 85) : level
                pixels[i + 2] = color ? (mortar ? 80 : 60) : level
            }
        }
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let bitmap = try #require(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let writer = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(writer, bitmap, nil)
        #expect(CGImageDestinationFinalize(writer))
    }

    private static func writePreview(_ values: [Float], to url: URL) throws {
        var samples = values.map { UInt16(($0 * 65535).rounded()).bigEndian }
        let data = samples.withUnsafeMutableBytes { Data($0) }
        let provider = try #require(CGDataProvider(data: data as CFData))
        let image = try #require(CGImage(width: 384, height: 256, bitsPerComponent: 16, bitsPerPixel: 16,
            bytesPerRow: 384 * 2, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: [.byteOrder16Big, CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)],
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let writer = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(writer, image, nil)
        #expect(CGImageDestinationFinalize(writer))
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}
