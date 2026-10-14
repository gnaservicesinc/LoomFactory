import CoreGraphics
import CoreML
import Foundation
import ImageIO
import LoomNative
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class NativeFactoryWorkflowTests: XCTestCase {
    func testNativeRunUsesTheBuildersSelectedResolutionPlanAndRecordsItsCounts() async throws {
        let (project, root) = try fixture(resolutions: [64, 128, 256])
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = try await project.snapshot()
        let originals = try Dictionary(uniqueKeysWithValues: snapshot.images.map { ($0.url, try Data(contentsOf: $0.url)) })
        let settings = TrainingSettings(imageSize: 32, sourceResolutionLevels: 2, epochs: 2,
            batchSize: 2, shuffleEachEpoch: false, modelWidth: 8, modelBlocks: 1,
            gradientAccumulationSteps: 3, validationFraction: 0)
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        let plan = try XCTUnwrap(snapshot.trainingPlan(datasetID: datasetID, map: .depth, settings: settings))
        XCTAssertEqual(plan.sourceMaterialCount, 1)
        XCTAssertEqual(plan.sourceImageCount, 6)
        XCTAssertEqual(plan.trainingPairCount, 2)
        XCTAssertEqual(plan.excludedPairCount, 1)
        let artifact = try await project.trainMap(datasetID: datasetID, map: .depth, settings: settings, progress: { _ in })
        let request = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("request.json"))
        let samples = try XCTUnwrap(request["samples"] as? [[String: Any]])
        XCTAssertEqual(Set(samples.compactMap { $0["id"] as? String }), Set(plan.training.map(\.id)))
        XCTAssertEqual(request["source_resolution_levels"] as? Int, 2)
        XCTAssertEqual(request["source_material_count"] as? Int, 1)
        XCTAssertEqual(request["source_image_count"] as? Int, 6)
        XCTAssertEqual(request["excluded_pair_count"] as? Int, 1)
        XCTAssertEqual(request["planned_training_pair_uses"] as? Int, plan.trainingSampleVisits)
        XCTAssertEqual(request["planned_optimizer_updates"] as? Int, plan.totalOptimizerUpdates)
        let descriptor = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("model.json"))
        let training = try XCTUnwrap(descriptor["training"] as? [String: Any])
        XCTAssertEqual(training["updates"] as? Int, plan.totalOptimizerUpdates)
        XCTAssertEqual(training["source_resolution_levels"] as? Int, 2)
        for (url, bytes) in originals { XCTAssertEqual(try Data(contentsOf: url), bytes) }
    }

    func testNativeTrainingPublishesRealCoreMLAndCompatibleWeightsPreservingOriginals() async throws {
        let (project, root) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let before = try pixels(root)
        let settings = TrainingSettings(imageSize: 32, epochs: 1, shuffleEachEpoch: false, modelWidth: 8, modelBlocks: 1, validationFraction: 0, modelArchitecture: NativeArchitecture.identifier)
        let artifact = try await project.trainMap(datasetID: ImageMetadataStore.categoryID("Bricks"), map: .depth,
                                                settings: settings, progress: { _ in })
        let weights = try NativeWeights.read(from: artifact.directory.appendingPathComponent(artifact.weightsPath!))
        XCTAssertEqual(weights.count, 8)
        let descriptor = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("model.json"))
        XCTAssertEqual((descriptor["training"] as? [String: Any])?["backend"] as? String, "swift-mpsgraph-metal")
        let compiled = try await MLModel.compileModel(at: artifact.directory.appendingPathComponent(artifact.coreMLPath!))
        defer { try? FileManager.default.removeItem(at: compiled) }
        let config = MLModelConfiguration(); config.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiled, configuration: config)
        let rgb = try MLMultiArray(shape: [1,3,32,32], dataType: .float32)
        for index in 0..<rgb.count { rgb[index] = NSNumber(value: Float(index % 255) / 255) }
        let result = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["rgb": rgb]))
        let actual = try XCTUnwrap(result.featureValue(for: "material_map")?.multiArrayValue)
        let session = try NativeTrainingSession(architecture: NativeArchitecture(mapKind: "depth", width: 8, blocks: 1),
            options: NativeTrainingOptions(), seed: 0, weights: weights)
        let expected = try session.predict(input: (0..<rgb.count).map { rgb[$0].floatValue }, batchSize: 1, height: 32, width: 32)
        for index in 0..<actual.count { XCTAssertEqual(actual[index].floatValue, expected[index], accuracy: 2e-5) }
        XCTAssertEqual(try pixels(root), before)
        let snapshot = try await project.snapshot()
        XCTAssertEqual(snapshot.artifacts.count, 1)
        XCTAssertEqual(snapshot.checkpoints.first?.status, "completed")
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(".loom-factory/runtime").path))
    }

    func testStopDuringNativeTrainingSavesCompletedUpdatesAndAllowsContinuation() async throws {
        let (project, root) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let (events, continuation) = AsyncStream<Void>.makeStream()
        let settings = TrainingSettings(imageSize: 32, epochs: 500, shuffleEachEpoch: false, modelWidth: 8, modelBlocks: 1, validationFraction: 0)
        let operation = Task {
            defer { continuation.finish() }
            return try await project.trainMap(datasetID: ImageMetadataStore.categoryID("Bricks"), map: .depth,
                settings: settings, progress: { event in if event.type == "checkpoint" { continuation.yield(()) } })
        }
        for await _ in events { break }
        let stopStarted = Date()
        await project.cancelTraining()
        do { _ = try await operation.value; XCTFail("Stop must cancel training") } catch is CancellationError {} catch { throw error }
        XCTAssertLessThan(Date().timeIntervalSince(stopStarted), 5)
        let stopped = try await project.snapshot()
        XCTAssertTrue(stopped.artifacts.isEmpty)
        let checkpoint = try XCTUnwrap(stopped.checkpoints.first)
        XCTAssertEqual(checkpoint.status, "cancelled")
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkpoint.directory.appendingPathComponent("model.json").path))
        let saved = try NativeFactoryTraining.json(checkpoint.directory.appendingPathComponent("checkpoint.json"))
        XCTAssertFalse(try NativeWeights.read(from: checkpoint.directory.appendingPathComponent(try XCTUnwrap(saved["weights_path"] as? String))).isEmpty)
        var resumed = settings; resumed.epochs = 1; resumed.continueLatestWeights = true
        let artifact = try await project.trainMap(datasetID: ImageMetadataStore.categoryID("Bricks"), map: .depth,
            settings: resumed, progress: { _ in })
        XCTAssertEqual(artifact.epochs, 1)
    }

    func testChangedMembershipStopsBeforeAnotherUpdateAndPreservesCheckpoint() async throws {
        let (project, root) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let settings = TrainingSettings(imageSize: 32, epochs: 5, shuffleEachEpoch: false, modelWidth: 8, modelBlocks: 1, validationFraction: 0)
        let target = root.appendingPathComponent("brick_disp.png")
        do {
            _ = try await project.trainMap(datasetID: ImageMetadataStore.categoryID("Bricks"), map: .depth, settings: settings,
                progress: { event in
                    if event.type == "checkpoint", event.update == 1 {
                        if var metadata = try? ImageMetadataStore.read(target) { metadata.ignored = true; try? ImageMetadataStore.write(metadata, to: target) }
                    }
                })
            XCTFail("Changed membership must stop training")
        } catch { XCTAssertTrue(error.localizedDescription.contains("source pair changed")) }
        let snapshot = try await project.snapshot()
        XCTAssertTrue(snapshot.artifacts.isEmpty)
        XCTAssertEqual(snapshot.checkpoints.first?.status, "stopped")
    }

    func testManualCheckpointAcknowledgesARealUpdateAndTrainingContinues() async throws {
        let (project, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        try addMaterialPairs(root)
        let snapshot = try await project.snapshot()
        let dataset = try XCTUnwrap(snapshot.datasets.first)
        let settings = TrainingSettings(imageSize: 32, epochs: 2,
            shuffleEachEpoch: false, modelWidth: 8, modelBlocks: 1, validationFraction: 0,
            checkpointEveryEpochs: 100)
        let plan = try XCTUnwrap(snapshot.trainingPlan(datasetID: dataset.id, map: .depth, settings: settings))
        let id = UUID().uuidString.lowercased()
        let directory = project.projectFolder.appendingPathComponent("runs/" + id)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let projectID = await project.document.id
        let control = TrainingRunControl(runID: id)
        let receiptTask = Task { try await control.request(stop: false) }
        try await waitForCheckpointRequest(control)
        let artifact = try await Task.detached {
            try NativeFactoryTraining.run(id: id, root: root, output: directory, projectID: projectID,
                dataset: dataset, map: .depth, plan: plan, settings: settings, resume: nil,
                cancellation: NativeCancellation(), progress: { _ in }, control: control)
        }.value
        let receipt = try await receiptTask.value
        XCTAssertEqual(receipt.completedUpdates, 1)
        XCTAssertFalse(receipt.stopped)
        XCTAssertEqual(artifact.epochs, 2)
        let descriptor = try NativeFactoryTraining.json(directory.appendingPathComponent("checkpoint.json"))
        XCTAssertEqual(descriptor["updates"] as? Int, 6)
        XCTAssertEqual(descriptor["optimizer_updates"] as? Int, 6)
        control.finish()
    }

    func testSaveAndStopMidEpochResumesOptimizerAndExactNextImage() async throws {
        let (project, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        try addMaterialPairs(root)
        let snapshot = try await project.snapshot()
        let dataset = try XCTUnwrap(snapshot.datasets.first)
        let settings = TrainingSettings(imageSize: 32, epochs: 3,
            shuffleEachEpoch: true, flipCrops: true, rotateCrops: true, seed: 19,
            modelWidth: 8, modelBlocks: 1, cropRefresh: .perSample, augmentationRefresh: .perSample,
            validationFraction: 0, checkpointEveryEpochs: 100)
        let baseline = try await project.trainMap(datasetID: dataset.id, map: .depth, settings: settings,
            start: .fresh, progress: { _ in })
        let expected = try NativeWeights.read(from: baseline.directory.appendingPathComponent("model.safetensors"))
        let plan = try XCTUnwrap(snapshot.trainingPlan(datasetID: dataset.id, map: .depth, settings: settings))
        let id = UUID().uuidString.lowercased()
        let directory = project.projectFolder.appendingPathComponent("runs/" + id)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let projectID = await project.document.id
        let control = TrainingRunControl(runID: id)
        let receiptTask = Task { try await control.request(stop: true) }
        try await waitForCheckpointRequest(control)
        do {
            _ = try await Task.detached {
                try NativeFactoryTraining.run(id: id, root: root, output: directory, projectID: projectID,
                    dataset: dataset, map: .depth, plan: plan, settings: settings, resume: nil,
                    cancellation: NativeCancellation(), progress: { _ in }, control: control)
            }.value
            XCTFail("Save & Stop must end the run after saving")
        } catch is TrainingSavedStop {} catch { throw error }
        let receipt = try await receiptTask.value
        XCTAssertTrue(receipt.stopped)
        XCTAssertEqual(receipt.completedUpdates, 1)
        let checkpoint = try NativeFactoryTraining.json(directory.appendingPathComponent("checkpoint.json"))
        XCTAssertEqual(checkpoint["status"] as? String, "saved_stopped")
        let cursor = try XCTUnwrap(checkpoint["resume_cursor"] as? [String: Any])
        XCTAssertEqual(cursor["nextWindow"] as? Int, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("artifact.json").path))
        control.finish()
        let resumed = try await project.trainMap(datasetID: dataset.id, map: .depth, settings: settings,
            start: .checkpoint(runID: id), progress: { _ in })
        XCTAssertEqual(resumed.epochs, 3)
        let actual = try NativeWeights.read(from: resumed.directory.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(actual.keys.sorted(), expected.keys.sorted())
        for name in expected.keys {
            for (left, right) in zip(try XCTUnwrap(actual[name]).values, expected[name]!.values) {
                XCTAssertEqual(left, right, accuracy: 2e-6)
            }
        }
        let resumedDescriptor = try NativeFactoryTraining.json(resumed.directory.appendingPathComponent("checkpoint.json"))
        XCTAssertEqual(resumedDescriptor["optimizer_updates"] as? Int, 9)
        XCTAssertEqual(resumedDescriptor["updates"] as? Int, 9)
    }

    private func waitForCheckpointRequest(_ control: TrainingRunControl) async throws {
        for _ in 0..<1_000 {
            if control.hasPendingRequest { return }
            try await Task.sleep(for: .milliseconds(1))
        }
        throw LoomError("Checkpoint request did not arrive")
    }

    private func addMaterialPairs(_ root: URL) throws {
        for index in 2...3 {
            for role in [MapRole.diffuse, .depth] {
                let original = root.appendingPathComponent(role == .diffuse ? "brick_diff.png" : "brick_disp.png")
                let destination = root.appendingPathComponent("brick\(index)_\(role.rawValue).png")
                try Data(contentsOf: original).write(to: destination)
                var metadata = ImageMetadata(pairingKey: "fixture_\(index)", role: role)
                metadata.category = "Bricks"; metadata.roleWasManuallySelected = true; metadata.pairingWasManuallySelected = true
                try ImageMetadataStore.write(metadata, to: destination)
            }
        }
    }

    private func fixture(resolutions: [Int] = [64]) throws -> (LoomProject, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("native-factory-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for size in resolutions {
          for stem in ["brick_diff", "brick_disp"] {
            let name = stem + (resolutions.count == 1 ? "" : "_\(size)") + ".png"
            var bytes = [UInt8](repeating: 0, count: size * size * 3)
            for index in bytes.indices {
                let valueIndex = name.contains("disp") ? index / 3 : index
                bytes[index] = UInt8((valueIndex * 7 + 31) % 255)
            }
            let provider = CGDataProvider(data: Data(bytes) as CFData)!
            let image = CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: size*3,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            let file = root.appendingPathComponent(name)
            let destination = CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(destination, image, nil); XCTAssertTrue(CGImageDestinationFinalize(destination))
            var metadata = ImageMetadata(pairingKey: "fixture_\(size)", role: name.contains("diff") ? .diffuse : .depth)
            metadata.category = "Bricks"; metadata.roleWasManuallySelected = true; metadata.pairingWasManuallySelected = true
            try ImageMetadataStore.write(metadata, to: file)
          }
        }
        return (try LoomProject(sourceFolder: root), root.resolvingSymlinksInPath().standardizedFileURL)
    }
    private func pixels(_ root: URL) throws -> [Data] { try ["brick_diff.png", "brick_disp.png"].map { try Data(contentsOf: root.appendingPathComponent($0)) } }
}
