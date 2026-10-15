import Foundation
import LoomNative
import Metal
import XCTest
@testable import LoomCore

final class CompactModelContractTests: XCTestCase {
    func testConservativeWholeFieldMemoryAdmissionScalesWithGridAndCurrentBudget() throws {
        let gib = UInt64(1 << 30)
        let architecture = try NativeArchitecture(mapKind: "depth", width: 32, blocks: 1,
            identifier: NativeArchitecture.compactIdentifier)
        let budget = CompactModelMemoryAdmission.budget(physical: 64 * gib,
            recommendedMetal: 48 * gib, allocatedMetal: 0)
        let oneK = CompactModelMemoryAdmission.forecast(width: 1024, height: 1024,
            featureWidth: 32, parameterCount: architecture.parameterCount)
        let twoK = CompactModelMemoryAdmission.forecast(width: 2048, height: 2048,
            featureWidth: 32, parameterCount: architecture.parameterCount)
        let fourK = CompactModelMemoryAdmission.forecast(width: 4096, height: 4096,
            featureWidth: 32, parameterCount: architecture.parameterCount)
        XCTAssertEqual(budget, 48 * gib)
        XCTAssertEqual(oneK, 8 * gib + UInt64(architecture.parameterCount) * 12)
        XCTAssertEqual(twoK, 32 * gib + UInt64(architecture.parameterCount) * 12)
        XCTAssertLessThan(oneK, budget); XCTAssertLessThan(twoK, budget)
        XCTAssertGreaterThan(fourK, budget)
        let busyBudget = CompactModelMemoryAdmission.budget(physical: 64 * gib,
            recommendedMetal: 48 * gib, allocatedMetal: 24 * gib)
        XCTAssertGreaterThan(twoK, busyBudget)
        XCTAssertEqual(CompactModelMemoryAdmission.forecast(width: Int.max, height: Int.max,
            featureWidth: 32, parameterCount: architecture.parameterCount), .max)
    }

    func testTrainedCompactCheckpointContinuationAndNativeModelTestingKeepTheirContract() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal GPU") }
        let folder = try temporaryDirectory(); defer {
            try? FileManager.default.removeItem(at: folder); NativeImages.clearDecodedPixelCache()
        }
        var originals: [URL: Data] = [:]
        for (name, role, channels) in [("brick_diff.png", MapRole.diffuse, 3), ("brick_disp.png", MapRole.depth, 1)] {
            let values = (0..<(64 * 64 * channels)).map { Float(($0 * 7 + 31) % 251) / 250 }
            let file = folder.appendingPathComponent(name)
            try ModelTestImage.writePNG(values, channels: channels, width: 64, height: 64, to: file)
            var metadata = ImageMetadata(pairingKey: "compact-workflow", role: role, category: "Bricks")
            metadata.roleWasManuallySelected = true; metadata.pairingWasManuallySelected = true
            try ImageMetadataStore.write(metadata, to: file)
            originals[file] = try Data(contentsOf: file)
        }
        let project = try LoomProject(sourceFolder: folder)
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        let settings = TrainingSettings(imageSize: 32, epochs: 1, shuffleEachEpoch: false, seed: 7,
            modelWidth: 8, modelBlocks: 1, validationFraction: 0, modelArchitecture: NativeArchitecture.compactIdentifier)
        let artifact = try await project.trainMap(datasetID: datasetID, map: .depth,
            settings: settings, start: .fresh, progress: { _ in })
        XCTAssertEqual(artifact.architecture, NativeArchitecture.compactIdentifier)
        let descriptor = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("model.json"))
        XCTAssertTrue(try TrainingRunFiles.nativeArchitecture(from: descriptor, map: .depth).isCompact)
        let checkpoint = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("checkpoint.json"))
        let checkpointFile = artifact.directory.appendingPathComponent(try XCTUnwrap(checkpoint["weights_path"] as? String))
        let before = try Data(contentsOf: checkpointFile)
        var requested = TrainingSettings(imageSize: 64, epochs: 1, seed: 9, modelWidth: 24, modelBlocks: 2,
            validationFraction: 0, modelArchitecture: NativeArchitecture.identifier)
        requested.flipCrops = false
        let continued = try await project.trainMap(datasetID: datasetID, map: .depth,
            settings: requested, start: .checkpoint(runID: artifact.id), progress: { _ in })
        let request = try NativeFactoryTraining.json(continued.directory.appendingPathComponent("request.json"))
        XCTAssertEqual(request["model_architecture"] as? String, NativeArchitecture.compactIdentifier)
        XCTAssertEqual(request["scalar_loss"] as? String, NativeScalarLoss.linearDetailV1.rawValue)
        XCTAssertEqual(request["width"] as? Int, 8); XCTAssertEqual(request["blocks"] as? Int, 1)
        XCTAssertEqual(request["image_size"] as? Int, 64)
        XCTAssertEqual(try Data(contentsOf: checkpointFile), before)
        let versions = try await project.modelVersions(datasetID: datasetID, map: .depth)
        XCTAssertEqual(versions.count, 2)
        XCTAssertTrue(versions.allSatisfy { $0.settings?.modelArchitecture == NativeArchitecture.compactIdentifier
            && $0.checkpoint?.modelArchitecture == NativeArchitecture.compactIdentifier
            && $0.settings?.scalarLoss == .linearDetailV1 })
        for source in [TrainingWeightSource.checkpoint, .model] {
            let result = try await project.testModel(datasetID: datasetID, map: .depth, runID: continued.id,
                source: source, inputURL: folder.appendingPathComponent("brick_diff.png"), progress: { _ in })
            let record = try NativeFactoryTraining.json(result.directory.appendingPathComponent("test.json"))
            XCTAssertEqual(record["architecture"] as? String, NativeArchitecture.compactIdentifier)
            XCTAssertEqual(record["stitching"] as? String, "whole-field-v1")
            XCTAssertTrue(FileManager.default.fileExists(atPath: result.outputURL.path))
        }
        for (file, bytes) in originals { XCTAssertEqual(try Data(contentsOf: file), bytes) }
    }

    func testCompactDescriptorAndObjectiveRemainDistinctFromExistingModels() async throws {
        let folder = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: folder) }
        let project = try LoomProject(sourceFolder: folder)
        let architecture = try NativeArchitecture(mapKind: "normal", width: 32, blocks: 1,
            identifier: NativeArchitecture.compactIdentifier)
        let settings = TrainingSettings(imageSize: 32, modelWidth: 32, modelBlocks: 1,
            modelArchitecture: architecture.identifier)
        let values = NativeFactoryTraining.settingsJSON(settings)
        let descriptor: [String: Any] = ["architecture": architecture.identifier,
            "architecture_config": architecture.descriptor, "training": values]
        XCTAssertEqual(try TrainingRunFiles.nativeArchitecture(from: descriptor, map: .normal), architecture)
        XCTAssertEqual(TrainingRunFiles.settings(from: values, descriptor: descriptor)?.scalarLoss, .linearDetailV1)
        let fresh = try await project.prepareTrainingStart(.fresh, datasetID: "bricks", map: .normal, settings: settings)
        XCTAssertEqual(fresh.settings.modelArchitecture, architecture.identifier)
        XCTAssertEqual(fresh.settings.scalarLoss, .linearDetailV1)
        do {
            _ = try await project.prepareTrainingStart(.fresh, datasetID: "bricks", map: .normal,
                settings: settings, scalarLossOverride: .logitCrossEntropyV2)
            XCTFail("A compact model cannot acquire a sigmoid/logit objective")
        } catch {}

        var altered = architecture.descriptor
        altered["head_activation"] = "sigmoid"
        var incompatible = descriptor; incompatible["architecture_config"] = altered
        XCTAssertThrowsError(try TrainingRunFiles.nativeArchitecture(from: incompatible, map: .normal))
        altered = architecture.descriptor; altered["normal_output_transform"] = "normalize"
        incompatible["architecture_config"] = altered
        XCTAssertThrowsError(try TrainingRunFiles.nativeArchitecture(from: incompatible, map: .normal))
        var incorrectObjective = values; incorrectObjective["scalar_loss"] = NativeScalarLoss.logitCrossEntropyV2.rawValue
        XCTAssertNil(TrainingRunFiles.settings(from: incorrectObjective, descriptor: descriptor))

        let residual = TrainingSettings(imageSize: 32, modelWidth: 8, modelBlocks: 1,
            modelArchitecture: NativeArchitecture.identifier)
        do {
            _ = try await project.prepareTrainingStart(.fresh, datasetID: "bricks", map: .depth,
                settings: residual, scalarLossOverride: .linearDetailV1)
            XCTFail("Existing models cannot silently acquire the compact objective")
        } catch {}
    }

    func testCompactWholeFieldTestPreservesExtendedRangeAndRawNormalValues() throws {
        let folder = try temporaryDirectory(); defer {
            try? FileManager.default.removeItem(at: folder); NativeImages.clearDecodedPixelCache()
        }
        let width = 64, height = 48, plane = width * height
        let input = folder.appendingPathComponent("input.png")
        try ModelTestImage.writePNG([Float](repeating: 0.3, count: plane * 3), channels: 3,
            width: width, height: height, to: input)
        let inputBytes = try Data(contentsOf: input)
        let weights = folder.appendingPathComponent("weights.safetensors")
        try Data([1, 2, 3]).write(to: weights)
        let architecture = try NativeArchitecture(mapKind: "normal", width: 32, blocks: 1,
            identifier: NativeArchitecture.compactIdentifier)
        let model = ModelTestSelection(runID: "compact-contract", source: .checkpoint, datasetID: "bricks",
            map: .normal, architecture: architecture, imageSize: 32, weightsURL: weights,
            weightsSHA256: try TrainingRunFiles.sha256(weights))
        let expected = [Float](repeating: -0.25, count: plane)
            + [Float](repeating: 0.25, count: plane) + [Float](repeating: 1.25, count: plane)
        var calls = 0
        let result = try ModelTestWorker.run(model: model, inputURL: input, output: folder.appendingPathComponent("test"),
            cancellation: NativeCancellation(), progress: { _ in }, compactMemoryBudget: .max) { pixels, h, w in
                calls += 1
                XCTAssertEqual(w, width); XCTAssertEqual(h, height)
                XCTAssertEqual(pixels.count, plane * 3)
                return expected
            }
        XCTAssertEqual(calls, 1, "Global compact attention must see the complete field")
        XCTAssertEqual(result.tileCount, 1)
        let expectedFile = folder.appendingPathComponent("expected.exr")
        try NativeFloatEXR.write(expected, channels: 3, width: width, height: height, to: expectedFile)
        XCTAssertEqual(try Data(contentsOf: result.outputURL), try Data(contentsOf: expectedFile),
            "The canonical EXR cannot clamp or normalize the raw prediction")
        let preview = try NativeImages.readTarget(from: result.previewURL, mapKind: "normal")
        for pixel in 0..<plane {
            XCTAssertEqual(preview.values[pixel * 3], 0)
            XCTAssertEqual(preview.values[pixel * 3 + 1], 0.25, accuracy: 1 / 65535)
            XCTAssertEqual(preview.values[pixel * 3 + 2], 1)
        }
        let record = try NativeFactoryTraining.json(result.directory.appendingPathComponent("test.json"))
        XCTAssertEqual(record["stitching"] as? String, "whole-field-v1")
        XCTAssertEqual(record["normal_output_transform"] as? String, "none")
        XCTAssertEqual(record["head_activation"] as? String, "linear")
        let admission = try XCTUnwrap(record["memory_admission"] as? [String: Any])
        XCTAssertEqual(admission["kind"] as? String, "conservative-forecast-advisory-v1")
        XCTAssertEqual(admission["attempted_over_budget"] as? Bool, false)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
    }

    func testCompactForecastAboveBudgetStillAttemptsExactWholeField() throws {
        let folder = try temporaryDirectory(); defer {
            try? FileManager.default.removeItem(at: folder); NativeImages.clearDecodedPixelCache()
        }
        // Exercise the same policy as an over-budget 4K forecast on a small
        // native image; the injected predictor avoids GPU allocation entirely.
        let width = 64, height = 48, plane = width * height
        let input = folder.appendingPathComponent("input.png")
        try ModelTestImage.writePNG([Float](repeating: 0.3, count: plane * 3), channels: 3,
            width: width, height: height, to: input)
        let inputBytes = try Data(contentsOf: input)
        let weights = folder.appendingPathComponent("weights.safetensors")
        let weightBytes = Data([1, 2, 3]); try weightBytes.write(to: weights)
        let architecture = try NativeArchitecture(mapKind: "depth", width: 32, blocks: 1,
            identifier: NativeArchitecture.compactIdentifier)
        let model = ModelTestSelection(runID: "compact-advisory", source: .checkpoint, datasetID: "bricks",
            map: .depth, architecture: architecture, imageSize: 32, weightsURL: weights,
            weightsSHA256: try TrainingRunFiles.sha256(weights))
        let messages = Messages()
        let expected = [Float](repeating: 0.25, count: plane)
        var calls = 0
        let result = try ModelTestWorker.run(model: model, inputURL: input, output: folder.appendingPathComponent("test"),
            cancellation: NativeCancellation(), progress: { messages.append($0.message) }, compactMemoryBudget: 1) { pixels, h, w in
                calls += 1
                XCTAssertEqual(w, width); XCTAssertEqual(h, height)
                XCTAssertEqual(pixels.count, plane * 3)
                return expected
            }
        XCTAssertEqual(calls, 1, "A conservative forecast cannot prevent a valid whole-field attempt")
        XCTAssertEqual(result.width, width); XCTAssertEqual(result.height, height)
        XCTAssertEqual(result.tileCount, 1)
        let expectedFile = folder.appendingPathComponent("expected.exr")
        try NativeFloatEXR.write(expected, channels: 1, width: width, height: height, to: expectedFile)
        XCTAssertEqual(try Data(contentsOf: result.outputURL), try Data(contentsOf: expectedFile))
        let record = try NativeFactoryTraining.json(result.directory.appendingPathComponent("test.json"))
        let admission = try XCTUnwrap(record["memory_admission"] as? [String: Any])
        XCTAssertEqual(admission["kind"] as? String, "conservative-forecast-advisory-v1")
        XCTAssertEqual(admission["budget_bytes"] as? UInt64, 1)
        XCTAssertEqual(admission["forecast_bytes"] as? UInt64,
            CompactModelMemoryAdmission.forecast(width: width, height: height,
                featureWidth: architecture.width, parameterCount: architecture.parameterCount))
        XCTAssertEqual(admission["attempted_over_budget"] as? Bool, true)
        XCTAssertTrue(messages.snapshot().contains { $0.contains("Attempting the complete native image field") && $0.contains("forecast budget") })
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
        XCTAssertEqual(try Data(contentsOf: weights), weightBytes)
    }

    func testCompactTestRejectsUnsupportedNativeGridBeforePrediction() throws {
        let folder = try temporaryDirectory(); defer {
            try? FileManager.default.removeItem(at: folder); NativeImages.clearDecodedPixelCache()
        }
        let input = folder.appendingPathComponent("input.png")
        try ModelTestImage.writePNG([Float](repeating: 0.3, count: 65 * 48 * 3), channels: 3,
            width: 65, height: 48, to: input)
        let weights = folder.appendingPathComponent("weights.safetensors"); try Data([1]).write(to: weights)
        let model = try ModelTestSelection(runID: "compact-contract", source: .checkpoint, datasetID: "bricks", map: .depth,
            architecture: NativeArchitecture(mapKind: "depth", width: 32, blocks: 1, identifier: NativeArchitecture.compactIdentifier),
            imageSize: 32, weightsURL: weights, weightsSHA256: TrainingRunFiles.sha256(weights))
        XCTAssertThrowsError(try ModelTestWorker.run(model: model, inputURL: input,
            output: folder.appendingPathComponent("test"), cancellation: NativeCancellation(), progress: { _ in }) { _, _, _ in
                XCTFail("Unsupported grids cannot be resized or silently tiled")
                return []
            })
        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.appendingPathComponent("test").path))
    }

    private func temporaryDirectory() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("compact-core-contract-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder.resolvingSymlinksInPath()
    }

    private final class Messages: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [String] = []
        func append(_ value: String) { lock.withLock { values.append(value) } }
        func snapshot() -> [String] { lock.withLock { values } }
    }
}
