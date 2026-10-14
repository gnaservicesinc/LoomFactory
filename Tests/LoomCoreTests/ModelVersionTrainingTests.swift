import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

@MainActor
final class ModelVersionTrainingTests: XCTestCase {
    func testRealTrainingReplacementAndSelectedExportContinuationPreserveAllEarlierWeights() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-native-versions-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let input = try image(at: root.appendingPathComponent("material_diff.png"), role: .diffuse)
        let target = try image(at: root.appendingPathComponent("material_disp.png"), role: .depth)
        let originals = try [input, target].map { try Data(contentsOf: $0) }
        let project = try LoomProject(sourceFolder: root)
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        let initial = TrainingSettings(imageSize: 32, epochs: 1, learningRate: 0.001, shuffleEachEpoch: false,
            seed: 7, modelWidth: 8, modelBlocks: 1, gradientClipPolicy: .fixed, validationFraction: 0)
        let first = try await project.trainMap(datasetID: datasetID, map: .depth, settings: initial,
            start: .fresh, runLabel: "Initial model", progress: { _ in })
        let firstFiles = try savedFiles(first.directory)
        let firstWeights = try TrainingRunFiles.sha256(first.directory.appendingPathComponent("model.safetensors"))
        let firstRequest = try NativeFactoryTraining.json(first.directory.appendingPathComponent("request.json"))
        XCTAssertEqual(firstRequest["run_label"] as? String, "Initial model")
        XCTAssertNil(firstRequest["resume_weights"])

        var replacementSettings = initial
        replacementSettings.seed = 91
        replacementSettings.continueLatestWeights = true
        let replacement = try await project.trainMap(datasetID: datasetID, map: .depth, settings: replacementSettings,
            start: .fresh, runLabel: "Different initialization", progress: { _ in })
        XCTAssertNotEqual(replacement.id, first.id)
        XCTAssertEqual(try savedFiles(first.directory), firstFiles)
        let replacementFiles = try savedFiles(replacement.directory)
        let replacementRequest = try NativeFactoryTraining.json(replacement.directory.appendingPathComponent("request.json"))
        XCTAssertNil(replacementRequest["resume_weights"], "Explicit fresh replacement must override the legacy continuation setting")
        let selectedReplacement = try await project.selectedModelArtifact(datasetID: datasetID, map: .depth)
        XCTAssertEqual(selectedReplacement?.id, replacement.id)

        var continuation = initial
        continuation.imageSize = 64; continuation.modelWidth = 16; continuation.modelBlocks = 2
        continuation.learningRate = 0.0003; continuation.optimizer = .adam
        continuation.gradientClipPolicy = .warmup
        continuation.gradientClipNorm = 0.5
        continuation.gradientClipWarmupStartNorm = 0.1
        continuation.gradientClipWarmupUpdates = 5
        let continued = try await project.trainMap(datasetID: datasetID, map: .depth, settings: continuation,
            start: .model(runID: first.id), runLabel: "Continue first with clipping warmup", progress: { _ in })
        let request = try NativeFactoryTraining.json(continued.directory.appendingPathComponent("request.json"))
        XCTAssertEqual(request["run_label"] as? String, "Continue first with clipping warmup")
        XCTAssertEqual(request["resume_weights"] as? String, first.directory.appendingPathComponent("model.safetensors").path)
        XCTAssertEqual(request["image_size"] as? Int, 32)
        XCTAssertEqual(request["width"] as? Int, 8)
        XCTAssertEqual(request["blocks"] as? Int, 1)
        XCTAssertEqual(try XCTUnwrap(request["learning_rate"] as? Double), 0.0003, accuracy: 1e-12)
        XCTAssertEqual(request["optimizer"] as? String, "adam")
        XCTAssertEqual(request["gradient_clip_policy"] as? String, "warmup")
        XCTAssertEqual(request["gradient_clip_norm"] as? Double, 0.5)
        XCTAssertEqual(request["gradient_clip_warmup_start_norm"] as? Double, 0.1)
        XCTAssertEqual(request["gradient_clip_warmup_updates"] as? Int, 5)
        let lineage = try XCTUnwrap(request["lineage"] as? [String: Any])
        XCTAssertEqual(lineage["parent_run_id"] as? String, first.id)
        XCTAssertEqual(lineage["source"] as? String, "model")
        XCTAssertEqual(lineage["weights_sha256"] as? String, firstWeights)

        let descriptor = try NativeFactoryTraining.json(continued.directory.appendingPathComponent("model.json"))
        XCTAssertEqual(descriptor["weights_sha256"] as? String,
            try TrainingRunFiles.sha256(continued.directory.appendingPathComponent("model.safetensors")))
        XCTAssertEqual(descriptor["coreml_sha256"] as? String,
            try TrainingRunFiles.sha256(continued.directory.appendingPathComponent("model.mlmodel")))
        let training = try XCTUnwrap(descriptor["training"] as? [String: Any])
        let history = try XCTUnwrap(training["history"] as? [[String: Any]])
        let epoch = try XCTUnwrap(history.first)
        XCTAssertEqual(history.count, 1)
        XCTAssertEqual(epoch["optimizer_updates"] as? Int, 1)
        XCTAssertNotNil(epoch["clipped_updates"] as? Int)
        XCTAssertTrue(try XCTUnwrap(epoch["mean_gradient_norm"] as? Double).isFinite)
        XCTAssertTrue(try XCTUnwrap(epoch["peak_gradient_norm"] as? Double).isFinite)
        XCTAssertEqual(try XCTUnwrap(epoch["gradient_clip_threshold"] as? Double), 0.1, accuracy: 1e-7)

        let selectedContinued = try await project.selectedModelArtifact(datasetID: datasetID, map: .depth)
        XCTAssertEqual(selectedContinued?.id, continued.id)
        let reopened = try LoomProject(sourceFolder: root)
        let restored = try await reopened.modelVersions(datasetID: datasetID, map: .depth)
        XCTAssertEqual(restored.count, 3)
        XCTAssertEqual(restored.filter(\.isActive).map(\.id), [continued.id])
        let savedContinued = try XCTUnwrap(restored.first { $0.id == continued.id })
        XCTAssertEqual(savedContinued.lineage?.parentRunID, first.id)
        XCTAssertEqual(savedContinued.lineage?.source, .model)
        XCTAssertEqual(savedContinued.lineage?.weightsSHA256, firstWeights)
        XCTAssertEqual(savedContinued.settings?.gradientClipPolicy, .warmup)
        XCTAssertEqual(savedContinued.history.first?.optimizerUpdates, 1)
        XCTAssertNotNil(savedContinued.history.first?.meanGradientNorm)
        XCTAssertEqual(try savedFiles(first.directory), firstFiles)
        XCTAssertEqual(try savedFiles(replacement.directory), replacementFiles)
        XCTAssertEqual(try [input, target].map { try Data(contentsOf: $0) }, originals)
    }

    private func savedFiles(_ directory: URL) throws -> [String: Data] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isRegularFileKey])
        var result: [String: Data] = [:]
        for file in files where try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
            result[file.lastPathComponent] = try Data(contentsOf: file)
        }
        return result
    }

    private func image(at file: URL, role: MapRole) throws -> URL {
        var bytes = [UInt8](repeating: 0, count: 32 * 32 * 3)
        for index in bytes.indices {
            let position = role == .depth ? index / 3 : index
            bytes[index] = UInt8((position * 7 + 31) % 255)
        }
        let provider = try XCTUnwrap(CGDataProvider(data: Data(bytes) as CFData))
        let image = try XCTUnwrap(CGImage(width: 32, height: 32, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: 96,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [], provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil); XCTAssertTrue(CGImageDestinationFinalize(destination))
        var metadata = ImageMetadata(pairingKey: "native-version-fixture", role: role, category: "Bricks")
        metadata.roleWasManuallySelected = true; metadata.pairingWasManuallySelected = true
        try ImageMetadataStore.write(metadata, to: file)
        return file
    }
}
