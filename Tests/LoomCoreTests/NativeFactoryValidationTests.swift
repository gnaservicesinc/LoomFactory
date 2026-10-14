import CoreGraphics
import Foundation
import ImageIO
import LoomNative
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class NativeFactoryValidationTests: XCTestCase {
    func testWholeMaterialHoldoutEarlyStoppingPreservesBestDeploymentAndLatestCheckpoint() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("native-validation-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var originals: [URL: Data] = [:]
        for (index, material) in ["brick_a", "brick_b"].enumerated() {
            let folder = root.appendingPathComponent(material, isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for role in [MapRole.diffuse, .depth] {
                let file = folder.appendingPathComponent(role.rawValue + ".png")
                try writeImage(to: file, role: role, materialIndex: index)
                let metadata = ImageMetadata(pairingKey: material, role: role, category: "Bricks",
                                             roleWasManuallySelected: true, pairingWasManuallySelected: true)
                try ImageMetadataStore.write(metadata, to: file)
                originals[file] = try Data(contentsOf: file)
            }
        }
        let project = try LoomProject(sourceFolder: root)
        // The update is below Float32 resolution, deliberately producing an
        // unchanged held-out score. This verifies control flow and artifacts;
        // it is not a material-quality or generalization experiment.
        let settings = TrainingSettings(imageSize: 32, epochs: 5, learningRate: 1e-20,
            shuffleEachEpoch: false, flipCrops: false, rotateCrops: false, modelWidth: 8, modelBlocks: 1,
            cropRefresh: .perRun, validationFraction: 0.5, earlyStoppingPatience: 1, exportBestWeights: true)
        let snapshot = try await project.snapshot()
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        let plan = try XCTUnwrap(snapshot.trainingPlan(datasetID: datasetID, map: .depth, settings: settings))
        XCTAssertEqual(plan.trainingPairCount, 1)
        XCTAssertEqual(plan.validationPairCount, 1)
        XCTAssertEqual(plan.trainingMaterialCount, 1)
        XCTAssertEqual(plan.validationMaterialCount, 1)
        let trainingFolders = Set(plan.training.map { $0.input.url.deletingLastPathComponent() })
        let validationFolders = Set(plan.validation.map { $0.input.url.deletingLastPathComponent() })
        XCTAssertTrue(trainingFolders.isDisjoint(with: validationFolders))
        XCTAssertEqual(Set(plan.training.map { $0.target.url.deletingLastPathComponent() }), trainingFolders)
        XCTAssertEqual(Set(plan.validation.map { $0.target.url.deletingLastPathComponent() }), validationFolders)

        let artifact = try await project.trainMap(datasetID: datasetID, map: .depth, settings: settings, progress: { _ in })
        let descriptor = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("model.json"))
        let training = try XCTUnwrap(descriptor["training"] as? [String: Any])
        let history = try XCTUnwrap(training["history"] as? [[String: Any]])
        XCTAssertEqual(artifact.epochs, 2)
        XCTAssertEqual(training["requested_epochs"] as? Int, 5)
        XCTAssertEqual(training["epochs"] as? Int, 2)
        XCTAssertEqual(training["updates"] as? Int, 2)
        XCTAssertEqual(training["training_sample_count"] as? Int, 1)
        XCTAssertEqual(training["validation_sample_count"] as? Int, 1)
        XCTAssertEqual(training["best_epoch"] as? Int, 1)
        XCTAssertEqual(training["export_epoch"] as? Int, 1)
        XCTAssertEqual(training["validation_crop"] as? String, "fixed-center-native-no-augmentation")
        XCTAssertEqual(history.count, 2)
        let firstValidation = try XCTUnwrap(history[0]["validation_loss"] as? Double)
        XCTAssertTrue(firstValidation.isFinite)
        XCTAssertEqual(history[1]["validation_loss"] as? Double, firstValidation)
        XCTAssertEqual(descriptor["final_validation_loss"] as? Double, firstValidation)
        let best = try NativeWeights.read(from: artifact.directory.appendingPathComponent("best.safetensors"))
        let deployed = try NativeWeights.read(from: artifact.directory.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(deployed, best)
        let checkpoint = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("checkpoint.json"))
        XCTAssertEqual(checkpoint["status"] as? String, "completed")
        XCTAssertEqual(checkpoint["epochs_completed"] as? Int, 2)
        XCTAssertEqual(checkpoint["updates"] as? Int, 2)
        XCTAssertEqual((checkpoint["history"] as? [[String: Any]])?.count, 2)
        let request = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("request.json"))
        let samples = try XCTUnwrap(request["samples"] as? [[String: Any]])
        XCTAssertEqual(samples.filter { $0["split"] as? String == "training" }.count, 1)
        XCTAssertEqual(samples.filter { $0["split"] as? String == "validation" }.count, 1)
        let requestedTraining = Set(samples.filter { $0["split"] as? String == "training" }.compactMap { ($0["input"] as? String).map { URL(fileURLWithPath: $0).deletingLastPathComponent() } })
        let requestedValidation = Set(samples.filter { $0["split"] as? String == "validation" }.compactMap { ($0["input"] as? String).map { URL(fileURLWithPath: $0).deletingLastPathComponent() } })
        XCTAssertEqual(requestedTraining, trainingFolders)
        XCTAssertEqual(requestedValidation, validationFolders)
        for (url, original) in originals { XCTAssertEqual(try Data(contentsOf: url), original) }
    }

    private func writeImage(to url: URL, role: MapRole, materialIndex: Int) throws {
        let channels = role == .diffuse ? 3 : 1
        var bytes = [UInt8](repeating: 0, count: 64 * 64 * channels)
        for index in bytes.indices { bytes[index] = UInt8((index * 13 + materialIndex * 17 + 23) % 255) }
        let provider = try XCTUnwrap(CGDataProvider(data: Data(bytes) as CFData))
        let image = try XCTUnwrap(CGImage(width: 64, height: 64, bitsPerComponent: 8, bitsPerPixel: channels * 8,
            bytesPerRow: 64 * channels, space: channels == 3 ? CGColorSpaceCreateDeviceRGB() : CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }
}
