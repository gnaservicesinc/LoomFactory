import CoreGraphics
import CoreML
import Foundation
import ImageIO
import LoomNative
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class PyramidWorkflowTests: XCTestCase {
    func testPyramidRunExportsTrainedWeightsAndCheckpointContinuationRetainsItsArchitecture() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("pyramid-workflow-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let originals = try fixtureImages(in: root)
        let project = try LoomProject(sourceFolder: root)
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        let settings = TrainingSettings(imageSize: 32, epochs: 1, shuffleEachEpoch: false,
            seed: 7, modelWidth: 8, modelBlocks: 1, validationFraction: 0,
            modelArchitecture: NativeArchitecture.pyramidIdentifier)
        let artifact = try await project.trainMap(datasetID: datasetID, map: .depth,
            settings: settings, start: .fresh, progress: { _ in })
        XCTAssertEqual(artifact.architecture, NativeArchitecture.pyramidIdentifier)
        let request = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("request.json"))
        XCTAssertEqual(request["model_architecture"] as? String, NativeArchitecture.pyramidIdentifier)
        let descriptor = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("model.json"))
        let architecture = try TrainingRunFiles.nativeArchitecture(from: descriptor, map: .depth)
        XCTAssertEqual(architecture.identifier, NativeArchitecture.pyramidIdentifier)
        XCTAssertEqual((descriptor["training"] as? [String: Any])?["updates"] as? Int, 1)
        let weights = try NativeWeights.read(from: artifact.directory.appendingPathComponent(try XCTUnwrap(artifact.weightsPath)))
        XCTAssertEqual(Set(weights.keys), Set(architecture.parameterShapes.keys))
        XCTAssertEqual(weights.values.reduce(0) { $0 + $1.values.count }, settings.parameterCount(map: .depth))
        try await verifyExport(artifact, architecture: architecture, weights: weights)

        let checkpoint = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("checkpoint.json"))
        XCTAssertEqual(checkpoint["architecture"] as? String, NativeArchitecture.pyramidIdentifier)
        XCTAssertEqual((checkpoint["settings"] as? [String: Any])?["model_architecture"] as? String,
            NativeArchitecture.pyramidIdentifier)
        let before = try Data(contentsOf: artifact.directory.appendingPathComponent("checkpoint.safetensors"))
        var requestedContinuation = settings
        requestedContinuation.modelArchitecture = NativeArchitecture.identifier
        requestedContinuation.modelWidth = 24; requestedContinuation.modelBlocks = 2
        let continued = try await project.trainMap(datasetID: datasetID, map: .depth,
            settings: requestedContinuation, start: .checkpoint(runID: artifact.id), progress: { _ in })
        let continuedRequest = try NativeFactoryTraining.json(continued.directory.appendingPathComponent("request.json"))
        XCTAssertEqual(continuedRequest["model_architecture"] as? String, NativeArchitecture.pyramidIdentifier)
        XCTAssertEqual(continuedRequest["width"] as? Int, 8)
        XCTAssertEqual(continuedRequest["blocks"] as? Int, 1)
        XCTAssertEqual(continuedRequest["resume_weights"] as? String,
            artifact.directory.appendingPathComponent("checkpoint.safetensors").path)
        XCTAssertEqual(try Data(contentsOf: artifact.directory.appendingPathComponent("checkpoint.safetensors")), before)
        let reopened = try LoomProject(sourceFolder: root)
        let versions = try await reopened.modelVersions(datasetID: datasetID, map: .depth)
        XCTAssertEqual(versions.count, 2)
        XCTAssertTrue(versions.allSatisfy { $0.settings?.modelArchitecture == NativeArchitecture.pyramidIdentifier })
        XCTAssertTrue(versions.allSatisfy { $0.checkpoint?.modelArchitecture == NativeArchitecture.pyramidIdentifier })
        XCTAssertEqual(versions.first { $0.id == continued.id }?.lineage?.source, .checkpoint)
        for (url, bytes) in originals { XCTAssertEqual(try Data(contentsOf: url), bytes) }
    }

    func testUnknownAndInconsistentPyramidDescriptorsAreRejectedBeforeLoadingWeights() throws {
        let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 1,
            identifier: NativeArchitecture.pyramidIdentifier)
        let settings: [String: Any] = ["image_size": 32, "width": 8, "blocks": 1,
            "model_architecture": NativeArchitecture.pyramidIdentifier]
        let descriptor: [String: Any] = ["architecture": architecture.identifier,
            "architecture_config": architecture.descriptor, "training": settings]
        XCTAssertEqual(try TrainingRunFiles.nativeArchitecture(from: descriptor, map: .depth), architecture)
        XCTAssertEqual(TrainingRunFiles.settings(from: settings, descriptor: descriptor)?.modelArchitecture,
            NativeArchitecture.pyramidIdentifier)

        var unknown = descriptor
        var unknownConfiguration = architecture.descriptor
        unknownConfiguration["id"] = "unregistered-network"
        unknown["architecture"] = "unregistered-network"; unknown["architecture_config"] = unknownConfiguration
        XCTAssertThrowsError(try TrainingRunFiles.nativeArchitecture(from: unknown, map: .depth))
        XCTAssertNil(TrainingRunFiles.settings(from: settings, descriptor: unknown))

        var mismatched = descriptor
        mismatched["architecture"] = NativeArchitecture.identifier
        XCTAssertThrowsError(try TrainingRunFiles.nativeArchitecture(from: mismatched, map: .depth))
        XCTAssertNil(TrainingRunFiles.settings(from: settings, descriptor: mismatched))

        var incorrectScales = descriptor
        var alteredConfiguration = architecture.descriptor
        alteredConfiguration["scale_channels"] = [8, 16, 64]
        incorrectScales["architecture_config"] = alteredConfiguration
        XCTAssertThrowsError(try TrainingRunFiles.nativeArchitecture(from: incorrectScales, map: .depth))
        var inconsistentSettings = settings; inconsistentSettings["model_architecture"] = NativeArchitecture.identifier
        XCTAssertNil(TrainingRunFiles.settings(from: inconsistentSettings, descriptor: descriptor))
        var inconsistentWidth = settings; inconsistentWidth["width"] = 16
        XCTAssertNil(TrainingRunFiles.settings(from: inconsistentWidth, descriptor: descriptor))
        var inconsistentBlocks = settings; inconsistentBlocks["blocks"] = 2
        XCTAssertNil(TrainingRunFiles.settings(from: inconsistentBlocks, descriptor: descriptor))
        var conflictingAliases = settings; conflictingAliases["modelArchitecture"] = NativeArchitecture.identifier
        XCTAssertNil(TrainingRunFiles.settings(from: conflictingAliases, descriptor: descriptor))
    }

    private func verifyExport(_ artifact: ModelArtifact, architecture: NativeArchitecture,
                              weights: [String: NativeTensor]) async throws {
        let compiled = try await MLModel.compileModel(at: artifact.directory.appendingPathComponent(try XCTUnwrap(artifact.coreMLPath)))
        defer { try? FileManager.default.removeItem(at: compiled) }
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        let session = try NativeTrainingSession(architecture: architecture,
            options: NativeTrainingOptions(), seed: 7, weights: weights)
        let rgb = try MLMultiArray(shape: [1, 3, 32, 32], dataType: .float32)
        let input = (0..<rgb.count).map { Float(($0 * 7 + 31) % 255) / 255 }
        for index in input.indices { rgb[index] = NSNumber(value: input[index]) }
        let prediction = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["rgb": rgb]))
        let actual = try XCTUnwrap(prediction.featureValue(for: "material_map")?.multiArrayValue)
        let expected = try session.predict(input: input, batchSize: 1, height: 32, width: 32)
        XCTAssertEqual(actual.count, expected.count)
        for index in expected.indices { XCTAssertEqual(actual[index].floatValue, expected[index], accuracy: 3e-5) }
    }

    private func fixtureImages(in root: URL) throws -> [URL: Data] {
        var originals: [URL: Data] = [:]
        for (name, role) in [("brick_diff.png", MapRole.diffuse), ("brick_disp.png", MapRole.depth)] {
            let byteCount = 32 * 32 * 3
            let bytes: [UInt8] = (0..<byteCount).map { index in
                let position: Int = role == .depth ? index / 3 : index
                let code = (position * 7 + 31) % 255
                return UInt8(code)
            }
            let provider = try XCTUnwrap(CGDataProvider(data: Data(bytes) as CFData))
            let image = try XCTUnwrap(CGImage(width: 32, height: 32, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: 96, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [], provider: provider,
                decode: nil, shouldInterpolate: false, intent: .defaultIntent))
            let file = root.appendingPathComponent(name)
            let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
            var metadata = ImageMetadata(pairingKey: "pyramid-workflow", role: role, category: "Bricks")
            metadata.roleWasManuallySelected = true; metadata.pairingWasManuallySelected = true
            try ImageMetadataStore.write(metadata, to: file)
            originals[file] = try Data(contentsOf: file)
        }
        return originals
    }
}
