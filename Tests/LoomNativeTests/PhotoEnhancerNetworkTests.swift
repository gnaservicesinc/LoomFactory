import CoreML
import Darwin
import LoomCore
import XCTest
@testable import LoomNative

final class PhotoEnhancerNetworkTests: XCTestCase {
    func testProductAndConstructorContractsCannotCrossWireMaterialAndRGB() throws {
        for product in NativeModelProduct.allCases {
            let identifier = product.isPhoto ? NativeArchitecture.photoCompactIdentifier : NativeArchitecture.compactIdentifier
            let architecture = try NativeArchitecture(mapKind: product.rawValue, width: 32, blocks: 1, identifier: identifier)
            XCTAssertEqual(architecture.outputChannels, product.channels)
            XCTAssertEqual(architecture.outputName, product.outputName)
            XCTAssertEqual(architecture.parameterCount, product.channels == 3 ? 1_000_515 : 999_937)
            XCTAssertThrowsError(try NativeArchitecture(mapKind: product.rawValue, width: 32, blocks: 1,
                identifier: product.isPhoto ? NativeArchitecture.compactIdentifier : NativeArchitecture.photoCompactIdentifier))
        }
    }
    func testCPUExportStartsAtIdentityAndKeepsUnboundedRGBResidualMeaning() throws {
        for product in [NativeModelProduct.photoRepair, .photoLighting] {
            let architecture = try NativeArchitecture(mapKind: product.rawValue, width: 2, blocks: 1, identifier: NativeArchitecture.photoCompactIdentifier)
            var weights = NativeCompactNetwork.initialWeights(architecture: architecture, seed: 77)
            XCTAssertTrue(weights["output.weight"]!.values.allSatisfy { $0 == 0 })
            XCTAssertEqual(weights["output.bias"]!.values, [0,0,0])
            let input = (0..<3*32*32).map { Float(($0*31)%251)/250 }
            XCTAssertEqual(try cpu(weights, architecture, input, 32).map(\.bitPattern), input.map(\.bitPattern))
            weights["output.bias"]!.values = [0.2,-0.3,0.4]
            let output = try cpu(weights, architecture, input, 32)
            for i in input.indices { XCTAssertEqual(output[i], input[i] + [Float(0.2),-0.3,0.4][i/(32*32)], accuracy: 1e-6) }
            XCTAssertTrue(output.contains { $0 < 0 }); XCTAssertTrue(output.contains { $0 > 1 })
        }
    }
    func testNative1KPhotoUpdateResumeAndCoreMLParityWhenExplicitlyEnabled() throws {
        guard ProcessInfo.processInfo.environment["LOOM_PHOTO_GPU_1K"] == "1" else { throw XCTSkip("Opt-in native 1K GPU update/resume/CoreML parity; do not run alongside material training.") }
        let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["LOOM_PHOTO_GPU_PROJECT"] ?? "/opt/ipde/sources_mats/.loom-factory")
        let lease: TrainingQueueLease
        do { lease = try TrainingQueueLease(projectFolder:root,controlFolder:root) }
        catch { throw XCTSkip("Material project is busy; no GPU test started: \(error.localizedDescription)") }
        defer { withExtendedLifetime(lease) {} }
        let architecture = try NativeArchitecture(mapKind: NativeModelProduct.photoRepair.rawValue, width: 16, blocks: 1, identifier: NativeArchitecture.photoCompactIdentifier)
        let options = try NativeTrainingOptions(weightDecay: 0, scalarLoss: .linearDetailV1)
        let input = (0..<3*1024*1024).map { Float($0%131)/130 }, target = input.map { $0*0.8+0.05 }
        let session = try NativeTrainingSession(architecture: architecture, options: options, seed: 77)
        XCTAssertEqual(try session.predict(input: input, batchSize: 1, height: 1024, width: 1024), input)
        let loss = try session.train(input: input, target: target, batchSize: 1, height: 1024, width: 1024)
        XCTAssertTrue(loss.isFinite); _ = try session.applyGradients(learningRate: 0.0001)
        let weights = try session.weights()
        let predicted = try session.predict(input: input, batchSize: 1, height: 1024, width: 1024)
        XCTAssertNotEqual(predicted, input)
        let restored = try NativeTrainingSession(architecture: architecture, options: options, seed: 77, weights: weights,
            optimizerState: session.optimizerState(), initialUpdateCount: 1)
        XCTAssertEqual(try restored.predict(input: input, batchSize: 1, height: 1024, width: 1024), predicted)
        let cpuValues = try cpu(weights, architecture, input, 1024)
        XCTAssertLessThan(zip(cpuValues, predicted).map { abs($0-$1) }.max()!, 0.0002)
        for worker in [session, restored] {
            _ = try worker.train(input: target, target: input, batchSize: 1, height: 1024, width: 1024)
            _ = try worker.applyGradients(learningRate: 0.0001)
        }
        XCTAssertEqual(try session.weights(), try restored.weights())
    }
    private func cpu(_ weights: [String: NativeTensor], _ architecture: NativeArchitecture, _ input: [Float], _ size: Int) throws -> [Float] {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let compiled = try NativeCoreML.export(weights: weights, architecture: architecture, imageSize: size, to: directory.appendingPathComponent("model.mlmodel"))
        defer { try? FileManager.default.removeItem(at: compiled) }
        let config = MLModelConfiguration(); config.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiled, configuration: config)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
        XCTAssertEqual(metadata?["loom.product"], architecture.mapKind)
        XCTAssertNil(model.modelDescription.outputDescriptionsByName["material_map"])
        let array = try MLMultiArray(shape: [1,3,NSNumber(value:size),NSNumber(value:size)], dataType: .float32)
        input.withUnsafeBufferPointer { array.dataPointer.assumingMemoryBound(to: Float.self).update(from: $0.baseAddress!, count: input.count) }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)])
        let output = try XCTUnwrap(model.prediction(from: provider).featureValue(for: "enhanced_rgb")?.multiArrayValue)
        let p = output.dataPointer.assumingMemoryBound(to: Float.self), strides = output.strides.map(\.intValue)
        return (0..<3).flatMap { c in (0..<size*size).map { i in p[c*strides[1]+(i/size)*strides[2]+(i%size)*strides[3]] } }
    }
}
