import CoreML
import Metal
import XCTest
@testable import LoomNative

final class NativePyramidTests: XCTestCase {
    func testArchitectureIdentityShapesAndParameterCountAreExplicit() throws {
        let architecture = try NativeArchitecture(mapKind: "depth", width: 38, blocks: 4,
                                                  identifier: NativeArchitecture.pyramidIdentifier)
        XCTAssertEqual(architecture.identifier, NativeArchitecture.pyramidIdentifier)
        XCTAssertEqual(architecture.parameterCount, 1_926_373)
        XCTAssertEqual(architecture.parameterShapes["encoder.half.weight"], [76, 38, 3, 3])
        XCTAssertEqual(architecture.parameterShapes["encoder.quarter.weight"], [152, 76, 3, 3])
        XCTAssertEqual(architecture.parameterShapes["body.0.conv1.weight"], [152, 152, 3, 3])
        XCTAssertEqual(architecture.parameterShapes["decoder.full.weight"], [38, 76, 3, 3])
        XCTAssertEqual(architecture.descriptor["id"] as? String, NativeArchitecture.pyramidIdentifier)
        XCTAssertEqual(architecture.descriptor["scale_channels"] as? [Int], [38, 76, 152])
        XCTAssertEqual(architecture.convolutionStride("encoder.half"), 2)
        XCTAssertEqual(architecture.convolutionStride("decoder.half"), 1)
        let legacy = try NativeArchitecture(mapKind: "depth", width: 2, blocks: 1)
        XCTAssertEqual(legacy.identifier, NativeArchitecture.identifier)
        XCTAssertEqual(legacy.parameterCount, 151)
        XCTAssertNil(legacy.parameterShapes["encoder.half.weight"])
        XCTAssertThrowsError(try NativeArchitecture(mapKind: "depth", width: 38, blocks: 4,
                                                    identifier: "loom-factory-material-unknown-v1"))
    }

    func testAllMapExportsMatchNativePredictionForOddRectangles() throws {
        try available()
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for kind in ["depth", "roughness", "normal"] {
            let architecture = try pyramid(kind)
            let model = try NativeTrainingSession(architecture: architecture,
                options: NativeTrainingOptions(), seed: 42)
            let weights = try model.weights()
            let compiled = try NativeCoreML.export(weights: weights, architecture: architecture,
                imageSize: 32, to: folder.appendingPathComponent(kind + ".mlmodel"))
            defer { try? FileManager.default.removeItem(at: compiled) }
            let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
            let exported = try MLModel(contentsOf: compiled, configuration: configuration)
            XCTAssertEqual(exported.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
                           ["loom.schema": "loom-factory-model-v1", "loom.architecture": NativeArchitecture.pyramidIdentifier])
            for (width, height) in [(32, 32), (35, 33), (38, 40)] {
                let input = sample(3 * height * width)
                let expected = try model.predict(input: input, batchSize: 1, height: height, width: width)
                let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
                array.dataPointer.assumingMemoryBound(to: Float.self).update(from: input, count: input.count)
                let provider = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)])
                let output = try XCTUnwrap(exported.prediction(from: provider).featureValue(for: "material_map")?.multiArrayValue)
                XCTAssertEqual(output.shape.map(\.intValue), [1, architecture.outputChannels, height, width])
                let maximumError = expected.indices.reduce(Float(0)) { max($0, abs(expected[$1] - output[$1].floatValue)) }
                XCTAssertLessThan(maximumError, 2e-5, "\(kind) \(width)x\(height) native/export parity")
            }
        }
    }

    func testAllMapTrainingOnOddRectanglesProducesFiniteUpdates() throws {
        try available()
        let height = 33, width = 35
        for kind in ["depth", "roughness", "normal"] {
            let architecture = try pyramid(kind)
            let options = try NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0, gradientClipNorm: 0)
            let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 42)
            let input = sample(3 * height * width)
            var target = Array(repeating: Float(kind == "roughness" ? 0.2 : 0.9), count: architecture.outputChannels * height * width)
            if kind == "normal" {
                target = Array(repeating: Float(0.5), count: 2 * height * width)
                    + Array(repeating: Float(1), count: height * width)
            }
            let before = try model.weights()
            let loss = try model.train(input: input, target: target, batchSize: 1, height: height, width: width)
            let norm = try model.applyGradients(learningRate: 0.002)
            XCTAssertTrue(loss.isFinite); XCTAssertTrue(norm.isFinite); XCTAssertGreaterThan(norm, 0)
            XCTAssertNotEqual(try model.weights(), before, "\(kind) must make a real optimizer update")
            XCTAssertEqual(model.updateCount, 1)
        }
    }

    func testPyramidLoRAKeepsFrozenBaseAndFusesStridedEncoderUpdates() throws {
        try available()
        let architecture = try pyramid("depth")
        let options = try NativeTrainingOptions(weightDecay: 0, gradientClipNorm: 0)
        let base = try NativeTrainingSession(architecture: architecture, options: options, seed: 42)
        let frozen = try base.weights()
        let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 17,
            weights: frozen, adapter: NativeAdapterConfiguration(rank: 2, alpha: 2))
        let width = 35, height = 33, input = sample(3 * width * height)
        let target = Array(repeating: Float(0.9), count: width * height)
        for _ in 0..<3 {
            _ = try model.train(input: input, target: target, batchSize: 1, height: height, width: width)
            _ = try model.applyGradients(learningRate: 0.002)
        }
        XCTAssertEqual(try model.weights(), frozen)
        let adapters = try model.adapterWeights()
        XCTAssertTrue(adapters["encoder.half.up.weight"]!.values.contains { $0 != 0 })
        XCTAssertTrue(adapters["encoder.quarter.up.weight"]!.values.contains { $0 != 0 })
        let fused = try NativeTrainingSession(architecture: architecture, options: options, seed: 99, weights: model.fusedWeights())
        let expected = try model.predict(input: input, batchSize: 1, height: height, width: width)
        let actual = try fused.predict(input: input, batchSize: 1, height: height, width: width)
        XCTAssertLessThan(zip(expected, actual).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }, 2e-5)
    }

    func testStridedEncoderAndNearestDecoderGradientsMatchFiniteDifferences() throws {
        try available()
        let architecture = try pyramid("depth")
        let options = try NativeTrainingOptions(weightDecay: 0, gradientClipNorm: 0)
        let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 42)
        let input = sample(3 * 32 * 32), target = Array(repeating: Float(0.9), count: 32 * 32)
        let weights = try model.weights()
        _ = try model.train(input: input, target: target, batchSize: 1, height: 32, width: 32)
        let gradients = try model.accumulatedGradients()
        for name in ["encoder.half.weight", "encoder.quarter.weight", "body.0.conv2.weight",
                     "decoder.half.weight", "decoder.full.weight"] {
            let gradient = gradients[name]!.values
            let index = gradient.indices.max { abs(gradient[$0]) < abs(gradient[$1]) }!
            XCTAssertGreaterThan(abs(gradient[index]), 1e-6, name + " must receive a teaching signal")
            let delta: Float = 0.001
            var losses: [Double] = []
            for sign: Float in [-1, 1] {
                var perturbed = weights
                var tensor = weights[name]!
                tensor.values[index] += sign * delta
                perturbed[name] = tensor
                let probe = try NativeTrainingSession(architecture: architecture, options: options, seed: 0, weights: perturbed)
                losses.append(try probe.evaluate(input: input, target: target, batchSize: 1, height: 32, width: 32))
            }
            let numerical = (losses[1] - losses[0]) / (2 * Double(delta))
            XCTAssertEqual(Double(gradient[index]), numerical, accuracy: 0.0005, name + " gradient")
        }
    }

    private func available() throws {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") }
    }
    private func pyramid(_ kind: String) throws -> NativeArchitecture {
        try NativeArchitecture(mapKind: kind, width: 2, blocks: 1, identifier: NativeArchitecture.pyramidIdentifier)
    }
    private func sample(_ count: Int) -> [Float] {
        (0..<count).map { Float(($0 * 7 + 3) % 101) / 100 }
    }
}
