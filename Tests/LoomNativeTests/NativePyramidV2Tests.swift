import CoreML
import Metal
import XCTest
@testable import LoomNative

final class NativePyramidV2Tests: XCTestCase {
    func testVersionRetainsPyramidContextCapacityAndNativeGrid() throws {
        for map in ["depth", "roughness", "normal"] {
            let old = try NativeArchitecture(mapKind: map, width: 38, blocks: 4, identifier: NativeArchitecture.pyramidIdentifier)
            let new = try NativeArchitecture(mapKind: map, width: 38, blocks: 4, identifier: NativeArchitecture.pyramidV2Identifier)
            XCTAssertTrue(new.isPyramid); XCTAssertTrue(new.usesLeakyActivation)
            XCTAssertFalse(new.isFullResolutionResidual)
            XCTAssertEqual(new.parameterShapes, old.parameterShapes)
            XCTAssertEqual(new.parameterCount, old.parameterCount)
            XCTAssertEqual(new.descriptor["scale_channels"] as? [Int], [38, 76, 152])
            XCTAssertEqual(new.descriptor["negative_slope"] as? Double, 0.1)
            XCTAssertTrue(new.supportsGrid(height: 1025, width: 2049))
            XCTAssertNil(old.descriptor["hidden_activation"])
        }
    }

    func testNegativeFeaturesSurviveAllEncoderBodyAndDecoderStages() throws {
        try available()
        let old = try NativeArchitecture(mapKind: "depth", width: 1, blocks: 1, identifier: NativeArchitecture.pyramidIdentifier)
        let new = try NativeArchitecture(mapKind: "depth", width: 1, blocks: 1, identifier: NativeArchitecture.pyramidV2Identifier)
        var weights = new.parameterShapes.mapValues { NativeTensor(shape: $0, values: Array(repeating: Float(0), count: $0.reduce(1, *))) }
        for name in new.convolutionNames {
            weights[name + ".weight"]!.values[4] = 1
            weights[name + ".bias"]!.values = Array(repeating: -0.1, count: weights[name + ".bias"]!.values.count)
        }
        weights["input.bias"]!.values = [-1]; weights["output.bias"]!.values = [0]
        let options = try NativeTrainingOptions(weightDecay: 0, gradientClipNorm: 0, scalarLoss: .logitCrossEntropyV2)
        let input = Array(repeating: Float(0.25), count: 3 * 33 * 35), target = Array(repeating: Float(0.9), count: 33 * 35)
        let legacy = try NativeTrainingSession(architecture: old, options: options, seed: 42, weights: weights)
        let model = try NativeTrainingSession(architecture: new, options: options, seed: 42, weights: weights)
        XCTAssertTrue(try legacy.predict(input: input, batchSize: 1, height: 33, width: 35).allSatisfy { $0 == 0.5 })
        // Independent scalar center-kernel calculation through all seven stages:
        // full -.075; half -.0175; quarter -.01175; residual -.0122925;
        // decoder half -.01297925; decoder full/logit -.018797925.
        let expected = 1 / (1 + exp(0.018797925))
        for value in try model.predict(input: input, batchSize: 1, height: 33, width: 35) {
            XCTAssertEqual(Double(value), expected, accuracy: 1e-7)
        }
        _ = try legacy.train(input: input, target: target, batchSize: 1, height: 33, width: 35)
        let dead = try legacy.accumulatedGradients()
        XCTAssertTrue(dead["encoder.quarter.weight"]!.values.allSatisfy { $0 == 0 })
        _ = try model.train(input: input, target: target, batchSize: 1, height: 33, width: 35)
        let gradients = try model.accumulatedGradients()
        for name in new.convolutionNames {
            XCTAssertTrue(gradients[name + ".weight"]!.values.contains { $0 != 0 }, name)
        }
        var losses: [Double] = []
        for sign: Float in [-1, 1] {
            var changed = weights; changed["input.bias"]!.values[0] += sign * 0.01
            let candidate = try NativeTrainingSession(architecture: new, options: options, seed: 42, weights: changed)
            losses.append(try candidate.evaluate(input: input, target: target, batchSize: 1, height: 33, width: 35))
        }
        XCTAssertEqual(Double(gradients["input.bias"]!.values[0]), (losses[1] - losses[0]) / 0.02, accuracy: 2e-5)
        _ = try model.applyGradients(learningRate: 0.001)
        let resumed = try NativeTrainingSession(architecture: new, options: options, seed: 9,
            weights: model.weights(), optimizerState: model.optimizerState(), initialUpdateCount: model.updateCount)
        for candidate in [model, resumed] {
            _ = try candidate.train(input: input, target: target, batchSize: 1, height: 33, width: 35)
            _ = try candidate.applyGradients(learningRate: 0.001)
        }
        XCTAssertEqual(try model.weights(), try resumed.weights())
        XCTAssertEqual(try model.optimizerState(), try resumed.optimizerState())
    }

    func testAllThreeProductsExportMatchingCoreMLAtNativeAndOddGrids() throws {
        try available()
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for map in ["depth", "roughness", "normal"] {
            let architecture = try NativeArchitecture(mapKind: map, width: 3, blocks: 2, identifier: NativeArchitecture.pyramidV2Identifier)
            let native = try NativeTrainingSession(architecture: architecture, options: NativeTrainingOptions(), seed: 42)
            let compiled = try NativeCoreML.export(weights: native.weights(), architecture: architecture,
                imageSize: 32, to: folder.appendingPathComponent(map + ".mlmodel"))
            defer { try? FileManager.default.removeItem(at: compiled) }
            let config = MLModelConfiguration(); config.computeUnits = .cpuOnly
            let model = try MLModel(contentsOf: compiled, configuration: config)
            for (height, width) in [(32, 32), (33, 35)] {
                let input = (0..<(3 * height * width)).map { Float(($0 * 13) % 101) / 100 }
                let expected = try native.predict(input: input, batchSize: 1, height: height, width: width)
                let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
                array.dataPointer.assumingMemoryBound(to: Float.self).update(from: input, count: input.count)
                let output = try XCTUnwrap(model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)]))
                    .featureValue(for: "material_map")?.multiArrayValue)
                XCTAssertEqual(output.shape.map(\.intValue), [1, architecture.outputChannels, height, width])
                XCTAssertLessThan(expected.indices.reduce(Float(0)) { max($0, abs(expected[$1] - output[$1].floatValue)) }, 2e-5, map)
            }
        }
    }
    func testLargeWrongNormalLogitsKeepACorrectiveGradient() throws {
        try available()
        let architecture = try NativeArchitecture(mapKind: "normal", width: 2, blocks: 1, identifier: NativeArchitecture.pyramidV2Identifier)
        var weights = architecture.parameterShapes.mapValues { NativeTensor(shape: $0, values: Array(repeating: Float(0), count: $0.reduce(1, *))) }
        weights["output.bias"]!.values = [-80, 80, 80]
        let model = try NativeTrainingSession(architecture: architecture, options: NativeTrainingOptions(), seed: 42, weights: weights)
        let input = Array(repeating: Float(0.4), count: 3 * 32 * 32)
        let target = Array(repeating: Float(0.5), count: 2 * 32 * 32) + Array(repeating: Float(1), count: 32 * 32)
        _ = try model.train(input: input, target: target, batchSize: 1, height: 32, width: 32)
        let gradient = try model.accumulatedGradients()["output.bias"]!.values
        XCTAssertTrue(gradient.allSatisfy { $0.isFinite && $0 != 0 })
        XCTAssertGreaterThan(gradient[1], 0)
        XCTAssertLessThan(gradient[2], 0)
    }
    private func available() throws {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") }
    }
}
