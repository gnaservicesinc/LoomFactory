import Foundation
import CryptoKit
import Metal
import XCTest
@testable import LoomNative

final class NativeScalarLossTests: XCTestCase {
    private func available() throws {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") }
    }
    private func options(_ loss: NativeScalarLoss) throws -> NativeTrainingOptions {
        try NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0,
            gradientClipNorm: 0, scalarLoss: loss)
    }
    private func zeroWeights(_ architecture: NativeArchitecture, bias: Float) -> [String: NativeTensor] {
        var weights = Dictionary(uniqueKeysWithValues: architecture.parameterShapes.map { name, shape in
            (name, NativeTensor(shape: shape, values: Array(repeating: Float(0), count: shape.reduce(1, *))))
        })
        weights["output.bias"]!.values = Array(repeating: bias, count: architecture.outputChannels)
        return weights
    }

    func testVersionedLossDefaultsPreserveLegacyContract() throws {
        XCTAssertEqual(try NativeTrainingOptions().scalarLoss, .absoluteV1)
        for loss in NativeScalarLoss.allCases {
            XCTAssertEqual(try JSONDecoder().decode(NativeScalarLoss.self, from: JSONEncoder().encode(loss)), loss)
        }
        XCTAssertThrowsError(try JSONDecoder().decode(NativeScalarLoss.self, from: Data("\"unknown-loss-v3\"".utf8)))
    }

    func testWhiteAndBlackSaturationHaveFiniteCorrectiveGradients() throws {
        try available()
        let input = Array(repeating: Float(0), count: 3 * 4 * 4)
        for identifier in [NativeArchitecture.identifier, NativeArchitecture.pyramidIdentifier] {
            let architecture = try NativeArchitecture(mapKind: "roughness", width: 2, blocks: 1, identifier: identifier)
            for (bias, targetValue, expectedGradient): (Float, Float, Float) in [(80, 0.25, 0.75), (-80, 0.75, -0.75)] {
                let target = Array(repeating: targetValue, count: 4 * 4)
                let weights = zeroWeights(architecture, bias: bias)
                let legacy = try NativeTrainingSession(architecture: architecture, options: options(.absoluteV1), seed: 1, weights: weights)
                let stable = try NativeTrainingSession(architecture: architecture, options: options(.logitCrossEntropyV2), seed: 1, weights: weights)
                XCTAssertEqual(try legacy.predict(input: input, batchSize: 1, height: 4, width: 4),
                    try stable.predict(input: input, batchSize: 1, height: 4, width: 4), "The loss version cannot alter inference")
                _ = try legacy.train(input: input, target: target, batchSize: 1, height: 4, width: 4)
                XCTAssertEqual(try legacy.accumulatedGradients()["output.bias"]!.values[0], 0, accuracy: 1e-30)
                let loss = try stable.train(input: input, target: target, batchSize: 1, height: 4, width: 4)
                XCTAssertEqual(loss, 60, accuracy: 1e-5)
                XCTAssertEqual(try stable.accumulatedGradients()["output.bias"]!.values[0], expectedGradient, accuracy: 1e-6)
                XCTAssertEqual(try stable.applyGradients(learningRate: 0.01), 0.75, accuracy: 1e-6)
                XCTAssertLessThan(try stable.evaluate(input: input, target: target, batchSize: 1, height: 4, width: 4), loss)
            }
        }
    }

    func testZeroLogitAndExtremeValuesMatchIndependentEquations() throws {
        try available()
        let architecture = try NativeArchitecture(mapKind: "depth", width: 2, blocks: 1)
        let input = Array(repeating: Float(0), count: 3 * 4 * 4)
        let nearOne = Float(1).nextDown, nearZero = Float(1) - Float(1).nextDown
        for (bias, targetValue): (Float, Float) in [(0, 0.25), (0, 0.75), (0.7, 0.3), (-1.2, 0.8),
            (10_000, 0.2), (-10_000, 0.8), (10_000, nearOne), (-10_000, nearZero),
            (100_000_000, nearOne), (-100_000_000, nearZero)] {
            let model = try NativeTrainingSession(architecture: architecture,
                options: options(.logitCrossEntropyV2), seed: 1, weights: zeroWeights(architecture, bias: bias))
            let loss = try model.train(input: input, target: Array(repeating: targetValue, count: 16), batchSize: 1, height: 4, width: 4)
            let z = Double(bias), y = Double(targetValue)
            let expectedLoss = max(z, 0) - z * y + log1p(exp(-abs(z)))
            let expectedGradient = 1 / (1 + exp(-z)) - y
            XCTAssertEqual(loss, expectedLoss, accuracy: max(1e-6, abs(expectedLoss) * 1e-6))
            XCTAssertEqual(Double(try model.accumulatedGradients()["output.bias"]!.values[0]), expectedGradient, accuracy: 1e-6)
        }
    }

    func testScalarLossChoicePreservesNormalLossAndGradients() throws {
        try available()
        let architecture = try NativeArchitecture(mapKind: "normal", width: 2, blocks: 1)
        let input = (0..<(3 * 5 * 5)).map { Float($0 % 17) / 16 }
        let target = Array(repeating: Float(0.5), count: 50) + Array(repeating: Float(1), count: 25)
        let legacy = try NativeTrainingSession(architecture: architecture, options: options(.absoluteV1), seed: 42)
        let expectedLoss = try legacy.train(input: input, target: target, batchSize: 1, height: 5, width: 5)
        let expectedGradients = try legacy.accumulatedGradients()
        for loss in NativeScalarLoss.allCases {
            let candidate = try NativeTrainingSession(architecture: architecture, options: options(loss), seed: 42)
            XCTAssertEqual(expectedLoss, try candidate.train(input: input, target: target, batchSize: 1, height: 5, width: 5))
            XCTAssertEqual(expectedGradients, try candidate.accumulatedGradients())
        }
    }

    func testStableLossOptimizerStateResumesExactly() throws {
        try available()
        let architecture = try NativeArchitecture(mapKind: "roughness", width: 3, blocks: 1,
            identifier: NativeArchitecture.pyramidIdentifier)
        let options = try NativeTrainingOptions(weightDecay: 0, scalarLoss: .logitCrossEntropyV2)
        let input = (0..<(3 * 7 * 7)).map { Float($0 % 17) / 16 }
        let target = (0..<(7 * 7)).map { Float($0 % 13) / 12 }
        let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 42)
        _ = try model.train(input: input, target: target, batchSize: 1, height: 7, width: 7)
        _ = try model.applyGradients(learningRate: 0.001)
        let resumed = try NativeTrainingSession(architecture: architecture, options: options, seed: 1,
            weights: model.weights(), optimizerState: model.optimizerState(), initialUpdateCount: model.updateCount)
        for candidate in [model, resumed] {
            _ = try candidate.train(input: input, target: target, batchSize: 1, height: 7, width: 7)
            _ = try candidate.applyGradients(learningRate: 0.001)
        }
        XCTAssertEqual(try model.weights(), try resumed.weights())
        XCTAssertEqual(try model.optimizerState(), try resumed.optimizerState())
    }

    /// An opt-in regression exercises the untouched saved white-map failure.
    /// The small diagnostic crop establishes recoverable gradients, not quality.
    func testSavedCollapsedRoughnessWeightsReceiveTeachingSignal() throws {
        try available()
        guard let directory = ProcessInfo.processInfo.environment["LOOM_NATIVE_ROUGHNESS_RUN_DIRECTORY"],
              let inputPath = ProcessInfo.processInfo.environment["LOOM_NATIVE_ROUGHNESS_INPUT"],
              let targetPath = ProcessInfo.processInfo.environment["LOOM_NATIVE_ROUGHNESS_TARGET"] else {
            throw XCTSkip("Set LOOM_NATIVE_ROUGHNESS_RUN_DIRECTORY, INPUT and TARGET for the saved roughness regression")
        }
        let folder = URL(fileURLWithPath: directory)
        let inputURL = URL(fileURLWithPath: inputPath), targetURL = URL(fileURLWithPath: targetPath)
        let sourceURLs = [inputURL, targetURL, folder.appendingPathComponent("model.safetensors"),
            folder.appendingPathComponent("model.json")]
        func hashes() throws -> [String] {
            try sourceURLs.map { SHA256.hash(data: try Data(contentsOf: $0)).map { String(format: "%02x", $0) }.joined() }
        }
        let originalHashes = try hashes()
        let descriptor = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("model.json"))) as! [String: Any]
        let config = try XCTUnwrap(descriptor["architecture_config"] as? [String: Any])
        let architecture = try NativeArchitecture(mapKind: "roughness", width: try XCTUnwrap(config["width"] as? Int),
            blocks: try XCTUnwrap(config["blocks"] as? Int), identifier: try XCTUnwrap(descriptor["architecture"] as? String))
        let weights = try NativeWeights.read(from: folder.appendingPathComponent("model.safetensors"))
        let pair = try NativeImages.pair(inputURL: inputURL, targetURL: targetURL,
            transform: NativeCropTransform(x: 480, y: 480, size: 64), mapKind: "roughness")
        let legacy = try NativeTrainingSession(architecture: architecture, options: options(.absoluteV1), seed: 1, weights: weights)
        let legacyLoss = try legacy.train(input: pair.input.values, target: pair.target.values, batchSize: 1, height: 64, width: 64)
        let legacyNorm = try legacy.applyGradients(learningRate: 0.0001)
        XCTAssertEqual(legacyNorm, 0, "Fixture must reproduce the old saturated failure")
        let stable = try NativeTrainingSession(architecture: architecture,
            options: NativeTrainingOptions(weightDecay: 0, scalarLoss: .logitCrossEntropyV2), seed: 1, weights: weights)
        let before = try stable.train(input: pair.input.values, target: pair.target.values, batchSize: 1, height: 64, width: 64)
        let norm = try stable.applyGradients(learningRate: 0.0001)
        let after = try stable.evaluate(input: pair.input.values, target: pair.target.values, batchSize: 1, height: 64, width: 64)
        XCTAssertTrue(norm.isFinite); XCTAssertGreaterThan(norm, 1)
        XCTAssertLessThan(after, before)
        XCTAssertNotEqual(try stable.weights(), weights)
        print("Saved roughness recovery: legacy L1=\(legacyLoss), legacy gradient norm=\(legacyNorm), v2 before=\(before), v2 after=\(after), v2 norm=\(norm)")
        for step in 2...100 {
            _ = try stable.train(input: pair.input.values, target: pair.target.values, batchSize: 1, height: 64, width: 64)
            _ = try stable.applyGradients(learningRate: 0.0001)
            if step % 10 == 0 {
                let prediction = try stable.predict(input: pair.input.values, batchSize: 1, height: 64, width: 64)
                let mae = zip(prediction, pair.target.values).reduce(0.0) { $0 + Double(abs($1.0 - $1.1)) } / Double(prediction.count)
                print("Saved roughness recovery step \(step): prediction=[\(prediction.min()!),\(prediction.max()!)], MAE=\(mae)")
            }
        }
        let prediction = try stable.predict(input: pair.input.values, batchSize: 1, height: 64, width: 64)
        let mae = zip(prediction, pair.target.values).reduce(0.0) { $0 + Double(abs($1.0 - $1.1)) } / Double(prediction.count)
        XCTAssertLessThan(prediction.min()!, 0.99, "Copied model must actually escape its white output")
        XCTAssertGreaterThan(prediction.max()! - prediction.min()!, 0.001, "Recovery must produce a varying map")
        XCTAssertLessThan(mae, legacyLoss, "The visible map must improve on the identical source crop")
        XCTAssertEqual(try hashes(), originalHashes, "Saved model and original image bytes must remain untouched")
    }
}
