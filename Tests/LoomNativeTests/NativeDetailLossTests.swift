import Metal
import XCTest
@testable import LoomNative

final class NativeDetailLossTests: XCTestCase {
    private func session(_ loss: NativeScalarLoss, width: Int = 3, weights: [String: NativeTensor]? = nil) throws -> NativeTrainingSession {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") }
        return try NativeTrainingSession(architecture: NativeArchitecture(mapKind: "depth", width: width, blocks: 1),
            options: NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0,
                gradientClipNorm: 0, scalarLoss: loss), seed: 42, weights: weights)
    }

    func testObjectiveMatchesIndependentSignedPixelDifferencesOnRectanglesAndTinyImages() throws {
        for (height, width) in [(7, 9), (1, 1), (1, 5), (3, 2)] {
            let model = try session(.nativeDetailV3)
            let legacy = try session(.logitCrossEntropyV2)
            let input = (0..<(3 * width * height)).map { Float(($0 * 7) % 19) / 18 }
            let target = (0..<(width * height)).map { Float(($0 * 11) % 23) / 22 }
            let predicted = try model.predict(input: input, batchSize: 1, height: height, width: width)
            XCTAssertEqual(predicted, try legacy.predict(input: input, batchSize: 1, height: height, width: width))
            var terms: [Double] = []
            for distance in [1, 2, 4] {
                for horizontal in [false, true] {
                    var sum = 0.0, count = 0
                    for y in 0..<height { for x in 0..<width {
                        let otherX = x + (horizontal ? distance : 0), otherY = y + (horizontal ? 0 : distance)
                        guard otherX < width, otherY < height else { continue }
                        let a = y * width + x, b = otherY * width + otherX
                        sum += abs((Double(predicted[b]) - Double(predicted[a])) - (Double(target[b]) - Double(target[a])))
                        count += 1
                    } }
                    if count > 0 { terms.append(sum / Double(count)) }
                }
            }
            let detail = terms.isEmpty ? 0 : terms.reduce(0, +) / Double(terms.count)
            let level = try legacy.evaluate(input: input, target: target, batchSize: 1, height: height, width: width)
            XCTAssertEqual(try model.evaluate(input: input, target: target, batchSize: 1, height: height, width: width),
                level + 2 * detail, accuracy: 2e-6)
        }
    }

    func testWrongSaturationRetainsCorrectiveGradient() throws {
        let architecture = try NativeArchitecture(mapKind: "depth", width: 3, blocks: 1)
        for bias: Float in [-80, 80] {
            var weights = Dictionary(uniqueKeysWithValues: architecture.parameterShapes.map {
                ($0.key, NativeTensor(shape: $0.value, values: Array(repeating: Float(0), count: $0.value.reduce(1, *))))
            })
            weights["output.bias"]!.values = [bias]
            let model = try session(.nativeDetailV3, weights: weights)
            let input = Array(repeating: Float(0), count: 3 * 9 * 7)
            let target = (0..<63).map { Float($0 % 7) / 6 }
            let before = try model.train(input: input, target: target, batchSize: 1, height: 7, width: 9)
            XCTAssertEqual(try model.accumulatedGradients()["output.bias"]!.values[0], bias > 0 ? 0.5 : -0.5, accuracy: 1e-6)
            _ = try model.applyGradients(learningRate: 0.1)
            XCTAssertLessThan(try model.evaluate(input: input, target: target, batchSize: 1, height: 7, width: 9), before)
        }
    }

    func testPyramidDetailTrainingResumesExactlyAndKeepsInferenceIdentity() throws {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") }
        let architecture = try NativeArchitecture(mapKind: "depth", width: 3, blocks: 1,
            identifier: NativeArchitecture.pyramidIdentifier)
        let options = try NativeTrainingOptions(weightDecay: 0, scalarLoss: .nativeDetailV3)
        let input = (0..<(3 * 9 * 7)).map { Float(($0 * 7) % 19) / 18 }
        let target = (0..<(9 * 7)).map { Float(($0 * 11) % 23) / 22 }
        let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 42)
        for _ in 0..<2 {
            _ = try model.train(input: input, target: target, batchSize: 1, height: 7, width: 9)
            _ = try model.applyGradients(learningRate: 0.001)
        }
        let resumed = try NativeTrainingSession(architecture: architecture, options: options, seed: 9,
            weights: model.weights(), optimizerState: model.optimizerState(), initialUpdateCount: model.updateCount)
        let inference = try NativeTrainingSession(architecture: architecture,
            options: NativeTrainingOptions(scalarLoss: .logitCrossEntropyV2), seed: 1, weights: model.weights())
        XCTAssertEqual(try model.predict(input: input, batchSize: 1, height: 7, width: 9),
            try inference.predict(input: input, batchSize: 1, height: 7, width: 9))
        for candidate in [model, resumed] {
            let loss = try candidate.train(input: input, target: target, batchSize: 1, height: 7, width: 9)
            let norm = try candidate.applyGradients(learningRate: 0.001)
            XCTAssertTrue(loss.isFinite); XCTAssertTrue(norm.isFinite); XCTAssertGreaterThan(norm, 0)
        }
        XCTAssertEqual(try model.weights(), try resumed.weights())
        XCTAssertEqual(try model.optimizerState(), try resumed.optimizerState())
    }

    func testDetailGradientMatchesNumericalDerivative() throws {
        let model = try session(.nativeDetailV3)
        let input = (0..<(3 * 7 * 9)).map { Float(($0 * 7) % 19) / 18 }
        let target = (0..<(7 * 9)).map { Float(($0 * 11) % 23) / 22 }
        let weights = try model.weights()
        _ = try model.train(input: input, target: target, batchSize: 1, height: 7, width: 9)
        let gradients = try model.accumulatedGradients()
        for name in ["input.weight", "body.0.conv2.weight", "output.weight", "output.bias"] {
            let gradient = gradients[name]!.values
            let index = gradient.indices.max { abs(gradient[$0]) < abs(gradient[$1]) }!
            var loss: [Double] = []
            let delta: Float = 0.0002
            for sign: Float in [-1, 1] {
                var perturbed = weights
                perturbed[name]!.values[index] += delta * sign
                let candidate = try session(.nativeDetailV3, weights: perturbed)
                loss.append(try candidate.evaluate(input: input, target: target, batchSize: 1, height: 7, width: 9))
            }
            XCTAssertEqual(Double(gradient[index]), (loss[1] - loss[0]) / (2 * Double(delta)), accuracy: 0.0015, name)
        }
    }
}
