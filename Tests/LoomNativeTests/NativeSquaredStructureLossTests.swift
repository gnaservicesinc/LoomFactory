import Metal
import XCTest
@testable import LoomNative

final class NativeSquaredStructureLossTests: XCTestCase {
    private func session(_ loss: NativeScalarLoss, map: String = "depth",
                         weights: [String: NativeTensor]? = nil) throws -> NativeTrainingSession {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Apple GPU required") }
        return try NativeTrainingSession(architecture: NativeArchitecture(mapKind: map, width: 2, blocks: 1,
            identifier: NativeArchitecture.compactIdentifier),
            options: NativeTrainingOptions(weightDecay: 0, gradientClipNorm: 0, scalarLoss: loss),
            seed: 37, weights: weights)
    }

    func testNativeOffsetsIncluding64MatchIndependentPixelReferenceAndGradient() throws {
        let size = 128
        let input = (0..<(3 * size * size)).map { Float(($0 * 7) % 251) / 250 }
        let target = (0..<(size * size)).map { Float(($0 / 13) % 17) / 16 }
        let model = try session(.squaredStructureV1)
        let weights = try model.weights()
        let prediction = try model.predict(input: input, batchSize: 1, height: size, width: size)
        let error = zip(prediction, target).map { Double($0) - Double($1) }
        var reference = error.reduce(0) { $0 + $1 * $1 } / Double(error.count)
        var differences: [Double] = []
        for offset in [1, 4, 16, 64] {
            for axis in [0, 1] {
                var sum = 0.0, count = 0
                for y in 0..<(size - (axis == 0 ? offset : 0)) {
                    for x in 0..<(size - (axis == 1 ? offset : 0)) {
                        let i = y * size + x
                        let delta = error[i + (axis == 0 ? offset * size : offset)] - error[i]
                        sum += delta * delta; count += 1
                    }
                }
                differences.append(sum / Double(count))
            }
        }
        reference += 4 * differences.reduce(0, +) / Double(differences.count)
        XCTAssertEqual(try model.evaluate(input: input, target: target, batchSize: 1, height: size, width: size), reference, accuracy: 2e-5)
        _ = try model.train(input: input, target: target, batchSize: 1, height: size, width: size)
        let gradient = try model.accumulatedGradients()["output.weight"]!.values
        let index = gradient.indices.max { abs(gradient[$0]) < abs(gradient[$1]) }!
        let step: Float = 0.0003
        var losses: [Double] = []
        for sign: Float in [-1, 1] {
            var perturbed = weights
            perturbed["output.weight"]!.values[index] += sign * step
            let candidate = try session(.squaredStructureV1, weights: perturbed)
            losses.append(try candidate.evaluate(input: input, target: target, batchSize: 1, height: size, width: size))
        }
        XCTAssertEqual(Double(gradient[index]), (losses[1] - losses[0]) / (2 * Double(step)), accuracy: 0.003)
    }

    func testObjectiveChangesNeitherInferenceNorLegacyChoiceAndExactMatchHasZeroLoss() throws {
        for map in ["depth", "normal", "roughness"] {
            let original = try session(.linearDetailV1, map: map)
            let alternate = try session(.squaredStructureV1, map: map, weights: original.weights())
            let input = (0..<(3 * 32 * 32)).map { Float(($0 * 11) % 251) / 250 }
            let prediction = try original.predict(input: input, batchSize: 1, height: 32, width: 32)
            XCTAssertEqual(try alternate.predict(input: input, batchSize: 1, height: 32, width: 32), prediction)
            var exactWeights = try original.weights()
            exactWeights["output.weight"]!.values = Array(repeating: 0, count: exactWeights["output.weight"]!.values.count)
            exactWeights["output.bias"]!.values = map == "normal" ? [0.5, 0.5, 1] : [0.5]
            let exact = try session(.squaredStructureV1, map: map, weights: exactWeights)
            let validTarget = try exact.predict(input: input, batchSize: 1, height: 32, width: 32)
            XCTAssertEqual(try exact.evaluate(input: input, target: validTarget, batchSize: 1, height: 32, width: 32), 0, accuracy: 1e-12)
        }
        XCTAssertEqual(try NativeTrainingOptions().scalarLoss, .absoluteV1)
    }
}
