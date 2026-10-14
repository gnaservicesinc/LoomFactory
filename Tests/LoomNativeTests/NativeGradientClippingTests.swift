import Foundation
import Metal
import XCTest
@testable import LoomNative

final class NativeGradientClippingTests: XCTestCase {
    func testWarmupUsesCompletedOptimizerUpdatesAndKeepsFixedAndOffBehavior() throws {
        let warmup = try NativeTrainingOptions(gradientClipNorm: 1, gradientClipPolicy: "warmup",
            gradientClipWarmupStartNorm: 0.25, gradientClipWarmupUpdates: 4)
        XCTAssertEqual(warmup.gradientClipThreshold(updateCount: 0), 0.25)
        XCTAssertEqual(warmup.gradientClipThreshold(updateCount: 2), 0.625)
        XCTAssertEqual(warmup.gradientClipThreshold(updateCount: 4), 1)
        XCTAssertEqual(warmup.gradientClipThreshold(updateCount: Int.max), 1)
        let fixed = try NativeTrainingOptions(gradientClipNorm: 0.1)
        XCTAssertEqual(fixed.gradientClipThreshold(updateCount: 0), 0.1)
        XCTAssertEqual(fixed.gradientClipThreshold(updateCount: Int.max), 0.1)
        for policy in ["fixed", "warmup", "off"] {
            let zero = try NativeTrainingOptions(gradientClipNorm: 0, gradientClipPolicy: policy)
            XCTAssertEqual(zero.gradientClipThreshold(updateCount: 0), 0)
        }
        let off = try NativeTrainingOptions(gradientClipNorm: .nan, gradientClipPolicy: "off",
            gradientClipWarmupStartNorm: .nan, gradientClipWarmupUpdates: -1)
        XCTAssertEqual(off.gradientClipThreshold(updateCount: 0), 0)
        XCTAssertThrowsError(try NativeTrainingOptions(gradientClipPolicy: "unknown"))
        XCTAssertThrowsError(try NativeTrainingOptions(gradientClipPolicy: "warmup", gradientClipWarmupStartNorm: 2))
        XCTAssertThrowsError(try NativeTrainingOptions(gradientClipPolicy: "warmup", gradientClipWarmupUpdates: 0))
    }

    func testAccumulatedGradientsUseOneGlobalDirectionPreservingScaleBeforeSGD() throws {
        try available()
        let options = try NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0,
            gradientClipNorm: 0.02)
        let model = try session(options: options)
        let input = sample(3 * 5 * 5)
        let highTarget = Array(repeating: Float(0.9), count: 5 * 5)
        let lowTarget = Array(repeating: Float(0.2), count: 5 * 5)
        _ = try model.train(input: input, target: highTarget, batchSize: 1, height: 5, width: 5, weight: 0.75)
        _ = try model.train(input: input, target: lowTarget, batchSize: 1, height: 5, width: 5, weight: 0.25)
        XCTAssertEqual(model.updateCount, 0, "Microbatches are not optimizer updates")
        try checkGlobalSGDUpdate(model, threshold: 0.02)
    }

    func testWarmupThresholdUsesRestoredCountAndAdvancesOnlyAfterApplyingAccumulation() throws {
        try available()
        let options = try NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0,
            gradientClipNorm: 0.02, gradientClipPolicy: "warmup",
            gradientClipWarmupStartNorm: 0.004, gradientClipWarmupUpdates: 4)
        let model = try session(options: options, initialUpdateCount: 3)
        let input = sample(3 * 5 * 5), target = Array(repeating: Float(0.9), count: 5 * 5)
        for _ in 0..<2 {
            _ = try model.train(input: input, target: target, batchSize: 1, height: 5, width: 5, weight: 0.5)
        }
        XCTAssertEqual(model.updateCount, 3)
        try checkGlobalSGDUpdate(model, threshold: 0.016)
        XCTAssertEqual(model.updateCount, 4)
        _ = try model.train(input: input, target: target, batchSize: 1, height: 5, width: 5)
        try checkGlobalSGDUpdate(model, threshold: 0.02)
        XCTAssertEqual(model.updateCount, 5)
    }

    private func available() throws {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") }
    }
    private func sample(_ count: Int) -> [Float] { (0..<count).map { Float($0 % 17) / 16 } }
    private func session(options: NativeTrainingOptions, initialUpdateCount: Int = 0) throws -> NativeTrainingSession {
        try NativeTrainingSession(architecture: NativeArchitecture(mapKind: "depth", width: 2, blocks: 1),
            options: options, seed: 42, initialUpdateCount: initialUpdateCount)
    }
    private func checkGlobalSGDUpdate(_ model: NativeTrainingSession, threshold: Double) throws {
        let before = try model.weights(), gradients = try model.accumulatedGradients()
        let squaredNorm = gradients.values.reduce(0.0) { sum, tensor in
            sum + tensor.values.reduce(0.0) { $0 + Double($1) * Double($1) }
        }
        let norm = sqrt(squaredNorm)
        XCTAssertGreaterThan(norm, threshold, "This fixture must actually trigger clipping")
        XCTAssertGreaterThan(gradients.values.filter { $0.values.contains { $0 != 0 } }.count, 1,
            "The fixture needs multiple gradient tensors to distinguish global from per-tensor clipping")
        let scale = Float(threshold / (norm + 1e-6)), rate: Float = 0.01
        XCTAssertEqual(try model.applyGradients(learningRate: Double(rate)), norm, accuracy: 1e-6)
        let after = try model.weights()
        for name in before.keys {
            for index in before[name]!.values.indices {
                let expected = before[name]!.values[index] - rate * scale * gradients[name]!.values[index]
                XCTAssertEqual(after[name]!.values[index], expected, accuracy: 2e-7, name)
            }
        }
        XCTAssertTrue(try model.accumulatedGradients().values.allSatisfy { $0.values.allSatisfy { $0 == 0 } })
    }
}
