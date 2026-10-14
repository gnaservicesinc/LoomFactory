import Foundation
import CryptoKit
import Metal
import XCTest
@testable import LoomNative

final class NativeTrainingTests: XCTestCase {
    private func available() throws { if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") } }
    private func sample(_ count: Int, offset: Int = 0) -> [Float] {
        (0..<count).map { Float(($0 + offset) % 17) / 16 }
    }
    private func architecture(_ map: String = "depth") throws -> NativeArchitecture {
        try NativeArchitecture(mapKind: map, width: 2, blocks: 1)
    }
    private func session(_ map: String = "depth", optimizer: String = "adamw") throws -> NativeTrainingSession {
        try NativeTrainingSession(architecture: architecture(map),
            options: NativeTrainingOptions(optimizer: optimizer, weightDecay: 0, gradientClipNorm: 0), seed: 42)
    }

    func testActualGPUGradientMatchesFiniteDifferenceAndReducesLoss() throws {
        try available()
        let model = try session()
        let input = sample(3 * 8 * 8), target = Array(repeating: Float(0.9), count: 8 * 8)
        let before = try model.evaluate(input: input, target: target, batchSize: 1, height: 8, width: 8)
        _ = try model.train(input: input, target: target, batchSize: 1, height: 8, width: 8)
        let gradients = try model.accumulatedGradients()
        let base = try model.weights()
        let epsilon: Float = 1e-3
        var plus = base, minus = base
        plus["output.bias"]!.values[0] += epsilon
        minus["output.bias"]!.values[0] -= epsilon
        let positive = try NativeTrainingSession(architecture: model.architecture, options: model.options, seed: 42, weights: plus)
        let negative = try NativeTrainingSession(architecture: model.architecture, options: model.options, seed: 42, weights: minus)
        let positiveLoss = try positive.evaluate(input: input, target: target, batchSize: 1, height: 8, width: 8)
        let negativeLoss = try negative.evaluate(input: input, target: target, batchSize: 1, height: 8, width: 8)
        XCTAssertEqual(Double(gradients["output.bias"]!.values[0]), (positiveLoss - negativeLoss) / Double(2 * epsilon), accuracy: 5e-4)
        let norm = try model.applyGradients(learningRate: 0.01)
        XCTAssertGreaterThan(norm, 0)
        XCTAssertEqual(model.updateCount, 1)
        let after = try model.evaluate(input: input, target: target, batchSize: 1, height: 8, width: 8)
        XCTAssertLessThan(after, before)
    }

    func testAccumulationMatchesCombinedBatchAndAllOptimizersUpdate() throws {
        try available()
        let input = sample(3 * 6 * 6), target = Array(repeating: Float(0.8), count: 6 * 6)
        for optimizer in ["adamw", "adam", "sgd", "lion"] {
            let split = try session(optimizer: optimizer), combined = try session(optimizer: optimizer)
            _ = try split.train(input: input, target: target, batchSize: 1, height: 6, width: 6, weight: 0.5)
            _ = try split.train(input: input, target: target, batchSize: 1, height: 6, width: 6, weight: 0.5)
            _ = try combined.train(input: input + input, target: target + target, batchSize: 2, height: 6, width: 6)
            _ = try split.applyGradients(learningRate: 0.001)
            _ = try combined.applyGradients(learningRate: 0.001)
            let left = try split.weights(), right = try combined.weights()
            for key in left.keys {
                for (a, b) in zip(left[key]!.values, right[key]!.values) { XCTAssertEqual(a, b, accuracy: 2e-6, optimizer + " " + key) }
            }
            XCTAssertEqual(split.updateCount, 1)
        }
    }

    func testTrueFrozenLoRATrainingAndFusionParity() throws {
        try available()
        let base = try session(), frozen = try base.weights()
        let adapter = try NativeAdapterConfiguration(rank: 2, alpha: 2)
        let adapted = try NativeTrainingSession(architecture: base.architecture, options: base.options, seed: 14, weights: frozen, adapter: adapter)
        let input = sample(3 * 8 * 8), target = Array(repeating: Float(0.95), count: 8 * 8)
        let initial = try base.predict(input: input, batchSize: 1, height: 8, width: 8)
        XCTAssertEqual(try adapted.predict(input: input, batchSize: 1, height: 8, width: 8), initial)
        for _ in 0..<3 {
            _ = try adapted.train(input: input, target: target, batchSize: 1, height: 8, width: 8)
            _ = try adapted.applyGradients(learningRate: 0.01)
        }
        XCTAssertEqual(try adapted.weights(), frozen, "Installed base is immutable during LoRA training")
        let delta = try adapted.adapterWeights()
        XCTAssertTrue(delta.keys.allSatisfy { $0.hasSuffix(".down.weight") || $0.hasSuffix(".up.weight") })
        XCTAssertTrue(delta.filter { $0.key.hasSuffix(".up.weight") }.values.contains { $0.values.contains { $0 != 0 } })
        let fused = try NativeTrainingSession(architecture: base.architecture, options: base.options, seed: 14, weights: adapted.fusedWeights())
        let expected = try adapted.predict(input: input, batchSize: 1, height: 8, width: 8)
        let actual = try fused.predict(input: input, batchSize: 1, height: 8, width: 8)
        for (a, b) in zip(expected, actual) { XCTAssertEqual(a, b, accuracy: 2e-6) }
    }

    func testNormalVectorsAndGradientsAreFinite() throws {
        try available()
        let model = try session("normal")
        let input = sample(3 * 5 * 5)
        let prediction = try model.predict(input: input, batchSize: 1, height: 5, width: 5)
        for index in 0..<25 {
            let norm = sqrt((0..<3).reduce(Float(0)) { $0 + pow(prediction[$1 * 25 + index] * 2 - 1, 2) })
            XCTAssertEqual(norm, 1, accuracy: 1e-5)
        }
        var target = Array(repeating: Float(0.5), count: 75)
        for index in 50..<75 { target[index] = 1 }
        let loss = try model.train(input: input, target: target, batchSize: 1, height: 5, width: 5)
        XCTAssertTrue(loss.isFinite)
        XCTAssertTrue(try model.applyGradients(learningRate: 0.001).isFinite)
    }

    func testCancellationPreservesLastFiniteUpdateAndCheckpointStateResumesExactly() throws {
        try available()
        let model = try session(), input = sample(3 * 5 * 5), target = sample(5 * 5, offset: 2)
        _ = try model.train(input: input, target: target, batchSize: 1, height: 5, width: 5)
        _ = try model.applyGradients(learningRate: 0.001)
        let resumed = try NativeTrainingSession(architecture: model.architecture, options: model.options, seed: 99,
            weights: model.weights(), optimizerState: model.optimizerState(), initialUpdateCount: model.updateCount)
        for candidate in [model, resumed] {
            _ = try candidate.train(input: input, target: target, batchSize: 1, height: 5, width: 5)
            _ = try candidate.applyGradients(learningRate: 0.001)
        }
        XCTAssertEqual(try model.weights(), try resumed.weights())
        let saved = try model.weights()
        _ = try model.train(input: input, target: target, batchSize: 1, height: 5, width: 5)
        model.cancellation.cancel()
        XCTAssertThrowsError(try model.applyGradients(learningRate: 0.001)) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertThrowsError(try model.train(input: input, target: target, batchSize: 1, height: 5, width: 5)) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(try model.weights(), saved)
        XCTAssertEqual(model.updateCount, 2)
    }

    func testLearningRateScheduleIncludesWarmupAndFinalCosineEndpoint() {
        XCTAssertEqual(NativeTrainingOptions.learningRate(base: 0.01, step: 0, totalSteps: 8, stepsPerEpoch: 2, warmupEpochs: 1, schedule: "cosine", minimumRatio: 0.1), 0.005)
        XCTAssertEqual(NativeTrainingOptions.learningRate(base: 0.01, step: 1, totalSteps: 8, stepsPerEpoch: 2, warmupEpochs: 1, schedule: "cosine", minimumRatio: 0.1), 0.01)
        XCTAssertEqual(NativeTrainingOptions.learningRate(base: 0.01, step: 7, totalSteps: 8, stepsPerEpoch: 2, warmupEpochs: 1, schedule: "cosine", minimumRatio: 0.1), 0.001, accuracy: 1e-12)
    }

    func testOptimizerUpdatesMatchIndependentScalarEquations() throws {
        try available()
        let architecture = try architecture()
        var weights = Dictionary(uniqueKeysWithValues: architecture.parameterShapes.map { name, shape in
            (name, NativeTensor(shape: shape, values: Array(repeating: Float(0), count: shape.reduce(1, *))))
        })
        weights["output.bias"]!.values[0] = 0.5
        let input = Array(repeating: Float(0), count: 3 * 4 * 4), target = Array(repeating: Float(1), count: 4 * 4)
        for optimizer in ["adamw", "adam", "sgd", "lion"] {
            let options = try NativeTrainingOptions(optimizer: optimizer, weightDecay: 0.2, beta1: 0.3,
                beta2: 0.6, momentum: 0.4, gradientClipNorm: 0)
            let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 0, weights: weights)
            var expected: Float = 0.5, first: Float = 0, second: Float = 0
            for step in 1...2 {
                let sigmoid = 1 / (1 + exp(-expected))
                var gradient = -sigmoid * (1 - sigmoid)
                if optimizer == "adam" || optimizer == "sgd" { gradient += expected * 0.2 }
                let direction: Float
                if optimizer == "sgd" { first = first * 0.4 + gradient; direction = first }
                else if optimizer == "lion" {
                    let blend = first * 0.3 + gradient * 0.7
                    direction = blend > 0 ? 1 : blend < 0 ? -1 : 0
                    first = first * 0.6 + gradient * 0.4
                } else {
                    first = first * 0.3 + gradient * 0.7
                    second = second * 0.6 + gradient * gradient * 0.4
                    let correctedFirst = first / Float(1 - pow(0.3, Double(step)))
                    let correctedSecond = second / Float(1 - pow(0.6, Double(step)))
                    direction = correctedFirst / (sqrt(correctedSecond) + 1e-8)
                }
                if optimizer == "adamw" || optimizer == "lion" { expected *= 1 - 0.01 * 0.2 }
                expected -= direction * 0.01
                _ = try model.train(input: input, target: target, batchSize: 1, height: 4, width: 4)
                _ = try model.applyGradients(learningRate: 0.01)
                XCTAssertEqual(try model.weights()["output.bias"]!.values[0], expected, accuracy: 2e-6, optimizer)
            }
        }
    }

    func testWideNetworkOwnedFeedsSurviveAllocationChurnAndMatchReloadedWeights() throws {
        try available()
        let architecture = try NativeArchitecture(mapKind: "depth", width: 164, blocks: 4)
        let options = try NativeTrainingOptions(gradientClipNorm: 0)
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let path = folder.appendingPathComponent("initial.safetensors")
        let input = sample(3 * 32 * 32), target = Array(repeating: Float(0.9), count: 32 * 32)
        for seed in [UInt64(42), 7, 3] {
            let generated = try NativeTrainingSession(architecture: architecture, options: options, seed: seed)
            try NativeWeights.write(generated.weights(), to: path)
            let reloaded = try NativeTrainingSession(architecture: architecture, options: options, seed: 0, weights: NativeWeights.read(from: path))
            autoreleasepool {
                // Reuse memory formerly occupied by random arrays and temporary
                // NSData. GPU state must depend only on owned Metal allocations.
                let churn = (0..<32).map { index in Data(repeating: UInt8(index), count: 1024 * 1024) }
                XCTAssertEqual(churn.reduce(0) { $0 + $1.count }, 32 * 1024 * 1024)
            }
            let generatedLoss = try generated.train(input: input, target: target, batchSize: 1, height: 32, width: 32)
            let reloadedLoss = try reloaded.train(input: input, target: target, batchSize: 1, height: 32, width: 32)
            XCTAssertEqual(generatedLoss, reloadedLoss, accuracy: 1e-6)
            let generatedNorm = try generated.applyGradients(learningRate: 0.001)
            let reloadedNorm = try reloaded.applyGradients(learningRate: 0.001)
            XCTAssertEqual(generatedNorm, reloadedNorm, accuracy: 1e-5)
            XCTAssertLessThan(generatedNorm, 100, "Bounded residual initialization must not produce arbitrary gradient data")
            XCTAssertEqual(try generated.weights(), try reloaded.weights())
        }
    }

    func testFullResolutionMaskedGradientsRemainZeroWithoutCPUReadback() throws {
        try available()
        let architecture = try NativeArchitecture(mapKind: "depth", width: 164, blocks: 4)
        let weights = Dictionary(uniqueKeysWithValues: architecture.parameterShapes.map { name, shape in
            (name, NativeTensor(shape: shape, values: Array(repeating: Float(0), count: shape.reduce(1, *))))
        })
        let options = try NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0, gradientClipNorm: 0)
        let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 42, weights: weights)
        let size = 1024
        let input = Array(repeating: Float(0), count: 3 * size * size)
        let target = Array(repeating: Float(1), count: size * size)
        var expectedBias: Float = 0
        for _ in 0..<2 {
            let sigmoid = 1 / (1 + exp(-expectedBias))
            let expectedNorm = sigmoid * (1 - sigmoid)
            _ = try model.train(input: input, target: target, batchSize: 1, height: size, width: size)
            // Read neither accumulated gradients nor weights between these
            // calls: a CPU materialization previously concealed bad GPU state.
            let norm = try model.applyGradients(learningRate: 0.001)
            XCTAssertEqual(norm, Double(expectedNorm), accuracy: 2e-6)
            expectedBias += expectedNorm * 0.001
        }
        let updated = try model.weights()
        XCTAssertEqual(updated["output.bias"]!.values[0], expectedBias, accuracy: 2e-6)
        for (name, tensor) in updated where name != "output.bias" {
            XCTAssertTrue(tensor.values.allSatisfy { $0 == 0 }, "Masked gradient changed \(name)")
        }
    }

    /// Set LOOM_NATIVE_REAL_PAIR_DIRECTORY to the unmodified Bark Willow 02
    /// source directory when validating a release on an Apple GPU. This tests
    /// the exact 1K/1.94M-parameter case whose small synthetic tests had passed.
    func testRealFullResolutionGradientAndUpdatesMatchFreshReloadWithoutCPUReadback() throws {
        try available()
        guard let directory = ProcessInfo.processInfo.environment["LOOM_NATIVE_REAL_PAIR_DIRECTORY"] else {
            throw XCTSkip("Set LOOM_NATIVE_REAL_PAIR_DIRECTORY to run the real 1K release regression")
        }
        let folder = URL(fileURLWithPath: directory, isDirectory: true)
        let inputURL = folder.appendingPathComponent("bark_willow_02_diff_1k.png")
        let targetURL = folder.appendingPathComponent("bark_willow_02_disp_1k.png")
        func hash(_ url: URL) throws -> String {
            SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
        }
        let originalHashes = try [inputURL, targetURL].map(hash)
        XCTAssertEqual(originalHashes, ["d5ba0df675904c7b94f1200f3e024fb39372065fd2940e25cc5d36a6e10a92a2",
                                        "e25841ea5e0322b25d99967bf02d66bc19df0beba5a62112fa0ca09fa1b4a4d0"])
        let pair = try NativeImages.pair(inputURL: inputURL, targetURL: targetURL,
            transform: NativeCropTransform(x: 0, y: 0, size: 1024), mapKind: "depth")
        let architecture = try NativeArchitecture(mapKind: "depth", width: 164, blocks: 4)
        let options = try NativeTrainingOptions()
        let generated = try NativeTrainingSession(architecture: architecture, options: options, seed: 42)
        let initial = try generated.weights()
        let reloaded = try NativeTrainingSession(architecture: architecture, options: options, seed: 7, weights: initial)
        let expectedLosses = [0.09165072441101074, 0.4429701566696167, 0.3618987798690796]
        let expectedNorms = [1.4644880294799805, 0.40909165143966675, 1.7472360134124756]
        for step in 0..<3 {
            for model in [generated, reloaded] {
                let loss = try model.train(input: pair.input.values, target: pair.target.values,
                    batchSize: 1, height: 1024, width: 1024)
                let norm = try model.applyGradients(learningRate: 0.001)
                XCTAssertEqual(loss, expectedLosses[step], accuracy: 2e-6)
                XCTAssertEqual(norm, expectedNorms[step], accuracy: 2e-5)
            }
        }
        XCTAssertEqual(try generated.weights(), try reloaded.weights())
        XCTAssertEqual(try [inputURL, targetURL].map(hash), originalHashes)
    }
}
