import CoreML
import CryptoKit
import Foundation
import Metal
import XCTest
@testable import LoomNative

/// Both executions retain the public float32 NCHW image/OIHW weight contract.
/// Compare NHWC/HWIO GPU execution with the established NCHW/OIHW graph path,
/// including gradients and saved state rather than only its forward output.
final class NativeChannelsLastParityTests: XCTestCase {
    private let size = 32

    private func available() throws {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") }
    }

    private func input(batch: Int, height: Int = 32, width: Int = 32, offset: Int = 0) -> [Float] {
        let area = height * width
        return (0..<(batch * 3 * area)).map { index in
            let image = index / (3 * area), channel = (index / area) % 3, pixel = index % area
            let code = (pixel * 11 + channel * 37 + image * 23 + offset * 17) % 101
            return 0.05 + Float(code) * 0.009
        }
    }

    private func scalarTarget(batch: Int, offset: Int = 0) -> [Float] {
        (0..<(batch * size * size)).map { 0.72 + Float(($0 * 7 + offset * 13) % 97) * 0.002 }
    }

    private func normalTarget(batch: Int) -> [Float] {
        let area = size * size
        var result = [Float](repeating: 0.5, count: batch * 3 * area)
        for image in 0..<batch {
            for pixel in 0..<area where pixel % 13 != 0 {
                let x = Float((pixel + image * 3) % 11 - 5) * 0.04
                let y = Float((pixel * 3 + image) % 17 - 8) * 0.025
                let length = sqrt(x * x + y * y + 1)
                let start = image * 3 * area + pixel
                result[start] = x / length * 0.5 + 0.5
                result[start + area] = y / length * 0.5 + 0.5
                result[start + 2 * area] = 1 / length * 0.5 + 0.5
            }
        }
        return result
    }

    private func assertClose(_ left: [Float], _ right: [Float], tolerance: Float, _ message: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(left.count, right.count, message, file: file, line: line)
        XCTAssertTrue(left.allSatisfy(\.isFinite), "Reference contains non-finite values: " + message, file: file, line: line)
        XCTAssertTrue(right.allSatisfy(\.isFinite), "Optimized path contains non-finite values: " + message, file: file, line: line)
        let maximumError = zip(left, right).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        XCTAssertLessThanOrEqual(maximumError, tolerance, message, file: file, line: line)
    }

    private func assertClose(_ left: [String: NativeTensor], _ right: [String: NativeTensor],
                             tolerance: Float, _ message: String,
                             file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertEqual(Set(left.keys), Set(right.keys), message, file: file, line: line)
        for name in left.keys.sorted() {
            let a = try XCTUnwrap(left[name], file: file, line: line)
            let b = try XCTUnwrap(right[name], file: file, line: line)
            XCTAssertEqual(a.shape, b.shape, message + " " + name, file: file, line: line)
            assertClose(a.values, b.values, tolerance: tolerance, message + " " + name, file: file, line: line)
        }
    }

    private func assertNormClose(_ reference: Double, _ optimized: Double, _ message: String,
                                 file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(reference.isFinite && optimized.isFinite, message, file: file, line: line)
        XCTAssertEqual(reference, optimized, accuracy: max(1e-6, abs(reference) * 1e-4), message, file: file, line: line)
    }

    func testFloat32PredictionGradientsAndPartialAccumulationMatchAcrossOptimizers() throws {
        try available()
        let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 2)
        for seed in [UInt64(42), 7] {
            for optimizer in ["adamw", "adam", "sgd", "lion"] {
                let options = try NativeTrainingOptions(optimizer: optimizer, weightDecay: 0.01, gradientClipNorm: 0.2)
                let reference = try NativeTrainingSession(architecture: architecture, options: options,
                    seed: seed, channelsLast: false)
                let optimized = try NativeTrainingSession(architecture: architecture, options: options,
                    seed: seed, channelsLast: true)
                let initial = try reference.weights()
                XCTAssertEqual(initial, try optimized.weights(), "Execution mode must not change initialized OIHW weights")
                for update in 0..<4 {
                    let label = "\(optimizer), seed \(seed), update \(update + 1)"
                    // A full two-image minibatch followed by one remaining
                    // image exercises the smaller final accumulation batch.
                    for batch in [2, 1] {
                        let pixels = input(batch: batch, offset: update + batch)
                        let target = scalarTarget(batch: batch, offset: update)
                        assertClose(try reference.predict(input: pixels, batchSize: batch, height: size, width: size),
                            try optimized.predict(input: pixels, batchSize: batch, height: size, width: size),
                            tolerance: 2e-5, label + " NCHW prediction, batch \(batch)")
                        let a = try reference.train(input: pixels, target: target, batchSize: batch,
                            height: size, width: size, weight: Float(batch) / 3)
                        let b = try optimized.train(input: pixels, target: target, batchSize: batch,
                            height: size, width: size, weight: Float(batch) / 3)
                        XCTAssertEqual(a, b, accuracy: 2e-6, label + " mean loss")
                        try assertClose(reference.accumulatedGradients(), optimized.accumulatedGradients(),
                            tolerance: 5e-6, label + " OIHW accumulated gradients, batch \(batch)")
                    }
                    assertNormClose(try reference.applyGradients(learningRate: 0.001),
                        try optimized.applyGradients(learningRate: 0.001), label + " unclipped gradient norm")
                    XCTAssertEqual(reference.updateCount, update + 1)
                    XCTAssertEqual(optimized.updateCount, update + 1)
                    try assertClose(reference.weights(), optimized.weights(), tolerance: 2e-6, label + " saved weights")
                    try assertClose(reference.optimizerState(), optimized.optimizerState(),
                        tolerance: 2e-6, label + " optimizer moments")
                }
                XCTAssertNotEqual(initial, try optimized.weights(), "\(optimizer) must perform actual updates")
            }
        }
    }

    func testNormalChannelNormalizationAndZeroVectorGradientsRemainFinite() throws {
        try available()
        let architecture = try NativeArchitecture(mapKind: "normal", width: 8, blocks: 2)
        let options = try NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0, gradientClipNorm: 1)
        let reference = try NativeTrainingSession(architecture: architecture, options: options,
            seed: 42, channelsLast: false)
        let optimized = try NativeTrainingSession(architecture: architecture, options: options,
            seed: 42, channelsLast: true)
        let pixels = input(batch: 2), target = normalTarget(batch: 2), area = size * size
        for update in 0..<2 {
            let a = try reference.predict(input: pixels, batchSize: 2, height: size, width: size)
            let b = try optimized.predict(input: pixels, batchSize: 2, height: size, width: size)
            assertClose(a, b, tolerance: 2e-5, "Normal NCHW output update \(update)")
            for values in [a, b] {
                for image in 0..<2 {
                    for pixel in 0..<area {
                        let start = image * 3 * area + pixel
                        let length = sqrt((0..<3).reduce(Float(0)) {
                            $0 + pow(values[start + $1 * area] * 2 - 1, 2)
                        })
                        XCTAssertEqual(length, 1, accuracy: 2e-5, "Normalize along channels, retaining NCHW output")
                    }
                }
            }
            let aLoss = try reference.train(input: pixels, target: target, batchSize: 2, height: size, width: size)
            let bLoss = try optimized.train(input: pixels, target: target, batchSize: 2, height: size, width: size)
            XCTAssertTrue(aLoss.isFinite && bLoss.isFinite)
            XCTAssertEqual(aLoss, bLoss, accuracy: 2e-6)
            try assertClose(reference.accumulatedGradients(), optimized.accumulatedGradients(),
                tolerance: 5e-6, "Normal gradient parity with zero target vectors")
            assertNormClose(try reference.applyGradients(learningRate: 0.0001),
                try optimized.applyGradients(learningRate: 0.0001), "Normal update gradient norm")
            try assertClose(reference.weights(), optimized.weights(), tolerance: 2e-6, "Normal updated OIHW weights")
        }

        let zeros = architecture.parameterShapes.mapValues {
            NativeTensor(shape: $0, values: [Float](repeating: 0, count: $0.reduce(1, *)))
        }
        let zeroTarget = [Float](repeating: 0.5, count: 3 * area)
        for execution in [false, true] {
            let zero = try NativeTrainingSession(architecture: architecture, options: options,
                seed: 0, weights: zeros, channelsLast: execution)
            let prediction = try zero.predict(input: input(batch: 1), batchSize: 1, height: size, width: size)
            XCTAssertTrue(prediction.allSatisfy { $0 == 0.5 }, "Encoded zero normal stays finite at normalize epsilon")
            XCTAssertEqual(try zero.train(input: input(batch: 1), target: zeroTarget,
                batchSize: 1, height: size, width: size), 1, accuracy: 1e-6)
            XCTAssertTrue(try zero.accumulatedGradients().values.allSatisfy { $0.values.allSatisfy { $0 == 0 } })
            XCTAssertEqual(try zero.applyGradients(learningRate: 0.001), 0)
            XCTAssertEqual(try zero.weights(), zeros)
        }
    }

    func testLoRAKeepsExactFrozenBaseAndOIHWAdapterFusionParity() throws {
        try available()
        let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 2)
        let options = try NativeTrainingOptions(gradientClipNorm: 0.2)
        let base = try NativeTrainingSession(architecture: architecture, options: options,
            seed: 42)
        let frozen = try base.weights(), adapter = try NativeAdapterConfiguration(rank: 2, alpha: 2)
        let reference = try NativeTrainingSession(architecture: architecture, options: options,
            seed: 14, weights: frozen, adapter: adapter, channelsLast: false)
        let optimized = try NativeTrainingSession(architecture: architecture, options: options,
            seed: 14, weights: frozen, adapter: adapter, channelsLast: true)
        XCTAssertEqual(try reference.adapterWeights(), try optimized.adapterWeights())
        let pixels = input(batch: 1), target = scalarTarget(batch: 1)
        let basePrediction = try base.predict(input: pixels, batchSize: 1, height: size, width: size)
        assertClose(basePrediction, try optimized.predict(input: pixels, batchSize: 1, height: size, width: size),
            tolerance: 2e-5, "Zero LoRA up-kernels preserve the installed base prediction")
        for update in 0..<4 {
            let a = try reference.train(input: pixels, target: target, batchSize: 1, height: size, width: size)
            let b = try optimized.train(input: pixels, target: target, batchSize: 1, height: size, width: size)
            XCTAssertEqual(a, b, accuracy: 2e-6, "LoRA loss at update \(update + 1)")
            try assertClose(reference.accumulatedGradients(), optimized.accumulatedGradients(),
                tolerance: 5e-6, "LoRA rank-kernel gradients")
            assertNormClose(try reference.applyGradients(learningRate: 0.001),
                try optimized.applyGradients(learningRate: 0.001), "LoRA gradient norm")
            XCTAssertEqual(try reference.weights(), frozen, "Reference base must remain exact")
            XCTAssertEqual(try optimized.weights(), frozen, "Optimized base must remain exact")
            try assertClose(reference.adapterWeights(), optimized.adapterWeights(), tolerance: 2e-6, "Updated LoRA weights")
        }
        let deltas = try optimized.adapterWeights()
        for name in architecture.convolutionNames {
            let shape = try XCTUnwrap(architecture.parameterShapes[name + ".weight"])
            XCTAssertEqual(try XCTUnwrap(deltas[name + ".down.weight"]).shape, [adapter.rank, shape[1], 3, 3])
            XCTAssertEqual(try XCTUnwrap(deltas[name + ".up.weight"]).shape, [shape[0], adapter.rank, 1, 1])
        }
        XCTAssertTrue(deltas.filter { $0.key.hasSuffix(".up.weight") }.values.contains { $0.values.contains { $0 != 0 } })
        try assertClose(reference.fusedWeights(), optimized.fusedWeights(), tolerance: 2e-6, "Exported OIHW fusion")
        let adaptedPrediction = try optimized.predict(input: pixels, batchSize: 1, height: size, width: size)
        assertClose(try reference.predict(input: pixels, batchSize: 1, height: size, width: size), adaptedPrediction,
            tolerance: 2e-5, "Trained LoRA prediction")
        for execution in [false, true] {
            let fused = try NativeTrainingSession(architecture: architecture, options: options,
                seed: 0, weights: optimized.fusedWeights(), channelsLast: execution)
            assertClose(adaptedPrediction, try fused.predict(input: pixels, batchSize: 1, height: size, width: size),
                tolerance: 2e-5, "Fused deployment weights preserve the LoRA result")
        }
    }

    func testTrainedChannelsLastWeightsDeployAsExistingCoreMLNCHWModel() throws {
        try available()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for map in ["depth", "roughness", "normal"] {
            let architecture = try NativeArchitecture(mapKind: map, width: 8, blocks: 2)
            let options = try NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0, gradientClipNorm: 1)
            let trained = try NativeTrainingSession(architecture: architecture, options: options,
                seed: 42, channelsLast: true)
            _ = try trained.train(input: input(batch: 1), target: map == "normal" ? normalTarget(batch: 1) : scalarTarget(batch: 1),
                batchSize: 1, height: size, width: size)
            _ = try trained.applyGradients(learningRate: 0.0001)
            let weights = try trained.weights()
            XCTAssertEqual(weights.mapValues(\.shape), architecture.parameterShapes, "Export retains original OIHW tensor interface")
            let compiled = try NativeCoreML.export(weights: weights, architecture: architecture,
                imageSize: size, to: directory.appendingPathComponent(map + ".mlmodel"))
            defer { try? FileManager.default.removeItem(at: compiled) }
            let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
            let deployed = try MLModel(contentsOf: compiled, configuration: configuration)
            for (width, height) in [(32, 32), (35, 33)] {
                let pixels = input(batch: 1, height: height, width: width)
                let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
                array.dataPointer.assumingMemoryBound(to: Float.self).update(from: pixels, count: pixels.count)
                let provider = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)])
                let result = try XCTUnwrap(deployed.prediction(from: provider).featureValue(for: "material_map")?.multiArrayValue)
                XCTAssertEqual(result.shape.map(\.intValue), [1, architecture.outputChannels, height, width])
                let expected = try trained.predict(input: pixels, batchSize: 1, height: height, width: width)
                let actual = (0..<result.count).map { result[$0].floatValue }
                assertClose(expected, actual, tolerance: 2e-5, "\(map) Core ML CPU output, \(width)x\(height)")
            }
        }
    }
    func testFullResolutionMaskedGradientsRemainZeroWithoutCPUReadback() throws {
        try available()
        let architecture = try NativeArchitecture(mapKind: "depth", width: 164, blocks: 4)
        let weights = Dictionary(uniqueKeysWithValues: architecture.parameterShapes.map { name, shape in
            (name, NativeTensor(shape: shape, values: Array(repeating: Float(0), count: shape.reduce(1, *))))
        })
        let options = try NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0, gradientClipNorm: 0)
        let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 42, weights: weights, channelsLast: true)
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
    func testRealFullResolutionChannelsLastUpdatesMatchTheEstablishedRegression() throws {
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
        let generated = try NativeTrainingSession(architecture: architecture, options: options, seed: 42, channelsLast: true)
        let initial = try generated.weights()
        let reloaded = try NativeTrainingSession(architecture: architecture, options: options, seed: 7, weights: initial, channelsLast: true)
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
        XCTAssertTrue(try generated.weights() == reloaded.weights(), "Fresh and reloaded GPU state must produce the same weights")
        XCTAssertEqual(try [inputURL, targetURL].map(hash), originalHashes)
    }
}
