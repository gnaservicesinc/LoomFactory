import CryptoKit
import Metal
import MetalPerformanceShadersGraph
import XCTest
@testable import LoomNative

final class NativeCompactNetworkTests: XCTestCase {
    func testExactHistoricalTensorContractAndSeparateLinearArchitecture() throws {
        for map in ["depth", "roughness", "normal"] {
            let architecture = try compact(map, width: 32)
            XCTAssertTrue(architecture.isCompact)
            XCTAssertFalse(architecture.isPyramid)
            XCTAssertTrue(architecture.usesLinearOutput)
            XCTAssertEqual(architecture.parameterCount, map == "normal" ? 1_000_515 : 999_937)
            XCTAssertEqual(architecture.parameterShapes.count, 142)
            XCTAssertEqual(architecture.convolutionNames.count, 50)
            XCTAssertEqual(architecture.adaptableConvolutionNames.count, 43)
            XCTAssertEqual(Set(architecture.convolutionNames).count, 50)
            XCTAssertEqual(architecture.parameterShapes["input.weight"], [32, 3, 3, 3])
            XCTAssertEqual(architecture.parameterShapes["input.bias"], [1, 32, 1, 1])
            XCTAssertEqual(architecture.parameterShapes["encoder.0.depthwise.weight"], [64, 1, 3, 3])
            XCTAssertEqual(architecture.parameterShapes["down.2.weight"], [256, 128, 2, 2])
            XCTAssertEqual(architecture.parameterShapes["middle.ffn_expand.weight"], [512, 256, 1, 1])
            XCTAssertEqual(architecture.parameterShapes["up.2.weight"], [128, 256, 1, 1])
            XCTAssertEqual(architecture.parameterShapes["decoder.0.norm1.weight"], [1, 32, 1, 1])
            XCTAssertEqual(architecture.parameterShapes["decoder.0.beta"], [1, 32, 1, 1])
            XCTAssertEqual(architecture.parameterShapes["output.weight"], [map == "normal" ? 3 : 1, 32, 3, 3])
            XCTAssertEqual(architecture.convolutionKernelSize("down.0"), 2)
            XCTAssertEqual(architecture.convolutionStride("down.0"), 2)
            XCTAssertEqual(architecture.convolutionKernelSize("up.0"), 1)
            XCTAssertEqual(architecture.convolutionStride("up.0"), 1)
            XCTAssertEqual(architecture.convolutionGroups("encoder.2.depthwise"), 256)
            XCTAssertEqual(architecture.convolutionGroups("encoder.2.expand"), 1)
            XCTAssertFalse(architecture.adaptableConvolutionNames.contains { $0.hasSuffix(".depthwise") })
            XCTAssertEqual(architecture.descriptor["head_activation"] as? String, "linear")
            XCTAssertEqual(architecture.descriptor["padding"] as? String, "reflect-v1")
            XCTAssertEqual(architecture.descriptor["normal_output_transform"] as? String, "none")
            XCTAssertEqual(architecture.descriptor["scale_channels"] as? [Int], [32, 64, 128, 256])
            XCTAssertTrue(architecture.supportsGrid(height: 32, width: 40))
            XCTAssertFalse(architecture.supportsGrid(height: 31, width: 40))
            XCTAssertFalse(architecture.supportsGrid(height: 33, width: 40))
        }
        XCTAssertThrowsError(try NativeArchitecture(mapKind: "depth", width: 32, blocks: 2,
            identifier: NativeArchitecture.compactIdentifier))
        let residual = try NativeArchitecture(mapKind: "depth", width: 2, blocks: 1)
        XCTAssertEqual(residual.parameterCount, 151)
        XCTAssertFalse(residual.usesLinearOutput)
        XCTAssertTrue(residual.supportsGrid(height: 33, width: 35))
    }

    func testHistoricalXavierAndResidualGainInitializationMatchesIndependentFloat32Fixture() throws {
        let architecture = try compact("depth", width: 32)
        let weights = NativeCompactNetwork.initialWeights(architecture: architecture, seed: 17)
        XCTAssertEqual(weights.mapValues(\.shape), architecture.parameterShapes)
        XCTAssertEqual(weights, NativeCompactNetwork.initialWeights(architecture: architecture, seed: 17))
        XCTAssertNotEqual(weights, NativeCompactNetwork.initialWeights(architecture: architecture, seed: 18))
        XCTAssertEqual(weights["decoder.0.attention.weight"]!.values.prefix(5).map(\.bitPattern),
            [0x3aa239e0, 0xbd882842, 0xbdd74c5a, 0xbe2d22e3, 0xbe975b26])
        XCTAssertEqual(weights["input.weight"]!.values.prefix(5).map(\.bitPattern),
            [0xbda729f1, 0x3d31073e, 0xbddbe373, 0x3d38dd7d, 0x3dff976e])
        XCTAssertEqual(weights["middle.depthwise.weight"]!.values.prefix(5).map(\.bitPattern),
            [0x3eda27d0, 0xbef6f68d, 0xbd8da9ca, 0xbef254aa, 0x3efec3ee])
        var digest = SHA256()
        for name in weights.keys.sorted() {
            digest.update(data: Data(name.utf8))
            weights[name]!.values.withUnsafeBytes { digest.update(bufferPointer: $0) }
        }
        // Independent Float32/UInt64 implementation of the historical equations;
        // checks the complete named stream, including random draw consumption.
        XCTAssertEqual(digest.finalize().map { String(format: "%02x", $0) }.joined(),
            "365aac8d1c9dc68c6c1378bc1eee803a53a29cfd51289976b10cf110c523349e")
        for name in NativeCompactNetwork.blockNames {
            XCTAssertTrue(weights[name + ".norm1.weight"]!.values.allSatisfy { $0 == 1 })
            XCTAssertTrue(weights[name + ".norm2.bias"]!.values.allSatisfy { $0 == 0 })
            XCTAssertTrue(weights[name + ".beta"]!.values.allSatisfy { $0 == Float(0.1) })
            XCTAssertTrue(weights[name + ".gamma"]!.values.allSatisfy { $0 == Float(0.1) })
        }
        XCTAssertEqual(weights["output.bias"]!.values, [0.5])
        let normal = try compact("normal", width: 2)
        XCTAssertEqual(NativeCompactNetwork.initialWeights(architecture: normal, seed: 17)["output.bias"]!.values,
            [0.5, 0.5, 1])
    }

    func testLinearScalarAndNormalHeadPreserveOutOfRangeComponentsAndNativeRectangles() throws {
        try available()
        let options = try NativeTrainingOptions(scalarLoss: .linearDetailV1)
        for map in ["depth", "roughness", "normal"] {
            let architecture = try compact(map, width: 2)
            var weights = architecture.parameterShapes.mapValues {
                NativeTensor(shape: $0, values: Array(repeating: 0, count: $0.reduce(1, *)))
            }
            let biases: [Float] = map == "normal" ? [-0.25, 0.4, 1.25] : [-0.25]
            weights["output.bias"]!.values = biases
            let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 0, weights: weights)
            let width = 40, height = 32, count = width * height
            let actual = try model.predict(input: samples(3 * count), batchSize: 1, height: height, width: width)
            XCTAssertEqual(actual.count, architecture.outputChannels * count)
            for channel in 0..<architecture.outputChannels {
                XCTAssertTrue(actual[(channel * count)..<((channel + 1) * count)].allSatisfy { $0 == biases[channel] })
            }
            XCTAssertThrowsError(try model.predict(input: samples(3 * 33 * width), batchSize: 1, height: 33, width: width))
        }
    }

    func testReflectionAndFullResolutionSkipPreserveConstantFieldAtEveryBorder() throws {
        try available()
        let architecture = try compact("depth", width: 2)
        var weights = architecture.parameterShapes.mapValues {
            NativeTensor(shape: $0, values: Array(repeating: 0, count: $0.reduce(1, *)))
        }
        // The full-resolution skip supplies input channel zero directly to a
        // 3x3 average output. Zero lower scales isolate the skip and padding.
        weights["input.weight"]!.values[4] = 1
        for index in 0..<9 { weights["output.weight"]!.values[index] = 1 / 9 }
        let model = try NativeTrainingSession(architecture: architecture,
            options: NativeTrainingOptions(scalarLoss: .linearDetailV1), seed: 0, weights: weights)
        let size = 32
        let input = Array(repeating: Float(0.75), count: size * size)
            + Array(repeating: Float(0.2), count: 2 * size * size)
        let actual = try model.predict(input: input, batchSize: 1, height: size, width: size)
        for value in actual { XCTAssertEqual(value, 0.75, accuracy: 2e-7) }
    }

    func testGradientsReachEarlyDepthwiseAttentionNormalizationAndBottleneckWeights() throws {
        try available()
        let architecture = try compact("depth", width: 4)
        let options = try NativeTrainingOptions(weightDecay: 0, gradientClipNorm: 0, scalarLoss: .linearDetailV1)
        let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 37)
        let size = 32, input = samples(3 * size * size), target = samples(size * size)
        let before = try model.weights()
        _ = try model.train(input: input, target: target, batchSize: 1, height: size, width: size)
        let gradients = try model.accumulatedGradients()
        XCTAssertEqual(gradients.mapValues(\.shape), architecture.parameterShapes)
        for gradient in gradients.values { XCTAssertTrue(gradient.values.allSatisfy(\.isFinite)) }
        for name in ["input.weight", "encoder.0.depthwise.weight", "encoder.0.norm1.weight",
                     "encoder.0.attention.weight", "middle.project.weight", "decoder.0.beta", "output.bias"] {
            XCTAssertTrue(gradients[name]!.values.contains { abs($0) > 1e-7 }, name)
        }
        _ = try model.applyGradients(learningRate: 0.0001)
        for name in ["input.weight", "middle.project.weight", "output.weight"] {
            XCTAssertNotEqual(try model.weights()[name], before[name], name)
        }
    }

    func testDenseLoRAKeepsDepthwiseBaseFrozenAndFusesAllKernelSizes() throws {
        try available()
        let architecture = try compact("depth", width: 2)
        let options = try NativeTrainingOptions(weightDecay: 0, gradientClipNorm: 1, scalarLoss: .linearDetailV1)
        let initial = NativeCompactNetwork.initialWeights(architecture: architecture, seed: 17)
        let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 93, weights: initial,
            adapter: NativeAdapterConfiguration(rank: 1, alpha: 1))
        let size = 32, input = samples(3 * size * size), target = samples(size * size)
        for _ in 0..<2 {
            _ = try model.train(input: input, target: target, batchSize: 1, height: size, width: size)
            _ = try model.applyGradients(learningRate: 0.0001)
        }
        XCTAssertEqual(try model.weights(), initial)
        let adapters = try model.adapterWeights()
        XCTAssertEqual(adapters.count, 86)
        XCTAssertEqual(adapters["down.0.down.weight"]?.shape, [1, 2, 2, 2])
        XCTAssertEqual(adapters["up.0.down.weight"]?.shape, [1, 4, 1, 1])
        XCTAssertNil(adapters["encoder.0.depthwise.down.weight"])
        let fused = try NativeTrainingSession(architecture: architecture, options: options, seed: 0, weights: model.fusedWeights())
        let expected = try model.predict(input: input, batchSize: 1, height: size, width: size)
        let actual = try fused.predict(input: input, batchSize: 1, height: size, width: size)
        XCTAssertLessThan(zip(expected, actual).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }, 2e-5)
    }

    func testLinearDetailObjectiveAndEveryPredictionGradientMatchIndependentSignedDifferenceEquations() throws {
        try available()
        let graph = MPSGraph(), height = 11, width = 13, channels = 2
        let shape = [1, channels, height, width].map(NSNumber.init(value:))
        let prediction = graph.placeholder(shape: shape, dataType: .float32, name: nil)
        let target = graph.placeholder(shape: shape, dataType: .float32, name: nil)
        let loss = graph.linearDetailLoss(prediction: prediction, target: target)
        let gradient = try XCTUnwrap(graph.gradients(of: loss, with: [prediction], name: nil)[prediction])
        let count = channels * height * width
        let values = (0..<count).map { Float(sin(Double($0 * 13 + 7)) * 0.7 + 0.3) }
        let targets = (0..<count).map { Float(cos(Double($0 * 7 + 3)) * 0.4 + 0.5) }
        let device = MPSGraphDevice(mtlDevice: try XCTUnwrap(MTLCreateSystemDefaultDevice()))
        func data(_ values: [Float]) -> MPSGraphTensorData {
            values.withUnsafeBytes { MPSGraphTensorData(device: device, data: Data($0), shape: shape, dataType: .float32) }
        }
        let result = graph.run(feeds: [prediction: data(values), target: data(targets)],
            targetTensors: [loss, gradient], targetOperations: nil)
        var actualLoss: Float = 0, actualGradient = Array(repeating: Float(0), count: count)
        withUnsafeMutableBytes(of: &actualLoss) { result[loss]!.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        actualGradient.withUnsafeMutableBytes { result[gradient]!.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        var expectedLoss = 0.0, expectedGradient = Array(repeating: 0.0, count: count)
        func sign(_ value: Double) -> Double { value > 0 ? 1 : value < 0 ? -1 : 0 }
        for index in 0..<count {
            let error = Double(values[index]) - Double(targets[index])
            expectedLoss += abs(error) / Double(count)
            expectedGradient[index] += sign(error) / Double(count)
        }
        for distance in [1, 2, 4, 8] {
            for horizontal in [false, true] {
                let pairCount = channels * (horizontal ? height * (width - distance) : (height - distance) * width)
                let coefficient = 4 / Double(distance * pairCount)
                for channel in 0..<channels { for y in 0..<height { for x in 0..<width {
                    let otherX = x + (horizontal ? distance : 0), otherY = y + (horizontal ? 0 : distance)
                    guard otherX < width, otherY < height else { continue }
                    let a = channel * width * height + y * width + x
                    let b = channel * width * height + otherY * width + otherX
                    let error = (Double(values[b]) - Double(values[a])) - (Double(targets[b]) - Double(targets[a]))
                    expectedLoss += abs(error) * coefficient
                    expectedGradient[b] += sign(error) * coefficient
                    expectedGradient[a] -= sign(error) * coefficient
                } } }
            }
        }
        XCTAssertEqual(Double(actualLoss), expectedLoss, accuracy: 3e-6)
        for index in 0..<count {
            XCTAssertEqual(Double(actualGradient[index]), expectedGradient[index], accuracy: 2e-7,
                "Independent linear/detail derivative at native component \(index)")
        }
    }

    func testGateDepthwiseNormalizationAttentionAndResidualGainGradientsMatchFiniteDifferences() throws {
        try available()
        let architecture = try compact("depth", width: 4)
        let options = try NativeTrainingOptions(weightDecay: 0, gradientClipNorm: 0, scalarLoss: .linearDetailV1)
        let size = 32, input = samples(3 * size * size), target = samples(size * size)
        let weights = NativeCompactNetwork.initialWeights(architecture: architecture, seed: 37)
        let graph = MPSGraph()
        let device = MPSGraphDevice(mtlDevice: try XCTUnwrap(MTLCreateSystemDefaultDevice()))
        let rgb = graph.placeholder(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32, name: nil)
        let targetTensor = graph.constant(target.withUnsafeBytes { Data($0) },
            shape: [1, 1, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        let parameters = architecture.parameterShapes.mapValues {
            graph.placeholder(shape: $0.map(NSNumber.init(value:)), dataType: .float32, name: nil)
        }
        let prediction = graph.compactMaterialNetwork(input: rgb, architecture: architecture,
            parameters: parameters, adapter: nil, channelsLast: false)
        // The independent output-gradient test above verifies the production
        // nonsmooth linear/detail objective. A smooth MSE probe isolates the
        // network derivative here without crossing L1 cusps or subtracting two
        // coarsely rounded Float32 reductions at tiny parameter perturbations.
        let error = graph.subtraction(prediction, targetTensor, name: nil)
        let loss = graph.mean(of: graph.square(with: error, name: nil), axes: [0, 1, 2, 3], name: nil)
        let names = ["input.weight", "encoder.0.expand.weight", "encoder.0.depthwise.weight",
                     "encoder.0.norm1.weight", "encoder.0.attention.weight", "middle.project.weight",
                     "decoder.0.beta", "decoder.0.gamma"]
        let derivatives = graph.gradients(of: loss, with: names.map { parameters[$0]! }, name: nil)
        func data(_ values: [Float], shape: [Int]) -> MPSGraphTensorData {
            values.withUnsafeBytes {
                MPSGraphTensorData(device: device, data: Data($0), shape: shape.map(NSNumber.init(value:)), dataType: .float32)
            }
        }
        var feeds = Dictionary(uniqueKeysWithValues: parameters.map { name, tensor in
            (tensor, data(weights[name]!.values, shape: weights[name]!.shape))
        })
        feeds[rgb] = data(input, shape: [1, 3, size, size])
        let outputs = graph.run(feeds: feeds, targetTensors: names.map { derivatives[parameters[$0]!]! }, targetOperations: nil)
        var gradients: [String: [Float]] = [:]
        for name in names {
            var values = Array(repeating: Float(0), count: weights[name]!.values.count)
            values.withUnsafeMutableBytes { outputs[derivatives[parameters[name]!]!]!.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
            gradients[name] = values
        }
        func independentObjective(_ candidateWeights: [String: NativeTensor]) throws -> Double {
            let probe = try NativeTrainingSession(architecture: architecture, options: options, seed: 0, weights: candidateWeights)
            let values = try probe.predict(input: input, batchSize: 1, height: size, width: size)
            return zip(values, target).reduce(0.0) { sum, pair in
                let difference = Double(pair.0) - Double(pair.1)
                return sum + difference * difference
            } / Double(target.count)
        }
        for name in names {
            let derivative = gradients[name]!
            let index = try XCTUnwrap(derivative.indices.max { abs(derivative[$0]) < abs(derivative[$1]) })
            XCTAssertGreaterThan(abs(derivative[index]), 1e-7, name + " must receive a teaching signal")
            let delta: Float = 0.001
            var loss: [Double] = []
            for sign: Float in [-1, 1] {
                var perturbed = weights
                perturbed[name]!.values[index] += sign * delta
                loss.append(try independentObjective(perturbed))
            }
            let numerical = (loss[1] - loss[0]) / (2 * Double(delta))
            XCTAssertEqual(Double(derivative[index]), numerical,
                accuracy: max(0.0001, abs(Double(derivative[index])) * 0.01), name + " smooth finite difference")
        }
    }

    func testHistoricalFullNativeCheckpointPredictsSavedFloat32FieldWithoutAnyTraining() throws {
        guard let path = ProcessInfo.processInfo.environment["LOOM_COMPACT_LEGACY_FIXTURE"], !path.isEmpty else {
            throw XCTSkip("Set LOOM_COMPACT_LEGACY_FIXTURE to the retained full-native-training/real-bark receipt directory")
        }
        try available()
        let folder = URL(fileURLWithPath: path, isDirectory: true)
        let checkpoint = folder.appendingPathComponent("known-crop-probe.safetensors")
        let checkpointBytes = try Data(contentsOf: checkpoint)
        let checkpointHash = SHA256.hash(data: checkpointBytes).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(checkpointHash, "5bc66a3742eb8eef26174f3fb4283ae9d5b7acab2b73ad1d0d2d4cac74cf45c7")
        func floats(_ name: String, count: Int) throws -> [Float] {
            let data = try Data(contentsOf: folder.appendingPathComponent(name), options: .mappedIfSafe)
            guard data.count == count * 4 else { throw NativeIOError.invalid("Historical fixture has an unexpected native grid") }
            return data.withUnsafeBytes { bytes in
                (0..<count).map { Float(bitPattern: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self).littleEndian) }
            }
        }
        let size = 1024, count = size * size
        let input = try floats("input.float32", count: 3 * count)
        let target = try floats("reference.float32", count: count)
        let expected = try floats("prediction-step-00200.float32", count: count)
        let architecture = try compact("depth", width: 32)
        let weights = try NativeWeights.read(from: checkpoint)
        XCTAssertEqual(weights.mapValues(\.shape), architecture.parameterShapes)
        let model = try NativeTrainingSession(architecture: architecture,
            options: NativeTrainingOptions(scalarLoss: .linearDetailV1), seed: 0, weights: weights)
        let actual = try model.predict(input: input, batchSize: 1, height: size, width: size)
        let errors = zip(actual, expected).map { abs(Double($0.0) - Double($0.1)) }
        let maximumError = errors.max()!, meanError = errors.reduce(0, +) / Double(count)
        let loss = try model.evaluate(input: input, target: target, batchSize: 1, height: size, width: size)
        print("compact legacy full-native parity: max=\(maximumError) mean=\(meanError) loss=\(loss)")
        XCTAssertLessThan(maximumError, 2e-5)
        XCTAssertLessThan(meanError, 1e-6)
        XCTAssertEqual(loss, 0.15394018590450287, accuracy: 2e-6)
        XCTAssertEqual(try model.weights(), weights)
        XCTAssertEqual(model.updateCount, 0)
        XCTAssertEqual(try Data(contentsOf: checkpoint), checkpointBytes)
    }

    private func compact(_ map: String, width: Int) throws -> NativeArchitecture {
        try NativeArchitecture(mapKind: map, width: width, blocks: 1, identifier: NativeArchitecture.compactIdentifier)
    }
    private func samples(_ count: Int) -> [Float] { (0..<count).map { Float(($0 * 13 + 7) % 97) / 96 } }
    private func available() throws { if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") } }
}
