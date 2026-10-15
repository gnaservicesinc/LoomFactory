import CoreML
import CryptoKit
import Metal
import XCTest
@testable import LoomNative

final class NativeResidualV2Tests: XCTestCase {
    func testDistinctIdentityRetainsFullResolutionWeightsAndExplicitActivationContract() throws {
        for map in ["depth", "roughness", "normal"] {
            let old = try NativeArchitecture(mapKind: map, width: 3, blocks: 2)
            let new = try architecture(map, width: 3, blocks: 2)
            XCTAssertEqual(new.identifier, "loom-factory-material-residual-v2")
            XCTAssertTrue(old.isFullResolutionResidual)
            XCTAssertTrue(new.isFullResolutionResidual)
            XCTAssertTrue(new.isResidualV2)
            XCTAssertFalse(old.isResidualV2)
            XCTAssertFalse(new.isPyramid)
            XCTAssertFalse(new.isCompact)
            XCTAssertFalse(new.usesLinearOutput)
            XCTAssertEqual(new.parameterShapes, old.parameterShapes)
            XCTAssertEqual(new.convolutionNames, old.convolutionNames)
            XCTAssertEqual(new.parameterCount, old.parameterCount)
            XCTAssertTrue(new.convolutionNames.allSatisfy { new.convolutionStride($0) == 1 })
            XCTAssertTrue(new.supportsGrid(height: 33, width: 35))
            XCTAssertEqual(new.descriptor["hidden_activation"] as? String, "leaky-relu")
            XCTAssertEqual(new.descriptor["negative_slope"] as? Double, 0.1)
            XCTAssertEqual(new.descriptor["padding"] as? String, "zero")
            XCTAssertNil(old.descriptor["hidden_activation"])
            XCTAssertNil(old.descriptor["negative_slope"])
        }
        XCTAssertFalse(try NativeArchitecture(mapKind: "depth", width: 2, blocks: 1,
            identifier: NativeArchitecture.pyramidIdentifier).isFullResolutionResidual)
        XCTAssertFalse(try NativeArchitecture(mapKind: "depth", width: 2, blocks: 1,
            identifier: NativeArchitecture.compactIdentifier).isFullResolutionResidual)
    }

    func testLegacyCoreMLNetworksRemainByteIdenticalToPreV2Constructors() throws {
        // Captured from fccff21's independent CPU-only constructors. Hash the
        // network field alone: dictionary metadata ordering is not significant.
        let expected = [
            NativeArchitecture.identifier: "8b7e1eec4a4b848614e9af2ad1dcdf5bd9c257e6f1a85d8e092a8e7994be1a7d",
            NativeArchitecture.pyramidIdentifier: "5491120f2a5aa86766912826fcbbb361767d2cfe86d62544e432bcaca485c797",
            NativeArchitecture.compactIdentifier: "8c30c672cc3036a60b20bfab90ac056d06e9da8375cdea6660226f1a139a4375"
        ]
        for (identifier, digest) in expected {
            let architecture = try NativeArchitecture(mapKind: "depth", width: 2, blocks: 1, identifier: identifier)
            let weights = architecture.parameterShapes.mapValues {
                NativeTensor(shape: $0, values: Array(repeating: 0, count: $0.reduce(1, *)))
            }
            let model = try NativeCoreML.modelData(weights: weights, mapKind: "depth", width: 2,
                blocks: 1, imageSize: 32, identifier: identifier)
            let network = try XCTUnwrap(wireMessage(model, field: 500))
            XCTAssertEqual(SHA256.hash(data: network).map { String(format: "%02x", $0) }.joined(), digest, identifier)
        }
    }

    func testNegativeHiddenFeaturesMatchIndependentForwardAndAnalyticGradients() throws {
        try available()
        let oldArchitecture = try NativeArchitecture(mapKind: "depth", width: 1, blocks: 1)
        let newArchitecture = try architecture("depth", width: 1, blocks: 1)
        let weights = negativeFeatureWeights(newArchitecture)
        let height = 7, width = 9, pixels = height * width
        let input = Array(repeating: Float(0.25), count: 3 * pixels)
        let target = Array(repeating: Float(0.9), count: pixels)
        // Every kernel has one nonzero center coefficient, so the equations
        // below hold independently at every pixel, including image borders.
        // input: -.75 -> -.075; branch: -.275 -> -.0275;
        // residual: -.1025 -> -.01025; output logit: -.01025.
        let logit = -0.01025, prediction = 1 / (1 + exp(-logit))
        let levelGradient = prediction - Double(Float(0.9))
        let expectedLoss = -logit * Double(Float(0.9)) + log1p(exp(logit))
        for channelsLast in [false, true] {
            let new = try NativeTrainingSession(architecture: newArchitecture, options: options(), seed: 1,
                weights: weights, channelsLast: channelsLast)
            let old = try NativeTrainingSession(architecture: oldArchitecture, options: options(), seed: 1,
                weights: weights, channelsLast: channelsLast)
            XCTAssertTrue(try old.predict(input: input, batchSize: 1, height: height, width: width).allSatisfy { $0 == 0.5 })
            for value in try new.predict(input: input, batchSize: 1, height: height, width: width) {
                XCTAssertEqual(Double(value), prediction, accuracy: 1e-7)
            }
            _ = try old.train(input: input, target: target, batchSize: 1, height: height, width: width)
            let dead = try old.accumulatedGradients()
            XCTAssertTrue(dead["input.weight"]!.values.allSatisfy { $0 == 0 })
            XCTAssertTrue(dead["body.0.conv1.weight"]!.values.allSatisfy { $0 == 0 })
            XCTAssertTrue(dead["body.0.conv2.weight"]!.values.allSatisfy { $0 == 0 })
            let loss = try new.train(input: input, target: target, batchSize: 1, height: height, width: width)
            XCTAssertEqual(loss, expectedLoss, accuracy: 1e-6)
            let gradients = try new.accumulatedGradients()
            for (name, expected) in ["input.bias": levelGradient * 0.011,
                "body.0.conv1.bias": levelGradient * 0.01,
                "body.0.conv2.bias": levelGradient * 0.1,
                "output.bias": levelGradient] {
                XCTAssertEqual(Double(gradients[name]!.values[0]), expected, accuracy: 1e-6, name)
            }
            for name in ["input.weight", "body.0.conv1.weight", "body.0.conv2.weight"] {
                XCTAssertTrue(gradients[name]!.values.contains { $0 != 0 }, name + " must retain a teaching signal")
            }
            _ = try new.applyGradients(learningRate: 0.01)
            XCTAssertLessThan(try new.evaluate(input: input, target: target, batchSize: 1, height: height, width: width), loss)
        }
    }

    func testNegativeBranchGradientsMatchFiniteDifferencesAndZeroCrossingsStayFinite() throws {
        try available()
        let architecture = try architecture("depth", width: 1, blocks: 1)
        let weights = negativeFeatureWeights(architecture)
        let input = Array(repeating: Float(0.25), count: 3 * 7 * 9)
        let target = Array(repeating: Float(0.9), count: 7 * 9)
        let model = try NativeTrainingSession(architecture: architecture, options: options(), seed: 1,
            weights: weights, channelsLast: false)
        _ = try model.train(input: input, target: target, batchSize: 1, height: 7, width: 9)
        let gradients = try model.accumulatedGradients()
        for name in ["input.bias", "body.0.conv1.bias", "body.0.conv2.bias", "output.weight"] {
            let index = name.hasSuffix(".weight") ? 4 : 0
            var losses: [Double] = []
            let delta: Float = 0.01
            for sign: Float in [-1, 1] {
                var changed = weights
                changed[name]!.values[index] += sign * delta
                let probe = try NativeTrainingSession(architecture: architecture, options: options(), seed: 1,
                    weights: changed, channelsLast: false)
                losses.append(try probe.evaluate(input: input, target: target, batchSize: 1, height: 7, width: 9))
            }
            XCTAssertEqual(Double(gradients[name]!.values[index]),
                (losses[1] - losses[0]) / (2 * Double(delta)), accuracy: 1e-5, name)
        }
        // The activation has a kink at zero; its chosen subgradient must stay
        // finite and preserve a path. Do not assert a smooth central derivative.
        var zeroCrossing = weights
        zeroCrossing["input.bias"]!.values = [-0.25]
        let zero = try NativeTrainingSession(architecture: architecture, options: options(), seed: 1,
            weights: zeroCrossing, channelsLast: false)
        _ = try zero.train(input: input, target: target, batchSize: 1, height: 7, width: 9)
        let atZero = try zero.accumulatedGradients()
        XCTAssertTrue(atZero.values.allSatisfy { $0.values.allSatisfy(\.isFinite) })
        XCTAssertNotEqual(atZero["input.bias"]!.values[0], 0)
    }

    func testFreshInitializationIsIdenticalButNewGraphAndResumedOptimizerRemainExplicit() throws {
        try available()
        let architecture = try architecture("depth", width: 3, blocks: 2)
        let adaptiveOptions = try NativeTrainingOptions(weightDecay: 0, scalarLoss: .logitCrossEntropyV2)
        let old = try NativeTrainingSession(architecture: NativeArchitecture(mapKind: "depth", width: 3, blocks: 2),
            options: adaptiveOptions, seed: 42, channelsLast: false)
        let new = try NativeTrainingSession(architecture: architecture, options: adaptiveOptions, seed: 42, channelsLast: false)
        XCTAssertEqual(try new.weights(), try old.weights(), "The same tensor initialization does not reinterpret a saved architecture identity")
        let input = (0..<(3 * 7 * 9)).map { Float(($0 * 7) % 31) / 30 }
        let target = (0..<(7 * 9)).map { Float(($0 * 11) % 29) / 28 }
        XCTAssertNotEqual(try new.predict(input: input, batchSize: 1, height: 7, width: 9),
            try old.predict(input: input, batchSize: 1, height: 7, width: 9))
        _ = try new.train(input: input, target: target, batchSize: 1, height: 7, width: 9)
        _ = try new.applyGradients(learningRate: 0.001)
        XCTAssertTrue(try new.optimizerState().values.contains { $0.values.contains { $0 != 0 } },
            "Continuation must restore real Adam moments, not just zero state")
        let resumed = try NativeTrainingSession(architecture: architecture, options: adaptiveOptions, seed: 9,
            weights: new.weights(), optimizerState: new.optimizerState(), initialUpdateCount: new.updateCount, channelsLast: false)
        for candidate in [new, resumed] {
            _ = try candidate.train(input: input, target: target, batchSize: 1, height: 7, width: 9)
            _ = try candidate.applyGradients(learningRate: 0.001)
        }
        XCTAssertEqual(try new.weights(), try resumed.weights())
        XCTAssertEqual(try new.optimizerState(), try resumed.optimizerState())
    }

    func testCoreMLExportPreservesLeakyFeaturesForAllProductsAndOddRectangles() throws {
        try available()
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for map in ["depth", "roughness", "normal"] {
            let architecture = try architecture(map, width: 3, blocks: 2)
            let native = try NativeTrainingSession(architecture: architecture, options: options(), seed: 42, channelsLast: false)
            let compiled = try NativeCoreML.export(weights: native.weights(), architecture: architecture,
                imageSize: 32, to: folder.appendingPathComponent(map + ".mlmodel"))
            defer { try? FileManager.default.removeItem(at: compiled) }
            let config = MLModelConfiguration(); config.computeUnits = .cpuOnly
            let model = try MLModel(contentsOf: compiled, configuration: config)
            XCTAssertEqual(model.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
                ["loom.schema": "loom-factory-model-v1", "loom.architecture": NativeArchitecture.residualV2Identifier])
            for (height, width) in [(32, 32), (33, 35)] {
                let input = (0..<(3 * width * height)).map { Float(($0 * 13) % 101) / 100 }
                let expected = try native.predict(input: input, batchSize: 1, height: height, width: width)
                let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
                array.dataPointer.assumingMemoryBound(to: Float.self).update(from: input, count: input.count)
                let provider = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)])
                let output = try XCTUnwrap(model.prediction(from: provider).featureValue(for: "material_map")?.multiArrayValue)
                XCTAssertEqual(output.shape.map(\.intValue), [1, architecture.outputChannels, height, width])
                let error = expected.indices.reduce(Float(0)) { max($0, abs(expected[$1] - output[$1].floatValue)) }
                XCTAssertLessThan(error, 2e-5, "\(map) \(height)x\(width)")
            }
        }
    }

    private func architecture(_ map: String, width: Int, blocks: Int) throws -> NativeArchitecture {
        try NativeArchitecture(mapKind: map, width: width, blocks: blocks, identifier: NativeArchitecture.residualV2Identifier)
    }
    private func options() throws -> NativeTrainingOptions {
        try NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0,
            gradientClipNorm: 0, scalarLoss: .logitCrossEntropyV2)
    }
    private func available() throws {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") }
    }
    private func negativeFeatureWeights(_ architecture: NativeArchitecture) -> [String: NativeTensor] {
        var weights = architecture.parameterShapes.mapValues {
            NativeTensor(shape: $0, values: Array(repeating: 0, count: $0.reduce(1, *)))
        }
        for name in ["input.weight", "body.0.conv1.weight", "body.0.conv2.weight", "output.weight"] {
            weights[name]!.values[4] = 1
        }
        weights["input.bias"]!.values = [-1]
        weights["body.0.conv1.bias"]!.values = [-0.2]
        return weights
    }

    private func wireMessage(_ encoded: Data, field: UInt64) -> Data? {
        let bytes = [UInt8](encoded)
        var index = 0
        func integer() -> UInt64? {
            var value: UInt64 = 0, shift = 0
            while index < bytes.count, shift < 64 {
                let byte = bytes[index]; index += 1
                value |= UInt64(byte & 0x7f) << shift
                if byte < 128 { return value }
                shift += 7
            }
            return nil
        }
        while index < bytes.count {
            guard let key = integer() else { return nil }
            switch key & 7 {
            case 0: guard integer() != nil else { return nil }
            case 1: index += 8
            case 2:
                guard let length = integer(), length <= UInt64(bytes.count - index) else { return nil }
                let end = index + Int(length)
                if key >> 3 == field { return Data(bytes[index..<end]) }
                index = end
            case 5: index += 4
            default: return nil
            }
        }
        return nil
    }
}
