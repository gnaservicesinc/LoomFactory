import CoreML
import Metal
import XCTest
@testable import LoomNative

final class NativeCompactCoreMLTests: XCTestCase {
    func testCompactExportsMatchNativeAllMapKindsAndFlexibleRectangles() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal GPU") }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for kind in ["depth", "roughness", "normal"] {
            let architecture = try NativeArchitecture(mapKind: kind, width: 32, blocks: 1,
                                                       identifier: NativeArchitecture.compactIdentifier)
            let native = try NativeTrainingSession(architecture: architecture, options: NativeTrainingOptions(scalarLoss: .linearDetailV1), seed: 42,
                                                   channelsLast: false)
            let compiled = try NativeCoreML.export(weights: native.weights(), architecture: architecture,
                                                   imageSize: 32, to: folder.appendingPathComponent(kind + ".mlmodel"))
            defer { try? FileManager.default.removeItem(at: compiled) }
            let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
            let exported = try MLModel(contentsOf: compiled, configuration: configuration)
            XCTAssertEqual(exported.modelDescription.inputDescriptionsByName["rgb"]?.multiArrayConstraint?.dataType, .float32)
            XCTAssertEqual(exported.modelDescription.outputDescriptionsByName["material_map"]?.multiArrayConstraint?.dataType, .float32)
            XCTAssertEqual(exported.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
                           ["loom.schema": "loom-factory-model-v1", "loom.architecture": NativeArchitecture.compactIdentifier])
            for (width, height) in [(32, 32), (48, 40)] {
                let input = (0..<(3 * width * height)).map { Float(($0 * 7 + 3) % 101) / 100 }
                let expected = try native.predict(input: input, batchSize: 1, height: height, width: width)
                let actual = try prediction(exported, input: input, width: width, height: height)
                XCTAssertEqual(actual.count, architecture.outputChannels * width * height)
                let error = zip(expected, actual).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
                XCTAssertLessThan(error, 5e-5, "\(kind) \(width)x\(height) native/export parity")
            }
        }
    }

    func testCompactExportKeepsRawScalarAndNormalCodesWithoutClamping() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal GPU") }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for kind in ["depth", "normal"] {
            let architecture = try NativeArchitecture(mapKind: kind, width: 32, blocks: 1,
                                                       identifier: NativeArchitecture.compactIdentifier)
            let native = try NativeTrainingSession(architecture: architecture, options: NativeTrainingOptions(scalarLoss: .linearDetailV1), seed: 42,
                                                   channelsLast: false)
            var weights = try native.weights()
            let output = weights["output.weight"]!
            weights["output.weight"] = NativeTensor(shape: output.shape, values: Array(repeating: 0, count: output.values.count))
            let codes: [Float] = kind == "normal" ? [-0.25, 0.5, 1.25] : [-0.25]
            weights["output.bias"] = NativeTensor(shape: weights["output.bias"]!.shape, values: codes)
            let compiled = try NativeCoreML.export(weights: weights, architecture: architecture,
                                                   imageSize: 32, to: folder.appendingPathComponent(kind + ".mlmodel"))
            defer { try? FileManager.default.removeItem(at: compiled) }
            let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
            let exported = try MLModel(contentsOf: compiled, configuration: configuration)
            let values = try prediction(exported, input: Array(repeating: 0.5, count: 3 * 32 * 32), width: 32, height: 32)
            for channel in codes.indices {
                XCTAssertEqual(Array(values[(channel * 32 * 32)..<((channel + 1) * 32 * 32)]),
                               Array(repeating: codes[channel], count: 32 * 32))
            }
        }
    }

    func testCompactExportRejectsDefaultGridThatWouldNeedPadding() throws {
        let architecture = try NativeArchitecture(mapKind: "depth", width: 32, blocks: 1,
                                                   identifier: NativeArchitecture.compactIdentifier)
        let weights = architecture.parameterShapes.mapValues { shape in
            NativeTensor(shape: shape, values: Array(repeating: 0, count: shape.reduce(1, *)))
        }
        XCTAssertThrowsError(try NativeCoreML.modelData(weights: weights, mapKind: "depth", width: 32,
                                                        blocks: 1, imageSize: 33, identifier: architecture.identifier))
    }

    private func prediction(_ model: MLModel, input: [Float], width: Int, height: Int) throws -> [Float] {
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        array.dataPointer.assumingMemoryBound(to: Float.self).update(from: input, count: input.count)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)])
        let output = try XCTUnwrap(model.prediction(from: provider).featureValue(for: "material_map")?.multiArrayValue)
        XCTAssertEqual(output.shape.map(\.intValue), [1, output.shape[1].intValue, height, width])
        return (0..<output.count).map { output[$0].floatValue }
    }
}
