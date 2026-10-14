import CoreGraphics
import CoreML
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LoomNative

final class NativeIOTests: XCTestCase {
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("loom-native-io-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func testSafetensorsRetainNamesShapesAndFloatBitPatterns() throws {
        let url = try directory().appendingPathComponent("model.safetensors")
        let tensors = ["input.weight": NativeTensor(shape: [1, 1, 2, 2], values: [0, -0.125, Float.leastNormalMagnitude, 1.234567]),
                       "input.bias": NativeTensor(shape: [1], values: [-0.0])]
        try NativeWeights.write(tensors, to: url, metadata: ["schema": "loom-factory-model-v1"])
        let restored = try NativeWeights.read(from: url)
        XCTAssertEqual(Set(restored.keys), Set(tensors.keys))
        for (name, original) in tensors {
            XCTAssertEqual(restored[name]?.shape, original.shape)
            XCTAssertEqual(restored[name]?.values.map(\.bitPattern), original.values.map(\.bitPattern))
        }
        let file = try Data(contentsOf: url)
        let length = file.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        XCTAssertEqual(length % 8, 0)
        let header = try XCTUnwrap(JSONSerialization.jsonObject(with: file.subdata(in: 8..<(8 + Int(length)))) as? [String: Any])
        XCTAssertEqual((header["__metadata__"] as? [String: String])?["schema"], "loom-factory-model-v1")
    }

    func testSafetensorsRejectInvalidShapesNonFiniteAndOverlappingOffsets() throws {
        let url = try directory().appendingPathComponent("invalid.safetensors")
        XCTAssertThrowsError(try NativeWeights.write(["weight": NativeTensor(shape: [2], values: [1])], to: url))
        XCTAssertThrowsError(try NativeWeights.write(["weight": NativeTensor(shape: [1], values: [.nan])], to: url))
        let header = Data(#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.utf8)
        var length = UInt64(header.count).littleEndian
        var file = Data(bytes: &length, count: 8); file.append(header); file.append(Data(repeating: 0, count: 4))
        try file.write(to: url)
        XCTAssertThrowsError(try NativeWeights.read(from: url))
    }

    func testSixteenBitPNGCodesArePreservedWithoutModifyingSource() throws {
        let url = try directory().appendingPathComponent("precision.png")
        let codes: [UInt16] = [1, 256, 32768, 65535]
        let data = codes.withUnsafeBytes { Data($0) }
        let provider = try XCTUnwrap(CGDataProvider(data: data as CFData))
        let image = try XCTUnwrap(CGImage(width: 2, height: 2, bitsPerComponent: 16, bitsPerPixel: 16,
                                        bytesPerRow: 4, space: CGColorSpaceCreateDeviceGray(),
                                        bitmapInfo: [.byteOrder16Little], provider: provider,
                                        decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let original = try Data(contentsOf: url)
        let scalar = try NativeImages.readTarget(from: url, mapKind: "depth")
        XCTAssertEqual(scalar.width, 2); XCTAssertEqual(scalar.height, 2); XCTAssertEqual(scalar.channels, 1)
        XCTAssertEqual(scalar.values, codes.map { Float($0) / 65535 })
        let rgb = try NativeImages.readRGB(from: url)
        XCTAssertEqual(rgb.values, scalar.values.flatMap { [$0, $0, $0] })
        XCTAssertEqual(try Data(contentsOf: url), original)
    }

    func testFloatEXRTargetAndLinearDiffuseUseSeparateConversions() throws {
        let url = try directory().appendingPathComponent("linear.exr")
        let pixels: [Float] = [0, 0.0001, 0.25, 1]
        let data = pixels.withUnsafeBytes { Data($0) }
        let provider = try XCTUnwrap(CGDataProvider(data: data as CFData))
        let image = try XCTUnwrap(CGImage(width: 2, height: 2, bitsPerComponent: 32, bitsPerPixel: 32,
                                        bytesPerRow: 8, space: try XCTUnwrap(CGColorSpace(name: CGColorSpace.linearGray)),
                                        bitmapInfo: [.byteOrder32Little, .floatComponents], provider: provider,
                                        decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, "com.ilm.openexr-image" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyDepth: 32] as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let original = try Data(contentsOf: url)
        let target = try NativeImages.readTarget(from: url, mapKind: "depth")
        for index in pixels.indices { XCTAssertEqual(target.values[index], pixels[index], accuracy: 1e-7) }
        let rgb = try NativeImages.readRGB(from: url)
        for index in pixels.indices {
            let value = target.values[index]
            let expected = value <= 0.0031308 ? value * 12.92 : 1.055 * pow(value, 1 / 2.4) - 0.055
            XCTAssertEqual(rgb.values[index * 3], expected, accuracy: 1e-7)
        }
        XCTAssertEqual(try Data(contentsOf: url), original)
    }

    func testFloatTIFFScalarPrecisionIsNotScaledOrQuantized() throws {
        let url = try directory().appendingPathComponent("precision.tiff")
        let pixels: [Float] = [0.0001234567, 0.23456789, 0.8765432, 1]
        let data = pixels.withUnsafeBytes { Data($0) }
        let provider = try XCTUnwrap(CGDataProvider(data: data as CFData))
        let image = try XCTUnwrap(CGImage(width: 2, height: 2, bitsPerComponent: 32, bitsPerPixel: 32,
                                        bytesPerRow: 8, space: try XCTUnwrap(CGColorSpace(name: CGColorSpace.linearGray)),
                                        bitmapInfo: [.byteOrder32Little, .floatComponents], provider: provider,
                                        decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.tiff.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let target = try NativeImages.readTarget(from: url, mapKind: "roughness")
        XCTAssertEqual(target.values.map(\.bitPattern), pixels.map(\.bitPattern))
    }

    func testCropRotationFlipsAndNormalVectorsMatchStoredCoordinates() throws {
        let input = NativeImage(width: 2, height: 2, channels: 3,
                                values: [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4])
        let target = NativeImage(width: 2, height: 2, channels: 3,
                                 values: [0.1, 0.2, 0.9, 0.3, 0.4, 0.9, 0.5, 0.6, 0.9, 0.7, 0.8, 0.9])
        let transform = NativeCropTransform(x: 0, y: 0, size: 2, horizontalFlip: true, quarterTurns: 1)
        let pair = try NativeImages.pair(input: input, target: target, transform: transform, mapKind: "normal")
        XCTAssertEqual(pair.input.shape, [1, 3, 2, 2])
        XCTAssertEqual(Array(pair.input.values.prefix(4)), [0.1, 0.3, 0.2, 0.4])
        let expectedX: [Float] = [0.8, 0.4, 0.6, 0.2]
        let expectedY: [Float] = [0.9, 0.5, 0.7, 0.3]
        for index in 0..<4 {
            XCTAssertEqual(pair.target.values[index], expectedX[index], accuracy: 1e-6)
            XCTAssertEqual(pair.target.values[4 + index], expectedY[index], accuracy: 1e-6)
            XCTAssertEqual(pair.target.values[8 + index], 0.9, accuracy: 1e-6)
        }
    }

    func testDirectStoredRegionMatchesFullImagePathForScalarAndNormalTransforms() throws {
        let folder = try directory()
        let inputURL = folder.appendingPathComponent("rgb16.png"), scalarURL = folder.appendingPathComponent("scalar16.png")
        var rgbCodes: [UInt16] = [], scalarCodes: [UInt16] = []
        for index in 0..<72 { rgbCodes.append(UInt16((index * 891 + 127) % 65536)) }
        for index in 0..<24 { scalarCodes.append(UInt16((index * 2411 + 5) % 65536)) }
        try writePNG(codes: rgbCodes, channels: 3, width: 6, height: 4, to: inputURL)
        try writePNG(codes: scalarCodes, channels: 1, width: 6, height: 4, to: scalarURL)
        let originalInput = try Data(contentsOf: inputURL), originalScalar = try Data(contentsOf: scalarURL)
        let decodedInput = try NativeImages.readRGB(from: inputURL)
        for kind in ["depth", "roughness", "normal"] {
            let targetURL = kind == "normal" ? inputURL : scalarURL
            let convention = kind == "normal" ? "DirectX" : "OpenGL"
            let decodedTarget = try NativeImages.readTarget(from: targetURL, mapKind: kind, normalConvention: convention)
            for turns in 0..<4 {
                let transform = NativeCropTransform(x: 2, y: 1, size: 3, horizontalFlip: true,
                                                    verticalFlip: turns % 2 == 0, quarterTurns: turns)
                let expected = try NativeImages.pair(input: decodedInput, target: decodedTarget, transform: transform,
                                                      mapKind: kind, exaggerate: kind == "depth")
                let actual = try NativeImages.pair(inputURL: inputURL, targetURL: targetURL, transform: transform,
                                                   mapKind: kind, normalConvention: convention, exaggerate: kind == "depth")
                XCTAssertEqual(actual.input, expected.input)
                XCTAssertEqual(actual.target, expected.target)
                XCTAssertEqual(actual.crop, transform)
            }
        }
        XCTAssertEqual(try Data(contentsOf: inputURL), originalInput)
        XCTAssertEqual(try Data(contentsOf: scalarURL), originalScalar)
    }

    func testDirectRegionValidatesScalarAgreementOutsideTheSelectedCrop() throws {
        let folder = try directory(), inputURL = folder.appendingPathComponent("input.png"), targetURL = folder.appendingPathComponent("target.png")
        let rgb = [UInt16](repeating: 2000, count: 4 * 4 * 3)
        var invalid = rgb; invalid[invalid.count - 1] = 2001
        try writePNG(codes: rgb, channels: 3, width: 4, height: 4, to: inputURL)
        try writePNG(codes: invalid, channels: 3, width: 4, height: 4, to: targetURL)
        XCTAssertThrowsError(try NativeImages.pair(inputURL: inputURL, targetURL: targetURL,
                                                 transform: NativeCropTransform(x: 0, y: 0, size: 2), mapKind: "depth")) {
            XCTAssertTrue($0.localizedDescription.contains("different color channels"))
        }
    }

    func testRawCodeCacheInvalidatesWhenThePhysicalImageChanges() throws {
        let folder = try directory(), inputURL = folder.appendingPathComponent("input.png"), targetURL = folder.appendingPathComponent("target.png")
        try writePNG(codes: [UInt16](repeating: 1000, count: 4 * 4 * 3), channels: 3, width: 4, height: 4, to: inputURL)
        try writePNG(codes: [UInt16](repeating: 2000, count: 4 * 4), channels: 1, width: 4, height: 4, to: targetURL)
        let transform = NativeCropTransform(x: 1, y: 1, size: 2)
        let first = try NativeImages.pair(inputURL: inputURL, targetURL: targetURL, transform: transform, mapKind: "depth")
        try writePNG(codes: [UInt16](repeating: 60000, count: 4 * 4), channels: 1, width: 4, height: 4, to: targetURL)
        let second = try NativeImages.pair(inputURL: inputURL, targetURL: targetURL, transform: transform, mapKind: "depth")
        XCTAssertEqual(first.input, second.input)
        XCTAssertEqual(first.target.values, [Float](repeating: Float(2000) / 65535, count: 4))
        XCTAssertEqual(second.target.values, [Float](repeating: Float(60000) / 65535, count: 4))
    }

    private func writePNG(codes: [UInt16], channels: Int, width: Int, height: Int, to url: URL) throws {
        let data = codes.withUnsafeBytes { Data($0) }
        let provider = try XCTUnwrap(CGDataProvider(data: data as CFData))
        let image = try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 16,
                                        bitsPerPixel: 16 * channels, bytesPerRow: width * 2 * channels,
                                        space: channels == 1 ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB(),
                                        bitmapInfo: [.byteOrder16Little], provider: provider, decode: nil,
                                        shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }

    func testAugmentationStreamsRefreshIndependentlyAndRestore() throws {
        var fixed = NativeAugmentationPlan(seed: 42, cropRefresh: "per_run", augmentationRefresh: "per_sample")
        let first = try fixed.transform(sampleID: "a", epoch: 1, width: 20, height: 20, size: 4, flip: true, rotate: true)
        let checkpoint = try JSONEncoder().encode(fixed)
        var restored = try JSONDecoder().decode(NativeAugmentationPlan.self, from: checkpoint)
        let second = try fixed.transform(sampleID: "a", epoch: 2, width: 20, height: 20, size: 4, flip: true, rotate: true)
        let resumed = try restored.transform(sampleID: "a", epoch: 2, width: 20, height: 20, size: 4, flip: true, rotate: true)
        XCTAssertEqual(first.x, second.x); XCTAssertEqual(first.y, second.y)
        XCTAssertEqual(second, resumed)
    }

    func testImageWorkHonorsAnAlreadyCancelledRun() throws {
        let cancellation = NativeCancellation(); cancellation.cancel()
        XCTAssertThrowsError(try NativeImages.readRGB(from: URL(fileURLWithPath: "/missing.png"), cancellation: cancellation)) {
            XCTAssertTrue($0 is CancellationError)
        }
        let input = NativeImage(width: 1, height: 1, channels: 3, values: [0, 0, 0])
        let target = NativeImage(width: 1, height: 1, channels: 1, values: [0])
        XCTAssertThrowsError(try NativeImages.pair(input: input, target: target, transform: NativeCropTransform(x: 0, y: 0, size: 1),
                                                 mapKind: "depth", cancellation: cancellation)) {
            XCTAssertTrue($0 is CancellationError)
        }
    }

    func testCoreMLScalarAndNormalExportMatchIndependentReferenceAndFlexibleShape() throws {
        for kind in ["depth", "roughness", "normal"] {
            let architecture = try NativeArchitecture(mapKind: kind, width: 2, blocks: 1)
            var weights: [String: NativeTensor] = [:]
            for (name, shape) in architecture.parameterShapes {
                let count = shape.reduce(1, *)
                let values = (0..<count).map { Float(($0 * 7 + name.count * 3) % 19 - 9) * 0.013 }
                weights[name] = NativeTensor(shape: shape, values: values)
            }
            let url = try directory().appendingPathComponent("\(kind).mlmodel")
            let compiled = try NativeCoreML.export(weights: weights, architecture: architecture, imageSize: 32, to: url)
            defer { try? FileManager.default.removeItem(at: compiled) }
            let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
            let model = try MLModel(contentsOf: compiled, configuration: configuration)
            for (width, height) in [(32, 32), (35, 33)] {
                let input = (0..<(3 * width * height)).map { Float(($0 * 11) % 101) / 100 }
                let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
                array.dataPointer.assumingMemoryBound(to: Float.self).update(from: input, count: input.count)
                let provider = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)])
                let prediction = try XCTUnwrap(model.prediction(from: provider).featureValue(for: "material_map")?.multiArrayValue)
                XCTAssertEqual(prediction.shape.map(\.intValue), [1, architecture.outputChannels, height, width])
                let expected = reference(input: input, width: width, height: height, architecture: architecture, weights: weights)
                var maximumError: Float = 0
                for index in expected.indices { maximumError = max(maximumError, abs(prediction[index].floatValue - expected[index])) }
                XCTAssertLessThan(maximumError, 2e-5, "\(kind) \(width)x\(height) export parity")
            }
        }
    }

    func testNormalExportZeroVectorMatchesNormalizeEpsilon() throws {
        let architecture = try NativeArchitecture(mapKind: "normal", width: 1, blocks: 1)
        let weights = architecture.parameterShapes.mapValues { NativeTensor(shape: $0, values: [Float](repeating: 0, count: $0.reduce(1, *))) }
        let compiled = try NativeCoreML.export(weights: weights, architecture: architecture, imageSize: 32,
                                             to: directory().appendingPathComponent("zero.mlmodel"))
        defer { try? FileManager.default.removeItem(at: compiled) }
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        let array = try MLMultiArray(shape: [1, 3, 32, 32], dataType: .float32)
        array.dataPointer.assumingMemoryBound(to: Float.self).initialize(repeating: 0, count: array.count)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)])
        let output = try XCTUnwrap(model.prediction(from: provider).featureValue(for: "material_map")?.multiArrayValue)
        for index in 0..<output.count { XCTAssertEqual(output[index].floatValue, 0.5, accuracy: 1e-6) }
    }

    // Independent scalar implementation verifies padding, OIHW weights, channel
    // ordering, residual additions and normal denominator without MPSGraph.
    private func reference(input: [Float], width: Int, height: Int, architecture: NativeArchitecture,
                           weights: [String: NativeTensor]) -> [Float] {
        let area = width * height
        func convolution(_ input: [Float], _ name: String) -> [Float] {
            let tensor = weights[name + ".weight"]!, bias = weights[name + ".bias"]!.values
            let outputChannels = tensor.shape[0], inputChannels = tensor.shape[1]
            var output = [Float](repeating: 0, count: outputChannels * area)
            for outputChannel in 0..<outputChannels {
                for y in 0..<height {
                    for x in 0..<width {
                        var value = bias[outputChannel]
                        for inputChannel in 0..<inputChannels {
                            for ky in 0..<3 {
                                let sy = y + ky - 1
                                guard sy >= 0 && sy < height else { continue }
                                for kx in 0..<3 {
                                    let sx = x + kx - 1
                                    guard sx >= 0 && sx < width else { continue }
                                    value += input[inputChannel * area + sy * width + sx] * tensor.values[((outputChannel * inputChannels + inputChannel) * 3 + ky) * 3 + kx]
                                }
                            }
                        }
                        output[outputChannel * area + y * width + x] = value
                    }
                }
            }
            return output
        }
        var value = convolution(input, "input").map { max(0, $0) }
        for index in 0..<architecture.blocks {
            let residual = convolution(convolution(value, "body.\(index).conv1").map { max(0, $0) }, "body.\(index).conv2")
            value = zip(value, residual).map { max(0, $0 + $1) }
        }
        value = convolution(value, "output").map { 1 / (1 + exp(-$0)) }
        if architecture.mapKind == "normal" {
            for pixel in 0..<area {
                let x = value[pixel] * 2 - 1, y = value[area + pixel] * 2 - 1, z = value[area * 2 + pixel] * 2 - 1
                let length = max(1e-6, sqrt(x * x + y * y + z * z))
                value[pixel] = x / length * 0.5 + 0.5
                value[area + pixel] = y / length * 0.5 + 0.5
                value[area * 2 + pixel] = z / length * 0.5 + 0.5
            }
        }
        return value
    }
}
