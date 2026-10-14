import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import LoomNative

final class NativeFloatEXRTests: XCTestCase {
    func testScalarFloat32ChannelsRetainAdjacentValuesSignedZeroAndExtendedRangeExactly() throws {
        let directory = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("depth.exr")
        let values: [Float] = [0.5, Float(0.5).nextUp, -0.25, 1.25, -0.0, 0, Float.leastNormalMagnitude, 0.12345679]
        try NativeFloatEXR.write(values, channels: 1, width: 4, height: 2, to: url)
        let decoded = try readUncompressedEXR(url)
        XCTAssertEqual(decoded.width, 4); XCTAssertEqual(decoded.height, 2)
        XCTAssertEqual(decoded.channelNames, ["Y"])
        XCTAssertEqual(decoded.channelTypes, [2], "EXR FLOAT channels must never become HALF or unsigned integer channels")
        XCTAssertEqual(decoded.planes[0], values.map(\.bitPattern))
        XCTAssertNotEqual(decoded.planes[0][0], decoded.planes[0][1], "Neighboring Float32 values must survive storage")
        let options = [kCGImageSourceShouldAllowFloat: true] as CFDictionary
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, options))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, options))
        XCTAssertEqual(image.width, 4); XCTAssertEqual(image.height, 2)
        XCTAssertEqual(image.bitsPerComponent, 32)
        XCTAssertTrue(image.bitmapInfo.contains(.floatComponents))
    }

    func testNormalRGBFloat32ChannelsKeepPlaneIdentityAndNativeImageIOValues() throws {
        let directory = try temporaryDirectory(); defer {
            try? FileManager.default.removeItem(at: directory); NativeImages.clearDecodedPixelCache()
        }
        let url = directory.appendingPathComponent("normal.exr"), width = 3, height = 2, count = width * height
        let red: [Float] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        let green: [Float] = [0.51, Float(0.51).nextUp, 0.53, 0.54, 0.55, 0.56]
        let blue: [Float] = [0.91, 0.92, 0.93, 0.94, 0.95, 0.96]
        try NativeFloatEXR.write(red + green + blue, channels: 3, width: width, height: height, to: url)
        let raw = try readUncompressedEXR(url)
        XCTAssertEqual(raw.channelNames, ["B", "G", "R"], "EXR payload channels are alphabetical")
        XCTAssertEqual(raw.channelTypes, [2, 2, 2])
        XCTAssertEqual(raw.planes, [blue, green, red].map { $0.map(\.bitPattern) })
        let image = try NativeImages.readTarget(from: url, mapKind: "normal", normalConvention: "OpenGL")
        XCTAssertEqual(image.channels, 3)
        for pixel in 0..<count {
            XCTAssertEqual(image.values[pixel * 3].bitPattern, red[pixel].bitPattern)
            XCTAssertEqual(image.values[pixel * 3 + 1].bitPattern, green[pixel].bitPattern)
            XCTAssertEqual(image.values[pixel * 3 + 2].bitPattern, blue[pixel].bitPattern)
        }
    }

    func testInvalidDimensionsAndCancellationPublishNoPartialFile() throws {
        let directory = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("depth.exr")
        XCTAssertThrowsError(try NativeFloatEXR.write([0.5], channels: 1, width: Int.max, height: 2, to: url))
        XCTAssertThrowsError(try NativeFloatEXR.write([0.5], channels: 2, width: 1, height: 1, to: url))
        let cancellation = NativeCancellation(); cancellation.cancel()
        XCTAssertThrowsError(try NativeFloatEXR.write([0.5], channels: 1, width: 1, height: 1, to: url,
            cancellation: cancellation)) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [])
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("native-float-exr-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.resolvingSymlinksInPath()
    }

    /// Independent narrow file inspection follows the public EXR layout. It
    /// verifies actual channel types, offsets and IEEE bits rather than trusting
    /// a decoder's display interpretation or the writer's implementation.
    private func readUncompressedEXR(_ url: URL) throws -> (width: Int, height: Int, channelNames: [String], channelTypes: [UInt32], planes: [[UInt32]]) {
        let data = try Data(contentsOf: url)
        var cursor = 0
        func integer<T: FixedWidthInteger>(_ type: T.Type, at offset: Int) throws -> T {
            guard offset >= 0, offset <= data.count - MemoryLayout<T>.size else { throw NativeIOError.invalid("Truncated EXR test file") }
            return data.withUnsafeBytes { T(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: T.self)) }
        }
        func text() throws -> String {
            guard let end = data[cursor...].firstIndex(of: 0) else { throw NativeIOError.invalid("Unterminated EXR test string") }
            let value = String(decoding: data[cursor..<end], as: UTF8.self); cursor = end + 1; return value
        }
        XCTAssertEqual(try integer(UInt32.self, at: 0), 20_000_630)
        XCTAssertEqual(try integer(UInt32.self, at: 4), 2)
        cursor = 8
        var width = 0, height = 0, names: [String] = [], types: [UInt32] = []
        while data[cursor] != 0 {
            let name = try text(), type = try text()
            let size = Int(try integer(UInt32.self, at: cursor)); cursor += 4
            let end = cursor + size
            XCTAssertLessThanOrEqual(end, data.count)
            if name == "channels" {
                XCTAssertEqual(type, "chlist")
                while data[cursor] != 0 {
                    names.append(try text()); types.append(try integer(UInt32.self, at: cursor)); cursor += 16
                }
            } else if name == "dataWindow" {
                XCTAssertEqual(try integer(Int32.self, at: cursor), 0)
                XCTAssertEqual(try integer(Int32.self, at: cursor + 4), 0)
                width = Int(try integer(Int32.self, at: cursor + 8)) + 1
                height = Int(try integer(Int32.self, at: cursor + 12)) + 1
            } else if name == "compression" { XCTAssertEqual(data[cursor], 0) }
            cursor = end
        }
        cursor += 1
        var planes = Array(repeating: [UInt32](repeating: 0, count: width * height), count: names.count)
        for y in 0..<height {
            let chunk = Int(try integer(UInt64.self, at: cursor + y * 8))
            XCTAssertEqual(try integer(Int32.self, at: chunk), Int32(y))
            XCTAssertEqual(try integer(Int32.self, at: chunk + 4), Int32(width * names.count * 4))
            for channel in names.indices {
                for x in 0..<width {
                    planes[channel][y * width + x] = try integer(UInt32.self, at: chunk + 8 + (channel * width + x) * 4)
                }
            }
        }
        return (width, height, names, types, planes)
    }
}
