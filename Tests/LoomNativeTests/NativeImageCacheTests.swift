import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LoomNative

final class NativeImageCacheTests: XCTestCase {
    func testRunPoliciesExposeTheirAdmittedBudgetAndNeverLoadAtConstruction() {
        let gib = 1024 * 1024 * 1024
        let stream = NativeImageCache(policy: .stream, physicalMemory: UInt64(64 * gib))
        XCTAssertEqual(stream.effectiveByteLimit, 0)
        XCTAssertEqual(stream.retainedByteCount, 0)
        let explicit = NativeImageCache(policy: .bounded(byteLimit: 32 * gib), physicalMemory: UInt64(64 * gib))
        XCTAssertEqual(explicit.effectiveByteLimit, 32 * gib, "An explicit budget may exceed the default recommendation")
        XCTAssertEqual(explicit.retainedByteCount, 0, "A cache must not preload the corpus before the first batch")
        let oversized = NativeImageCache(policy: .bounded(byteLimit: Int.max), physicalMemory: UInt64(32 * gib))
        XCTAssertEqual(oversized.effectiveByteLimit, 16 * gib)
        let negative = NativeImageCache(policy: .bounded(byteLimit: -1), physicalMemory: UInt64(64 * gib))
        XCTAssertEqual(negative.effectiveByteLimit, 0)
    }

    func testStreamingAndRetainedPairsProduceIdenticalNativePixelsAndSeparateMemoryChoices() throws {
        let folder = try directory()
        let input = folder.appendingPathComponent("input.png"), target = folder.appendingPathComponent("target.png")
        try writePNG((0..<48).map { UInt16($0 * 1201) }, channels: 1, to: input)
        try writePNG((0..<48).map { UInt16($0 * 997) }, channels: 1, to: target)
        let originals = try [input, target].map { try Data(contentsOf: $0) }
        let stream = NativeImageCache(policy: .stream)
        let retained = NativeImageCache(policy: .bounded(byteLimit: 1024 * 1024))
        let crop = NativeCropTransform(x: 3, y: 1, size: 4, horizontalFlip: true, quarterTurns: 1)
        for _ in 0..<3 {
            let a = try NativeImages.pair(inputURL: input, targetURL: target, transform: crop, mapKind: "depth", imageCache: stream)
            let b = try NativeImages.pair(inputURL: input, targetURL: target, transform: crop, mapKind: "depth", imageCache: retained)
            XCTAssertEqual(a.input, b.input); XCTAssertEqual(a.target, b.target); XCTAssertEqual(a.crop, b.crop)
            XCTAssertEqual(stream.retainedByteCount, 0)
            XCTAssertGreaterThan(retained.retainedByteCount, 0)
            XCTAssertLessThanOrEqual(retained.retainedByteCount, retained.effectiveByteLimit)
        }
        stream.clear()
        XCTAssertGreaterThan(retained.retainedByteCount, 0, "Clearing one run must leave another run's cache intact")
        retained.clear()
        XCTAssertEqual(retained.retainedByteCount, 0)
        XCTAssertEqual(try [input, target].map { try Data(contentsOf: $0) }, originals)
    }

    func testConcurrentPairsRespectTheBudgetAndPreserveEverySelectedPixel() throws {
        let folder = try directory()
        let input = folder.appendingPathComponent("input.png"), target = folder.appendingPathComponent("target.png")
        let inputCodes = (0..<48).map { UInt16($0 * 1201) }
        let targetCodes = (0..<48).map { UInt16($0 * 997) }
        try writePNG(inputCodes, channels: 1, to: input)
        try writePNG(targetCodes, channels: 1, to: target)
        // A source may exceed the retention budget; it still decodes and trains
        // rather than accumulating corpus-sized storage or failing admission.
        let cache = NativeImageCache(policy: .bounded(byteLimit: 64))
        let results = PairResults()
        DispatchQueue.concurrentPerform(iterations: 12) { index in
            let crop = NativeCropTransform(x: index % 5, y: index % 3, size: 4)
            results.set(index, Result { try autoreleasepool {
                try NativeImages.pair(inputURL: input, targetURL: target, transform: crop, mapKind: "depth", imageCache: cache)
            } })
        }
        for index in 0..<12 {
            let pair = try XCTUnwrap(results.get(index)).get()
            var expectedInput: [Float] = [], expectedTarget: [Float] = []
            for y in 0..<4 {
                for x in 0..<4 {
                    let source = (index % 3 + y) * 8 + index % 5 + x
                    expectedInput.append(Float(inputCodes[source]) / 65535)
                    expectedTarget.append(Float(targetCodes[source]) / 65535)
                }
            }
            XCTAssertEqual(pair.input.values, expectedInput + expectedInput + expectedInput)
            XCTAssertEqual(pair.target.values, expectedTarget)
        }
        XCTAssertLessThanOrEqual(cache.retainedByteCount, cache.effectiveByteLimit)
        XCTAssertEqual(cache.retainedByteCount, 0, "An oversized image must never enter the retained cache")
    }

    func testChangedFileIsValidatedAgainAfterItsPixelsCouldNotFitTheCache() throws {
        let folder = try directory()
        let input = folder.appendingPathComponent("input.png"), target = folder.appendingPathComponent("target.png")
        try writePNG([UInt16](repeating: 2000, count: 48), channels: 1, to: input)
        let valid = [UInt16](repeating: 3000, count: 48 * 3)
        try writePNG(valid, channels: 3, to: target)
        let cache = NativeImageCache(policy: .stream)
        let crop = NativeCropTransform(x: 0, y: 0, size: 4)
        _ = try NativeImages.pair(inputURL: input, targetURL: target, transform: crop, mapKind: "depth", imageCache: cache)
        var invalid = valid
        invalid[invalid.count - 1] = 3001
        try writePNG(invalid, channels: 3, to: target)
        XCTAssertThrowsError(try NativeImages.pair(inputURL: input, targetURL: target, transform: crop, mapKind: "depth", imageCache: cache)) {
            XCTAssertTrue($0.localizedDescription.contains("different color channels"), "The full source must be checked beyond the selected crop")
        }
        XCTAssertEqual(cache.retainedByteCount, 0)
    }

    private func directory() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("loom-image-cache-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        return folder
    }

    private func writePNG(_ codes: [UInt16], channels: Int, to url: URL) throws {
        let data = codes.withUnsafeBytes { Data($0) }
        let provider = try XCTUnwrap(CGDataProvider(data: data as CFData))
        let image = try XCTUnwrap(CGImage(width: 8, height: 6, bitsPerComponent: 16, bitsPerPixel: channels * 16,
            bytesPerRow: 8 * channels * 2, space: channels == 1 ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: [.byteOrder16Little], provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }

    private final class PairResults: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Int: Result<NativeImagePair, Error>] = [:]
        func set(_ index: Int, _ value: Result<NativeImagePair, Error>) { lock.withLock { values[index] = value } }
        func get(_ index: Int) -> Result<NativeImagePair, Error>? { lock.withLock { values[index] } }
    }
}
