import Foundation
import XCTest
@testable import LoomNative

final class NativePixelValidationCacheTests: XCTestCase {
    func testWholeImageValidationSurvivesPixelEvictionButChangedFilesAreNotTrusted() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let first = folder.appendingPathComponent("first"), second = folder.appendingPathComponent("second")
        try Data([1]).write(to: first); try Data([2]).write(to: second)
        let key = try NativeRawPixelKey(first), other = try NativeRawPixelKey(second)
        let cache = NativeRawPixelCache(limit: 1)
        func pixels(_ key: NativeRawPixelKey) -> NativeRawPixels {
            NativeRawPixels(key: key, width: 1, height: 1, channels: 1, bits: 8,
                components: 1, bitsPerPixel: 8, bytesPerRow: 1, bitmapInfo: 0,
                floating: false, leadingAlpha: false, premultiplied: false,
                hasAlpha: false, storage: Data([1]))
        }
        cache.insert(pixels(key)); cache.markValidated(key, role: "depth")
        cache.insert(pixels(other))
        XCTAssertNil(cache.read(key), "The decoded-pixel budget must still evict its old image")
        XCTAssertTrue(cache.isValidated(key, role: "depth"), "Eviction must not rescan an unchanged full image")
        XCTAssertFalse(cache.isValidated(key, role: "roughness"), "Each role requires its own validation")
        try Data([1, 2]).write(to: first)
        XCTAssertFalse(cache.isValidated(try NativeRawPixelKey(first), role: "depth"), "A changed identity must be checked again")
        cache.clear()
        XCTAssertFalse(cache.isValidated(key, role: "depth"), "Run teardown must release both caches")
    }

    func testValidationRecordsAreBoundedEvenWhenImagesCannotFitPixelCache() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let keys = try (0..<3).map { index in
            let url = folder.appendingPathComponent(String(index))
            try Data([UInt8(index)]).write(to: url)
            return try NativeRawPixelKey(url)
        }
        let cache = NativeRawPixelCache(limit: 0, validationLimit: 2)
        for key in keys { cache.markValidated(key, role: "depth") }
        XCTAssertFalse(cache.isValidated(keys[0], role: "depth"))
        XCTAssertTrue(cache.isValidated(keys[1], role: "depth"))
        XCTAssertTrue(cache.isValidated(keys[2], role: "depth"))
    }
}
