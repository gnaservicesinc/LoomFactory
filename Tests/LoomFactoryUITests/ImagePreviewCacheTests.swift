import AppKit
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

struct ImagePreviewCacheTests {
    @Test
    func concurrentViewsShareOneDecodeAndCacheStaysWithinItsMemoryBudget() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-preview-cache-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("source.png")
        try writePreviewFixture(url, size: 256, value: 100)
        let original = try Data(contentsOf: url)
        let cache = FactoryPreviewCache(maximumBytes: 2 * 256 * 256 * 4, maximumImages: 2)
        let request = FactoryPreviewRequest(url: url, revision: "one", maximumPixels: 256, applyOrientationTransform: true)
        let images = await withTaskGroup(of: FactoryBitmapBox?.self, returning: [FactoryBitmapBox?].self) { group in
            for _ in 0..<12 { group.addTask { await cache.image(for: request) } }
            var result: [FactoryBitmapBox?] = []
            for await image in group { result.append(image) }
            return result
        }
        #expect(images.count == 12)
        #expect(images.allSatisfy { $0?.bitmap.width == 256 })
        var stats = await cache.statistics()
        #expect(stats.decodes == 1)
        #expect(stats.cachedImages == 1)
        for index in 0..<8 {
            _ = await cache.image(for: .init(url: url, revision: "version-\(index)", maximumPixels: 256, applyOrientationTransform: true))
        }
        stats = await cache.statistics()
        #expect(stats.cachedImages <= 2)
        #expect(stats.cachedBytes <= 2 * 256 * 256 * 4)
        #expect(stats.pendingRequests == 0)
        #expect(try Data(contentsOf: url) == original)
    }

    @Test
    func replacingSourceAtSameURLInvalidatesCachedPreviewWithoutChangingRevision() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-preview-replace-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("source.png")
        try writePreviewFixture(url, size: 128, value: 100)
        let cache = FactoryPreviewCache()
        let request = FactoryPreviewRequest(url: url, revision: "same", maximumPixels: 256, applyOrientationTransform: true)
        let first = await cache.image(for: request)
        #expect(first?.bitmap.width == 128)
        try writePreviewFixture(url, size: 64, value: 200)
        let second = await cache.image(for: request)
        #expect(second?.bitmap.width == 64)
        let stats = await cache.statistics()
        #expect(stats.decodes == 2)
    }

    @Test
    func alreadyCancelledPreviewRequestsDoNotDecode() async throws {
        let cache = FactoryPreviewCache()
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return await cache.image(for: .init(url: URL(fileURLWithPath: "/does-not-exist.png"), revision: "",
                maximumPixels: 256, applyOrientationTransform: true))
        }
        let result = await task.value
        #expect(result == nil)
        let stats = await cache.statistics()
        #expect(stats.decodes == 0)
        #expect(stats.pendingRequests == 0)
    }
}

private func writePreviewFixture(_ url: URL, size: Int, value: UInt8) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let provider = try #require(CGDataProvider(data: Data(repeating: value, count: size * size * 4) as CFData))
    let image = try #require(CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let writer = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(writer, image, nil)
    #expect(CGImageDestinationFinalize(writer))
}
