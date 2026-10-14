import AppKit
import Darwin
import ImageIO
import LoomCore
import SwiftUI

struct FactoryPreviewRequest: Hashable, Sendable {
    let url: URL
    let revision: String
    let maximumPixels: Int
    let applyOrientationTransform: Bool
    var priority: TaskPriority { maximumPixels <= 128 ? .utility : .userInitiated }

    /// Source signatures are resolved away from the UI executor, including cache hits.
    func resolved() -> Self {
        var info = stat()
        let signature = stat(url.path, &info) == 0
            ? "\(info.st_dev)|\(info.st_ino)|\(info.st_size)|\(info.st_mtimespec.tv_sec)|\(info.st_mtimespec.tv_nsec)|\(info.st_ctimespec.tv_sec)|\(info.st_ctimespec.tv_nsec)"
            : "unavailable"
        return Self(url: url, revision: revision + "|" + signature, maximumPixels: maximumPixels,
                    applyOrientationTransform: applyOrientationTransform)
    }
}

final class FactoryBitmapBox: @unchecked Sendable {
    let bitmap: CGImage
    var cost: Int { bitmap.bytesPerRow * bitmap.height }
    init(_ bitmap: CGImage) { self.bitmap = bitmap }
}

/// A serial decoder bounds expensive ImageIO work. Two decoders allow a selected
/// preview to progress while a thumbnail is decoding, without an unbounded queue
/// of simultaneous image allocations during fast scrolling.
private actor FactoryPreviewDecoder {
    func decode(_ request: FactoryPreviewRequest) -> FactoryBitmapBox? {
        guard !Task.isCancelled else { return nil }
        return autoreleasepool {
            guard let source = CGImageSourceCreateWithURL(request.url as CFURL, [
                kCGImageSourceShouldCache: false
            ] as CFDictionary), !Task.isCancelled,
                  let bitmap = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: request.applyOrientationTransform,
                    kCGImageSourceThumbnailMaxPixelSize: max(1, request.maximumPixels),
                    kCGImageSourceShouldCacheImmediately: true
                  ] as CFDictionary), !Task.isCancelled else { return nil }
            return FactoryBitmapBox(bitmap)
        }
    }
}

actor FactoryPreviewCache {
    static let shared = FactoryPreviewCache()
    struct Statistics: Sendable {
        let hits: Int
        let decodes: Int
        let cachedBytes: Int
        let cachedImages: Int
        let pendingRequests: Int
    }
    private struct Entry { let image: FactoryBitmapBox; var access: UInt64 }
    private struct Loading {
        let id: UUID
        let task: Task<FactoryBitmapBox?, Never>
        var readers: Set<UUID>
    }
    private let maximumBytes: Int
    private let maximumImages: Int
    private let decoders = [FactoryPreviewDecoder(), FactoryPreviewDecoder()]
    private var nextDecoder = 0
    private var cache: [FactoryPreviewRequest: Entry] = [:]
    private var loading: [FactoryPreviewRequest: Loading] = [:]
    private var access: UInt64 = 0
    private var cachedBytes = 0
    private var hits = 0
    private var decodes = 0

    init(maximumBytes: Int = 128 * 1024 * 1024, maximumImages: Int = 256) {
        self.maximumBytes = max(0, maximumBytes)
        self.maximumImages = max(0, maximumImages)
    }

    func image(for request: FactoryPreviewRequest) async -> FactoryBitmapBox? {
        guard !Task.isCancelled else { return nil }
        let key = await Task.detached(priority: request.priority) { request.resolved() }.value
        guard !Task.isCancelled else { return nil }
        access &+= 1
        if var entry = cache[key] {
            entry.access = access
            cache[key] = entry
            hits += 1
            return entry.image
        }
        let reader = UUID()
        let task: Task<FactoryBitmapBox?, Never>
        if var existing = loading[key] {
            existing.readers.insert(reader)
            loading[key] = existing
            task = existing.task
        } else {
            let id = UUID()
            let decoder = decoders[nextDecoder]
            nextDecoder = (nextDecoder + 1) % decoders.count
            decodes += 1
            task = Task.detached(priority: request.priority) {
                let result = await decoder.decode(key)
                await self.complete(key, id: id, image: result)
                return result
            }
            loading[key] = Loading(id: id, task: task, readers: [reader])
        }
        let result = await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            Task { await self.release(key, reader: reader) }
        }
        release(key, reader: reader)
        return Task.isCancelled ? nil : result
    }

    private func release(_ key: FactoryPreviewRequest, reader: UUID) {
        guard var entry = loading[key] else { return }
        entry.readers.remove(reader)
        if entry.readers.isEmpty {
            entry.task.cancel()
            loading.removeValue(forKey: key)
        } else { loading[key] = entry }
    }

    private func complete(_ key: FactoryPreviewRequest, id: UUID, image: FactoryBitmapBox?) {
        guard loading[key]?.id == id else { return }
        loading.removeValue(forKey: key)
        guard let image, image.cost <= maximumBytes, maximumImages > 0 else { return }
        access &+= 1
        cache[key] = Entry(image: image, access: access)
        cachedBytes += image.cost
        while cachedBytes > maximumBytes || cache.count > maximumImages {
            guard let oldest = cache.min(by: { $0.value.access < $1.value.access }) else { break }
            cachedBytes -= oldest.value.image.cost
            cache.removeValue(forKey: oldest.key)
        }
    }

    func statistics() -> Statistics {
        Statistics(hits: hits, decodes: decodes, cachedBytes: cachedBytes,
                   cachedImages: cache.count, pendingRequests: loading.count)
    }
}

struct FactoryImage: View {
    let url: URL
    let revision: String
    var maximumPixels = 1400
    var interpolation: Image.Interpolation = .high
    var applyOrientationTransform = true
    @State private var image: NSImage?
    @State private var couldNotRead = false

    private var request: FactoryPreviewRequest {
        FactoryPreviewRequest(url: url, revision: revision, maximumPixels: maximumPixels,
                              applyOrientationTransform: applyOrientationTransform)
    }

    var body: some View {
        ZStack {
            Color.primary.opacity(0.035)
            if let image {
                Image(nsImage: image).resizable().interpolation(interpolation)
                    .aspectRatio(contentMode: .fit)
            } else if couldNotRead {
                Label("Image could not be read", systemImage: "photo.badge.exclamationmark")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                ProgressView().controlSize(.small)
            }
        }
        .clipped()
        .task(id: request, priority: request.priority) {
            image = nil
            couldNotRead = false
            let result = await FactoryPreviewCache.shared.image(for: request)
            guard !Task.isCancelled else { return }
            image = result.map { NSImage(cgImage: $0.bitmap, size: .zero) }
            couldNotRead = result == nil
        }
    }
}

extension SourceImage {
    /// This key is safe to read while SwiftUI builds rows. The model appends the
    /// file signature prepared during the background folder scan.
    var previewRevision: String { "\(width)x\(height)|\(mapTypeInference?.sourceSignature ?? "")" }
}
