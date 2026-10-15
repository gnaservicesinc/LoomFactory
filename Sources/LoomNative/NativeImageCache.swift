import Foundation

/// Retained original numeric pixels are optional. Both modes still prepare the
/// next bounded batches while the training owner uses the current batch.
public enum NativeImageCachePolicy: Sendable, Equatable {
    case stream
    case bounded(byteLimit: Int)
}

/// One run owns its cache so another source project's memory choice cannot
/// change it. Images enter on demand; construction never loads the corpus.
public final class NativeImageCache: @unchecked Sendable {
    /// Requested limits are admitted up to half of physical RAM. Callers show
    /// this effective limit when a request exceeds that safety ceiling.
    public let effectiveByteLimit: Int
    let rawPixels: NativeRawPixelCache

    public convenience init(policy: NativeImageCachePolicy) {
        self.init(policy: policy, physicalMemory: ProcessInfo.processInfo.physicalMemory)
    }

    init(policy: NativeImageCachePolicy, physicalMemory: UInt64) {
        let ceiling = Int(min(UInt64(Int.max), physicalMemory / 2))
        switch policy {
        case .stream: effectiveByteLimit = 0
        case .bounded(let requested): effectiveByteLimit = min(ceiling, max(0, requested))
        }
        rawPixels = NativeRawPixelCache(limit: effectiveByteLimit)
    }

    public var retainedByteCount: Int { rawPixels.retainedByteCount }

    /// Join preparation workers before clearing so they cannot repopulate it.
    public func clear() { rawPixels.clear() }
}
