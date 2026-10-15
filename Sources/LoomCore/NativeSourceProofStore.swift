import CryptoKit
import Darwin
import Foundation
import LoomNative

/// Capture inexpensive file identities at run start, then hash each source on
/// its first preparation worker visit. Unrelated files can hash concurrently;
/// two visits to the same source share one proof. No image pixels are retained.
final class NativeSourceProofStore: @unchecked Sendable {
    typealias Hasher = @Sendable (URL, NativeCancellation) throws -> String
    private let condition = NSCondition()
    private let identities: [String: SourceFileIdentity]
    private let urls: [String: URL]
    private let hasher: Hasher
    private var working: Set<String> = []
    private var completed: [String: Result<SourceProof, Error>] = [:]

    init(urls: [URL], cancellation: NativeCancellation? = nil,
         hasher: @escaping Hasher = { try NativeFactoryTraining.sha($0, cancellation: $1) }) throws {
        var identities: [String: SourceFileIdentity] = [:], unique: [String: URL] = [:]
        for url in urls where identities[url.path] == nil {
            try cancellation?.check()
            identities[url.path] = try SourceFileIdentity(url)
            unique[url.path] = url
        }
        self.identities = identities; self.urls = unique; self.hasher = hasher
    }

    @discardableResult
    func ensure(_ url: URL, cancellation: NativeCancellation) throws -> SourceProof {
        try cancellation.check()
        guard let identity = identities[url.path] else {
            throw LoomError("Source image is outside this training plan: \(url.lastPathComponent).")
        }
        let path = url.path
        condition.lock()
        while working.contains(path) {
            _ = condition.wait(until: Date(timeIntervalSinceNow: 0.05))
            if cancellation.isCancelled { condition.unlock(); throw CancellationError() }
        }
        if let result = completed[path] {
            condition.unlock()
            let proof = try result.get()
            try cancellation.check(); try proof.check(url)
            return proof
        }
        working.insert(path)
        condition.unlock()
        let result = Result {
            try cancellation.check(); try identity.check(url)
            let hash = try hasher(url, cancellation)
            try cancellation.check(); try identity.check(url)
            return SourceProof(identity: identity, hash: hash)
        }
        condition.lock()
        completed[path] = result; working.remove(path)
        condition.broadcast(); condition.unlock()
        return try result.get()
    }

    /// This check never hashes or waits for another worker. The initial stat
    /// identity also catches edits before an image's first lazy SHA visit.
    func check(_ url: URL) throws {
        guard let identity = identities[url.path] else {
            throw LoomError("Source image is outside this training plan: \(url.lastPathComponent).")
        }
        try identity.check(url)
    }

    func snapshots() -> [String: SourceProof] {
        condition.lock(); defer { condition.unlock() }
        return completed.compactMapValues { try? $0.get() }
    }

    /// Never hashes, waits for a worker, or visits an unprepared source. An
    /// early weights/optimizer checkpoint can omit its exact replay cursor
    /// until the whole training plan has acquired first-visit proofs.
    func fingerprintIfComplete(pairs: [TrainingPair], validationIDs: Set<String>) -> String? {
        let proofs = snapshots()
        var rows: [String] = []
        rows.reserveCapacity(pairs.count)
        for pair in pairs {
            guard let input = proofs[pair.input.url.path], let target = proofs[pair.target.url.path] else { return nil }
            rows.append("\(pair.id)|\(pair.input.width)x\(pair.input.height)|\(input.hash)|\(target.hash)|\(validationIDs.contains(pair.id))")
        }
        return SHA256.hash(data: Data(rows.sorted().joined(separator: "\n").utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// Match the original checkpoint fingerprint exactly when admitting an
    /// exact cursor resume. New-run checkpoints use fingerprintIfComplete so
    /// saving early weights never blocks on the rest of the corpus.
    func fingerprint(pairs: [TrainingPair], validationIDs: Set<String>, cancellation: NativeCancellation) throws -> String {
        var rows: [String] = []
        rows.reserveCapacity(pairs.count)
        for pair in pairs {
            let input = try ensure(pair.input.url, cancellation: cancellation)
            let target = try ensure(pair.target.url, cancellation: cancellation)
            rows.append("\(pair.id)|\(pair.input.width)x\(pair.input.height)|\(input.hash)|\(target.hash)|\(validationIDs.contains(pair.id))")
        }
        return SHA256.hash(data: Data(rows.sorted().joined(separator: "\n").utf8)).map { String(format: "%02x", $0) }.joined()
    }

    func verifyAll(cancellation: NativeCancellation, beforeEach: (URL) throws -> Void = { _ in }) throws {
        for path in urls.keys.sorted() {
            try cancellation.check()
            let url = urls[path]!
            try beforeEach(url)
            let proof = try ensure(url, cancellation: cancellation)
            try proof.verifyHash(url, cancellation: cancellation)
        }
    }
}

struct SourceFileIdentity: Sendable, Equatable {
    let size: Int64, seconds: Int, nanos: Int, device: Int32, inode: UInt64
    init(_ url: URL) throws {
        var value = stat()
        guard stat(url.path, &value) == 0 else { throw LoomError("A source image is missing: \(url.lastPathComponent)") }
        size = value.st_size; seconds = value.st_mtimespec.tv_sec; nanos = value.st_mtimespec.tv_nsec
        device = value.st_dev; inode = value.st_ino
    }
    func check(_ url: URL) throws {
        guard (try? SourceFileIdentity(url)) == self else {
            throw LoomError("Source pixels changed during training: \(url.lastPathComponent). Refresh the source folder.")
        }
    }
}

struct SourceProof: Sendable {
    let identity: SourceFileIdentity
    let hash: String
    init(identity: SourceFileIdentity, hash: String) { self.identity = identity; self.hash = hash }
    init(_ url: URL, cancellation: NativeCancellation) throws {
        let identity = try SourceFileIdentity(url)
        let hash = try NativeFactoryTraining.sha(url, cancellation: cancellation)
        try identity.check(url)
        self.init(identity: identity, hash: hash)
    }
    func check(_ url: URL) throws { try identity.check(url) }
    func verifyHash(_ url: URL, cancellation: NativeCancellation) throws {
        try check(url)
        guard try NativeFactoryTraining.sha(url, cancellation: cancellation) == hash else {
            throw LoomError("Source pixels changed during training: \(url.lastPathComponent).")
        }
        try check(url)
    }
    func json(path: String) -> [String: Any] {
        ["path": path, "size": identity.size, "mtime_seconds": identity.seconds,
         "mtime_nanoseconds": identity.nanos, "device": identity.device, "inode": identity.inode, "sha256": hash]
    }
}
