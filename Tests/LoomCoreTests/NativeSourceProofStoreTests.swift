import CryptoKit
import Foundation
import LoomNative
import XCTest
@testable import LoomCore

final class NativeSourceProofStoreTests: XCTestCase {
    func testConstructionAndCheapChecksDoNotHashTheCorpus() throws {
        let folder = try directory()
        let first = try file("first", in: folder), unused = try file("unused", in: folder)
        let calls = Counts()
        let store = try NativeSourceProofStore(urls: [first, unused]) { url, cancellation in
            calls.add(url.path)
            return try NativeFactoryTraining.sha(url, cancellation: cancellation)
        }
        XCTAssertTrue(calls.snapshot().isEmpty)
        XCTAssertTrue(store.snapshots().isEmpty)
        let pair = TrainingPair(id: "material", input: image(first, role: .diffuse), target: image(unused, role: .depth))
        XCTAssertNil(store.fingerprintIfComplete(pairs: [pair], validationIDs: []))
        try store.check(first); try store.check(unused)
        XCTAssertTrue(calls.snapshot().isEmpty)
        let proof = try store.ensure(first, cancellation: NativeCancellation())
        XCTAssertEqual(proof.hash, try NativeFactoryTraining.sha(first))
        XCTAssertEqual(calls.snapshot(), [first.path: 1])
        XCTAssertEqual(Set(store.snapshots().keys), [first.path])
        XCTAssertNil(store.fingerprintIfComplete(pairs: [pair], validationIDs: []))
        XCTAssertEqual(calls.snapshot(), [first.path: 1], "An early checkpoint must not hash the rest of the corpus")
        _ = try store.ensure(first, cancellation: NativeCancellation())
        XCTAssertEqual(calls.snapshot(), [first.path: 1], "A worker revisiting an unchanged file must reuse its SHA")
    }

    func testDifferentSourcesHashConcurrentlyAndSharedSourcesAreHashedOnce() throws {
        let folder = try directory()
        let first = try file("first", in: folder), second = try file("second", in: folder)
        let firstBegan = expectation(description: "First source is hashing")
        let secondBegan = expectation(description: "Independent source hashes concurrently")
        let duplicateBegan = expectation(description: "A second visit waits for the same source")
        let finished = expectation(description: "All source proof visits finish")
        finished.expectedFulfillmentCount = 3
        let release = DispatchSemaphore(value: 0)
        let cancellation = NativeCancellation(), calls = Counts(), results = ProofResults()
        let store = try NativeSourceProofStore(urls: [first, second]) { url, cancellation in
            calls.add(url.path)
            if url == first { firstBegan.fulfill(); release.wait() }
            else { secondBegan.fulfill() }
            return try NativeFactoryTraining.sha(url, cancellation: cancellation)
        }
        defer { release.signal() }
        DispatchQueue.global().async {
            results.set(0, Result { try store.ensure(first, cancellation: cancellation) }); finished.fulfill()
        }
        wait(for: [firstBegan], timeout: 2)
        DispatchQueue.global().async {
            duplicateBegan.fulfill()
            results.set(1, Result { try store.ensure(first, cancellation: cancellation) }); finished.fulfill()
        }
        DispatchQueue.global().async {
            results.set(2, Result { try store.ensure(second, cancellation: cancellation) }); finished.fulfill()
        }
        wait(for: [secondBegan, duplicateBegan], timeout: 2)
        release.signal()
        wait(for: [finished], timeout: 2)
        XCTAssertEqual(calls.snapshot(), [first.path: 1, second.path: 1])
        let a = try XCTUnwrap(results.get(0)).get(), b = try XCTUnwrap(results.get(1)).get()
        XCTAssertEqual(a.hash, b.hash)
        _ = try XCTUnwrap(results.get(2)).get()
    }

    func testEditsBeforeFirstHashAndAfterCompletedHashAreRejected() throws {
        let folder = try directory()
        let first = try file("first", in: folder), second = try file("second", in: folder)
        let store = try NativeSourceProofStore(urls: [first, second])
        _ = try store.ensure(first, cancellation: NativeCancellation())
        try Data("changed-before-first-hash".utf8).write(to: second)
        XCTAssertThrowsError(try store.ensure(second, cancellation: NativeCancellation()))
        try Data("changed-after-completed-hash".utf8).write(to: first)
        XCTAssertThrowsError(try store.check(first))
        XCTAssertThrowsError(try store.ensure(first, cancellation: NativeCancellation()))
        XCTAssertThrowsError(try store.verifyAll(cancellation: NativeCancellation()))
    }

    func testFingerprintMatchesLegacySortedRowsAndFinalVerificationIncludesUnvisitedSources() throws {
        let folder = try directory()
        let input = try file("input", in: folder), target = try file("target", in: folder)
        let pair = TrainingPair(id: "material", input: image(input, role: .diffuse), target: image(target, role: .depth))
        let store = try NativeSourceProofStore(urls: [input, target])
        let row = "material|64x32|\(try NativeFactoryTraining.sha(input))|\(try NativeFactoryTraining.sha(target))|true"
        let expected = SHA256.hash(data: Data(row.utf8)).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(try store.fingerprint(pairs: [pair], validationIDs: [pair.id], cancellation: NativeCancellation()), expected)
        XCTAssertEqual(store.fingerprintIfComplete(pairs: [pair], validationIDs: [pair.id]), expected)
        XCTAssertEqual(store.snapshots().count, 2)
        try store.verifyAll(cancellation: NativeCancellation())

        let untouched = try NativeSourceProofStore(urls: [input, target])
        try untouched.verifyAll(cancellation: NativeCancellation())
        XCTAssertEqual(untouched.snapshots().count, 2, "A source never visited for training must still be included in the final proof")
    }

    func testCancellationDoesNotStartHashingAndHashErrorsRemainVisible() throws {
        enum Failure: Error { case cannotRead }
        let folder = try directory(), calls = Counts()
        let url = try file("source", in: folder)
        let store = try NativeSourceProofStore(urls: [url]) { url, _ in calls.add(url.path); throw Failure.cannotRead }
        let cancelled = NativeCancellation(); cancelled.cancel()
        XCTAssertThrowsError(try store.ensure(url, cancellation: cancelled)) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertTrue(calls.snapshot().isEmpty)
        XCTAssertThrowsError(try store.ensure(url, cancellation: NativeCancellation())) { XCTAssertTrue($0 is Failure) }
        XCTAssertThrowsError(try store.ensure(url, cancellation: NativeCancellation())) { XCTAssertTrue($0 is Failure) }
        XCTAssertEqual(calls.snapshot(), [url.path: 1])
        XCTAssertTrue(store.snapshots().isEmpty)
    }

    private func directory() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("loom-source-proofs-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        return folder
    }
    private func file(_ name: String, in folder: URL) throws -> URL {
        let url = folder.appendingPathComponent(name)
        try Data(name.utf8).write(to: url)
        return url
    }
    private func image(_ url: URL, role: MapRole) -> SourceImage {
        SourceImage(id: url.path, url: url, displayName: url.lastPathComponent, role: role,
            category: "Bricks", verified: true, note: "", pairingKey: "material", width: 64, height: 32)
    }
    private final class Counts: @unchecked Sendable {
        let lock = NSLock()
        var counts: [String: Int] = [:]
        func add(_ path: String) { lock.withLock { counts[path, default: 0] += 1 } }
        func snapshot() -> [String: Int] { lock.withLock { counts } }
    }
    private final class ProofResults: @unchecked Sendable {
        let lock = NSLock()
        var results: [Int: Result<SourceProof, Error>] = [:]
        func set(_ index: Int, _ value: Result<SourceProof, Error>) { lock.withLock { results[index] = value } }
        func get(_ index: Int) -> Result<SourceProof, Error>? { lock.withLock { results[index] } }
    }
}
