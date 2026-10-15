import CryptoKit
import Foundation
import XCTest
@testable import LoomNative

final class NativeFileHashTests: XCTestCase {
    func testStreamingHashMatchesOriginalBytesAcrossChunkBoundariesAndRepeatedReads() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("loom-hash-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        let data = Data((0..<(3 * 1_048_576 + 17)).map { UInt8($0 % 251) })
        try data.write(to: url)
        let expected = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        for _ in 0..<4 { XCTAssertEqual(try NativeFileHash.sha256(url), expected) }
        XCTAssertEqual(try Data(contentsOf: url), data)
        try Data().write(to: url)
        XCTAssertEqual(try NativeFileHash.sha256(url), SHA256.hash(data: Data()).map { String(format: "%02x", $0) }.joined())
    }

    func testCancellationIsCheckedBeforeOpeningBetweenChunksAndAfterReading() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("loom-hash-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        let stopped = NativeCancellation(); stopped.cancel()
        XCTAssertThrowsError(try NativeFileHash.sha256(url) { try stopped.check() }) { XCTAssertTrue($0 is CancellationError) }
        try Data(repeating: 77, count: 2 * 1_048_576 + 1).write(to: url)
        var checks = 0
        XCTAssertThrowsError(try NativeFileHash.sha256(url) {
            checks += 1
            if checks == 3 { throw CancellationError() }
        }) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(checks, 3)
        XCTAssertThrowsError(try NativeFileHash.sha256(url.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)))
    }
}
