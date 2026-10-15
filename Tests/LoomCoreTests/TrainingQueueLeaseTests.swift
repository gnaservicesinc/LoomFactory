import Darwin
import Foundation
import XCTest
@testable import LoomCore

final class TrainingQueueLeaseTests: XCTestCase {
    func testGuiOperationAlsoExcludesAQueueUntilItFinishes() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("operation-lease-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let operation = try XCTUnwrap(TrainingQueueLease.requireAvailable(projectFolder: folder))
        try withExtendedLifetime(operation) {
            XCTAssertThrowsError(try TrainingQueueLease(projectFolder: folder, controlFolder: folder))
            XCTAssertNil(try TrainingQueueLease.requireAvailable(projectFolder: folder), "The owning process can run its nested serial work")
        }
    }

    func testLeaseExcludesAnotherRunnerAndAllowsItsOwnTrainingProcess() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("training-lease-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        do {
            let lease = try TrainingQueueLease(projectFolder: folder, controlFolder: folder)
            try withExtendedLifetime(lease) {
                XCTAssertThrowsError(try TrainingQueueLease(projectFolder: folder, controlFolder: folder))
                XCTAssertNoThrow(try TrainingQueueLease.requireAvailable(projectFolder: folder))
                // A different owner must block the GUI while the OS lock is held.
                let data = try JSONSerialization.data(withJSONObject: ["pid": getpid() + 1, "control_folder": folder.path])
                try data.write(to: folder.appendingPathComponent("training-queue.lock"))
                XCTAssertThrowsError(try TrainingQueueLease.requireAvailable(projectFolder: folder))
            }
        }
        // A stale receipt cannot block training after the process releases its lock.
        XCTAssertNoThrow(try TrainingQueueLease.requireAvailable(projectFolder: folder))
        let next = try TrainingQueueLease(projectFolder: folder, controlFolder: folder)
        withExtendedLifetime(next) {}
    }
}
