import Darwin
import Foundation
import XCTest
@testable import LoomCore

final class TrainingQueueLeaseTests: XCTestCase {
    func testCategorizationCannotStartGPUInferenceUnderAnotherActorsOperationLease() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("categorizer-lease-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let project = try LoomProject(sourceFolder: folder)
        let lease = try XCTUnwrap(TrainingQueueLease.requireAvailable(projectFolder: project.projectFolder))
        do {
            _ = try await project.categorize(using: folder.appendingPathComponent("missing.mlmodel"), progress: { _, _ in
                XCTFail("Categorization must not touch sources or run inference while the project is busy")
            })
            XCTFail("GPU-backed categorization must hold the shared operation lease")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Another model operation"), error.localizedDescription) }
        withExtendedLifetime(lease) {}
    }

    func testGuiOperationAlsoExcludesAQueueUntilItFinishes() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("operation-lease-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let operation = try XCTUnwrap(TrainingQueueLease.requireAvailable(projectFolder: folder))
        try withExtendedLifetime(operation) {
            XCTAssertThrowsError(try TrainingQueueLease(projectFolder: folder, controlFolder: folder))
            XCTAssertThrowsError(try TrainingQueueLease.requireAvailable(projectFolder: folder),
                "Independent project actors in the same app cannot overlap model operations")
        }
    }

    func testLeaseExcludesAnotherRunnerAndAllowsOnlyItsScopedTrainingOperation() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("training-lease-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        do {
            let lease = try TrainingQueueLease(projectFolder: folder, controlFolder: folder)
            XCTAssertThrowsError(try TrainingQueueLease(projectFolder: folder, controlFolder: folder))
            XCTAssertThrowsError(try TrainingQueueLease.requireAvailable(projectFolder: folder))
            try await lease.withLease {
                let detachedWasBlocked = await Task.detached {
                    do { _ = try TrainingQueueLease.requireAvailable(projectFolder: folder); return false }
                    catch { return true }
                }.value
                XCTAssertTrue(detachedWasBlocked, "An unrelated task cannot borrow the queue's lease")
                let concurrentScopeWasBlocked = await Task.detached {
                    do { try await lease.withLease {}; return false }
                    catch { return true }
                }.value
                XCTAssertTrue(concurrentScopeWasBlocked, "A queue can lend its lease to only one operation at a time")
                XCTAssertNil(try TrainingQueueLease.requireAvailable(projectFolder: folder))
                XCTAssertThrowsError(try TrainingQueueLease.requireAvailable(projectFolder: folder),
                    "A permit cannot admit a second overlapping operation")
            }
            try await lease.withLease {
                let borrowed = try await LeaseTestOperation().borrow(folder)
                XCTAssertTrue(borrowed, "Each serial queue operation receives a permit across its project actor call")
            }
            let gate = LeaseTestGate()
            let delayed = try await lease.withLease {
                Task {
                    await gate.wait()
                    do { _ = try TrainingQueueLease.requireAvailable(projectFolder: folder); return false }
                    catch { return true }
                }
            }
            await gate.open()
            let expiredPermitWasBlocked = await delayed.value
            XCTAssertTrue(expiredPermitWasBlocked, "An inherited task cannot use a permit after its borrowing scope ends")
            try await lease.withLease {
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

private actor LeaseTestOperation {
    func borrow(_ folder: URL) throws -> Bool { try TrainingQueueLease.requireAvailable(projectFolder: folder) == nil }
}

private actor LeaseTestGate {
    private var opened = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if opened { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func open() {
        opened = true
        for waiter in waiters { waiter.resume() }
        waiters.removeAll()
    }
}
