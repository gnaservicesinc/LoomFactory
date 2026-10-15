import Foundation
import LoomNative
import XCTest
@testable import LoomCore

final class NativePreparationPipelineTests: XCTestCase {
    func testTwoWorkersOverlapWithBoundedPendingResultsAndFIFOConsumption() throws {
        let pipeline = NativePreparationPipeline<Int>(cancellation: NativeCancellation(), capacity: 2)
        let firstBegan = expectation(description: "First preparation is blocked")
        let secondBegan = expectation(description: "Second preparation overlaps the first")
        let releaseFirst = DispatchSemaphore(value: 0)
        defer { releaseFirst.signal(); pipeline.close() }
        try pipeline.submit {
            firstBegan.fulfill()
            releaseFirst.wait()
            return 1
        }
        wait(for: [firstBegan], timeout: 2)
        try pipeline.submit {
            secondBegan.fulfill()
            return 2
        }
        // A serial queue cannot reach this point while the first worker holds
        // its semaphore. Completing the second job must not free its slot.
        wait(for: [secondBegan], timeout: 2)
        XCTAssertThrowsError(try pipeline.submit { 3 })
        releaseFirst.signal()
        XCTAssertEqual(try pipeline.take(), 1)
        try pipeline.submit { 3 }
        XCTAssertEqual(try pipeline.take(), 2)
        XCTAssertEqual(try pipeline.take(), 3)
        XCTAssertThrowsError(try pipeline.take())
    }

    func testParallelPreparationPreservesErrorsInSubmissionOrderAndReusesCapacity() throws {
        enum Failure: Error { case missingFirstSource }
        let pipeline = NativePreparationPipeline<Int>(cancellation: NativeCancellation(), capacity: 2)
        let releaseFirst = DispatchSemaphore(value: 0)
        let secondBegan = expectation(description: "Later successful work overlaps the earlier failure")
        defer { releaseFirst.signal(); pipeline.close() }
        try pipeline.submit {
            releaseFirst.wait()
            throw Failure.missingFirstSource
        }
        try pipeline.submit { secondBegan.fulfill(); return 2 }
        wait(for: [secondBegan], timeout: 2)
        releaseFirst.signal()
        XCTAssertThrowsError(try pipeline.take()) { XCTAssertTrue($0 is Failure) }
        try pipeline.submit { 3 }
        XCTAssertEqual(try pipeline.take(), 2)
        XCTAssertEqual(try pipeline.take(), 3)
    }

    func testCancellationAndCloseJoinBothActiveWorkers() throws {
        let cancellation = NativeCancellation()
        let pipeline = NativePreparationPipeline<Int>(cancellation: cancellation, capacity: 2)
        let began = expectation(description: "Both preparation workers began")
        began.expectedFulfillmentCount = 2
        let drained = expectation(description: "Both preparation workers drained")
        drained.expectedFulfillmentCount = 2
        defer { cancellation.cancel(); pipeline.close() }
        for index in 0..<2 {
            try pipeline.submit {
                began.fulfill()
                defer { drained.fulfill() }
                while !cancellation.isCancelled { Thread.sleep(forTimeInterval: 0.005) }
                try cancellation.check()
                return index
            }
        }
        wait(for: [began], timeout: 2)
        cancellation.cancel()
        XCTAssertThrowsError(try pipeline.take()) { XCTAssertTrue($0 is CancellationError) }
        pipeline.close()
        wait(for: [drained], timeout: 0.1)
        XCTAssertThrowsError(try pipeline.submit { 3 })
        pipeline.close()
    }

    func testCloseJoinsTwoWorkersWithoutCancellingTheirSharedRun() throws {
        let cancellation = NativeCancellation()
        let pipeline = NativePreparationPipeline<Int>(cancellation: cancellation, capacity: 2)
        let began = expectation(description: "Both preparations began")
        began.expectedFulfillmentCount = 2
        let release = DispatchSemaphore(value: 0)
        let closing = expectation(description: "Owner began closing two workers")
        let closed = expectation(description: "Owner joined both workers")
        defer { release.signal(); release.signal(); pipeline.close() }
        for index in 0..<2 {
            try pipeline.submit { began.fulfill(); release.wait(); return index }
        }
        wait(for: [began], timeout: 2)
        DispatchQueue.global().async {
            closing.fulfill()
            pipeline.close()
            closed.fulfill()
        }
        wait(for: [closing], timeout: 2)
        release.signal(); release.signal()
        wait(for: [closed], timeout: 2)
        XCTAssertFalse(cancellation.isCancelled)
        XCTAssertThrowsError(try pipeline.take())
        XCTAssertThrowsError(try pipeline.submit { 3 })
    }

    func testLookaheadPreparesWhileOwnerIsBusyAndBoundsPendingStorage() throws {
        let cancellation = NativeCancellation()
        let pipeline = NativePreparationPipeline<[Float]>(cancellation: cancellation)
        defer { pipeline.close() }
        let prepared = expectation(description: "Next image prepared while owner is busy")
        try pipeline.submit {
            prepared.fulfill()
            return [0.125, 0.75]
        }
        wait(for: [prepared], timeout: 2)
        XCTAssertThrowsError(try pipeline.submit { [1] })
        XCTAssertEqual(try pipeline.take(), [0.125, 0.75])
        try pipeline.submit { [0.25] }
        XCTAssertEqual(try pipeline.take(), [0.25])
    }

    func testPreparationErrorIsPreservedAndNextBatchCanProceed() throws {
        let pipeline = NativePreparationPipeline<Int>(cancellation: NativeCancellation())
        defer { pipeline.close() }
        enum Failure: Error { case missingSource }
        try pipeline.submit { throw Failure.missingSource }
        XCTAssertThrowsError(try pipeline.take()) { XCTAssertTrue($0 is Failure) }
        try pipeline.submit { 7 }
        XCTAssertEqual(try pipeline.take(), 7)
    }

    func testStopInterruptsWaitAndCloseDrainsWorkerBeforeReturning() throws {
        let cancellation = NativeCancellation()
        let pipeline = NativePreparationPipeline<Int>(cancellation: cancellation)
        let began = expectation(description: "Preparation began")
        let finished = expectation(description: "Preparation drained")
        try pipeline.submit {
            began.fulfill()
            defer { finished.fulfill() }
            while !cancellation.isCancelled { Thread.sleep(forTimeInterval: 0.005) }
            try cancellation.check()
            return 1
        }
        wait(for: [began], timeout: 2)
        cancellation.cancel()
        XCTAssertThrowsError(try pipeline.take()) { XCTAssertTrue($0 is CancellationError) }
        pipeline.close()
        wait(for: [finished], timeout: 0.1)
        XCTAssertThrowsError(try pipeline.submit { 2 }) { XCTAssertTrue($0 is CancellationError) }
    }

    func testCloseJoinsPendingPreparationWithoutCancellingTheRun() throws {
        let cancellation = NativeCancellation()
        let pipeline = NativePreparationPipeline<Int>(cancellation: cancellation)
        let began = expectation(description: "Preparation began")
        let release = DispatchSemaphore(value: 0)
        let closing = expectation(description: "Owner started closing")
        let closed = expectation(description: "Owner finished closing")
        try pipeline.submit {
            began.fulfill()
            release.wait()
            return 99
        }
        wait(for: [began], timeout: 2)
        DispatchQueue.global().async {
            closing.fulfill()
            pipeline.close()
            closed.fulfill()
        }
        wait(for: [closing], timeout: 2)
        release.signal()
        wait(for: [closed], timeout: 2)
        XCTAssertFalse(cancellation.isCancelled)
        XCTAssertThrowsError(try pipeline.take())
        XCTAssertThrowsError(try pipeline.submit { 100 })
        pipeline.close()
    }
}
