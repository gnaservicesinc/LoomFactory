import Foundation
import XCTest
@testable import LoomCore

final class TrainingProgressTests: XCTestCase {
    func testStartupAndFirstCycleDoNotInventSteadySpeedOrETA() {
        var tracker = TrainingProgressTracker(totalEpochs: 3, trainingImages: 8,
            validationImages: 2, stepsPerEpoch: 4, startSeconds: 100)
        let startup = tracker.snapshot(now: 105)
        XCTAssertEqual(startup.stage, .checkingSources)
        XCTAssertEqual(startup.totalSteps, 12)
        XCTAssertEqual(startup.elapsedSeconds, 5)
        XCTAssertNil(startup.lastCycleSeconds)
        XCTAssertNil(startup.averageCycleSeconds)
        XCTAssertNil(startup.totalRemainingSeconds)
        XCTAssertNil(startup.currentWorkRemainingSeconds)
        XCTAssertFalse(startup.estimatesIncludeValidation)
        XCTAssertFalse(startup.estimatesIncludeCheckpointing)

        tracker.beginEpoch(epoch: 1, now: 105)
        tracker.begin(stage: .training, image: "Leather/leather_col2.png", index: 2, now: 105)
        tracker.completeStep(stepInEpoch: 1, completedSteps: 1, completedImages: 2, cycleSeconds: 90, now: 195)
        let first = tracker.snapshot(now: 195)
        XCTAssertEqual(first.currentImage, "Leather/leather_col2.png")
        XCTAssertEqual(first.imageIndex, 2)
        XCTAssertEqual(first.completedImages, 2)
        XCTAssertEqual(first.lastCycleSeconds, 90)
        XCTAssertNil(first.averageCycleSeconds)
        XCTAssertNil(first.epochRemainingSeconds)
        XCTAssertNil(first.totalRemainingSeconds)
        XCTAssertNil(first.currentWorkRemainingSeconds)
    }

    func testSecondCycleExcludesWarmupButKeepsItsElapsedTime() {
        var tracker = TrainingProgressTracker(totalEpochs: 3, trainingImages: 8,
            validationImages: 0, stepsPerEpoch: 4, startSeconds: 0)
        tracker.beginEpoch(epoch: 1, now: 5)
        tracker.completeStep(stepInEpoch: 1, completedSteps: 1, completedImages: 2, cycleSeconds: 90, now: 95)
        tracker.completeStep(stepInEpoch: 2, completedSteps: 2, completedImages: 4, cycleSeconds: 2, now: 97)
        let progress = tracker.snapshot(now: 97)
        XCTAssertEqual(progress.averageCycleSeconds, 2)
        XCTAssertEqual(progress.lastCycleSeconds, 2)
        XCTAssertEqual(progress.epochRemainingSeconds, 4)
        XCTAssertEqual(progress.totalRemainingSeconds, 20)
        XCTAssertEqual(progress.elapsedSeconds, 97)
        XCTAssertEqual(progress.epochElapsedSeconds, 92)
        XCTAssertTrue(progress.estimatesIncludeValidation, "No validation stage is pending in this run")
        XCTAssertFalse(progress.estimatesIncludeCheckpointing)
    }

    func testRollingSpeedAdaptsToRecentCycles() throws {
        var tracker = TrainingProgressTracker(totalEpochs: 2, trainingImages: 40,
            validationImages: 0, stepsPerEpoch: 40, startSeconds: 0)
        tracker.beginEpoch(epoch: 1, now: 0)
        tracker.completeStep(stepInEpoch: 1, completedSteps: 1, completedImages: 1, cycleSeconds: 500, now: 500)
        for index in 2...22 {
            tracker.completeStep(stepInEpoch: index, completedSteps: index, completedImages: index,
                cycleSeconds: Double(index - 1), now: Double(500 + index * index))
        }
        let progress = tracker.snapshot(now: 1000)
        // The most recent 20 durations are 2...21; startup and the next old
        // duration have both left the average.
        XCTAssertEqual(try XCTUnwrap(progress.averageCycleSeconds), 11.5, accuracy: 0.000_001)
        XCTAssertEqual(progress.lastCycleSeconds, 21)
        XCTAssertEqual(try XCTUnwrap(progress.totalRemainingSeconds), 58 * 11.5, accuracy: 0.000_001)
    }

    func testValidationAndCheckpointCostsAreObservedAndNotCountedTwice() {
        var tracker = TrainingProgressTracker(totalEpochs: 5, trainingImages: 10,
            validationImages: 2, stepsPerEpoch: 10, checkpointEveryEpochs: 2, startSeconds: 0)
        tracker.beginEpoch(epoch: 1, now: 0)
        tracker.completeStep(stepInEpoch: 1, completedSteps: 1, completedImages: 1, cycleSeconds: 20, now: 20)
        tracker.completeStep(stepInEpoch: 10, completedSteps: 10, completedImages: 10, cycleSeconds: 2, now: 40)
        tracker.checkpointCompleted(durationSeconds: 3, epochComplete: false, now: 43)
        let trainingOnly = tracker.snapshot(now: 43)
        XCTAssertEqual(trainingOnly.totalRemainingSeconds, 92, "40 steps × 2 seconds plus 4 future saves × 3 seconds")
        XCTAssertTrue(trainingOnly.estimatesIncludeCheckpointing)
        XCTAssertFalse(trainingOnly.estimatesIncludeValidation)

        tracker.begin(stage: .validating, image: "holdout-a.png", index: 1, now: 43)
        tracker.completeValidation(durationSeconds: 6, now: 49)
        let validated = tracker.snapshot(now: 49)
        XCTAssertEqual(validated.imagesPerEpoch, 2)
        XCTAssertEqual(validated.totalRemainingSeconds, 116, "Whole validation passes are estimated once per pending epoch")
        XCTAssertEqual(validated.epochRemainingSeconds, 0, "Epoch 1 validation finished and no epoch save is scheduled")
        XCTAssertTrue(validated.estimatesIncludeValidation)

        tracker.beginEpoch(epoch: 2, now: 50)
        tracker.completeStep(stepInEpoch: 10, completedSteps: 20, completedImages: 20, cycleSeconds: 2, now: 70)
        tracker.begin(stage: .training, now: 70)
        XCTAssertEqual(tracker.snapshot(now: 70).epochRemainingSeconds, 9)
        tracker.begin(stage: .validating, image: "holdout-b.png", index: 2, now: 70)
        let checking = tracker.snapshot(now: 72)
        XCTAssertEqual(checking.epochRemainingSeconds, 7)
        XCTAssertEqual(checking.totalRemainingSeconds, 94)
        XCTAssertEqual(checking.currentWorkRemainingSeconds, 4)
        tracker.completeValidation(durationSeconds: 6, now: 76)
        XCTAssertEqual(tracker.snapshot(now: 76).totalRemainingSeconds, 90)
        tracker.begin(stage: .checkpointing, now: 76)
        XCTAssertEqual(tracker.snapshot(now: 77).epochRemainingSeconds, 2)
        XCTAssertEqual(tracker.snapshot(now: 77).totalRemainingSeconds, 89)
        XCTAssertEqual(tracker.snapshot(now: 77).currentWorkRemainingSeconds, 2)
        tracker.checkpointCompleted(durationSeconds: 3, now: 79)
        XCTAssertEqual(tracker.snapshot(now: 79).epochRemainingSeconds, 0)
        XCTAssertEqual(tracker.snapshot(now: 79).currentWorkRemainingSeconds, 0)
        XCTAssertEqual(tracker.snapshot(now: 79).totalRemainingSeconds, 87,
            "30 steps × 2 seconds, 3 validation passes × 6 seconds, and 3 remaining saves × 3 seconds")
    }

    func testEpochAndStageTimersUseMonotonicClockAndFreezeWhenStopped() {
        var tracker = TrainingProgressTracker(totalEpochs: 2, trainingImages: 4,
            validationImages: 0, stepsPerEpoch: 2, startSeconds: 100)
        tracker.beginEpoch(epoch: 1, now: 110)
        tracker.begin(stage: .preparingImage, image: "a.png", index: 1, now: 111)
        tracker.begin(stage: .preparingImage, image: "b.png", index: 2, now: 113)
        let preparing = tracker.snapshot(now: 115)
        XCTAssertEqual(preparing.stageElapsedSeconds, 4, "Changing the image alone does not restart the stage")
        XCTAssertEqual(preparing.epochElapsedSeconds, 5)
        tracker.begin(stage: .stopped, now: 117)
        let stopped = tracker.snapshot(now: 10_000)
        XCTAssertEqual(stopped.elapsedSeconds, 17)
        XCTAssertEqual(stopped.epochElapsedSeconds, 7)
        XCTAssertEqual(stopped.stageElapsedSeconds, 0)
        XCTAssertNil(stopped.totalRemainingSeconds)
        XCTAssertNil(stopped.epochRemainingSeconds)
        XCTAssertEqual(stopped.currentWorkRemainingSeconds, 0)
    }

    func testCycleCountdownPersistsThroughPreparationTrainingAndOptimization() {
        var tracker = TrainingProgressTracker(totalEpochs: 2, trainingImages: 8,
            validationImages: 0, stepsPerEpoch: 4, startSeconds: 0)
        tracker.beginEpoch(epoch: 1, now: 0)
        tracker.completeStep(stepInEpoch: 1, completedSteps: 1, completedImages: 2, cycleSeconds: 50, now: 50)
        tracker.completeStep(stepInEpoch: 2, completedSteps: 2, completedImages: 4, cycleSeconds: 2, now: 52)
        tracker.begin(stage: .preparingImage, image: "a.png", index: 5, now: 53)
        XCTAssertEqual(tracker.snapshot(now: 54).epochRemainingSeconds, 3)
        XCTAssertEqual(tracker.snapshot(now: 54).totalRemainingSeconds, 11)
        XCTAssertEqual(tracker.snapshot(now: 54).currentWorkRemainingSeconds, 1)
        tracker.begin(stage: .training, image: "a.png", index: 5, now: 55)
        XCTAssertEqual(tracker.snapshot(now: 55).epochRemainingSeconds, 2)
        XCTAssertEqual(tracker.snapshot(now: 55).totalRemainingSeconds, 10)
        XCTAssertEqual(tracker.snapshot(now: 55).currentWorkRemainingSeconds, 0)
        // A second microbatch belongs to the same optimizer step. A long step
        // cannot consume the estimate for the steps that have not started.
        tracker.begin(stage: .preparingImage, image: "b.png", index: 6, now: 57)
        tracker.begin(stage: .training, image: "b.png", index: 6, now: 59)
        tracker.begin(stage: .optimizing, now: 60)
        XCTAssertEqual(tracker.snapshot(now: 62).epochRemainingSeconds, 2)
        XCTAssertEqual(tracker.snapshot(now: 62).totalRemainingSeconds, 10)
        XCTAssertEqual(tracker.snapshot(now: 62).currentWorkRemainingSeconds, 0)
        tracker.completeStep(stepInEpoch: 3, completedSteps: 3, completedImages: 6, cycleSeconds: 10, now: 63)
        XCTAssertEqual(tracker.snapshot(now: 63).averageCycleSeconds, 6)
        XCTAssertEqual(tracker.snapshot(now: 63).epochRemainingSeconds, 6)
        XCTAssertEqual(tracker.snapshot(now: 63).totalRemainingSeconds, 30)
        XCTAssertEqual(tracker.snapshot(now: 63).currentWorkRemainingSeconds, 0)
    }

    func testSlowValidationAndCheckpointCannotConsumeFuturePasses() {
        var tracker = TrainingProgressTracker(totalEpochs: 3, trainingImages: 2,
            validationImages: 1, stepsPerEpoch: 2, startSeconds: 0)
        tracker.beginEpoch(epoch: 1, now: 0)
        tracker.completeStep(stepInEpoch: 1, completedSteps: 1, completedImages: 1, cycleSeconds: 10, now: 10)
        tracker.completeStep(stepInEpoch: 2, completedSteps: 2, completedImages: 2, cycleSeconds: 2, now: 12)
        tracker.begin(stage: .validating, now: 12)
        tracker.completeValidation(durationSeconds: 6, now: 18)
        tracker.begin(stage: .checkpointing, now: 18)
        tracker.checkpointCompleted(durationSeconds: 3, now: 21)
        tracker.beginEpoch(epoch: 2, now: 21)
        tracker.completeStep(stepInEpoch: 2, completedSteps: 4, completedImages: 4, cycleSeconds: 2, now: 25)
        tracker.begin(stage: .validating, now: 25)
        XCTAssertEqual(tracker.snapshot(now: 35).epochRemainingSeconds, 3)
        XCTAssertEqual(tracker.snapshot(now: 35).currentWorkRemainingSeconds, 0)
        XCTAssertEqual(tracker.snapshot(now: 35).totalRemainingSeconds, 19,
            "4 seconds of training, 6 seconds for future validation, and 9 seconds for remaining saves")
        tracker.completeValidation(durationSeconds: 10, now: 35)
        tracker.begin(stage: .checkpointing, now: 35)
        XCTAssertEqual(tracker.snapshot(now: 40).epochRemainingSeconds, 0)
        XCTAssertEqual(tracker.snapshot(now: 40).currentWorkRemainingSeconds, 0)
        XCTAssertEqual(tracker.snapshot(now: 40).totalRemainingSeconds, 18,
            "4 seconds of training, 8 seconds for future validation, and 6 seconds for future saves")
    }

    func testExportHasNoFabricatedETAAndCompletionHasNoRemainingWork() {
        var tracker = TrainingProgressTracker(totalEpochs: 1, trainingImages: 2,
            validationImages: 0, stepsPerEpoch: 2, startSeconds: 0)
        tracker.beginEpoch(epoch: 1, now: 0)
        tracker.completeStep(stepInEpoch: 1, completedSteps: 1, completedImages: 1, cycleSeconds: 10, now: 10)
        tracker.completeStep(stepInEpoch: 2, completedSteps: 2, completedImages: 2, cycleSeconds: 2, now: 12)
        tracker.checkpointCompleted(durationSeconds: 3, now: 15)
        tracker.begin(stage: .verifyingSources, now: 15)
        XCTAssertNil(tracker.snapshot(now: 16).totalRemainingSeconds)
        XCTAssertNil(tracker.snapshot(now: 16).currentWorkRemainingSeconds)
        tracker.begin(stage: .exporting, now: 16)
        XCTAssertNil(tracker.snapshot(now: 25).totalRemainingSeconds)
        XCTAssertNil(tracker.snapshot(now: 25).currentWorkRemainingSeconds)
        tracker.begin(stage: .checkpointing, now: 25)
        XCTAssertEqual(tracker.snapshot(now: 26).totalRemainingSeconds, 2)
        tracker.checkpointCompleted(durationSeconds: 3, now: 28)
        XCTAssertEqual(tracker.snapshot(now: 28).totalRemainingSeconds, 0)
        tracker.begin(stage: .completed, now: 29)
        let completed = tracker.snapshot(now: 200)
        XCTAssertEqual(completed.elapsedSeconds, 29)
        XCTAssertEqual(completed.epochRemainingSeconds, 0)
        XCTAssertEqual(completed.totalRemainingSeconds, 0)
        XCTAssertEqual(completed.currentWorkRemainingSeconds, 0)
    }

    func testInvalidTimingAndCounterValuesCannotProduceInvalidProgress() {
        var tracker = TrainingProgressTracker(totalEpochs: 2, trainingImages: 4,
            validationImages: 0, stepsPerEpoch: 2, startSeconds: .nan)
        tracker.beginEpoch(epoch: 10, now: 10)
        tracker.begin(stage: .training, image: "a.png", index: 10, completedImages: -1, now: -20)
        tracker.completeStep(stepInEpoch: 20, completedSteps: 30, completedImages: -2, cycleSeconds: .nan, now: .infinity)
        tracker.completeStep(stepInEpoch: -1, completedSteps: -1, completedImages: -1, cycleSeconds: -1, now: 5)
        let progress = tracker.snapshot(now: .nan)
        XCTAssertEqual(progress.epoch, 2)
        XCTAssertEqual(progress.imageIndex, 4)
        XCTAssertEqual(progress.completedImages, 0)
        XCTAssertEqual(progress.stepInEpoch, 2)
        XCTAssertEqual(progress.completedSteps, 4)
        XCTAssertEqual(progress.elapsedSeconds, 10)
        XCTAssertNil(progress.lastCycleSeconds)
        XCTAssertNil(progress.averageCycleSeconds)
        XCTAssertNil(progress.totalRemainingSeconds)
    }
}
