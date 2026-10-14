import AppKit
import Foundation
import LoomCore
import SwiftUI
import Testing
@testable import LoomFactory

@Suite(.serialized)
struct TrainingProgressViewTests {
    private let recordedAt = Date(timeIntervalSince1970: 1_000_000)

    @Test
    func imageAndStepCountersDescribeCurrentEpochAndWholeRun() {
        let progress = snapshot()
        let values = TrainingProgressPresentation(progress: progress, isRunning: true, now: recordedAt)
        #expect(values.epoch == "Epoch 2 of 500")
        #expect(values.epochSteps == "13 / 22")
        #expect(values.epochImage == "26 / 44")
        #expect(values.steps == "\(35.formatted()) / \(11_000.formatted())")
        #expect(values.remainingSteps == 10_965.formatted())
        #expect(abs(values.epochFraction - 13.0 / 22.0) < 0.000001)
        #expect(abs(values.runFraction - 35.0 / 11_000.0) < 0.000001)
        // Seventy visits include the previous epoch; they are not the image position.
        #expect(progress.completedImages == 70)
        #expect(values.lastCycle == "\(2.8.formatted(.number.precision(.fractionLength(2)))) s / step")
        #expect(values.averageCycle == "\(2.5.formatted(.number.precision(.fractionLength(2)))) s / step")
    }

    @Test
    func elapsedTimesAndEstimatesAdvanceFromTheSameRecordedSnapshot() {
        let values = TrainingProgressPresentation(progress: snapshot(), isRunning: true,
                                                  now: recordedAt.addingTimeInterval(5))
        #expect(values.stageElapsed == "7s")
        #expect(values.epochElapsed == "37s")
        #expect(values.elapsed == "1m 40s")
        #expect(values.epochRemaining == "25s")
        #expect(values.totalRemaining == "7h 37m 7s")
        #expect(values.estimatesCoverStages)
        #expect(values.epochRemainingTitle == "Epoch time left")
        #expect(values.totalRemainingTitle == "Estimated total left")
        #expect(values.estimateScopeNote == nil)

        let exhausted = TrainingProgressPresentation(progress: snapshot(), isRunning: true,
                                                     now: recordedAt.addingTimeInterval(30_000))
        #expect(exhausted.epochRemaining == "0s")
        #expect(exhausted.totalRemaining == "7h 36m 42s", "An overdue stage cannot consume the time for unstarted work")
        let clockMovedBack = TrainingProgressPresentation(progress: snapshot(), isRunning: true,
                                                          now: recordedAt.addingTimeInterval(-100))
        #expect(clockMovedBack.elapsed == "1m 35s")
        #expect(clockMovedBack.epochRemaining == "30s")
    }

    @Test
    func stoppedOrArchivedRunsFreezeElapsedTimeAndStopCountingDown() {
        let later = recordedAt.addingTimeInterval(3_600)
        var progress = snapshot()
        let archived = TrainingProgressPresentation(progress: progress, isRunning: false, now: later)
        #expect(archived.stageElapsed == "2s")
        #expect(archived.epochElapsed == "32s")
        #expect(archived.elapsed == "1m 35s")
        #expect(archived.epochRemaining == "30s")
        #expect(archived.totalRemaining == "7h 37m 12s")

        progress.stage = .stopped
        let stopped = TrainingProgressPresentation(progress: progress, isRunning: true, now: later)
        #expect(stopped.elapsed == "1m 35s")
        #expect(stopped.epochElapsed == "32s")
        #expect(stopped.epochRemaining == "—")
        #expect(stopped.totalRemaining == "—")
        progress.stage = .completed
        let completed = TrainingProgressPresentation(progress: progress, isRunning: true, now: later)
        #expect(completed.elapsed == "1m 35s")
        #expect(completed.epochRemaining == "Complete")
        #expect(completed.totalRemaining == "Complete")
        #expect(completed.remainingSteps == "0")
    }

    @Test
    func startupMeasuresRateAndOnlyClaimsKnownEstimateCoverage() {
        var progress = TrainingProgress()
        progress.recordedAt = recordedAt
        var values = TrainingProgressPresentation(progress: progress, isRunning: true, now: recordedAt)
        #expect(values.lastCycle == "Measuring…")
        #expect(values.averageCycle == "Measuring…")
        #expect(values.epochRemaining == "Measuring…")
        #expect(values.totalRemaining == "Measuring…")
        #expect(values.epochFraction == 0)
        #expect(values.runFraction == 0)
        #expect(values.epochRemainingTitle == "Epoch training left")
        #expect(values.totalRemainingTitle == "Training time left")
        #expect(values.estimateScopeNote?.contains("validation and saving") == true)

        progress.estimatesIncludeValidation = true
        values = TrainingProgressPresentation(progress: progress, isRunning: true, now: recordedAt)
        #expect(!values.estimatesCoverStages)
        #expect(values.estimateScopeNote?.contains("include saving") == true)
        #expect(values.estimateScopeNote?.contains("validation") == false)
        progress.estimatesIncludeValidation = false
        progress.estimatesIncludeCheckpointing = true
        values = TrainingProgressPresentation(progress: progress, isRunning: true, now: recordedAt)
        #expect(values.estimateScopeNote?.contains("include validation") == true)
        #expect(values.estimateScopeNote?.contains("saving") == false)
        progress.estimatesIncludeValidation = true
        values = TrainingProgressPresentation(progress: progress, isRunning: true, now: recordedAt)
        #expect(values.estimatesCoverStages)
        #expect(values.estimateScopeNote == nil)
    }

    @Test
    func malformedTimingAndCounterValuesRemainSafeToPresent() {
        var progress = TrainingProgress()
        progress.recordedAt = recordedAt
        progress.totalSteps = Int.min
        progress.completedSteps = Int.max
        progress.stepInEpoch = -1
        progress.stepsPerEpoch = 22
        progress.imageIndex = -1
        progress.imagesPerEpoch = 44
        progress.lastCycleSeconds = .infinity
        progress.averageCycleSeconds = .nan
        progress.epochRemainingSeconds = -1
        progress.totalRemainingSeconds = .nan
        let values = TrainingProgressPresentation(progress: progress, isRunning: false, now: recordedAt)
        #expect(values.remainingSteps == "0")
        #expect(values.epochSteps == "0 / 22")
        #expect(values.epochImage == "0 / 44")
        #expect(values.epochFraction == 0)
        #expect(values.runFraction == 0)
        #expect(values.lastCycle == "Measuring…")
        #expect(values.averageCycle == "Measuring…")
        #expect(values.epochRemaining == "Measuring…")
        #expect(values.totalRemaining == "Measuring…")
        #expect(TrainingProgressPresentation.duration(.infinity) == "—")
        #expect(TrainingProgressPresentation.duration(-1) == "0s")
        #expect(!TrainingProgressPresentation.duration(Double.greatestFiniteMagnitude).isEmpty)
        progress.totalSteps = 22
        progress.completedSteps = 23
        let pastEnd = TrainingProgressPresentation(progress: progress, isRunning: false, now: recordedAt)
        #expect(pastEnd.runFraction == 1)
        #expect(pastEnd.remainingSteps == "0")
    }

    @Test @MainActor
    func progressRendersCompactlyInTheNative290PointSidebar() async throws {
        _ = NSApplication.shared
        let width: CGFloat = 290
        let view = NSHostingView(rootView: TrainingProgressView(progress: snapshot(), isRunning: false)
            .padding(14).frame(width: width).background(.background).preferredColorScheme(.dark))
        let window = NSWindow(contentRect: NSRect(x: -20_000, y: -20_000, width: width, height: 700),
                              styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFrontRegardless()
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(150))
        view.layoutSubtreeIfNeeded()
        let height = ceil(view.fittingSize.height)
        #expect(height >= 250)
        #expect(height <= 450)
        window.setContentSize(NSSize(width: width, height: height))
        view.frame = NSRect(x: 0, y: 0, width: width, height: height)
        view.layoutSubtreeIfNeeded()
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1_000)
        let directory: URL
        if let destination = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            directory = URL(fileURLWithPath: destination, isDirectory: true)
        } else {
            directory = FileManager.default.temporaryDirectory.appendingPathComponent("loom-training-progress-render", isDirectory: true)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try png.write(to: directory.appendingPathComponent("training-progress-44-images-500-epochs.png"))
    }

    private func snapshot() -> TrainingProgress {
        var progress = TrainingProgress()
        progress.stage = .training
        progress.recordedAt = recordedAt
        progress.currentImage = "Leather Red 03 / leather_red_03_col2_4k.png"
        progress.epoch = 2
        progress.totalEpochs = 500
        progress.stepInEpoch = 13
        progress.stepsPerEpoch = 22
        progress.imageIndex = 26
        progress.imagesPerEpoch = 44
        progress.completedImages = 70
        progress.completedSteps = 35
        progress.totalSteps = 11_000
        progress.elapsedSeconds = 95.5
        progress.epochElapsedSeconds = 32.5
        progress.stageElapsedSeconds = 2.5
        progress.lastCycleSeconds = 2.8
        progress.averageCycleSeconds = 2.5
        progress.epochRemainingSeconds = 30
        progress.totalRemainingSeconds = 27_432
        progress.currentWorkRemainingSeconds = 30
        progress.estimatesIncludeValidation = true
        progress.estimatesIncludeCheckpointing = true
        return progress
    }
}
