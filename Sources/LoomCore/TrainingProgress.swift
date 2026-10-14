import Foundation

public enum TrainingStage: String, Codable, Sendable, CaseIterable {
    case checkingSources, preparingModel, preparingImage, training, optimizing
    case validating, checkpointing, verifyingSources, exporting, completed, stopped

    public var displayName: String {
        switch self {
        case .checkingSources: "Checking source images"
        case .preparingModel: "Preparing model"
        case .preparingImage: "Preparing image"
        case .training: "Training image"
        case .optimizing: "Updating model"
        case .validating: "Validating held-out images"
        case .checkpointing: "Saving checkpoint"
        case .verifyingSources: "Verifying original images"
        case .exporting: "Exporting model"
        case .completed: "Complete"
        case .stopped: "Stopped"
        }
    }
}

/// A measured snapshot of a run. Image position is local to the current epoch;
/// completed image visits and optimizer steps are cumulative across the run.
public struct TrainingProgress: Codable, Sendable, Equatable {
    public var stage: TrainingStage
    public var currentImage: String?
    public var epoch: Int
    public var totalEpochs: Int
    public var imageIndex: Int
    public var imagesPerEpoch: Int
    public var completedImages: Int
    public var stepInEpoch: Int
    public var stepsPerEpoch: Int
    public var completedSteps: Int
    public var totalSteps: Int
    public var elapsedSeconds: Double
    public var epochElapsedSeconds: Double
    public var lastCycleSeconds: Double?
    public var averageCycleSeconds: Double?
    public var epochRemainingSeconds: Double?
    public var totalRemainingSeconds: Double?
    public var recordedAt: Date
    public var stageElapsedSeconds: Double
    public var estimatesIncludeValidation: Bool
    public var estimatesIncludeCheckpointing: Bool
    /// The portion of the ETA assigned to the work currently in progress.
    /// Live displays may count down this amount without consuming future work.
    public var currentWorkRemainingSeconds: Double?
    /// Work inherited from the selected weights. Optional for old saved logs.
    public var priorEpochs: Int?
    public var priorSteps: Int?
    public var priorImages: Int?
    public var priorElapsedSeconds: Double?

    public init(stage: TrainingStage = .checkingSources, currentImage: String? = nil,
                epoch: Int = 0, totalEpochs: Int = 0, imageIndex: Int = 0, imagesPerEpoch: Int = 0,
                completedImages: Int = 0, stepInEpoch: Int = 0, stepsPerEpoch: Int = 0,
                completedSteps: Int = 0, totalSteps: Int = 0, elapsedSeconds: Double = 0,
                epochElapsedSeconds: Double = 0, lastCycleSeconds: Double? = nil,
                averageCycleSeconds: Double? = nil, epochRemainingSeconds: Double? = nil,
                totalRemainingSeconds: Double? = nil, recordedAt: Date = Date(),
                stageElapsedSeconds: Double = 0, estimatesIncludeValidation: Bool = false,
                estimatesIncludeCheckpointing: Bool = false, currentWorkRemainingSeconds: Double? = nil,
                priorEpochs: Int? = nil, priorSteps: Int? = nil, priorImages: Int? = nil,
                priorElapsedSeconds: Double? = nil) {
        self.stage = stage; self.currentImage = currentImage
        self.epoch = epoch; self.totalEpochs = totalEpochs
        self.imageIndex = imageIndex; self.imagesPerEpoch = imagesPerEpoch
        self.completedImages = completedImages; self.stepInEpoch = stepInEpoch
        self.stepsPerEpoch = stepsPerEpoch; self.completedSteps = completedSteps
        self.totalSteps = totalSteps; self.elapsedSeconds = elapsedSeconds
        self.epochElapsedSeconds = epochElapsedSeconds; self.lastCycleSeconds = lastCycleSeconds
        self.averageCycleSeconds = averageCycleSeconds; self.epochRemainingSeconds = epochRemainingSeconds
        self.totalRemainingSeconds = totalRemainingSeconds; self.recordedAt = recordedAt
        self.stageElapsedSeconds = stageElapsedSeconds
        self.estimatesIncludeValidation = estimatesIncludeValidation
        self.estimatesIncludeCheckpointing = estimatesIncludeCheckpointing
        self.currentWorkRemainingSeconds = currentWorkRemainingSeconds
        self.priorEpochs = priorEpochs; self.priorSteps = priorSteps; self.priorImages = priorImages
        self.priorElapsedSeconds = priorElapsedSeconds
    }
}

/// Uses a monotonic clock supplied by the trainer so deterministic tests need no
/// sleeping. A cycle includes preparation, GPU work, and its optimizer update.
struct TrainingProgressTracker {
    private let totalEpochs: Int
    private let trainingImages: Int
    private let validationImages: Int
    private let stepsPerEpoch: Int
    private let totalSteps: Int
    private let checkpointEveryEpochs: Int
    private let startSeconds: Double
    private var epochStartedSeconds: Double?
    private var stageStartedSeconds: Double
    private var stoppedSeconds: Double?
    private var lastSeconds: Double
    private var stage = TrainingStage.checkingSources
    private var currentImage: String?
    private var epoch = 0
    private var imageIndex = 0
    private var completedImages = 0
    private var stepInEpoch = 0
    private var completedSteps = 0
    private var lastCycleSeconds: Double?
    private var cycleStartedSeconds: Double?
    private var cycleActive = false
    private var cycles: [Double] = []
    private var validCycleCount = 0
    private var validationDurations: [Double] = []
    private var checkpointDurations: [Double] = []
    private var validatedEpochs: Set<Int> = []
    private var checkpointedEpochs: Set<Int> = []
    private var checkpointActive = false
    private var finalCheckpointCompleted = false
    private var exportStarted = false
    private var priorEpochs = 0
    private var priorSteps = 0
    private var priorImages = 0
    private var priorElapsedSeconds = 0.0
    private var restoredElapsedSeconds = 0.0
    private var restoredEpochElapsedSeconds = 0.0
    private static let timingWindow = 20

    init(totalEpochs: Int, trainingImages: Int, validationImages: Int, stepsPerEpoch: Int,
         checkpointEveryEpochs: Int = 1, startSeconds: Double) {
        self.totalEpochs = max(0, totalEpochs)
        self.trainingImages = max(0, trainingImages)
        self.validationImages = max(0, validationImages)
        self.stepsPerEpoch = max(0, stepsPerEpoch)
        let (count, overflow) = max(0, totalEpochs).multipliedReportingOverflow(by: max(0, stepsPerEpoch))
        self.totalSteps = overflow ? Int.max : count
        self.checkpointEveryEpochs = max(1, checkpointEveryEpochs)
        let start = startSeconds.isFinite ? startSeconds : 0
        self.startSeconds = start; self.stageStartedSeconds = start; self.lastSeconds = start
    }

    mutating func beginEpoch(epoch: Int, now: Double) {
        let now = clock(now)
        self.epoch = min(totalEpochs, max(0, epoch))
        epochStartedSeconds = now
        restoredEpochElapsedSeconds = 0
        stepInEpoch = 0; imageIndex = 0; currentImage = nil
        cycleActive = false
    }

    mutating func restoreEpochPosition(stepInEpoch: Int, imageIndex: Int, elapsedSeconds: Double) {
        self.stepInEpoch = min(stepsPerEpoch, max(0, stepInEpoch))
        self.imageIndex = min(trainingImages, max(0, imageIndex))
        restoredEpochElapsedSeconds = elapsedSeconds.isFinite ? max(0, elapsedSeconds) : 0
    }

    /// Restore measured work without treating past steps as new timing samples.
    /// ETA still uses measurements from the current machine and training plan.
    mutating func restore(epoch: Int = 0, stepInEpoch: Int = 0, completedSteps: Int = 0,
                          completedImages: Int = 0, elapsedSeconds: Double = 0,
                          priorEpochs: Int = 0, priorSteps: Int = 0, priorImages: Int = 0,
                          priorElapsedSeconds: Double = 0, now: Double) {
        _ = clock(now)
        self.epoch = min(totalEpochs, max(0, epoch))
        self.stepInEpoch = min(stepsPerEpoch, max(0, stepInEpoch))
        self.completedSteps = min(totalSteps, max(0, completedSteps))
        self.completedImages = max(0, completedImages)
        self.priorEpochs = max(0, priorEpochs); self.priorSteps = max(0, priorSteps)
        self.priorImages = max(0, priorImages)
        self.priorElapsedSeconds = priorElapsedSeconds.isFinite ? max(0, priorElapsedSeconds) : 0
        restoredElapsedSeconds = elapsedSeconds.isFinite ? max(0, elapsedSeconds) : 0
        // Already-completed epochs have no pending validation or scheduled save.
        let completedEpochs = stepsPerEpoch > 0 ? self.completedSteps / stepsPerEpoch : 0
        if completedEpochs > 0 {
            validatedEpochs.formUnion(1...completedEpochs)
            checkpointedEpochs.formUnion((1...completedEpochs).filter { checkpointScheduled($0) })
        }
    }

    mutating func begin(stage: TrainingStage, image: String? = nil, index: Int? = nil,
                        completedImages: Int? = nil, now: Double) {
        let now = clock(now)
        if self.stage != stage || (stage == .checkpointing && !checkpointActive) { stageStartedSeconds = now }
        self.stage = stage; self.currentImage = image
        if stage == .checkpointing { checkpointActive = true }
        if stage == .preparingImage || stage == .training || stage == .optimizing {
            if !cycleActive { cycleStartedSeconds = now }
            cycleActive = true
        } else { cycleActive = false }
        if let index { imageIndex = min(stage == .validating ? validationImages : trainingImages, max(0, index)) }
        if let completedImages { self.completedImages = max(self.completedImages, max(0, completedImages)) }
        if stage == .exporting { exportStarted = true }
        if stage == .completed || stage == .stopped { stoppedSeconds = now }
    }

    mutating func completeStep(stepInEpoch: Int, completedSteps: Int, completedImages: Int,
                               cycleSeconds: Double, now: Double) {
        _ = clock(now)
        cycleActive = false
        self.stepInEpoch = min(stepsPerEpoch, max(self.stepInEpoch, max(0, stepInEpoch)))
        self.completedSteps = min(totalSteps, max(self.completedSteps, max(0, completedSteps)))
        self.completedImages = max(self.completedImages, max(0, completedImages))
        guard cycleSeconds.isFinite, cycleSeconds >= 0 else { return }
        lastCycleSeconds = cycleSeconds
        validCycleCount += 1
        // Compilation and initial GPU allocation belong to elapsed time but
        // would distort the remaining work once a steady cycle has completed.
        if validCycleCount == 2 { cycles.removeAll(keepingCapacity: true) }
        append(cycleSeconds, to: &cycles)
    }

    /// Duration is the whole validation pass, including its image preparation.
    mutating func completeValidation(durationSeconds: Double, now: Double) {
        _ = clock(now)
        if epoch > 0 { validatedEpochs.insert(epoch) }
        guard durationSeconds.isFinite, durationSeconds >= 0 else { return }
        append(durationSeconds, to: &validationDurations)
    }

    /// The first-update safety save supplies a timing without consuming an
    /// epoch checkpoint. A save after exporting is the final run checkpoint.
    mutating func checkpointCompleted(durationSeconds: Double, epochComplete: Bool = true, now: Double) {
        _ = clock(now)
        checkpointActive = false
        if epochComplete {
            if exportStarted { finalCheckpointCompleted = true }
            else if epoch > 0 { checkpointedEpochs.insert(epoch) }
        }
        guard durationSeconds.isFinite, durationSeconds >= 0 else { return }
        append(durationSeconds, to: &checkpointDurations)
    }

    func snapshot(now: Double, date: Date = Date()) -> TrainingProgress {
        let now = stoppedSeconds ?? (now.isFinite ? max(lastSeconds, now) : lastSeconds)
        let stageElapsed = max(0, now - stageStartedSeconds)
        let averageCycle = validCycleCount >= 2 ? mean(cycles) : nil
        let validationMean = mean(validationDurations)
        let checkpointMean = mean(checkpointDurations)
        let terminal = stage == .completed || stage == .stopped
        let cycleElapsed = cycleActive ? cycleStartedSeconds.map { max(0, now - $0) } ?? 0 : 0
        let validationPending = epoch > 0 && !validatedEpochs.contains(epoch)
        let epochSavePending = epoch > 0 && checkpointScheduled(epoch) && !checkpointedEpochs.contains(epoch)
        let currentSavePending = (epochSavePending && stepInEpoch == stepsPerEpoch)
            || (exportStarted && !finalCheckpointCompleted)
        var currentWorkRemaining: Double?
        if terminal { currentWorkRemaining = 0 }
        else {
            switch stage {
            case .preparingImage, .training, .optimizing:
                if let averageCycle {
                    currentWorkRemaining = cycleActive && stepInEpoch < stepsPerEpoch
                        ? max(0, averageCycle - cycleElapsed) : 0
                }
            case .validating:
                if validationImages == 0 { currentWorkRemaining = 0 }
                else if let validationMean {
                    currentWorkRemaining = validationPending ? max(0, validationMean - stageElapsed) : 0
                }
            case .checkpointing:
                if let checkpointMean {
                    currentWorkRemaining = checkpointActive && currentSavePending
                        ? max(0, checkpointMean - stageElapsed) : 0
                }
            default: break
            }
        }
        var epochRemaining: Double?
        var totalRemaining: Double?
        if stage == .completed {
            epochRemaining = 0; totalRemaining = 0
        } else if !terminal, let averageCycle {
            let epochStepsLeft = max(0, stepsPerEpoch - stepInEpoch)
            let stepsLeft = max(0, totalSteps - completedSteps)
            // Count down only the current cycle. A slower-than-average cycle
            // cannot consume the estimate for steps that have not started.
            let cycleCountdown = min(averageCycle, cycleElapsed)
            epochRemaining = averageCycle * Double(epochStepsLeft) - (epochStepsLeft > 0 ? cycleCountdown : 0)
            totalRemaining = averageCycle * Double(stepsLeft) - (epochStepsLeft > 0 && stepsLeft > 0 ? cycleCountdown : 0)
            if validationImages > 0, let validationMean {
                let passesLeft = max(0, totalEpochs - validatedEpochs.count)
                let activeElapsed = stage == .validating && validationPending ? stageElapsed : 0
                let currentRemaining = validationPending ? max(0, validationMean - activeElapsed) : 0
                epochRemaining! += currentRemaining
                totalRemaining! += max(0, validationMean * Double(passesLeft) - min(validationMean, activeElapsed))
            }
            if let checkpointMean {
                let scheduled = totalEpochs / checkpointEveryEpochs + (totalEpochs % checkpointEveryEpochs == 0 ? 0 : 1)
                let completedScheduled = checkpointedEpochs.filter { checkpointScheduled($0) }.count
                let savesLeft = Double(max(0, scheduled - completedScheduled)) + (finalCheckpointCompleted ? 0 : 1)
                let activeElapsed = stage == .checkpointing && checkpointActive && currentSavePending ? stageElapsed : 0
                epochRemaining! += epochSavePending ? max(0, checkpointMean - activeElapsed) : 0
                totalRemaining! += max(0, checkpointMean * savesLeft - min(checkpointMean, activeElapsed))
            }
            // Verification and Core ML export have no measured duration yet.
            // Their stage timer remains live instead of showing a false ETA.
            if stage == .exporting || stage == .verifyingSources { totalRemaining = nil }
        }
        return TrainingProgress(stage: stage, currentImage: currentImage,
            epoch: adding(epoch, priorEpochs), totalEpochs: adding(totalEpochs, priorEpochs), imageIndex: imageIndex,
            imagesPerEpoch: stage == .validating ? validationImages : trainingImages,
            completedImages: adding(completedImages, priorImages), stepInEpoch: stepInEpoch, stepsPerEpoch: stepsPerEpoch,
            completedSteps: adding(completedSteps, priorSteps), totalSteps: adding(totalSteps, priorSteps),
            elapsedSeconds: priorElapsedSeconds + restoredElapsedSeconds + max(0, now - startSeconds),
            epochElapsedSeconds: restoredEpochElapsedSeconds + (epochStartedSeconds.map { max(0, now - $0) } ?? 0),
            lastCycleSeconds: lastCycleSeconds, averageCycleSeconds: averageCycle,
            epochRemainingSeconds: finite(epochRemaining), totalRemainingSeconds: finite(totalRemaining),
            recordedAt: date, stageElapsedSeconds: stageElapsed,
            estimatesIncludeValidation: validationImages == 0 || validationMean != nil,
            estimatesIncludeCheckpointing: checkpointMean != nil,
            currentWorkRemainingSeconds: finite(currentWorkRemaining),
            priorEpochs: priorEpochs > 0 ? priorEpochs : nil, priorSteps: priorSteps > 0 ? priorSteps : nil,
            priorImages: priorImages > 0 ? priorImages : nil,
            priorElapsedSeconds: priorElapsedSeconds > 0 ? priorElapsedSeconds : nil)
    }

    private func checkpointScheduled(_ epoch: Int) -> Bool {
        epoch > 0 && (epoch % checkpointEveryEpochs == 0 || epoch == totalEpochs)
    }

    private func adding(_ first: Int, _ second: Int) -> Int {
        let (total, overflow) = first.addingReportingOverflow(second)
        return overflow ? Int.max : total
    }

    private mutating func clock(_ now: Double) -> Double {
        if now.isFinite { lastSeconds = max(lastSeconds, now) }
        return lastSeconds
    }

    private func append(_ duration: Double, to values: inout [Double]) {
        values.append(duration)
        if values.count > Self.timingWindow { values.removeFirst(values.count - Self.timingWindow) }
    }

    private func mean(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        // Divide each term before summation to keep large valid durations from
        // overflowing when they are averaged.
        return finite(values.reduce(0) { $0 + $1 / Double(values.count) })
    }

    private func finite(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value >= 0 else { return nil }
        return value
    }
}
