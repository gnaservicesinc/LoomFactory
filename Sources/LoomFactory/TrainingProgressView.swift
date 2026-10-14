import Foundation
import LoomCore
import SwiftUI

/// The training sidebar reports work and time from the trainer's current snapshot.
struct TrainingProgressView: View {
    let progress: TrainingProgress
    let isRunning: Bool

    var body: some View {
        if isRunning {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                content(now: context.date)
            }
        } else {
            content(now: progress.recordedAt)
        }
    }

    private func content(now: Date) -> some View {
        let values = TrainingProgressPresentation(progress: progress, isRunning: isRunning, now: now)
        return VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 5) {
                Text(progress.stage.displayName)
                    .font(.callout.weight(.semibold))
                    .accessibilityIdentifier("training.progress.stage")
                Text("\(values.stageElapsed) in this stage")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                    .accessibilityIdentifier("training.progress.stageElapsed")
            }
            if let image = progress.currentImage, !image.isEmpty {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Current image").font(.caption).foregroundStyle(.secondary)
                    Text(image).font(.callout.weight(.medium)).textSelection(.enabled)
                        .lineLimit(3)
                        .truncationMode(.middle)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("training.progress.image")
                }
                .padding(10).frame(maxWidth: .infinity, alignment: .leading)
                .background(FactoryStyle.accent.opacity(0.07), in: RoundedRectangle(cornerRadius: 8))
            }
            if progress.totalEpochs > 0 {
                VStack(alignment: .leading, spacing: 7) {
                    Text(progress.epoch == 0 ? "\(progress.totalEpochs.formatted()) epochs planned" : values.epoch)
                        .font(.callout.weight(.semibold)).monospacedDigit()
                        .accessibilityIdentifier("training.progress.epoch")
                    if progress.stepsPerEpoch > 0 {
                        ProgressView(value: values.epochFraction).progressViewStyle(.linear)
                            .accessibilityLabel("Current epoch progress")
                        metric("Steps this epoch", values.epochSteps, identifier: "epochSteps")
                    }
                    if progress.imagesPerEpoch > 0 {
                        metric(progress.stage == .validating ? "Validation image" : "Image in epoch", values.epochImage, identifier: "epochImage")
                    }
                }
            }
            if progress.totalSteps > 0 {
                VStack(alignment: .leading, spacing: 7) {
                    metric("Steps completed", values.steps, identifier: "steps")
                    metric("Steps left", values.remainingSteps, identifier: "remainingSteps")
                    ProgressView(value: values.runFraction).progressViewStyle(.linear)
                        .accessibilityLabel("All training steps progress")
                }
            }
            if let priorSteps = progress.priorSteps, priorSteps > 0 {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Previous training included").font(.caption.weight(.semibold))
                    metric("Completed epochs", (progress.priorEpochs ?? 0).formatted(), identifier: "priorEpochs")
                    metric("Optimizer updates", priorSteps.formatted(), identifier: "priorSteps")
                    if let duration = progress.priorElapsedSeconds {
                        metric("Training elapsed", TrainingProgressPresentation.duration(duration), identifier: "priorElapsed")
                    }
                }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                    .background(FactoryStyle.accent.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
            }
            Divider()
            VStack(alignment: .leading, spacing: 5) {
                metric("Last cycle", values.lastCycle, identifier: "lastCycle")
                metric("Average cycle", values.averageCycle, identifier: "averageCycle")
                metric(values.epochRemainingTitle, values.epochRemaining, identifier: "epochRemaining")
                metric(values.totalRemainingTitle, values.totalRemaining, identifier: "totalRemaining")
                if progress.epoch > 0 {
                    metric("Epoch elapsed", values.epochElapsed, identifier: "epochElapsed")
                }
                metric(progress.priorSteps == nil ? "Run elapsed" : "Total training elapsed", values.elapsed, identifier: "elapsed")
            }
            if let note = values.estimateScopeNote, isRunning {
                Text(note)
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if isRunning, progress.totalRemainingSeconds != nil {
                Text("Time left is estimated; final export is measured separately.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .help("A cycle is one training step, including image preparation.")
    }

    private func metric(_ title: String, _ value: String, identifier: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(title).foregroundStyle(.secondary)
            Spacer(minLength: 0)
            Text(value).monospacedDigit().multilineTextAlignment(.trailing)
        }
        .font(.caption)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("training.progress.\(identifier)")
    }
}

/// Keep formatting and anchored countdowns deterministic for snapshot validation.
struct TrainingProgressPresentation {
    private let progress: TrainingProgress
    private let age: Double

    init(progress: TrainingProgress, isRunning: Bool, now: Date) {
        self.progress = progress
        let active = isRunning && progress.stage != .completed && progress.stage != .stopped
        age = active ? max(0, now.timeIntervalSince(progress.recordedAt)) : 0
    }

    var epoch: String {
        "Epoch \(max(0, progress.epoch).formatted()) of \(max(0, progress.totalEpochs).formatted())"
    }
    var epochSteps: String {
        "\(max(0, progress.stepInEpoch).formatted()) / \(max(0, progress.stepsPerEpoch).formatted())"
    }
    var epochImage: String {
        "\(max(0, progress.imageIndex).formatted()) / \(max(0, progress.imagesPerEpoch).formatted())"
    }
    var steps: String {
        "\(max(0, progress.completedSteps).formatted()) / \(max(0, progress.totalSteps).formatted())"
    }
    var remainingSteps: String {
        progress.stage == .completed ? "0" : max(0, max(0, progress.totalSteps) - max(0, progress.completedSteps)).formatted()
    }
    var epochFraction: Double { fraction(progress.stepInEpoch, of: progress.stepsPerEpoch) }
    var runFraction: Double { fraction(progress.completedSteps, of: progress.totalSteps) }
    var elapsed: String { Self.duration(progress.elapsedSeconds + age) }
    var epochElapsed: String { Self.duration(progress.epochElapsedSeconds + age) }
    var stageElapsed: String { Self.duration(progress.stageElapsedSeconds + age) }
    var lastCycle: String { Self.cycle(progress.lastCycleSeconds) }
    var averageCycle: String { Self.cycle(progress.averageCycleSeconds) }
    var epochRemaining: String { remaining(progress.epochRemainingSeconds) }
    var totalRemaining: String { remaining(progress.totalRemainingSeconds) }
    var estimatesCoverStages: Bool {
        progress.estimatesIncludeValidation && progress.estimatesIncludeCheckpointing
    }
    var epochRemainingTitle: String { estimatesCoverStages ? "Epoch time left" : "Epoch training left" }
    var totalRemainingTitle: String { estimatesCoverStages ? "Estimated total left" : "Training time left" }
    var estimateScopeNote: String? {
        if estimatesCoverStages { return nil }
        let stages = switch (progress.estimatesIncludeValidation, progress.estimatesIncludeCheckpointing) {
        case (false, false): "validation and saving"
        case (false, true): "validation"
        case (true, false): "saving"
        case (true, true): ""
        }
        return "Estimates will include \(stages) when those timings are available."
    }

    private func remaining(_ seconds: Double?) -> String {
        if progress.stage == .completed { return "Complete" }
        if progress.stage == .stopped { return "—" }
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "Measuring…" }
        let budget = progress.currentWorkRemainingSeconds ?? 0
        let countdown = budget.isFinite ? min(age, max(0, budget)) : 0
        return Self.duration(max(0, seconds - countdown))
    }

    private func fraction(_ value: Int, of total: Int) -> Double {
        guard total > 0 else { return 0 }
        return min(1, max(0, Double(value) / Double(total)))
    }

    private static func cycle(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "Measuring…" }
        return "\(seconds.formatted(.number.precision(.fractionLength(2)))) s / step"
    }

    static func duration(_ seconds: Double) -> String {
        guard seconds.isFinite else { return "—" }
        // Bound the conversion as well as the display: malformed event values cannot trap Int.
        let whole = Int(min(Double(Int.max / 2), max(0, seconds)).rounded(.down))
        let hours = whole / 3600, minutes = (whole % 3600) / 60, remainder = whole % 60
        if hours > 0 { return "\(hours.formatted())h \(minutes)m \(remainder)s" }
        if minutes > 0 { return "\(minutes)m \(remainder)s" }
        return "\(remainder)s"
    }
}
