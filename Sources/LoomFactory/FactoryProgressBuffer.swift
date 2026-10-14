import Foundation
import LoomCore

/// Source checking and restored history can emit many events at once. Deliver
/// bounded batches to the UI, keeping every event in the trainer's saved log.
final class FactoryProgressBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [TrainingEvent] = []
    private var measurements: TrainingEvent?
    private var scheduled = false

    func append(_ event: TrainingEvent, deliver: @escaping @MainActor @Sendable ([TrainingEvent], TrainingEvent?) -> Void) {
        let shouldSchedule = lock.withLock {
            let saved = measurements
            measurements = TrainingEvent(type: "measurements", message: "",
                progress: event.progress ?? saved?.progress, epoch: event.epoch ?? saved?.epoch,
                loss: event.loss ?? saved?.loss, validationLoss: event.validationLoss ?? saved?.validationLoss,
                learningRate: event.learningRate ?? saved?.learningRate, update: event.update ?? saved?.update,
                totalUpdates: event.totalUpdates ?? saved?.totalUpdates,
                parameterCount: event.parameterCount ?? saved?.parameterCount,
                effectiveBatchSize: event.effectiveBatchSize ?? saved?.effectiveBatchSize,
                gradientNorm: event.gradientNorm ?? saved?.gradientNorm,
                gradientClipThreshold: event.gradientClipThreshold ?? saved?.gradientClipThreshold,
                trainingProgress: event.trainingProgress ?? saved?.trainingProgress,
                cycleTiming: event.cycleTiming ?? saved?.cycleTiming)
            events.append(event)
            if events.count > 300 { events.removeFirst(events.count - 300) }
            guard !scheduled else { return false }
            scheduled = true
            return true
        }
        guard shouldSchedule else { return }
        Task {
            try? await Task.sleep(for: .milliseconds(30))
            let batch = lock.withLock {
                let values = events
                events.removeAll(keepingCapacity: true)
                scheduled = false
                return (values, measurements)
            }
            await deliver(batch.0, batch.1)
        }
    }
}
