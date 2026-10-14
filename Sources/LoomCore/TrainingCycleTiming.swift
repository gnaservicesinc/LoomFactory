import Foundation

/// Measured wall time, kept separate from user-facing work counters.
public struct TrainingCycleTiming: Codable, Sendable, Equatable {
    public let preparationSeconds: Double
    public let preparationWaitSeconds: Double
    public let trainingSeconds: Double
    public let optimizerSeconds: Double
    public let totalSeconds: Double
    public init(preparationSeconds: Double, preparationWaitSeconds: Double,
                trainingSeconds: Double, optimizerSeconds: Double, totalSeconds: Double) {
        self.preparationSeconds = preparationSeconds; self.preparationWaitSeconds = preparationWaitSeconds
        self.trainingSeconds = trainingSeconds; self.optimizerSeconds = optimizerSeconds; self.totalSeconds = totalSeconds
    }
}
