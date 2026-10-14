import Foundation
import LoomCore
import Testing
@testable import LoomFactory

struct FactoryProgressBufferTests {
    @Test @MainActor
    func thousandsOfVerificationEventsCannotEraseTrainingMeasurements() async throws {
        let buffer = FactoryProgressBuffer()
        var delivered: [TrainingEvent] = []
        var summary: TrainingEvent?
        let deliver: @MainActor @Sendable ([TrainingEvent], TrainingEvent?) -> Void = { events, measurements in
            delivered = events
            summary = measurements
        }
        buffer.append(.init(type: "started", message: "Training", parameterCount: 2_000_000, effectiveBatchSize: 4), deliver: deliver)
        buffer.append(.init(type: "epoch_completed", message: "Validated", epoch: 8, loss: 0.2,
            validationLoss: 0.3, learningRate: 0.0001, update: 120, totalUpdates: 120,
            gradientNorm: 0.7, gradientClipThreshold: 1), deliver: deliver)
        for index in 0..<3000 {
            buffer.append(.init(type: "status", message: "Checking original \(index)"), deliver: deliver)
        }
        for _ in 0..<100 {
            if summary != nil { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(delivered.count == 300)
        #expect(delivered.allSatisfy { $0.type == "status" })
        let measurements = try #require(summary)
        let metrics = TrainingRunMetrics(events: [measurements] + delivered)
        #expect(metrics.epoch == 8)
        #expect(metrics.update == 120)
        #expect(metrics.totalUpdates == 120)
        #expect(metrics.trainingLoss == 0.2)
        #expect(metrics.validationLoss == 0.3)
        #expect(metrics.learningRate == 0.0001)
        #expect(metrics.parameterCount == 2_000_000)
        #expect(metrics.effectiveBatchSize == 4)
        #expect(metrics.gradientNorm == 0.7)
        #expect(metrics.gradientClipThreshold == 1)
    }
}
