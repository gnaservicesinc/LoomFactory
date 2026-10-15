import Foundation
import XCTest
@testable import LoomCore

final class TrainingContinuationHistoryTests: XCTestCase {
    func testObjectiveLabelsSurviveSeveralBranchesWithoutRelabellingOldLosses() {
        let first: [String: Any] = ["settings": ["scalar_loss": "logit-cross-entropy-v2"],
            "prior_history": [["epoch": 1, "loss": 0.2, "scalar_loss": "absolute-v1"]],
            "prior_epochs": 1, "epochs_completed": 1,
            "history": [["epoch": 1, "loss": 0.6]]]
        let branch = TrainingContinuationHistory(descriptor: first, source: .checkpoint,
            runID: "v2-parent", resumingCursor: false)
        var next = branch.json
        next["scalar_loss"] = "native-detail-v3"
        next["epochs"] = 1
        next["history"] = [["epoch": 1, "loss": 0.7]]
        let continued = TrainingContinuationHistory(descriptor: ["training": next], source: .model,
            runID: "v3-parent", resumingCursor: false)
        XCTAssertEqual(continued.rows.compactMap { $0["scalar_loss"] as? String },
            ["absolute-v1", "logit-cross-entropy-v2", "native-detail-v3"])
        XCTAssertEqual(continued.rows.compactMap { $0["loss"] as? Double }, [0.2, 0.6, 0.7])
    }

    func testLegacyTimingRecoveryUsesOnlyAMeasuredMatchingSavedUpdate() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("legacy-progress-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = TrainingProgress(stage: .checkpointing, completedImages: 14, completedSteps: 7,
            elapsedSeconds: 120, epochElapsedSeconds: 9)
        let later = TrainingProgress(stage: .training, completedSteps: 8, elapsedSeconds: 125)
        var log = Data()
        for progress in [first, later] {
            let value = try JSONSerialization.jsonObject(with: JSONEncoder().encode(progress))
            log.append(try JSONSerialization.data(withJSONObject: ["training_progress": value])); log.append(10)
        }
        try log.write(to: directory.appendingPathComponent("training.log"))
        let recovered = try XCTUnwrap(TrainingContinuationHistory.measuredProgress(in: directory, completedUpdates: 7))
        XCTAssertEqual(recovered.elapsedSeconds, 120)
        XCTAssertEqual(recovered.epochElapsedSeconds, 9)
        XCTAssertEqual(recovered.completedImages, 14)
        XCTAssertNil(TrainingContinuationHistory.measuredProgress(in: directory, completedUpdates: 6))
    }

    func testNewSchedulePreservesAllEarlierMeasuredWorkAndHistory() {
        let priorRows: [[String: Any]] = [["epoch": 1, "loss": 0.9, "update": 4, "run_id": "ancestor"]]
        let localRows: [[String: Any]] = [["epoch": 1, "loss": 0.6, "update": 3], ["epoch": 2, "loss": 0.3, "update": 6]]
        let training: [String: Any] = [
            "prior_history": priorRows,
            "prior_epochs": 1, "prior_updates": 4, "prior_images": 8, "prior_duration_seconds": 30.0,
            "epochs": 2, "updates": 6, "completed_images": 12, "elapsed_seconds": 40.0,
            "history": localRows
        ]
        let restored = TrainingContinuationHistory(descriptor: ["training": training], source: .model,
            runID: "parent", resumingCursor: false)
        XCTAssertEqual(restored.epochs, 3)
        XCTAssertEqual(restored.updates, 10)
        XCTAssertEqual(restored.images, 20)
        XCTAssertEqual(restored.elapsedSeconds, 70)
        XCTAssertEqual(restored.rows.count, 3)
        XCTAssertEqual(restored.rows.last?["epoch"] as? Int, 3)
        XCTAssertEqual(restored.rows.last?["update"] as? Int, 10)
        XCTAssertEqual(restored.rows.last?["run_id"] as? String, "parent")
    }

    func testExactInterruptedCursorDoesNotCountItsLocalHistoryTwice() {
        let descriptor: [String: Any] = ["prior_epochs": 3, "prior_updates": 10, "prior_images": 20,
            "prior_duration_seconds": 70.0, "elapsed_seconds": 22.0,
            "prior_history": [["epoch": 3, "loss": 0.3]], "history": [["epoch": 1, "loss": 0.2]],
            "updates": 4, "epochs_completed": 1, "completed_images": 7]
        let restored = TrainingContinuationHistory(descriptor: descriptor, source: .checkpoint,
            runID: "interrupted", resumingCursor: true)
        XCTAssertEqual(restored.epochs, 3)
        XCTAssertEqual(restored.updates, 10)
        XCTAssertEqual(restored.images, 20)
        XCTAssertEqual(restored.elapsedSeconds, 70)
        XCTAssertEqual(restored.restoredRunElapsedSeconds, 22)
        XCTAssertEqual(restored.rows.count, 1)
    }

    func testLegacyCheckpointRecoversImageVisitsFromItsSavedPosition() {
        let descriptor: [String: Any] = ["epochs_completed": 2, "updates": 7,
            "history": [["epoch": 1], ["epoch": 2]],
            "resume_cursor": ["epoch": 2, "imageCount": 2]]
        let restored = TrainingContinuationHistory(descriptor: descriptor, source: .checkpoint,
            runID: "old", resumingCursor: false, trainingImages: 6)
        XCTAssertEqual(restored.images, 14)
        XCTAssertEqual(restored.updates, 7)
        XCTAssertEqual(restored.epochs, 2)
        XCTAssertEqual(restored.elapsedSeconds, 0, "Missing timing must not be invented")
    }
}
