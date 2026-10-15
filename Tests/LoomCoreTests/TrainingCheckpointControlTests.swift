import Foundation
import LoomNative
import XCTest
@testable import LoomCore

final class TrainingCheckpointControlTests: XCTestCase {
    func testRequestsQueuedWhileSavingAreNotAcknowledgedByTheEarlierSave() async throws {
        let control = TrainingRunControl()
        let first = Task { try await control.request(stop: false) }
        try await waitForPending(control)
        XCTAssertEqual(control.beginRequest(), false)
        let second = Task { try await control.request(stop: true) }
        // The pending Save & Stop must not borrow a Save Checkpoint receipt.
        for _ in 0..<20 { await Task.yield() }
        let directory = URL(fileURLWithPath: "/tmp/checkpoint-control")
        control.complete(.success(.init(runDirectory: directory, completedUpdates: 7, stopped: false)))
        let firstReceipt = try await first.value
        XCTAssertFalse(firstReceipt.stopped)
        try await waitForPending(control)
        XCTAssertEqual(control.beginRequest(), true)
        control.complete(.success(.init(runDirectory: directory, completedUpdates: 8, stopped: true)))
        let secondReceipt = try await second.value
        XCTAssertTrue(secondReceipt.stopped)
        control.finish()
    }

    func testSaveFailureCompletesTheRequestAndAllowsRetry() async throws {
        let control = TrainingRunControl()
        let request = Task { try await control.request(stop: true) }
        try await waitForPending(control)
        XCTAssertEqual(control.beginRequest(), true)
        control.complete(.failure(LoomError("Disk full")))
        do { _ = try await request.value; XCTFail("A failed save must not report success") }
        catch { XCTAssertEqual(error.localizedDescription, "Disk full") }
        let retry = Task { try await control.request(stop: true) }
        try await waitForPending(control)
        XCTAssertEqual(control.beginRequest(), true)
        control.complete(.success(.init(runDirectory: URL(fileURLWithPath: "/tmp/retry"), completedUpdates: 9, stopped: true)))
        let retryReceipt = try await retry.value
        XCTAssertTrue(retryReceipt.stopped)
    }

    func testFinishedRunRejectsPendingAndFutureRequests() async throws {
        let control = TrainingRunControl()
        let pending = Task { try await control.request(stop: false) }
        try await waitForPending(control)
        control.finish()
        do { _ = try await pending.value; XCTFail("Pending request must finish") } catch {}
        do { _ = try await control.request(stop: true); XCTFail("Finished run cannot save") } catch {}
    }

    func testCheckpointPublicationPreservesLastGoodOnFailureAndEverySuccessfulRevision() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("checkpoint-publication-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = ["test.weight": NativeTensor(shape: [2], values: [1, 2])]
        let optimizer = ["first.test.weight": NativeTensor(shape: [2], values: [0.1, 0.2])]
        try TrainingCheckpointFiles.write(weights: weights, optimizer: optimizer, bestWeights: weights,
            descriptor: ["updates": 1], directory: directory)
        let file = directory.appendingPathComponent("checkpoint.json")
        let before = try Data(contentsOf: file)
        let old = try NativeFactoryTraining.json(file)
        // Tensor writes succeed, but descriptor serialization fails afterwards.
        XCTAssertThrowsError(try TrainingCheckpointFiles.write(weights: weights, optimizer: optimizer,
            descriptor: ["invalid": Double.nan], directory: directory))
        XCTAssertEqual(try Data(contentsOf: file), before)
        let oldWeights = directory.appendingPathComponent(try XCTUnwrap(old["weights_path"] as? String))
        XCTAssertEqual(try NativeWeights.read(from: oldWeights), weights)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil).count, 5)
        try TrainingCheckpointFiles.write(weights: weights, optimizer: optimizer,
            descriptor: ["updates": 2], directory: directory)
        let saved = try NativeFactoryTraining.json(file)
        XCTAssertEqual(saved["updates"] as? Int, 2)
        XCTAssertTrue(FileManager.default.fileExists(atPath: oldWeights.path))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil).count, 8)
        let archives = try CheckpointArchive.descriptors(in: directory).filter { $0.lastPathComponent != "checkpoint.json" }
        XCTAssertEqual(archives.count, 2)
        XCTAssertEqual(try archives.map { try NativeFactoryTraining.json($0)["updates"] as? Int }.compactMap { $0 }.sorted(), [1, 2])
        let currentWeights = directory.appendingPathComponent(try XCTUnwrap(saved["weights_path"] as? String))
        XCTAssertEqual(saved["weights_sha256"] as? String, try TrainingRunFiles.sha256(currentWeights))
    }

    func testCompatibilityRetainsCheckpointScheduleFreedomButDetectsTrainingChanges() {
        let saved = ["epochs": 500, "checkpoint_every_epochs": 100, "batch_size": 2, "seed": 3]
        XCTAssertTrue(TrainingCheckpointCursor.compatibleSettings(saved,
            current: ["epochs": 500, "checkpoint_every_epochs": 1, "batch_size": 2, "seed": 3]))
        XCTAssertFalse(TrainingCheckpointCursor.compatibleSettings(saved,
            current: ["epochs": 600, "checkpoint_every_epochs": 1, "batch_size": 2, "seed": 3]))
        XCTAssertFalse(TrainingCheckpointCursor.compatibleSettings(saved,
            current: ["epochs": 500, "checkpoint_every_epochs": 100, "batch_size": 1, "seed": 3]))
        var withExecutionSettings = saved
        withExecutionSettings["requested_seed"] = -1
        withExecutionSettings["cache_images_in_memory"] = 1
        withExecutionSettings["image_cache_limit_mb"] = 4096
        XCTAssertTrue(TrainingCheckpointCursor.compatibleSettings(saved, current: withExecutionSettings))
    }

    func testLegacyCheckpointLossDefaultsPreventMixingObjectiveHistories() {
        let saved: [String: Any] = ["optimizer": "adamw", "epochs": 500]
        let legacy: [String: Any] = ["optimizer": "adamw", "epochs": 500, "scalar_loss": "absolute-v1"]
        var stable = legacy; stable["scalar_loss"] = "logit-cross-entropy-v2"
        XCTAssertTrue(TrainingCheckpointCursor.compatibleSettings(saved, current: legacy))
        XCTAssertFalse(TrainingCheckpointCursor.compatibleSettings(saved, current: stable))
        XCTAssertFalse(TrainingCheckpointCursor.compatibleSettings(saved, current: stable, keys: ["optimizer", "scalar_loss"]))
        var detail = stable; detail["scalar_loss"] = "native-detail-v3"
        XCTAssertFalse(TrainingCheckpointCursor.compatibleSettings(stable, current: detail))
        XCTAssertFalse(TrainingCheckpointCursor.compatibleSettings(stable, current: detail, keys: ["optimizer", "scalar_loss"]))
        XCTAssertTrue(TrainingCheckpointCursor.compatibleSettings(detail, current: detail))
    }

    private func waitForPending(_ control: TrainingRunControl) async throws {
        for _ in 0..<1_000 {
            if control.hasPendingRequest { return }
            try await Task.sleep(for: .milliseconds(1))
        }
        XCTFail("Checkpoint request was not queued")
        throw LoomError("Checkpoint request timed out")
    }
}
