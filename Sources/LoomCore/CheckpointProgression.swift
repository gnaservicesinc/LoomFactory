import Foundation
import LoomNative

public struct CheckpointTestOutput: Identifiable, Sendable {
    public var id: String { checkpoint.id }
    public let checkpoint: TrainingCheckpoint
    public let result: ModelTestResult?
    public let issue: String?
}

public struct CheckpointProgression: Sendable {
    public let directory: URL
    public let outputs: [CheckpointTestOutput]
    public let checkpointCount: Int
    public let cancelled: Bool
    public var results: [ModelTestResult] { outputs.compactMap(\.result) }
}

extension LoomProject {
    /// Holds the project GPU lease for the entire sweep. Only one network and
    /// one image's prediction are resident at a time; original runs are read-only.
    public func testAllCheckpoints(datasetID: String, map: MapRole, runID: String, inputURL: URL,
                                   boundaryHandling: ModelTestBoundaryHandling = .context,
                                   progress: @escaping @Sendable (TrainingEvent) -> Void,
                                   didSave: @escaping @Sendable (CheckpointProgression) -> Void = { _ in }) async throws -> CheckpointProgression {
        let lease = try TrainingQueueLease.requireAvailable(projectFolder: projectFolder)
        defer { withExtendedLifetime(lease) {} }
        guard !operationActive else { throw LoomError("Stop the current operation before testing checkpoints.") }
        let directory = try TrainingRunFiles.runDirectory(runID, projectFolder: projectFolder)
        let checkpoints = try loadCheckpointVersions(directory: directory)
        guard !checkpoints.isEmpty else { throw LoomError("This run has no saved checkpoints to test.") }
        let output = projectFolder.appendingPathComponent("model-tests/progression-" + UUID().uuidString.lowercased(), isDirectory: true)
        let projectID = document.id, projectFolder = self.projectFolder
        let cancellation = NativeCancellation()
        operationActive = true; activeTraining = cancellation
        defer { operationActive = false; activeTraining = nil }
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await Task.detached(priority: .userInitiated) {
                try CheckpointProgressionWorker.run(checkpoints: checkpoints, inputURL: inputURL, output: output,
                    cancellation: cancellation, boundaryHandling: boundaryHandling, progress: progress, didSave: didSave) { checkpoint in
                    try ModelTestSelection.load(runID: runID, source: .checkpoint, datasetID: datasetID, map: map,
                        projectFolder: projectFolder, projectID: projectID, checkpointID: checkpoint.revisionID)
                }
            }.value
        } onCancel: { cancellation.cancel() }
    }
}

enum CheckpointProgressionWorker {
    typealias Runner = (ModelTestSelection, URL, URL, NativeCancellation, ModelTestBoundaryHandling,
                       @escaping @Sendable (TrainingEvent) -> Void) throws -> ModelTestResult

    static func run(checkpoints: [TrainingCheckpoint], inputURL: URL, output: URL,
                    cancellation: NativeCancellation, boundaryHandling: ModelTestBoundaryHandling,
                    progress: @escaping @Sendable (TrainingEvent) -> Void,
                    didSave: @escaping @Sendable (CheckpointProgression) -> Void = { _ in },
                    load: (TrainingCheckpoint) throws -> ModelTestSelection,
                    runner: Runner = { model, input, output, cancellation, boundary, progress in
                        try ModelTestWorker.run(model: model, inputURL: input, output: output,
                            cancellation: cancellation, progress: progress, boundaryHandling: boundary)
                    }) throws -> CheckpointProgression {
        let ordered = checkpoints.sorted {
            if $0.totalUpdatesCompleted != $1.totalUpdatesCompleted { return $0.totalUpdatesCompleted < $1.totalUpdatesCompleted }
            if $0.totalEpochsCompleted != $1.totalEpochsCompleted { return $0.totalEpochsCompleted < $1.totalEpochsCompleted }
            if $0.modifiedAt != $1.modifiedAt { return $0.modifiedAt < $1.modifiedAt }
            return $0.id < $1.id
        }
        try cancellation.check()
        let input = inputURL.standardizedFileURL.resolvingSymlinksInPath()
        let hash = try NativeFactoryTraining.sha(input, cancellation: cancellation)
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        var outputs: [CheckpointTestOutput] = []
        var cancelled = false
        func snapshot() -> CheckpointProgression {
            .init(directory: output, outputs: outputs, checkpointCount: ordered.count, cancelled: cancelled)
        }
        func publish(status: String) throws {
            let entries: [[String: Any]] = outputs.map { entry in
                var value: [String: Any] = ["checkpoint_id": entry.checkpoint.revisionID as Any? ?? NSNull(),
                    "epochs_completed": entry.checkpoint.totalEpochsCompleted,
                    "updates_completed": entry.checkpoint.totalUpdatesCompleted,
                    "saved_at": ISO8601DateFormatter().string(from: entry.checkpoint.modifiedAt),
                    "error": entry.issue as Any? ?? NSNull()]
                if let result = entry.result {
                    let folder = result.directory.lastPathComponent + "/"
                    value["output_path"] = folder + result.outputURL.lastPathComponent
                    value["preview_path"] = folder + result.previewURL.lastPathComponent
                    value["test_record_path"] = folder + "test.json"
                }
                return value
            }
            try NativeFactoryTraining.writeJSON(["schema": "loom-factory-checkpoint-progression-v1",
                "run_id": ordered.first?.runID ?? "", "input_path": input.path, "input_sha256": hash,
                "status": status, "checkpoint_count": ordered.count, "completed_count": outputs.compactMap(\.result).count,
                "boundary_handling": boundaryHandling.rawValue, "outputs": entries,
                "display": "Fixed [0,1] PNG previews; canonical Float32 EXR outputs retain raw predictions. No per-output normalization."],
                to: output.appendingPathComponent("progression.json"))
            didSave(snapshot())
        }
        try publish(status: "running")
        for (index, checkpoint) in ordered.enumerated() {
            do {
                try cancellation.check()
                guard hash == (try NativeFactoryTraining.sha(input, cancellation: cancellation)) else {
                    throw LoomError("The input image changed during the checkpoint sweep. Choose the completed image and run again.")
                }
                let destination = output.appendingPathComponent(String(format: "%03d-epoch-%06d-update-%09d", index + 1,
                    checkpoint.totalEpochsCompleted, checkpoint.totalUpdatesCompleted), isDirectory: true)
                let model = try load(checkpoint)
                let result = try autoreleasepool {
                    try runner(model, input, destination, cancellation, boundaryHandling) { event in
                        progress(.init(type: "status", message: "Checkpoint \(index + 1) of \(ordered.count) · epoch \(checkpoint.totalEpochsCompleted) · \(event.message)",
                            progress: (Double(index) + (event.progress ?? 0)) / Double(ordered.count)))
                    }
                }
                guard result.inputSHA256 == hash else { throw LoomError("The input image changed during the checkpoint sweep.") }
                outputs.append(.init(checkpoint: checkpoint, result: result, issue: nil))
            } catch {
                if cancellation.isCancelled { cancelled = true; break }
                outputs.append(.init(checkpoint: checkpoint, result: nil, issue: error.localizedDescription))
                // Never compare outputs from different versions of the input.
                if (try? NativeFactoryTraining.sha(input, cancellation: cancellation)) != hash {
                    try publish(status: "input-changed")
                    return snapshot()
                }
            }
            try publish(status: "running")
        }
        try publish(status: cancelled ? "cancelled" : outputs.contains(where: { $0.issue != nil }) ? "completed-with-errors" : "completed")
        progress(.init(type: "completed", message: "\(cancelled ? "Stopped; kept" : "Saved") \(outputs.compactMap(\.result).count) of \(ordered.count) checkpoint outputs", progress: 1))
        return snapshot()
    }
}
