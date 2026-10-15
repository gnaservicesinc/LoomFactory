import Foundation
import LoomNative

extension LoomProject {
    /// Exports the exact checkpoint as a separate immutable deployment version.
    /// The source run, its final export and its continuation state stay intact.
    @discardableResult
    public func selectCheckpointModelVersion(runID: String, datasetID: String, map: MapRole,
                                            checkpointID: String? = nil,
                                            progress: @escaping @Sendable (TrainingEvent) -> Void = { _ in }) async throws -> ModelArtifact {
        let artifact = try await exportCheckpointModelVersion(runID: runID, datasetID: datasetID, map: map,
            checkpointID: checkpointID, progress: progress)
        try activateModelVersion(runID: artifact.id, datasetID: datasetID, map: map)
        return artifact
    }

    /// Creates a reviewable Core ML candidate without changing the user's
    /// selected production model. Overnight queues use this non-selecting path.
    @discardableResult
    public func exportCheckpointModelVersion(runID: String, datasetID: String, map: MapRole,
                                            checkpointID: String? = nil,
                                            progress: @escaping @Sendable (TrainingEvent) -> Void = { _ in }) async throws -> ModelArtifact {
        let operationLease = try TrainingQueueLease.requireAvailable(projectFolder: projectFolder)
        defer { withExtendedLifetime(operationLease) {} }
        guard !operationActive else { throw LoomError("Finish or stop the current project operation before choosing a checkpoint.") }
        let source = try ModelTestSelection.load(runID: runID, source: .checkpoint, datasetID: datasetID,
            map: map, projectFolder: projectFolder, projectID: document.id, checkpointID: checkpointID)
        guard let version = try modelVersions(datasetID: datasetID, map: map).first(where: { $0.id == runID }),
              version.checkpointVersions.contains(where: { checkpointID == nil || $0.revisionID == checkpointID }),
              version.settings != nil else {
            throw LoomError("Choose a saved checkpoint belonging to this source project.")
        }
        let cancellation = NativeCancellation(), folder = projectFolder, projectID = document.id
        operationActive = true; activeTraining = cancellation
        defer { operationActive = false; activeTraining = nil }
        let artifact = try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await Task.detached(priority: .userInitiated) {
                try CheckpointModelExport.run(source: source, version: version, projectFolder: folder,
                    projectID: projectID, cancellation: cancellation, progress: progress)
            }.value
        } onCancel: { cancellation.cancel() }
        return artifact
    }
}

enum CheckpointModelExport {
    static func run(source: ModelTestSelection, version: ModelVersion, projectFolder: URL, projectID: String,
                    cancellation: NativeCancellation, progress: @escaping @Sendable (TrainingEvent) -> Void) throws -> ModelArtifact {
        try cancellation.check()
        let original = try TrainingRunFiles.runDirectory(source.runID, projectFolder: projectFolder)
        let descriptor = try TrainingRunFiles.descriptor(at: original, source: .checkpoint, checkpointID: source.checkpointID)
        guard descriptor["weights_sha256"] as? String == source.weightsSHA256 else {
            throw LoomError("The selected checkpoint changed before export. Choose the saved checkpoint again.")
        }
        guard let settings = TrainingRunFiles.settings(from: descriptor["settings"] as? [String: Any] ?? [:], descriptor: descriptor),
              let checkpoint = version.checkpointVersions.first(where: { source.checkpointID == nil || $0.revisionID == source.checkpointID }),
              checkpoint.updatesCompleted > 0 || checkpoint.epochsCompleted > 0 else {
            throw LoomError("This checkpoint has no completed training updates to use as a model.")
        }
        let id = "checkpoint-model-" + UUID().uuidString.lowercased()
        let destination = projectFolder.appendingPathComponent("runs/" + id, isDirectory: true)
        let staging = projectFolder.appendingPathComponent("checkpoint-exports/" + id, isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }
        let weightsURL = staging.appendingPathComponent("model.safetensors")
        progress(.init(type: "status", message: "Preserving selected checkpoint weights…"))
        try FileManager.default.copyItem(at: source.weightsURL, to: weightsURL)
        guard try NativeFactoryTraining.sha(weightsURL, cancellation: cancellation) == source.weightsSHA256,
              try NativeFactoryTraining.sha(source.weightsURL, cancellation: cancellation) == source.weightsSHA256 else {
            throw LoomError("The selected checkpoint changed before export. Choose the saved checkpoint again.")
        }
        try cancellation.check()
        progress(.init(type: "status", message: "Exporting selected checkpoint as a model…"))
        let weights = try NativeWeights.read(from: weightsURL)
        let compiled = try NativeCoreML.export(weights: weights, architecture: source.architecture,
            imageSize: source.imageSize, to: staging.appendingPathComponent("model.mlmodel"))
        defer { try? FileManager.default.removeItem(at: compiled) }
        try cancellation.check()
        let lineage = TrainingLineage(parentRunID: source.runID, source: .checkpoint, weightsSHA256: source.weightsSHA256)
        let createdAt = Date()
        let checkpointLabel = source.checkpointID == nil ? "checkpoint" :
            "checkpoint epoch \(checkpoint.totalEpochsCompleted) · update \(checkpoint.totalUpdatesCompleted)"
        let label = (version.label.isEmpty ? version.datasetName + " · " + version.map.displayName : version.label) + " · " + checkpointLabel
        try TrainingRunFiles.writeMetadata(.init(projectID: projectID, runID: id, datasetID: source.datasetID,
            map: source.map, label: label, createdAt: createdAt, settings: settings, lineage: lineage,
            status: "checkpoint_export"), at: staging)
        var training = descriptor["settings"] as? [String: Any] ?? [:]
        for key in ["history", "prior_history", "prior_epochs", "prior_updates", "prior_images", "prior_duration_seconds",
                    "updates", "elapsed_seconds", "completed_images"] {
            if let value = descriptor[key] { training[key] = value }
        }
        training["requested_epochs"] = settings.epochs
        training["epochs"] = checkpoint.epochsCompleted
        training["total_epochs"] = checkpoint.totalEpochsCompleted
        training["export_epoch"] = checkpoint.epochsCompleted
        training["exported_from_checkpoint"] = true
        let history = descriptor["history"] as? [[String: Any]] ?? []
        let loss = history.last?["loss"] as? Double
        let lineageJSON: [String: Any] = ["parent_run_id": source.runID, "source": "checkpoint", "weights_sha256": source.weightsSHA256]
        let model: [String: Any] = ["schema": "loom-factory-model-v1", "status": "completed", "kind": "map",
            "project_id": projectID, "dataset_id": source.datasetID, "dataset_name": version.datasetName,
            "map": source.map.rawValue, "architecture": source.architecture.identifier,
            "architecture_config": source.architecture.descriptor, "image_size": source.imageSize,
            "input_name": "rgb", "output_name": "material_map", "input_type": "multiarray",
            "input_shape": [1, 3, source.imageSize, source.imageSize], "weights_path": "model.safetensors",
            "coreml_path": "model.mlmodel", "weights_sha256": source.weightsSHA256,
            "coreml_sha256": try NativeFactoryTraining.sha(staging.appendingPathComponent("model.mlmodel"), cancellation: cancellation),
            "parameter_count": settings.parameterCount(map: source.map), "training": training,
            "final_loss": loss as Any? ?? NSNull(), "lineage": lineageJSON,
            "run_id": id, "run_label": label, "created_at": ISO8601DateFormatter().string(from: createdAt),
            "checkpoint_status": descriptor["status"] ?? "saved",
            "checkpoint_updates": checkpoint.totalUpdatesCompleted,
            "checkpoint_id": source.checkpointID as Any? ?? NSNull(),
            "quality_note": "Exact checkpoint export for human review; export completion does not qualify prediction quality."]
        try NativeFactoryTraining.writeJSON(model, to: staging.appendingPathComponent("model.json"))
        // An export sees no new training images. Membership follows the immutable parent lineage.
        try NativeFactoryTraining.writeJSON(["schema": "loom-factory-checkpoint-export-v1", "project_id": projectID,
            "dataset_id": source.datasetID, "map_kind": source.map.rawValue, "samples": [], "lineage": lineageJSON],
            to: staging.appendingPathComponent("request.json"))
        let artifact = ModelArtifact(id: id, kind: "map", datasetID: source.datasetID, datasetName: version.datasetName,
            map: source.map, directory: destination, createdAt: createdAt, epochs: checkpoint.epochsCompleted,
            loss: loss, coreMLPath: "model.mlmodel", weightsPath: "model.safetensors", architecture: source.architecture.identifier)
        try JSONEncoder().encode(artifact).write(to: staging.appendingPathComponent("artifact.json"), options: .atomic)
        try cancellation.check()
        try FileManager.default.moveItem(at: staging, to: destination)
        progress(.init(type: "completed", message: "Checkpoint saved as a separate model version", progress: 1))
        return artifact
    }
}
