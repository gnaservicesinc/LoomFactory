import Foundation

public struct CheckpointCleanupSummary: Sendable, Equatable {
    public var runCount: Int = 0
    public var checkpointCount: Int = 0
    public var fileCount: Int = 0
    public var byteCount: Int64 = 0
    public var formattedSize: String { ByteCountFormatter.string(fromByteCount: byteCount, countStyle: .file) }
    public var message: String {
        "Removed \(checkpointCount) older checkpoints from \(runCount) approved models (\(formattedSize))."
    }
}

extension LoomProject {
    /// A read-only estimate. Execution always rebuilds the plan under the project lease.
    public func approvedCheckpointCleanupPreview() throws -> CheckpointCleanupSummary {
        CheckpointCleanup.summary(try approvedCheckpointCleanupPlans())
    }

    /// Permanently removes only obsolete checkpoint revisions of currently approved
    /// exports. Trash would keep consuming the same disk space.
    public func cleanupApprovedCheckpoints() throws -> CheckpointCleanupSummary {
        guard !operationActive, activeTraining == nil, activeProcess?.isRunning != true else {
            throw LoomError("Finish or save and stop the current operation before cleaning checkpoints.")
        }
        try CheckpointCleanup.validateRoot(projectFolder)
        let lease = try TrainingQueueLease.requireExclusive(projectFolder: projectFolder)
        defer { withExtendedLifetime(lease) {} }
        return try CheckpointCleanup.execute(approvedCheckpointCleanupPlans())
    }

    private func approvedCheckpointCleanupPlans() throws -> [CheckpointCleanup.Plan] {
        let management = try loadModelManagement()
        return try loadArtifacts().filter {
            $0.kind == "map" && modelApprovalState(for: $0, management: management).approved
        }.map { try CheckpointCleanup.plan(artifact: $0, projectFolder: projectFolder, projectID: document.id) }
    }
}

enum CheckpointCleanup {
    struct Plan {
        let directory: URL
        let checkpointCount: Int
        let files: [URL]
        let byteCount: Int64
        var historyData: Data? = nil
    }

    static func validateRoot(_ directory: URL) throws {
        let values = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true,
              directory.standardizedFileURL == directory.standardizedFileURL.resolvingSymlinksInPath() else {
            throw LoomError("The checkpoint folder was replaced by a link. Reopen the source folder.")
        }
    }

    static func plan(artifact: ModelArtifact, projectFolder: URL, projectID: String) throws -> Plan {
        try validateRoot(projectFolder)
        try validateRoot(projectFolder.appendingPathComponent("runs", isDirectory: true))
        let directory = try TrainingRunFiles.ownedRun(artifact.directory, projectFolder: projectFolder)
        let archives = try CheckpointArchive.descriptors(in: directory).filter { $0.lastPathComponent != "checkpoint.json" }
        guard !archives.isEmpty else { return Plan(directory: directory, checkpointCount: 0, files: [], byteCount: 0) }
        let latest = try TrainingRunFiles.descriptor(at: directory, source: .checkpoint)
        try validateIdentity(latest, artifact: artifact, projectID: projectID)
        // Hash every retained tensor, including optimizer and best-validation state,
        // before removing any recovery point. Never fall back to an older checkpoint.
        var protected = try referencedFiles(latest, directory: directory)
        let model = try TrainingRunFiles.descriptor(at: directory, source: .model)
        for key in ["weights_path", "coreml_path", "optimizer_path", "best_weights_path"] {
            if let path = model[key] as? String {
                let file = directory.appendingPathComponent(path).standardizedFileURL
                if key == "coreml_path", (try file.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true {
                    try validateRoot(file)
                    guard file.path.hasPrefix(directory.path + "/") else { throw LoomError("The native model is outside its run.") }
                    protected.insert(file)
                } else { protected.insert(try TrainingRunFiles.containedFile(path, directory: directory)) }
            }
        }
        let latestID = latest["checkpoint_id"] as? String
        var tensors: Set<URL> = [], descriptors: [URL] = []
        let historyURL = directory.appendingPathComponent("pruned-checkpoints.json")
        var history: [String: Any] = [:]
        var previousHistoryBytes = 0
        if FileManager.default.fileExists(atPath: historyURL.path) {
            let file = try TrainingRunFiles.containedFile(historyURL.lastPathComponent, directory: directory)
            let data = try Data(contentsOf: file)
            guard let existing = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw LoomError("The saved checkpoint history cannot be read safely.")
            }
            history = existing; previousHistoryBytes = data.count
        }
        for archive in archives {
            let file = try TrainingRunFiles.containedFile(archive.lastPathComponent, directory: directory)
            guard let descriptor = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any],
                  let id = descriptor["checkpoint_id"] as? String,
                  try CheckpointArchive.filename(id) == archive.lastPathComponent else {
                throw LoomError("A checkpoint revision has an invalid identity; no checkpoints were removed.")
            }
            try validateIdentity(descriptor, artifact: artifact, projectID: projectID)
            if id == latestID {
                guard NSDictionary(dictionary: descriptor).isEqual(to: latest) else {
                    throw LoomError("The latest checkpoint and its saved revision disagree; no checkpoints were removed.")
                }
                continue
            }
            guard (descriptor["updates"] as? Int ?? 0) <= (latest["updates"] as? Int ?? 0) else {
                throw LoomError("A checkpoint is newer than the latest checkpoint alias; no checkpoints were removed.")
            }
            descriptors.append(file)
            // Keep compact display metadata so existing A/B progressions reopen
            // after their inference tensors have been removed.
            var metadata = descriptor
            for key in ["history", "prior_history", "resume_cursor"] { metadata.removeValue(forKey: key) }
            history[id] = metadata
            // Already removed files are allowed so an interrupted cleanup can retry.
            // Only conventional checkpoint tensor names are eligible for deletion.
            for key in ["weights_path", "optimizer_path", "best_weights_path"] {
                guard let path = descriptor[key] as? String else { continue }
                guard isCheckpointTensor(path) else {
                    throw LoomError("A checkpoint references an unexpected tensor file; no checkpoints were removed.")
                }
                let candidate = directory.appendingPathComponent(path)
                if FileManager.default.fileExists(atPath: candidate.path) {
                    tensors.insert(try TrainingRunFiles.containedFile(path, directory: directory))
                } else if (try? candidate.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true {
                    throw LoomError("A checkpoint tensor was replaced by a link.")
                }
            }
        }
        // Tensor files first: on interruption the descriptor still identifies any
        // remaining files, allowing a subsequent cleanup to finish the same work.
        let files = tensors.subtracting(protected).sorted { $0.path < $1.path } + descriptors
        let bytes = try files.reduce(Int64(0)) { $0 + Int64(try $1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) }
        let historyData = descriptors.isEmpty ? nil : try JSONSerialization.data(withJSONObject: history, options: [.sortedKeys])
        let historyGrowth = (historyData?.count ?? previousHistoryBytes) - previousHistoryBytes
        return Plan(directory: directory, checkpointCount: descriptors.count, files: files,
            byteCount: max(0, bytes - Int64(historyGrowth)), historyData: historyData)
    }

    private static func validateIdentity(_ descriptor: [String: Any], artifact: ModelArtifact, projectID: String) throws {
        guard descriptor["schema"] as? String == "loom-factory-checkpoint-v1",
              descriptor["project_id"] as? String == projectID, descriptor["kind"] as? String == "map",
              descriptor["dataset_id"] as? String == artifact.datasetID,
              descriptor["map"] as? String == artifact.map?.rawValue,
              (descriptor["run_id"] as? String).map({ $0 == artifact.id }) ?? true else {
            throw LoomError("A checkpoint does not belong to this approved model; no checkpoints were removed.")
        }
    }

    private static func referencedFiles(_ descriptor: [String: Any], directory: URL) throws -> Set<URL> {
        guard descriptor["weights_path"] as? String != nil else { throw LoomError("The latest checkpoint has no saved weights.") }
        var result: Set<URL> = []
        for prefix in ["weights", "optimizer", "best_weights"] {
            guard let path = descriptor[prefix + "_path"] as? String else {
                guard descriptor[prefix + "_sha256"] == nil else { throw LoomError("The latest checkpoint is missing a tensor path.") }
                continue
            }
            let file = try TrainingRunFiles.containedFile(path, directory: directory)
            guard let hash = descriptor[prefix + "_sha256"] as? String, hash == (try TrainingRunFiles.sha256(file)) else {
                throw LoomError("The latest checkpoint's \(prefix) checksum changed; no checkpoints were removed.")
            }
            result.insert(file)
        }
        return result
    }

    private static func isCheckpointTensor(_ path: String) -> Bool {
        guard !path.contains("/"), !path.contains("\\"), path.hasSuffix(".safetensors") else { return false }
        return ["checkpoint.safetensors", "optimizer.safetensors", "best.safetensors"].contains(path)
            || ["checkpoint-", "optimizer-", "best-"].contains { path.hasPrefix($0) }
    }

    static func summary(_ plans: [Plan]) -> CheckpointCleanupSummary {
        plans.reduce(into: CheckpointCleanupSummary()) { result, plan in
            if !plan.files.isEmpty { result.runCount += 1 }
            result.checkpointCount += plan.checkpointCount
            result.fileCount += plan.files.count
            result.byteCount += plan.byteCount
        }
    }

    static func execute(_ plans: [Plan]) throws -> CheckpointCleanupSummary {
        // All runs have been validated before the first deletion. Recheck file
        // containment immediately before unlinking; never recurse into directories.
        var removed = 0
        do {
            for plan in plans {
                if let history = plan.historyData {
                    try validateRoot(plan.directory.deletingLastPathComponent().deletingLastPathComponent())
                    try validateRoot(plan.directory.deletingLastPathComponent())
                    try validateRoot(plan.directory)
                    let file = plan.directory.appendingPathComponent("pruned-checkpoints.json")
                    if FileManager.default.fileExists(atPath: file.path) {
                        _ = try TrainingRunFiles.containedFile(file.lastPathComponent, directory: plan.directory)
                    }
                    try history.write(to: file, options: .atomic)
                }
                for file in plan.files {
                    try validateRoot(plan.directory.deletingLastPathComponent().deletingLastPathComponent())
                    try validateRoot(plan.directory.deletingLastPathComponent())
                    try validateRoot(plan.directory)
                    _ = try TrainingRunFiles.containedFile(file.lastPathComponent, directory: plan.directory)
                    try FileManager.default.removeItem(at: file)
                    removed += 1
                }
            }
        } catch {
            throw LoomError("Checkpoint cleanup stopped after removing \(removed) files: \(error.localizedDescription). The retained checkpoint and exported model were preserved; run cleanup again to finish.")
        }
        return summary(plans)
    }
}
