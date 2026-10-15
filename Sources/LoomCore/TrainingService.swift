import Foundation
import LoomNative

extension LoomProject {
    public func saveTrainingCheckpoint(stopAfterSaving: Bool = false) async throws -> TrainingCheckpointReceipt {
        guard let control = activeTrainingControl else {
            throw LoomError("No model training run is active.")
        }
        return try await control.request(stop: stopAfterSaving)
    }

    public func trainMap(datasetID: String, map: MapRole, settings: TrainingSettings,
                         start: TrainingStart? = nil, runLabel: String = "",
                         scalarLossOverride: NativeScalarLoss? = nil,
                         progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> ModelArtifact {
        guard !operationActive else { throw LoomError("An operation is already running in this source project. Stop it before starting another.") }
        guard MapRole.targets.contains(map) else { throw LoomError("Choose depth, roughness or normal training.") }
        var requestedStart = start ?? .fresh
        if start == nil, settings.continueLatestWeights == true {
            guard let checkpoint = try loadCheckpoints().first(where: { $0.datasetID == datasetID && $0.map == map }) else {
                throw LoomError("This dataset has no saved weights for this training type yet. Choose a fresh model to start a new run.")
            }
            requestedStart = .checkpoint(runID: checkpoint.directory.lastPathComponent)
        }
        let prepared = try prepareTrainingStart(requestedStart, datasetID: datasetID, map: map,
            settings: settings, scalarLossOverride: scalarLossOverride)
        var runSettings = prepared.settings
        if let issue = runSettings.validationIssue { throw LoomError(issue) }
        runSettings.requestedSeed = runSettings.seed
        runSettings.resolveRandomSeedForRun()
        let settings = runSettings
        let current = try snapshot()
        guard let dataset = current.datasets.first(where: { $0.id == datasetID }),
              let plan = try current.trainingPlan(datasetID: datasetID, map: map, settings: settings) else {
            throw LoomError("This dataset no longer has images in the source folder. Choose a current dataset.")
        }
        guard !plan.training.isEmpty else {
            throw LoomError("\(dataset.name) has no matching diffuse/\(map.displayName.lowercased()) pairs at \(settings.imageSize) × \(settings.imageSize). Check the dataset and map types, or choose a crop size that fits.")
        }
        let id = UUID().uuidString.lowercased()
        let output = projectFolder.appendingPathComponent("runs/" + id, isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var metadata = TrainingRunMetadata(projectID: document.id, runID: id, datasetID: datasetID, map: map,
            label: runLabel.trimmingCharacters(in: .whitespacesAndNewlines), createdAt: .now, settings: settings,
            lineage: prepared.resume.map { TrainingLineage(parentRunID: $0.runID, source: $0.source, weightsSHA256: $0.sha256) },
            status: "running")
        try TrainingRunFiles.writeMetadata(metadata, at: output)
        let cancellation = NativeCancellation()
        let control = TrainingRunControl(runID: id)
        operationActive = true; activeTraining = cancellation; activeTrainingControl = control
        defer { control.finish(); operationActive = false; activeTraining = nil; activeTrainingControl = nil }
        let root = sourceFolder, projectID = document.id, continued = prepared.resume?.weights
        // The actor remains available to Stop while the serial GPU worker runs.
        do {
            let artifact = try await withTaskCancellationHandler {
                try Task.checkCancellation()
                return try await Task.detached(priority: .userInitiated) {
                    try NativeFactoryTraining.run(id: id, root: root, output: output, projectID: projectID,
                        dataset: dataset, map: map, plan: plan, settings: settings, resume: continued,
                        cancellation: cancellation, progress: progress, control: control)
                }.value
            } onCancel: { cancellation.cancel() }
            metadata.status = "completed"
            do { try TrainingRunFiles.writeMetadata(metadata, at: output) }
            catch { progress(.init(type: "warning", message: "Model saved; its run label could not be refreshed: \(error.localizedDescription)")) }
            // A finished run is a candidate. Only a user's selection changes an
            // explicit assignment; legacy unassigned groups still use the latest.
            return artifact
        } catch {
            metadata.status = error is TrainingSavedStop ? "saved_stopped" : error is CancellationError ? "cancelled" : "failed"
            try? TrainingRunFiles.writeMetadata(metadata, at: output)
            throw error
        }
    }

    // Package assembly uses native system tools only. Training never starts a child process.
    func execute(executable: URL, arguments: [String], environment: [String: String], log: URL,
                 progress: @escaping @Sendable (TrainingEvent) -> Void) async throws {
        try Task.checkCancellation()
        let process = Process(); process.executableURL = executable; process.arguments = arguments
        process.environment = ProcessInfo.processInfo.environment.merging(environment) { _, new in new }
        process.currentDirectoryURL = sourceFolder
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log)
        process.standardOutput = handle; process.standardError = handle
        activeProcess = process
        defer { activeProcess = nil; try? handle.close() }
        try process.run()
        let status = await withTaskCancellationHandler {
            await Task.detached { process.waitUntilExit(); return process.terminationStatus }.value
        } onCancel: { if process.isRunning { process.terminate() } }
        try Task.checkCancellation()
        guard status == 0 else { throw LoomError("The package operation could not finish. Run details: \(log.path)") }
    }
}
