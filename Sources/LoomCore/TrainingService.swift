import Foundation
import LoomNative

extension LoomProject {
    public func trainMap(datasetID: String, map: MapRole, settings: TrainingSettings,
                         progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> ModelArtifact {
        guard !operationActive else { throw LoomError("An operation is already running in this source project. Stop it before starting another.") }
        guard MapRole.targets.contains(map) else { throw LoomError("Choose depth, roughness or normal training.") }
        if let issue = settings.validationIssue { throw LoomError(issue) }
        let current = try snapshot()
        guard let dataset = current.datasets.first(where: { $0.id == datasetID }),
              let plan = try current.trainingPlan(datasetID: datasetID, map: map, settings: settings) else {
            throw LoomError("This dataset no longer has images in the source folder. Choose a current dataset.")
        }
        guard !plan.training.isEmpty else {
            throw LoomError("\(dataset.name) has no matching diffuse/\(map.displayName.lowercased()) pairs at \(settings.imageSize) × \(settings.imageSize). Check the dataset and map types, or choose a crop size that fits.")
        }
        var resume: URL?
        if settings.continueLatestWeights == true {
            guard let checkpoint = try loadCheckpoints().first(where: { $0.datasetID == datasetID && $0.map == map }) else {
                throw LoomError("This dataset has no saved weights for this training type yet. Turn off Continue latest run to start a new model.")
            }
            guard settings.modelWidth == checkpoint.modelWidth, settings.modelBlocks == checkpoint.modelBlocks,
                  settings.imageSize == checkpoint.imageSize, settings.exaggerateFeatures == checkpoint.exaggerateFeatures else {
                throw LoomError("Continue with the saved model size, native crop dimensions and depth transform. Choose a new model to change them.")
            }
            resume = checkpoint.directory.appendingPathComponent("checkpoint.safetensors")
        }
        let id = UUID().uuidString.lowercased()
        let output = projectFolder.appendingPathComponent("runs/" + id, isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let cancellation = NativeCancellation()
        operationActive = true; activeTraining = cancellation
        defer { operationActive = false; activeTraining = nil }
        let root = sourceFolder, projectID = document.id, continued = resume
        // The actor remains available to Stop while the serial GPU worker runs.
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await Task.detached(priority: .userInitiated) {
                try NativeFactoryTraining.run(id: id, root: root, output: output, projectID: projectID,
                    dataset: dataset, map: map, plan: plan, settings: settings, resume: continued,
                    cancellation: cancellation, progress: progress)
            }.value
        } onCancel: { cancellation.cancel() }
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
