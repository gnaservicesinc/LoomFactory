import Foundation

extension LoomProject {
    public func trainMap(datasetID: String, map: MapRole, settings: TrainingSettings,
                         progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> ModelArtifact {
        guard MapRole.targets.contains(map) else { throw LoomError("Choose depth, roughness or normal training.") }
        let current = try snapshot()
        guard let dataset = current.datasets.first(where: { $0.id == datasetID }) else {
            throw LoomError("This dataset no longer has images in the source folder. Choose a current dataset.")
        }
        let pairs = try TrainingPairSelector.select(images: current.images, dataset: dataset.name, map: map, size: settings.imageSize)
        let samples: [[String: Any]] = pairs.map { pair in
            var item: [String: Any] = ["id": pair.id, "input": pair.input.url.path, "target": pair.target.url.path]
            if map == .normal { item["normal_convention"] = pair.target.normalConvention == "directx" ? "DirectX" : "OpenGL" }
            return item
        }
        guard !samples.isEmpty else { throw LoomError("\(dataset.name) has no reviewed diffuse/\(map.displayName.lowercased()) pairs at \(settings.imageSize) × \(settings.imageSize). Review the images or choose a crop size that fits.") }
        return try await runTraining(dataset: dataset, map: map, samples: samples, settings: settings, progress: progress)
    }

    func runTraining(dataset: DatasetCategory, map: MapRole, samples: [[String: Any]],
                     settings: TrainingSettings, progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> ModelArtifact {
        guard !operationActive else { throw LoomError("An operation is already running in this source project. Stop it before starting another.") }
        operationActive = true
        defer { operationActive = false }
        guard (32...8192).contains(settings.imageSize), (1...100_000).contains(settings.epochs),
              (1...64).contains(settings.batchSize), settings.learningRate.isFinite, settings.learningRate > 0 else {
            throw LoomError("Choose a positive epoch count, learning rate and supported image size.")
        }
        let id = UUID().uuidString.lowercased()
        let output = projectFolder.appendingPathComponent("runs/" + id, isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var request: [String: Any] = ["schema": "loom-factory-training-v1", "project_id": document.id, "require_image_metadata": true,
            "source_root": sourceFolder.path, "output_directory": output.path, "samples": samples,
            "image_size": settings.imageSize, "epochs": settings.epochs, "batch_size": settings.batchSize,
            "learning_rate": settings.learningRate, "shuffle_each_epoch": settings.shuffleEachEpoch,
            "flip_crops": settings.flipCrops, "rotate_crops": settings.rotateCrops,
            "exaggerate_features": map == .depth && settings.exaggerateFeatures,
            "seed": settings.seed, "width": settings.modelWidth, "device": "auto"]
        request["dataset_id"] = dataset.id; request["display_name"] = dataset.name
        request["map_kind"] = map.rawValue
        if settings.continueLatestWeights == true {
            let runs = projectFolder.appendingPathComponent("runs", isDirectory: true)
            let previous = try FileManager.default.contentsOfDirectory(at: runs, includingPropertiesForKeys: [.contentModificationDateKey])
                .filter { $0 != output }
                .sorted { ((try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
                    > ((try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast) }
            for directory in previous {
                let descriptor = directory.appendingPathComponent("checkpoint.json")
                guard let bytes = try? Data(contentsOf: descriptor),
                      let checkpoint = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
                      checkpoint["project_id"] as? String == document.id,
                      checkpoint["dataset_id"] as? String == dataset.id,
                      checkpoint["map"] as? String == map.rawValue,
                      checkpoint["kind"] as? String == "map",
                      FileManager.default.fileExists(atPath: directory.appendingPathComponent("checkpoint.safetensors").path) else { continue }
                request["resume_weights"] = directory.appendingPathComponent("checkpoint.safetensors").path
                break
            }
            guard request["resume_weights"] != nil else { throw LoomError("This dataset has no saved weights for this training type yet. Turn off Continue latest run to start a new model.") }
        }
        let file = output.appendingPathComponent("request.json")
        try JSONSerialization.data(withJSONObject: request, options: [.prettyPrinted, .sortedKeys]).write(to: file, options: .atomic)
        let runtime = try runtimeDirectory()
        let python = projectFolder.appendingPathComponent("runtime/bin/python3")
        if !FileManager.default.isExecutableFile(atPath: python.path) {
            progress(.init(type: "setup", message: "Preparing the training runtime in this source folder…"))
            let setup = runtime.appendingPathComponent("setup_runtime.sh")
            guard FileManager.default.fileExists(atPath: setup.path) else { throw LoomError("The Loom Factory training runtime is missing. Reinstall Loom Factory.") }
            try await execute(executable: URL(fileURLWithPath: "/bin/bash"), arguments: [setup.path, sourceFolder.path],
                              environment: [:], log: output.appendingPathComponent("setup.log"), progress: progress)
        }
        try Task.checkCancellation()
        progress(.init(type: "start", message: "Training \(dataset.name) from reviewed image metadata"))
        try await execute(executable: python, arguments: ["-m", "loom_factory", "train-map", "--request", file.path],
                          environment: ["PYTHONPATH": runtime.path], log: output.appendingPathComponent("training.log"), progress: progress)
        let descriptor = output.appendingPathComponent("model.json")
        guard let result = try JSONSerialization.jsonObject(with: Data(contentsOf: descriptor)) as? [String: Any] else {
            throw LoomError("Training finished without a model descriptor. The run log is in the source project.")
        }
        let coreML = result["coreml_path"] as? String ?? "Model.mlpackage"
        let weights = result["weights_path"] as? String ?? "model.safetensors"
        guard FileManager.default.fileExists(atPath: output.appendingPathComponent(coreML).path),
              FileManager.default.fileExists(atPath: output.appendingPathComponent(weights).path) else {
            throw LoomError("The native model export did not finish. The original maps and run weights are preserved.")
        }
        let artifact = ModelArtifact(id: id, kind: "map", datasetID: dataset.id, datasetName: dataset.name,
            map: map, directory: output, epochs: settings.epochs,
            loss: (result["final_loss"] as? NSNumber)?.doubleValue ?? ((result["training"] as? [String: Any])?["history"] as? [[String: Any]])?.last?["loss"] as? Double,
            coreMLPath: coreML, weightsPath: weights, architecture: (result["architecture"] as? [String: Any])?["id"] as? String ?? result["architecture"] as? String ?? "loom-native-v1")
        try JSONEncoder().encode(artifact).write(to: output.appendingPathComponent("artifact.json"), options: .atomic)
        progress(.init(type: "completed", message: "Model saved in this source project", progress: 1))
        return artifact
    }

    func runtimeDirectory() throws -> URL {
        if let override = ProcessInfo.processInfo.environment["LOOM_FACTORY_RUNTIME"] {
            let directory = URL(fileURLWithPath: override, isDirectory: true)
            if FileManager.default.fileExists(atPath: directory.appendingPathComponent("loom_factory/__main__.py").path) { return directory }
        }
        if let directory = Bundle.main.resourceURL?.appendingPathComponent("Runtime", isDirectory: true),
           FileManager.default.fileExists(atPath: directory.appendingPathComponent("loom_factory/__main__.py").path) { return directory }
        let development = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Runtime")
        guard FileManager.default.fileExists(atPath: development.appendingPathComponent("loom_factory/__main__.py").path) else {
            throw LoomError("The Loom Factory runtime is not installed.")
        }
        return development
    }

    func execute(executable: URL, arguments: [String], environment: [String: String], log: URL,
                 progress: @escaping @Sendable (TrainingEvent) -> Void) async throws {
        let process = Process(); process.executableURL = executable; process.arguments = arguments
        var variables = ProcessInfo.processInfo.environment; variables.merge(environment) { _, new in new }
        variables["PYTHONUNBUFFERED"] = "1"; variables["PYTHONDONTWRITEBYTECODE"] = "1"
        process.environment = variables; process.currentDirectoryURL = sourceFolder
        let pipe = Pipe(); process.standardOutput = pipe; process.standardError = pipe
        activeProcess = process
        defer { activeProcess = nil }
        try process.run()
        let handle = try FileHandle(forWritingTo: createLog(log))
        defer { try? handle.close() }
        let status: Int32 = try await Task.detached {
            var pending = Data()
            while true {
                let chunk = pipe.fileHandleForReading.availableData
                if chunk.isEmpty { break }
                try handle.write(contentsOf: chunk); pending.append(chunk)
                while let newline = pending.firstIndex(of: 10) {
                    let line = Data(pending[..<newline]); pending.removeSubrange(...newline)
                    if let json = try? JSONSerialization.jsonObject(with: line) as? [String: Any] {
                        progress(TrainingEvent(type: json["type"] as? String ?? json["event"] as? String ?? "progress",
                            message: json["message"] as? String ?? "Training…", progress: (json["progress"] as? NSNumber)?.doubleValue,
                            epoch: json["epoch"] as? Int, loss: (json["loss"] as? NSNumber)?.doubleValue))
                    }
                }
            }
            process.waitUntilExit()
            return process.terminationStatus
        }.value
        try Task.checkCancellation()
        guard status == 0 else {
            let contents = (try? String(contentsOf: log, encoding: .utf8)) ?? ""
            let reason = contents.split(separator: "\n").last.map(String.init) ?? "The operation was stopped."
            throw LoomError("\(reason)\nRun details: \(log.path)")
        }
    }
}

private func createLog(_ url: URL) throws -> URL {
    guard FileManager.default.createFile(atPath: url.path, contents: nil) else { throw LoomError("Could not create the project run log.") }
    return url
}
