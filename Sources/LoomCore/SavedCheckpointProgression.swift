import Foundation

extension LoomProject {
    /// Reopens the most recent sweep for these exact input bytes and this run.
    /// All returned paths remain inside the saved progression directory.
    public func latestCheckpointProgression(runID: String, inputURL: URL) throws -> CheckpointProgression? {
        let run = try TrainingRunFiles.runDirectory(runID, projectFolder: projectFolder)
        let checkpoints = try loadCheckpointVersions(directory: run)
        let input = inputURL.standardizedFileURL.resolvingSymlinksInPath()
        let hash = try TrainingRunFiles.sha256(input)
        let root = projectFolder.appendingPathComponent("model-tests", isDirectory: true)
        guard FileManager.default.fileExists(atPath: root.path) else { return nil }
        let folders = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.contentModificationDateKey])
            .filter { $0.lastPathComponent.hasPrefix("progression-") }
            .sorted { ((try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
                > ((try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast) }
        for folder in folders {
            guard let file = try? TrainingRunFiles.containedFile("progression.json", directory: folder),
                  let manifest = TrainingRunFiles.optionalJSON(file),
                  manifest["schema"] as? String == "loom-factory-checkpoint-progression-v1",
                  manifest["run_id"] as? String == runID, manifest["input_sha256"] as? String == hash,
                  manifest["input_path"] as? String == input.path,
                  let count = manifest["checkpoint_count"] as? Int,
                  let rows = manifest["outputs"] as? [[String: Any]] else { continue }
            let outputs = rows.compactMap { row -> CheckpointTestOutput? in
                let id = row["checkpoint_id"] as? String
                guard let checkpoint = checkpoints.first(where: { $0.revisionID == id }) else { return nil }
                do {
                    if let issue = row["error"] as? String { return .init(checkpoint: checkpoint, result: nil, issue: issue) }
                    let recordURL = try TrainingRunFiles.containedFile(row["test_record_path"] as? String ?? "", directory: folder)
                    guard let record = TrainingRunFiles.optionalJSON(recordURL),
                          record["schema"] as? String == "loom-factory-model-test-v1", record["run_id"] as? String == runID,
                          record["input_sha256"] as? String == hash, record["checkpoint_id"] as? String == id,
                          record["weight_source"] as? String == TrainingWeightSource.checkpoint.rawValue,
                          let map = (record["map"] as? String).flatMap(MapRole.init(rawValue:)), map == checkpoint.map,
                          let width = record["width"] as? Int, let height = record["height"] as? Int,
                          let tileSize = record["tile_size"] as? Int, let tileCount = record["tile_count"] as? Int else {
                        throw LoomError("The saved checkpoint test record is inconsistent.")
                    }
                    let output = try TrainingRunFiles.containedFile(row["output_path"] as? String ?? "", directory: folder)
                    let preview = try TrainingRunFiles.containedFile(row["preview_path"] as? String ?? "", directory: folder)
                    return .init(checkpoint: checkpoint, result: .init(directory: recordURL.deletingLastPathComponent(),
                        inputURL: input, outputURL: output, runID: runID, source: .checkpoint, map: map,
                        width: width, height: height, tileSize: tileSize, tileCount: tileCount, note: record["note"] as? String ?? "",
                        previewURL: preview, checkpointID: id, inputSHA256: hash, boundaryHandling: record["stitching"] as? String ?? "",
                        modelName: record["model_name"] as? String, epochsCompleted: row["epochs_completed"] as? Int,
                        updatesCompleted: row["updates_completed"] as? Int), issue: nil)
                } catch { return .init(checkpoint: checkpoint, result: nil, issue: error.localizedDescription) }
            }
            return .init(directory: folder, outputs: outputs, checkpointCount: count, cancelled: manifest["status"] as? String == "cancelled")
        }
        return nil
    }
}
