import Foundation

public struct TrainingCheckpoint: Identifiable, Sendable, Equatable {
    public var id: String { directory.path }
    public let directory: URL
    public let datasetID: String
    public let map: MapRole
    public let modelWidth: Int
    public let modelBlocks: Int
    public let imageSize: Int
    public let exaggerateFeatures: Bool
    public let epochsCompleted: Int
    public let status: String
    public let modifiedAt: Date
    public var parameterCount: Int { TrainingSettings(modelWidth: modelWidth, modelBlocks: modelBlocks).parameterCount(map: map) }
}

extension LoomProject {
    func loadCheckpoints() throws -> [TrainingCheckpoint] {
        let runs = projectFolder.appendingPathComponent("runs", isDirectory: true)
        guard FileManager.default.fileExists(atPath: runs.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: runs, includingPropertiesForKeys: nil).compactMap { directory in
            let path = directory.appendingPathComponent("checkpoint.json")
            guard let data = try? Data(contentsOf: path),
                  let descriptor = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  descriptor["schema"] as? String == "loom-factory-checkpoint-v1",
                  descriptor["project_id"] as? String == document.id,
                  descriptor["kind"] as? String == "map",
                  let datasetID = descriptor["dataset_id"] as? String,
                  let mapName = descriptor["map"] as? String, let map = MapRole(rawValue: mapName), MapRole.targets.contains(map),
                  let architecture = descriptor["architecture_config"] as? [String: Any],
                  architecture["id"] as? String == "loom-factory-material-residual-v1",
                  let width = architecture["width"] as? Int, (8...256).contains(width),
                  let blocks = architecture["blocks"] as? Int, (1...12).contains(blocks),
                  let savedSettings = descriptor["settings"] as? [String: Any],
                  let size = savedSettings["image_size"] as? Int, (32...8192).contains(size),
                  FileManager.default.fileExists(atPath: directory.appendingPathComponent("checkpoint.safetensors").path) else { return nil }
            let modified = (try? path.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return TrainingCheckpoint(directory: directory, datasetID: datasetID, map: map, modelWidth: width,
                modelBlocks: blocks, imageSize: size, exaggerateFeatures: savedSettings["exaggerate_features"] as? Bool ?? false,
                epochsCompleted: descriptor["epochs_completed"] as? Int ?? 0, status: descriptor["status"] as? String ?? "saved", modifiedAt: modified)
        }.sorted { $0.modifiedAt > $1.modifiedAt }
    }
}
