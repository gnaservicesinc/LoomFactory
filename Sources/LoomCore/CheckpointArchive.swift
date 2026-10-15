import Foundation

/// Checkpoint history lives beside its immutable tensors, within the owned run.
/// Only descriptors are copied; large tensors are retained once per save.
enum CheckpointArchive {
    static func filename(_ id: String) throws -> String {
        guard TrainingRunFiles.validRunID(id), id.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") }) else {
            throw LoomError("Choose a saved checkpoint from this run.")
        }
        return "checkpoint-\(id).json"
    }

    static func descriptors(in directory: URL) throws -> [URL] {
        let latest = directory.appendingPathComponent("checkpoint.json")
        let archives = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("checkpoint-") && $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return (FileManager.default.fileExists(atPath: latest.path) ? [latest] : []) + archives
    }

    static func preserveLatest(in directory: URL) throws {
        let latest = directory.appendingPathComponent("checkpoint.json")
        guard FileManager.default.fileExists(atPath: latest.path) else { return }
        let file = try TrainingRunFiles.containedFile("checkpoint.json", directory: directory)
        let data = try Data(contentsOf: file)
        guard var value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LoomError("The previous checkpoint descriptor cannot be preserved.")
        }
        let id = try value["checkpoint_id"] as? String ?? "legacy-" + TrainingRunFiles.sha256(file)
        let target = directory.appendingPathComponent(try filename(id))
        guard !FileManager.default.fileExists(atPath: target.path) else { return }
        value["checkpoint_id"] = id
        try publish(value, revisionID: id, in: directory)
    }

    static func publish(_ descriptor: [String: Any], revisionID: String, in directory: URL) throws {
        guard descriptor["checkpoint_id"] as? String == revisionID else {
            throw LoomError("The checkpoint revision identity is inconsistent.")
        }
        let target = directory.appendingPathComponent(try filename(revisionID))
        guard !FileManager.default.fileExists(atPath: target.path) else {
            throw LoomError("A saved checkpoint revision already exists and cannot be replaced.")
        }
        let data = try JSONSerialization.data(withJSONObject: descriptor, options: [.sortedKeys, .prettyPrinted])
        let temporary = directory.appendingPathComponent(".checkpoint-descriptor-" + UUID().uuidString.lowercased())
        defer { try? FileManager.default.removeItem(at: temporary) }
        try data.write(to: temporary, options: .atomic)
        try FileManager.default.moveItem(at: temporary, to: target)
    }
}
