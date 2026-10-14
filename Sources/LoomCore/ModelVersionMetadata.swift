import Foundation

/// Human-facing names and notes live outside immutable training artifacts.
struct ModelVersionMetadata: Codable {
    var schema = "loom-factory-model-version-metadata-v1"
    let projectID: String
    var versions: [Entry] = []
    struct Entry: Codable {
        let runID: String
        let name: String
        let notes: String
    }
}

extension LoomProject {
    public func updateModelVersionMetadata(runID: String, name: String, notes: String) throws {
        guard !operationActive else { throw LoomError("Finish or stop the current operation before editing a saved run.") }
        let cleanName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanName.isEmpty else { throw LoomError("Give this model or checkpoint a name.") }
        guard try modelVersions().contains(where: { $0.id == runID }) else {
            throw LoomError("Choose a saved model or checkpoint belonging to this source project.")
        }
        var metadata = try loadModelVersionMetadata()
        metadata.versions.removeAll { $0.runID == runID }
        metadata.versions.append(.init(runID: runID, name: cleanName, notes: notes))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(to: projectFolder.appendingPathComponent("model-version-metadata.json"), options: .atomic)
    }

    func loadModelVersionMetadata() throws -> ModelVersionMetadata {
        let file = projectFolder.appendingPathComponent("model-version-metadata.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return .init(projectID: document.id) }
        let properties = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard properties.isRegularFile == true, properties.isSymbolicLink != true,
              file.standardizedFileURL == file.standardizedFileURL.resolvingSymlinksInPath() else {
            throw LoomError("The saved model names and notes were replaced by a link.")
        }
        let value = try JSONDecoder().decode(ModelVersionMetadata.self, from: Data(contentsOf: file))
        guard value.schema == "loom-factory-model-version-metadata-v1", value.projectID == document.id,
              value.versions.allSatisfy({ TrainingRunFiles.validRunID($0.runID) }),
              Set(value.versions.map(\.runID)).count == value.versions.count else {
            throw LoomError("The saved model names and notes do not belong to this source project.")
        }
        return value
    }
}
