import Foundation

public struct TrainingRunDeletion: Sendable, Equatable {
    public let runIDs: [String]
    public let trashURLs: [URL]
    public var count: Int { runIDs.count }
}

extension LoomProject {
    /// Removes only this project's map-training runs. Packages and classifiers
    /// are independent outputs and remain available.
    public func deleteAllTrainingRuns() throws -> TrainingRunDeletion {
        try deleteTrainingRuns(runIDs: modelVersions().map(\.id))
    }

    /// Moves complete run folders, including their checkpoints and exported
    /// weights, to the macOS Trash. Originals are never part of a run folder.
    public func deleteTrainingRuns(runIDs: [String]) throws -> TrainingRunDeletion {
        try deleteTrainingRuns(runIDs: runIDs, moveToTrash: { folder in
            var trashed: NSURL?
            try FileManager.default.trashItem(at: folder, resultingItemURL: &trashed)
            return trashed as URL?
        })
    }

    // A supplied mover lets isolated tests exercise exactly the same validation
    // and selection handling without filling the user's Trash.
    func deleteTrainingRuns(runIDs: [String], moveToTrash: @Sendable (URL) throws -> URL?) throws -> TrainingRunDeletion {
        guard !operationActive, activeTraining == nil, activeProcess?.isRunning != true else {
            throw LoomError("Finish or stop the current project operation before deleting training runs.")
        }
        let ids = Array(Set(runIDs)).sorted()
        guard !ids.isEmpty else { return TrainingRunDeletion(runIDs: [], trashURLs: []) }
        try validateRunDeletionRoot(projectFolder)
        let runs = projectFolder.appendingPathComponent("runs", isDirectory: true)
        try validateRunDeletionRoot(runs)
        // Validate the complete request before moving any folder.
        let folders = try ids.map { id -> URL in
            let folder = try TrainingRunFiles.runDirectory(id, projectFolder: projectFolder)
            try validateTrainingRunForDeletion(folder)
            return folder
        }
        let selectionFile = projectFolder.appendingPathComponent("model-selection.json")
        var selection = try deletionModelSelection(at: selectionFile)
        let originalSelection = selection
        var removed: [String] = [], trash: [URL] = []
        var failure: Error?
        for folder in folders {
            do {
                // Recheck immediately before each move; a folder replacement
                // discovered after the first move must not follow a link.
                try validateTrainingRunForDeletion(folder)
                if let destination = try moveToTrash(folder) { trash.append(destination) }
                removed.append(folder.lastPathComponent)
            } catch { failure = error; break }
        }
        if var updated = selection {
            updated.models.removeAll { removed.contains($0.runID) }
            selection = updated
        }
        if selection != originalSelection, let selection {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(selection).write(to: selectionFile, options: .atomic)
        }
        if let failure { throw failure }
        return TrainingRunDeletion(runIDs: removed, trashURLs: trash)
    }

    private func validateRunDeletionRoot(_ folder: URL) throws {
        let properties = try folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard properties.isDirectory == true, properties.isSymbolicLink != true,
              folder.standardizedFileURL == folder.standardizedFileURL.resolvingSymlinksInPath() else {
            throw LoomError("The training run folder was replaced by a link. Reopen the source folder before deleting runs.")
        }
    }

    private func validateTrainingRunForDeletion(_ folder: URL) throws {
        _ = try TrainingRunFiles.ownedRun(folder, projectFolder: projectFolder)
        var hasProjectIdentity = false, isMapRun = false
        if let metadata = try TrainingRunFiles.metadata(at: folder) {
            guard metadata.projectID == document.id, MapRole.targets.contains(metadata.map) else {
                throw LoomError("This training run belongs to another source project.")
            }
            hasProjectIdentity = true; isMapRun = true
        }
        for name in ["request.json", "checkpoint.json", "model.json"] {
            let file = folder.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: file.path) else { continue }
            _ = try TrainingRunFiles.containedFile(name, directory: folder)
            guard let descriptor = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any] else {
                throw LoomError("The training run's identity could not be read safely.")
            }
            if let projectID = descriptor["project_id"] as? String {
                guard projectID == document.id else { throw LoomError("This training run belongs to another source project.") }
                hasProjectIdentity = true
            }
            let map = (descriptor["map"] as? String ?? descriptor["map_kind"] as? String).flatMap(MapRole.init(rawValue:))
            if let kind = descriptor["kind"] as? String, kind != "map" {
                throw LoomError("Only map-training runs can be deleted from training history.")
            }
            if let map, MapRole.targets.contains(map) { isMapRun = true }
        }
        guard hasProjectIdentity, isMapRun else {
            throw LoomError("Choose a map-training run owned by this source project.")
        }
        // Run symlinks are unnecessary. Reject them rather than allow a run to
        // hide references to source images or another project's files.
        guard let entries = FileManager.default.enumerator(at: folder,
            includingPropertiesForKeys: [.isSymbolicLinkKey], options: []) else {
            throw LoomError("The training run could not be checked before deletion.")
        }
        for case let entry as URL in entries {
            guard try entry.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
                throw LoomError("This run contains a link. Remove the link before deleting the run.")
            }
        }
    }

    private func deletionModelSelection(at file: URL) throws -> DeletionModelSelection? {
        guard FileManager.default.fileExists(atPath: file.path) else { return nil }
        let properties = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard properties.isRegularFile == true, properties.isSymbolicLink != true,
              file.standardizedFileURL == file.standardizedFileURL.resolvingSymlinksInPath() else {
            throw LoomError("The saved model selection was replaced by a link.")
        }
        let value = try JSONDecoder().decode(DeletionModelSelection.self, from: Data(contentsOf: file))
        guard value.schema == "loom-factory-model-selection-v1", value.projectID == document.id,
              value.models.allSatisfy({ MapRole.targets.contains($0.map) && TrainingRunFiles.validRunID($0.runID) }),
              Set(value.models.map { $0.datasetID + "\u{0}" + $0.map.rawValue }).count == value.models.count else {
            throw LoomError("The saved model selection does not belong to this source project.")
        }
        return value
    }
}

private struct DeletionModelSelection: Codable, Equatable {
    var schema: String
    var projectID: String
    var models: [Entry]
    struct Entry: Codable, Equatable {
        var datasetID: String
        var map: MapRole
        var runID: String
    }
}
