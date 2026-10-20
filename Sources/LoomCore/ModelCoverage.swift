import CryptoKit
import Darwin
import Foundation
import LoomNative

/// Deployment products have stable identities separate from source image map roles.
/// A future multistage product can add its real contract without changing group identity.
public enum ModelProductID: String, Codable, CaseIterable, Sendable, Identifiable {
    case classifier, depth, roughness, normal
    public var id: String { rawValue }
    public var displayName: String {
        switch self {
        case .classifier: "Category classifier"
        case .depth: "Depth"
        case .roughness: "Roughness"
        case .normal: "Normal"
        }
    }
    public var map: MapRole? {
        switch self {
        case .classifier: nil
        case .depth: .depth
        case .roughness: .roughness
        case .normal: .normal
        }
    }
    public static var categoryProducts: [Self] { [.depth, .roughness, .normal] }
}

public enum ModelGroupID: Codable, Hashable, Sendable {
    case project
    case dataset(String)
}

public struct ModelSlot: Codable, Hashable, Sendable, Identifiable {
    public let group: ModelGroupID
    public let product: ModelProductID
    public init(group: ModelGroupID, product: ModelProductID) { self.group = group; self.product = product }
    public var datasetID: String? { if case .dataset(let id) = group { id } else { nil } }
    public var id: String { (datasetID.map { "dataset:\($0)" } ?? "project") + ":" + product.rawValue }
    var isValid: Bool {
        switch group {
        case .project: product == .classifier
        case .dataset(let id): !id.isEmpty && !id.contains("\u{0}") && product != .classifier
        }
    }
}

public enum ModelCoverageStatus: String, Codable, Sendable {
    case missing, available, selected, approved, unavailable, stale
}

public struct ModelCoverageCandidate: Identifiable, Sendable, Equatable {
    public let artifact: ModelArtifact
    public let isSelected: Bool
    public let isApproved: Bool
    public let approvalIsStale: Bool
    /// A classifier may still categorize a partially tagged folder while its
    /// catalog is incomplete for that folder's current deployment coverage.
    public let canChoose: Bool
    public let availabilityIssue: String?
    public var id: String { artifact.directory.lastPathComponent }
    public var isAvailable: Bool { availabilityIssue == nil }
}

public struct ModelCoverageEntry: Identifiable, Sendable, Equatable {
    public let slot: ModelSlot
    public let groupName: String
    public let isRequired: Bool
    public let candidates: [ModelCoverageCandidate]
    public let selectedRunID: String?
    public let effectiveRunID: String?
    public let status: ModelCoverageStatus
    public let detail: String?
    /// Saved training runs without a completed deployment artifact.
    /// A completed model's continuation checkpoint is not a second version.
    public let checkpointCount: Int
    public let pairCount: Int
    public var id: String { slot.id }
    public var usesLatestByDefault: Bool { selectedRunID == nil && effectiveRunID != nil }
    public var selectedArtifact: ModelArtifact? { candidates.first { $0.id == effectiveRunID }?.artifact }
    public var availableCount: Int { candidates.filter(\.isAvailable).count }
    public var approvedCount: Int { candidates.filter(\.isApproved).count }
}

struct ModelManagement: Codable {
    var schema = "loom-factory-model-management-v1"
    let projectID: String
    var classifierRunID: String?
    var requirements: [Requirement] = []
    var approvals: [Approval] = []
    struct Requirement: Codable { let slot: ModelSlot; let required: Bool }
    struct Approval: Codable {
        let slot: ModelSlot
        let runID: String
        let fingerprint: String
        let approvedAt: Date
    }
}

extension LoomProject {
    public var managementPersistenceURLs: [URL] {
        ["model-selection.json", "model-management.json", "model-version-metadata.json"].map { projectFolder.appendingPathComponent($0) }
    }

    public func modelCoverage() throws -> [ModelCoverageEntry] {
        try snapshot().modelCoverage
    }

    /// Approval records the human's decision about these exact saved files.
    /// It neither changes the chosen deployment version nor asserts learned quality.
    @discardableResult
    public func setModelApproval(runID: String, approved: Bool) throws -> CheckpointCleanupSummary {
        guard !operationActive else { throw LoomError("Finish or stop the current project operation before reviewing models.") }
        var management = try loadModelManagement()
        var cleanup: CheckpointCleanup.Plan?
        var cleanupLease: TrainingQueueLease?
        defer { withExtendedLifetime(cleanupLease) {} }
        if approved {
            guard let artifact = try loadArtifacts().first(where: { $0.directory.lastPathComponent == runID }),
                  let slot = modelSlot(for: artifact) else { throw LoomError("Choose a completed model from this source project.") }
            if artifact.kind == "map" {
                try CheckpointCleanup.validateRoot(projectFolder)
                cleanupLease = try TrainingQueueLease.requireExclusive(projectFolder: projectFolder)
            }
            let datasets = artifact.kind == "classifier" ? try snapshot().datasets : nil
            let fingerprint = try ManagedArtifactFiles.fingerprint(artifact, projectFolder: projectFolder,
                projectID: document.id, datasets: datasets)
            if artifact.kind == "map" {
                cleanup = try CheckpointCleanup.plan(artifact: artifact, projectFolder: projectFolder, projectID: document.id)
            }
            management.approvals.removeAll { $0.runID == runID }
            management.approvals.append(.init(slot: slot, runID: runID, fingerprint: fingerprint, approvedAt: .now))
        } else {
            management.approvals.removeAll { $0.runID == runID }
        }
        try saveModelManagement(management)
        do { return try CheckpointCleanup.execute(cleanup.map { [$0] } ?? []) }
        catch { throw LoomError("The model is approved, but cleanup could not finish: \(error.localizedDescription)") }
    }

    public func setModelRequirement(slot: ModelSlot, required: Bool) throws {
        guard !operationActive else { throw LoomError("Finish or stop the current project operation before changing model requirements.") }
        guard slot.isValid, slot.product != .classifier || required else {
            throw LoomError("Texture Loom always requires its category classifier.")
        }
        var management = try loadModelManagement()
        management.requirements.removeAll { $0.slot == slot }
        if !required { management.requirements.append(.init(slot: slot, required: false)) }
        try saveModelManagement(management)
    }

    public func selectClassifierVersion(runID: String) throws {
        guard !operationActive else { throw LoomError("Finish or stop the current project operation before choosing a classifier.") }
        guard let artifact = try loadArtifacts().first(where: { $0.kind == "classifier" && $0.directory.lastPathComponent == runID }) else {
            throw LoomError("Choose an imported classifier from this source project.")
        }
        _ = try ManagedArtifactFiles.fingerprint(artifact, projectFolder: projectFolder, projectID: document.id,
            datasets: nil)
        var management = try loadModelManagement()
        management.classifierRunID = runID
        try saveModelManagement(management)
    }

    public func selectedClassifierArtifact() throws -> ModelArtifact? {
        guard let artifact = try selectedClassifierArtifact(artifacts: loadArtifacts()) else { return nil }
        // Categorization itself establishes current groups, so choosing its
        // classifier must not depend on having already tagged every group.
        _ = try ManagedArtifactFiles.fingerprint(artifact, projectFolder: projectFolder, projectID: document.id)
        return artifact
    }

    func selectedClassifierArtifact(artifacts: [ModelArtifact]) throws -> ModelArtifact? {
        let chosen = try loadModelManagement().classifierRunID
        let candidates = artifacts.filter { $0.kind == "classifier" }.sorted { $0.createdAt > $1.createdAt }
        if let chosen {
            guard let artifact = candidates.first(where: { $0.directory.lastPathComponent == chosen }) else {
                throw LoomError("The selected classifier version is missing. Choose another saved classifier before exporting.")
            }
            return artifact
        }
        return candidates.first
    }

    func loadModelCoverage(datasets: [DatasetCategory], artifacts: [ModelArtifact], checkpoints: [TrainingCheckpoint]) throws -> [ModelCoverageEntry] {
        let management = try loadModelManagement(), selection = try loadModelSelection()
        let slots: [(ModelSlot, String, Int)] = [(ModelSlot(group: .project, product: .classifier), "Texture Loom", 0)] +
            datasets.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }.flatMap { dataset in
                ModelProductID.categoryProducts.map { product in
                    (ModelSlot(group: .dataset(dataset.id), product: product), dataset.name,
                        product.map.flatMap { dataset.pairCounts[$0.rawValue] } ?? 0)
                }
            }
        return slots.map { slot, groupName, pairCount in
            let chosen = slot.product == .classifier ? management.classifierRunID :
                selection.models.first { $0.datasetID == slot.datasetID && $0.map == slot.product.map }?.runID
            let matching = artifacts.filter { modelSlot(for: $0) == slot }.sorted { $0.createdAt > $1.createdAt }
            let candidates = matching.map { artifact in
                let approval = management.approvals.first { $0.slot == slot && $0.runID == artifact.directory.lastPathComponent }
                let fingerprint: String?, issue: String?, canChoose: Bool
                do {
                    fingerprint = try ManagedArtifactFiles.fingerprint(artifact, projectFolder: projectFolder,
                        projectID: document.id)
                    canChoose = true
                    if slot.product == .classifier {
                        do {
                            _ = try ManagedArtifactFiles.fingerprint(artifact, projectFolder: projectFolder,
                                projectID: document.id, datasets: datasets)
                            issue = nil
                        } catch { issue = error.localizedDescription }
                    } else { issue = nil }
                } catch { fingerprint = nil; issue = error.localizedDescription; canChoose = false }
                let approved = approval != nil && approval?.fingerprint == fingerprint && issue == nil
                return ModelCoverageCandidate(artifact: artifact, isSelected: artifact.directory.lastPathComponent == chosen,
                    isApproved: approved, approvalIsStale: approval != nil && !approved,
                    canChoose: canChoose, availabilityIssue: issue)
            }
            // Default behavior stays compatible with older projects; it remains visibly unchosen.
            let effective = chosen ?? candidates.first?.id
            let current = candidates.first { $0.id == effective }
            let status: ModelCoverageStatus
            let detail: String?
            if chosen != nil && current == nil {
                status = .unavailable; detail = "The chosen model is missing. Choose another saved version."
            } else if let current, !current.isAvailable {
                status = .stale; detail = current.availabilityIssue
            } else if let current, chosen != nil {
                status = current.approvalIsStale ? .stale : current.isApproved ? .approved : .selected
                detail = current.approvalIsStale ? "The saved files changed after approval. Review and approve this version again." : nil
            } else {
                status = candidates.isEmpty ? .missing : .available
                detail = nil
            }
            return ModelCoverageEntry(slot: slot, groupName: groupName,
                isRequired: management.requirements.first { $0.slot == slot }?.required ?? true,
                candidates: candidates, selectedRunID: chosen, effectiveRunID: effective, status: status, detail: detail,
                checkpointCount: checkpoints.filter { checkpoint in
                    checkpoint.datasetID == slot.datasetID && checkpoint.map == slot.product.map &&
                        !matching.contains { $0.directory.standardizedFileURL.resolvingSymlinksInPath().path ==
                            checkpoint.directory.standardizedFileURL.resolvingSymlinksInPath().path }
                }.count,
                pairCount: pairCount)
        }
    }

    func modelApprovalState(for artifact: ModelArtifact, management: ModelManagement) -> (approved: Bool, stale: Bool) {
        guard let slot = modelSlot(for: artifact),
              let approval = management.approvals.first(where: { $0.slot == slot && $0.runID == artifact.directory.lastPathComponent }) else { return (false, false) }
        let fingerprint = try? ManagedArtifactFiles.fingerprint(artifact, projectFolder: projectFolder, projectID: document.id)
        return (approval.fingerprint == fingerprint, approval.fingerprint != fingerprint)
    }

    func loadModelManagement() throws -> ModelManagement {
        let file = projectFolder.appendingPathComponent("model-management.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return ModelManagement(projectID: document.id) }
        try ManagedArtifactFiles.regularUnlinked(file)
        let value = try JSONDecoder().decode(ModelManagement.self, from: Data(contentsOf: file))
        guard value.schema == "loom-factory-model-management-v1", value.projectID == document.id,
              value.classifierRunID.map(TrainingRunFiles.validRunID) ?? true,
              value.requirements.allSatisfy({ $0.slot.isValid && ($0.slot.product != .classifier || $0.required) }),
              Set(value.requirements.map(\.slot)).count == value.requirements.count,
              value.approvals.allSatisfy({ $0.slot.isValid && TrainingRunFiles.validRunID($0.runID) &&
                  $0.fingerprint.count == 64 && $0.fingerprint.allSatisfy { "0123456789abcdef".contains($0) } }),
              Set(value.approvals.map(\.runID)).count == value.approvals.count else {
            throw LoomError("The saved model management settings do not belong to this source project.")
        }
        return value
    }

    private func saveModelManagement(_ value: ModelManagement) throws {
        let file = projectFolder.appendingPathComponent("model-management.json")
        if FileManager.default.fileExists(atPath: file.path) { try ManagedArtifactFiles.regularUnlinked(file) }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: file, options: .atomic)
    }
}

private func modelSlot(for artifact: ModelArtifact) -> ModelSlot? {
    if artifact.kind == "classifier" { return ModelSlot(group: .project, product: .classifier) }
    guard artifact.kind == "map", let datasetID = artifact.datasetID, let map = artifact.map,
          let product = ModelProductID.categoryProducts.first(where: { $0.map == map }) else { return nil }
    return ModelSlot(group: .dataset(datasetID), product: product)
}

/// File revisions include inode and change time so same-size replacements cannot retain approval.
/// The cache avoids rehashing large weights on every UI refresh; all paths are revalidated first.
private enum ManagedArtifactFiles {
    private final class Cache: @unchecked Sendable {
        let lock = NSLock()
        var hashes: [String: (revision: String, hash: String)] = [:]
    }
    private static let cache = Cache()

    static func regularUnlinked(_ file: URL) throws {
        let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true,
              file.standardizedFileURL == file.standardizedFileURL.resolvingSymlinksInPath() else {
            throw LoomError("A saved model file was removed or replaced by a link.")
        }
    }

    static func fingerprint(_ artifact: ModelArtifact, projectFolder: URL, projectID: String,
                            datasets: [DatasetCategory]? = nil) throws -> String {
        let directory = try TrainingRunFiles.ownedRun(artifact.directory, projectFolder: projectFolder)
        guard artifact.id == directory.lastPathComponent else { throw LoomError("The saved model's run identity changed.") }
        let descriptor = try TrainingRunFiles.descriptor(at: directory, source: .model)
        guard descriptor["project_id"] as? String == projectID,
              descriptor["kind"] as? String == artifact.kind,
              descriptor["architecture"] as? String == artifact.architecture,
              let corePath = artifact.coreMLPath, descriptor["coreml_path"] as? String == corePath else {
            throw LoomError("The completed model does not match this source project's saved version.")
        }
        guard let inputName = descriptor["input_name"] as? String, !inputName.isEmpty,
              let outputName = descriptor["output_name"] as? String, !outputName.isEmpty else {
            throw LoomError("The model descriptor is missing its native input or output name.")
        }
        var hashes = [try fileHash(directory.appendingPathComponent("model.json"))]
        let native = try containedArtifact(corePath, directory: directory)
        let nativeHash = try treeHash(native)
        if let declared = descriptor["coreml_sha256"] as? String, declared != nativeHash {
            throw LoomError("The completed model's native export checksum changed.")
        }
        hashes.append(nativeHash)
        if artifact.kind == "map" {
            guard let datasetID = artifact.datasetID, let map = artifact.map,
                  descriptor["dataset_id"] as? String == datasetID, descriptor["map"] as? String == map.rawValue,
                  let weightsPath = artifact.weightsPath, descriptor["weights_path"] as? String == weightsPath else {
                throw LoomError("The completed model does not match this dataset and product.")
            }
            _ = try TrainingRunFiles.nativeArchitecture(from: descriptor, map: map)
            let weightsHash = try fileHash(TrainingRunFiles.containedFile(weightsPath, directory: directory))
            if let declared = descriptor["weights_sha256"] as? String, declared != weightsHash {
                throw LoomError("The completed model's weights checksum changed.")
            }
            hashes.append(weightsHash)
        } else if artifact.kind == "classifier" {
            guard let labels = descriptor["labels"] as? [[String: String]], !labels.isEmpty,
                  labels.allSatisfy({ label in
                      guard let id = label["id"], let name = label["name"],
                            let sourceLabel = label["sourceLabel"] ?? label["id"] else { return false }
                      return ImageMetadataStore.categoryID(name) == id &&
                          !sourceLabel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                  }),
                  Set(labels.compactMap { $0["id"] }).count == labels.count,
                  Set(labels.compactMap { $0["sourceLabel"] ?? $0["id"] }).count == labels.count else {
                throw LoomError("The classifier has an inconsistent dataset label catalog.")
            }
            if let datasets, Set(labels.compactMap { $0["id"] }) != Set(datasets.map(\.id)) {
                throw LoomError("The classifier does not match every dataset currently tagged in this source folder.")
            }
        } else { throw LoomError("This model product is not supported by Texture Loom.") }
        return digest(Data(hashes.joined(separator: "\n").utf8))
    }

    private static func containedArtifact(_ path: String, directory: URL) throws -> URL {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
            throw LoomError("The saved model references a file outside its run.")
        }
        let url = directory.appendingPathComponent(path).standardizedFileURL
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey])
        guard values.isSymbolicLink != true, values.isDirectory == true || values.isRegularFile == true,
              url == url.resolvingSymlinksInPath(), url.path.hasPrefix(directory.path + "/") else {
            throw LoomError("A saved model file was removed or replaced by a link.")
        }
        return url
    }

    private static func treeHash(_ url: URL) throws -> String {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey])
        if values.isDirectory != true { return try fileHash(url) }
        guard let enumerator = FileManager.default.enumerator(at: url,
            includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey], options: []) else {
            throw LoomError("The native model package could not be read.")
        }
        var files: [URL] = []
        for case let entry as URL in enumerator {
            let properties = try entry.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey])
            guard properties.isSymbolicLink != true else { throw LoomError("The native model package contains a link.") }
            if properties.isRegularFile == true { files.append(entry) }
            else if properties.isDirectory != true { throw LoomError("The native model package contains an unsupported file.") }
        }
        guard !files.isEmpty else { throw LoomError("The native model package is empty.") }
        let records = try files.sorted { $0.path < $1.path }.map { file in
            String(file.path.dropFirst(url.path.count + 1)) + ":" + (try fileHash(file))
        }
        return digest(Data(records.joined(separator: "\n").utf8))
    }

    private static func fileHash(_ file: URL) throws -> String {
        try regularUnlinked(file)
        var info = stat()
        guard lstat(file.path, &info) == 0 else { throw LoomError("The saved model file could not be read.") }
        let revision = "\(info.st_dev):\(info.st_ino):\(info.st_size):\(info.st_mtimespec.tv_sec):\(info.st_mtimespec.tv_nsec):\(info.st_ctimespec.tv_sec):\(info.st_ctimespec.tv_nsec)"
        cache.lock.lock(); let saved = cache.hashes[file.path]; cache.lock.unlock()
        if saved?.revision == revision { return saved!.hash }
        let value = try TrainingRunFiles.sha256(file)
        var after = stat()
        guard lstat(file.path, &after) == 0, after.st_ino == info.st_ino, after.st_size == info.st_size,
              after.st_mtimespec.tv_sec == info.st_mtimespec.tv_sec, after.st_mtimespec.tv_nsec == info.st_mtimespec.tv_nsec,
              after.st_ctimespec.tv_sec == info.st_ctimespec.tv_sec, after.st_ctimespec.tv_nsec == info.st_ctimespec.tv_nsec else {
            throw LoomError("The saved model changed while it was being checked. Try again.")
        }
        cache.lock.lock(); cache.hashes[file.path] = (revision, value); cache.lock.unlock()
        return value
    }

    private static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
}
