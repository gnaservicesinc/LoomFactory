import CryptoKit
import Foundation
import LoomNative

public enum TrainingStart: Sendable, Equatable {
    case fresh
    case checkpoint(runID: String)
    case model(runID: String)
}

public enum TrainingWeightSource: String, Codable, Sendable, Equatable {
    case checkpoint, model
}

public struct TrainingLineage: Codable, Sendable, Equatable {
    public let parentRunID: String
    public let source: TrainingWeightSource
    public let weightsSHA256: String
    public init(parentRunID: String, source: TrainingWeightSource, weightsSHA256: String) {
        self.parentRunID = parentRunID; self.source = source; self.weightsSHA256 = weightsSHA256
    }
}

public struct TrainingMetric: Sendable, Equatable {
    public let epoch: Int
    public let loss: Double?
    public let validationLoss: Double?
    public let learningRate: Double?
    public let update: Int?
    public let meanGradientNorm: Double?
    public let peakGradientNorm: Double?
    public let clippedUpdates: Int?
    public let optimizerUpdates: Int?
    public let gradientClipThreshold: Double?
    public init(epoch: Int, loss: Double? = nil, validationLoss: Double? = nil,
                learningRate: Double? = nil, update: Int? = nil, meanGradientNorm: Double? = nil,
                peakGradientNorm: Double? = nil, clippedUpdates: Int? = nil, optimizerUpdates: Int? = nil,
                gradientClipThreshold: Double? = nil) {
        self.epoch = epoch; self.loss = loss; self.validationLoss = validationLoss
        self.learningRate = learningRate; self.update = update
        self.meanGradientNorm = meanGradientNorm; self.peakGradientNorm = peakGradientNorm
        self.clippedUpdates = clippedUpdates; self.optimizerUpdates = optimizerUpdates
        self.gradientClipThreshold = gradientClipThreshold
    }
}

/// One immutable run directory. Its checkpoint can differ from the best weights exported by that run.
public struct ModelVersion: Identifiable, Sendable, Equatable {
    public let id: String
    public let directory: URL
    public let datasetID: String
    public let datasetName: String
    public let map: MapRole
    public let status: String
    public let createdAt: Date
    public let label: String
    public let notes: String
    public let inheritedEpochs: Int
    public let settings: TrainingSettings?
    public let checkpoint: TrainingCheckpoint?
    public let artifact: ModelArtifact?
    public let history: [TrainingMetric]
    public let lineage: TrainingLineage?
    public let isActive: Bool
    public let isApproved: Bool
    public let approvalIsStale: Bool
    /// The recorded run settings, including optimizer and data selection.
    public var savedSettings: TrainingSettings? { settings }
    public var exportedEpochsCompleted: Int { (artifact?.epochs ?? 0) + inheritedEpochs }
    public init(id: String, directory: URL, datasetID: String, datasetName: String, map: MapRole,
                status: String, createdAt: Date = .now, label: String = "", settings: TrainingSettings? = nil,
                checkpoint: TrainingCheckpoint? = nil, artifact: ModelArtifact? = nil,
                history: [TrainingMetric] = [], lineage: TrainingLineage? = nil, isActive: Bool = false,
                isApproved: Bool = false, approvalIsStale: Bool = false, notes: String = "", inheritedEpochs: Int = 0) {
        self.id = id; self.directory = directory; self.datasetID = datasetID; self.datasetName = datasetName
        self.map = map; self.status = status; self.createdAt = createdAt; self.label = label; self.notes = notes
        self.inheritedEpochs = inheritedEpochs
        self.settings = settings; self.checkpoint = checkpoint; self.artifact = artifact
        self.history = history; self.lineage = lineage; self.isActive = isActive
        self.isApproved = isApproved; self.approvalIsStale = approvalIsStale
    }
}

struct TrainingRunMetadata: Codable, Sendable {
    var schema = "loom-factory-run-version-v1"
    let projectID: String
    let runID: String
    let datasetID: String
    let map: MapRole
    let label: String
    let createdAt: Date
    let settings: TrainingSettings
    let lineage: TrainingLineage?
    var status: String
}

struct TrainingResume: Sendable {
    let weights: URL
    let sha256: String
    let runID: String
    let source: TrainingWeightSource
}

struct ModelSelection: Codable {
    var schema = "loom-factory-model-selection-v1"
    let projectID: String
    var models: [Entry]
    struct Entry: Codable {
        let datasetID: String
        let map: MapRole
        let runID: String
    }
}

extension LoomProject {
    public func modelVersions(datasetID: String? = nil, map: MapRole? = nil) throws -> [ModelVersion] {
        try loadModelVersions(artifacts: loadArtifacts(), checkpoints: loadCheckpoints()).filter {
            (datasetID == nil || $0.datasetID == datasetID) && (map == nil || $0.map == map)
        }
    }

    /// Changes only which completed version the next package export will contain.
    public func selectModelVersion(runID: String, datasetID: String, map: MapRole) throws {
        guard !operationActive else { throw LoomError("Finish or stop the current project operation before changing the exported model.") }
        try activateModelVersion(runID: runID, datasetID: datasetID, map: map)
    }

    public func selectedModelArtifact(datasetID: String, map: MapRole) throws -> ModelArtifact? {
        try selectedModelArtifact(datasetID: datasetID, map: map, artifacts: loadArtifacts())
    }

    func selectedModelArtifact(datasetID: String, map: MapRole, artifacts: [ModelArtifact]) throws -> ModelArtifact? {
        let selection = try loadModelSelection()
        let candidates = artifacts.filter { $0.kind == "map" && $0.datasetID == datasetID && $0.map == map }.sorted { $0.createdAt > $1.createdAt }
        if let chosen = selection.models.first(where: { $0.datasetID == datasetID && $0.map == map }) {
            guard let artifact = candidates.first(where: { $0.directory.lastPathComponent == chosen.runID }) else {
                throw LoomError("The selected \(map.displayName.lowercased()) version is missing. Choose another saved version before exporting.")
            }
            return artifact
        }
        // Projects created before selection was introduced keep their latest-completed export behavior.
        return candidates.first
    }

    func activateModelVersion(runID: String, datasetID: String, map: MapRole) throws {
        guard MapRole.targets.contains(map),
              let artifact = try loadArtifacts().first(where: { $0.kind == "map" && $0.datasetID == datasetID &&
                  $0.map == map && $0.directory.lastPathComponent == runID }) else {
            throw LoomError("Choose a completed model version from this dataset and map.")
        }
        try TrainingRunFiles.validateArtifact(artifact, projectFolder: projectFolder, projectID: document.id)
        var selection = try loadModelSelection()
        selection.models.removeAll { $0.datasetID == datasetID && $0.map == map }
        selection.models.append(.init(datasetID: datasetID, map: map, runID: runID))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(selection).write(to: projectFolder.appendingPathComponent("model-selection.json"), options: .atomic)
    }

    func loadModelVersions(artifacts: [ModelArtifact], checkpoints: [TrainingCheckpoint]) throws -> [ModelVersion] {
        let runs = projectFolder.appendingPathComponent("runs", isDirectory: true)
        guard FileManager.default.fileExists(atPath: runs.path) else { return [] }
        let selection = try loadModelSelection()
        let management = try loadModelManagement()
        let displayMetadata = try loadModelVersionMetadata()
        let folders = try FileManager.default.contentsOfDirectory(at: runs, includingPropertiesForKeys: nil)
        return folders.compactMap { folder -> ModelVersion? in
            guard (try? TrainingRunFiles.ownedRun(folder, projectFolder: projectFolder)) != nil else { return nil }
            let checkpoint = checkpoints.first { $0.directory == folder }
            let artifact = artifacts.first { $0.kind == "map" && $0.directory == folder }
            let metadata = try? TrainingRunFiles.metadata(at: folder)
            let model = TrainingRunFiles.optionalJSON(folder.appendingPathComponent("model.json"))
            let savedCheckpoint = TrainingRunFiles.optionalJSON(folder.appendingPathComponent("checkpoint.json"))
            let request = TrainingRunFiles.optionalJSON(folder.appendingPathComponent("request.json"))
            let descriptors = [model, savedCheckpoint, request].compactMap { $0 }
            guard metadata?.projectID == document.id || descriptors.contains(where: { $0["project_id"] as? String == document.id }) else { return nil }
            // Do not surface a run copied from a different source project as local history.
            guard descriptors.allSatisfy({ ($0["project_id"] as? String).map { $0 == document.id } ?? true }),
                  metadata?.projectID == nil || metadata?.projectID == document.id else { return nil }
            guard let datasetID = metadata?.datasetID ?? artifact?.datasetID ?? checkpoint?.datasetID ?? request?["dataset_id"] as? String,
                  let map = metadata?.map ?? artifact?.map ?? checkpoint?.map ?? (request?["map_kind"] as? String).flatMap(MapRole.init(rawValue:)),
                  MapRole.targets.contains(map) else { return nil }
            let progress = TrainingRunFiles.optionalJSON(folder.appendingPathComponent("progress.json"))
            let training = model?["training"] as? [String: Any]
            let settings = metadata?.settings ?? TrainingRunFiles.settings(from: request ?? training ?? savedCheckpoint?["settings"] as? [String: Any] ?? [:],
                descriptor: model ?? savedCheckpoint)
            let localHistory = progress?["history"] as? [[String: Any]] ?? training?["history"] as? [[String: Any]] ?? savedCheckpoint?["history"] as? [[String: Any]] ?? []
            let priorHistory = progress?["prior_history"] as? [[String: Any]] ?? training?["prior_history"] as? [[String: Any]] ?? savedCheckpoint?["prior_history"] as? [[String: Any]] ?? []
            let priorEpochs = progress?["prior_epochs"] as? Int ?? training?["prior_epochs"] as? Int ?? savedCheckpoint?["prior_epochs"] as? Int ?? 0
            let priorUpdates = progress?["prior_updates"] as? Int ?? training?["prior_updates"] as? Int ?? savedCheckpoint?["prior_updates"] as? Int ?? 0
            let historyRows = priorHistory + localHistory.enumerated().map { index, row in
                var shifted = row
                shifted["epoch"] = (row["epoch"] as? Int ?? index + 1) + priorEpochs
                if let update = row["update"] as? Int ?? row["updates"] as? Int { shifted["update"] = update + priorUpdates }
                return shifted
            }
            let history = historyRows.enumerated().map { index, row in
                TrainingMetric(epoch: row["epoch"] as? Int ?? index + 1, loss: row["loss"] as? Double,
                    validationLoss: row["validation_loss"] as? Double, learningRate: row["learning_rate"] as? Double,
                    update: row["update"] as? Int ?? row["updates"] as? Int,
                    meanGradientNorm: row["mean_gradient_norm"] as? Double, peakGradientNorm: row["peak_gradient_norm"] as? Double,
                    clippedUpdates: row["clipped_updates"] as? Int, optimizerUpdates: row["optimizer_updates"] as? Int,
                    gradientClipThreshold: row["gradient_clip_threshold"] as? Double)
            }
            let selected = selection.models.first { $0.datasetID == datasetID && $0.map == map }
            let active = selected?.runID ?? artifacts.first { $0.kind == "map" && $0.datasetID == datasetID && $0.map == map }?.directory.lastPathComponent
            let created = metadata?.createdAt ?? artifact?.createdAt ?? checkpoint?.modifiedAt ??
                (try? folder.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            let recordedStatus = progress?["status"] as? String ?? checkpoint?.status ?? metadata?.status ?? "interrupted"
            let status = ["running", "training"].contains(recordedStatus) && activeTrainingControl?.runID != folder.lastPathComponent
                ? "inactive in this window" : recordedStatus
            let approval = artifact.map { modelApprovalState(for: $0, management: management) } ?? (approved: false, stale: false)
            let display = displayMetadata.versions.first { $0.runID == folder.lastPathComponent }
            return ModelVersion(id: folder.lastPathComponent, directory: folder, datasetID: datasetID,
                datasetName: artifact?.datasetName ?? model?["dataset_name"] as? String ?? savedCheckpoint?["dataset_name"] as? String ?? request?["display_name"] as? String ?? datasetID,
                map: map, status: artifact == nil ? status : "completed",
                createdAt: created, label: display?.name ?? metadata?.label ?? request?["run_label"] as? String ?? "", settings: settings,
                checkpoint: checkpoint, artifact: artifact, history: history, lineage: metadata?.lineage,
                isActive: artifact != nil && active == folder.lastPathComponent,
                isApproved: approval.approved, approvalIsStale: approval.stale, notes: display?.notes ?? "", inheritedEpochs: priorEpochs)
        }.sorted { $0.createdAt > $1.createdAt }
    }

    func prepareTrainingStart(_ start: TrainingStart, datasetID: String, map: MapRole,
                              settings: TrainingSettings) throws -> (settings: TrainingSettings, resume: TrainingResume?) {
        guard start != .fresh else {
            var fresh = settings
            if map != .normal { fresh.scalarLoss = .logitCrossEntropyV2 }
            return (fresh, nil)
        }
        let runID: String, source: TrainingWeightSource
        switch start {
        case .fresh: return (settings, nil)
        case .checkpoint(let id): runID = id; source = .checkpoint
        case .model(let id): runID = id; source = .model
        }
        let folder = try TrainingRunFiles.runDirectory(runID, projectFolder: projectFolder)
        let descriptor = try TrainingRunFiles.descriptor(at: folder, source: source)
        guard let saved = TrainingRunFiles.settings(from: (source == .checkpoint ? descriptor["settings"] : descriptor["training"]) as? [String: Any] ?? [:], descriptor: descriptor) else {
            throw LoomError("This saved version has no readable native model settings.")
        }
        var resolved = settings
        resolved.modelArchitecture = saved.modelArchitecture
        resolved.scalarLoss = saved.scalarLoss
        resolved.modelWidth = saved.modelWidth; resolved.modelBlocks = saved.modelBlocks
        resolved.imageSize = saved.imageSize; resolved.exaggerateFeatures = saved.exaggerateFeatures
        let path = source == .checkpoint ? (descriptor["weights_path"] as? String ?? "checkpoint.safetensors") : descriptor["weights_path"] as? String
        guard let path else { throw LoomError("This saved model version has no trainable weights.") }
        let weights = try TrainingRunFiles.containedFile(path, directory: folder)
        return (resolved, try TrainingRunFiles.resume(weights: weights, projectFolder: projectFolder, projectID: document.id,
            datasetID: datasetID, map: map, settings: resolved))
    }

    func loadModelSelection() throws -> ModelSelection {
        let file = projectFolder.appendingPathComponent("model-selection.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return ModelSelection(projectID: document.id, models: []) }
        let selection = try JSONDecoder().decode(ModelSelection.self, from: Data(contentsOf: file))
        guard selection.schema == "loom-factory-model-selection-v1", selection.projectID == document.id,
              selection.models.allSatisfy({ MapRole.targets.contains($0.map) && TrainingRunFiles.validRunID($0.runID) }),
              Set(selection.models.map { $0.datasetID + "\u{0}" + $0.map.rawValue }).count == selection.models.count else {
            throw LoomError("The saved model selection does not belong to this source project.")
        }
        return selection
    }
}

enum TrainingRunFiles {
    static func metadata(at directory: URL) throws -> TrainingRunMetadata? {
        let file = directory.appendingPathComponent("run-version.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return nil }
        let value = try JSONDecoder().decode(TrainingRunMetadata.self, from: Data(contentsOf: file))
        guard value.schema == "loom-factory-run-version-v1", value.runID == directory.lastPathComponent else {
            throw LoomError("The saved run version identity is inconsistent.")
        }
        return value
    }

    static func writeMetadata(_ value: TrainingRunMetadata, at directory: URL) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: directory.appendingPathComponent("run-version.json"), options: .atomic)
    }

    static func validRunID(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".." && !value.contains("/") && !value.contains("\\")
    }

    static func runDirectory(_ id: String, projectFolder: URL) throws -> URL {
        guard validRunID(id) else { throw LoomError("Choose a saved run from this source project.") }
        let directory = projectFolder.appendingPathComponent("runs", isDirectory: true).appendingPathComponent(id, isDirectory: true)
        return try ownedRun(directory, projectFolder: projectFolder)
    }

    @discardableResult static func ownedRun(_ directory: URL, projectFolder: URL) throws -> URL {
        let runs = projectFolder.appendingPathComponent("runs", isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
        let canonical = directory.standardizedFileURL.resolvingSymlinksInPath()
        let properties = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard canonical.deletingLastPathComponent() == runs, properties.isDirectory == true, properties.isSymbolicLink != true,
              directory.standardizedFileURL.path == canonical.path else {
            throw LoomError("The saved weights must belong to an unchanged run in this source project.")
        }
        return canonical
    }

    static func containedFile(_ path: String, directory: URL) throws -> URL {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else {
            throw LoomError("The saved model references a file outside its run.")
        }
        let file = directory.appendingPathComponent(path).standardizedFileURL
        let canonical = file.resolvingSymlinksInPath()
        let properties = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard canonical.path.hasPrefix(directory.resolvingSymlinksInPath().path + "/"), file == canonical,
              properties.isRegularFile == true, properties.isSymbolicLink != true else {
            throw LoomError("A saved model file was removed or replaced by a link.")
        }
        return canonical
    }

    static func descriptor(at directory: URL, source: TrainingWeightSource) throws -> [String: Any] {
        let file = try containedFile(source == .checkpoint ? "checkpoint.json" : "model.json", directory: directory)
        guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any],
              value["schema"] as? String == (source == .checkpoint ? "loom-factory-checkpoint-v1" : "loom-factory-model-v1"),
              source == .checkpoint || value["status"] as? String == "completed" else {
            throw LoomError("This saved model version is incomplete or uses an unsupported format.")
        }
        return value
    }

    static func resume(weights: URL, projectFolder: URL, projectID: String, datasetID: String, map: MapRole,
                       settings: TrainingSettings, expectedSHA256: String? = nil) throws -> TrainingResume {
        let directory = try ownedRun(weights.deletingLastPathComponent(), projectFolder: projectFolder)
        let checkpointDescriptor = optionalJSON(directory.appendingPathComponent("checkpoint.json"))
        let checkpointPath = checkpointDescriptor?["weights_path"] as? String ?? "checkpoint.safetensors"
        let source: TrainingWeightSource = weights.standardizedFileURL.resolvingSymlinksInPath()
            == directory.appendingPathComponent(checkpointPath).standardizedFileURL.resolvingSymlinksInPath() ? .checkpoint : .model
        let descriptor = try descriptor(at: directory, source: source)
        let architecture = try nativeArchitecture(from: descriptor, map: map)
        let nativeSettings = (source == .checkpoint ? descriptor["settings"] : descriptor["training"]) as? [String: Any] ?? [:]
        guard descriptor["project_id"] as? String == projectID, descriptor["dataset_id"] as? String == datasetID,
              descriptor["kind"] as? String == "map", descriptor["map"] as? String == map.rawValue,
              architecture.identifier == settings.modelArchitecture,
              architecture.width == settings.modelWidth, architecture.blocks == settings.modelBlocks,
              (nativeSettings["model_architecture"] as? String).map({ $0 == architecture.identifier }) ?? true,
              (nativeSettings["image_size"] as? Int ?? descriptor["image_size"] as? Int) == settings.imageSize,
              (nativeSettings["exaggerate_features"] as? Bool ?? false) == settings.exaggerateFeatures else {
            throw LoomError("The saved weights do not match this project, dataset, map or native model settings.")
        }
        let declaredPath = descriptor["weights_path"] as? String ?? (source == .checkpoint ? "checkpoint.safetensors" : "")
        let file = try containedFile(declaredPath, directory: directory)
        guard file == weights.standardizedFileURL.resolvingSymlinksInPath() else { throw LoomError("The saved weights do not match their model descriptor.") }
        let hash = try sha256(file)
        if let savedHash = descriptor["weights_sha256"] as? String, savedHash != hash {
            throw LoomError("The saved weights checksum changed. Choose another model version.")
        }
        if source == .checkpoint, descriptor["weights_sha256"] as? String == nil {
            throw LoomError("This saved checkpoint has no weights checksum.")
        }
        if let expectedSHA256, expectedSHA256 != hash {
            throw LoomError("The selected weights changed before training could start.")
        }
        return TrainingResume(weights: file, sha256: hash, runID: directory.lastPathComponent, source: source)
    }

    static func validateArtifact(_ artifact: ModelArtifact, projectFolder: URL, projectID: String) throws {
        let directory = try ownedRun(artifact.directory, projectFolder: projectFolder)
        let descriptor = try descriptor(at: directory, source: .model)
        guard artifact.kind == "map", let datasetID = artifact.datasetID, let map = artifact.map,
              descriptor["project_id"] as? String == projectID, descriptor["dataset_id"] as? String == datasetID,
              descriptor["map"] as? String == map.rawValue, descriptor["architecture"] as? String == artifact.architecture,
              let weightsPath = artifact.weightsPath, descriptor["weights_path"] as? String == weightsPath,
              let coreMLPath = artifact.coreMLPath, descriptor["coreml_path"] as? String == coreMLPath else {
            throw LoomError("The completed model does not match this source project's saved version.")
        }
        _ = try nativeArchitecture(from: descriptor, map: map)
        let weights = try containedFile(weightsPath, directory: directory)
        let native = try containedFile(coreMLPath, directory: directory)
        if let hash = descriptor["weights_sha256"] as? String, hash != (try sha256(weights)) {
            throw LoomError("The completed model's weights checksum changed.")
        }
        if let hash = descriptor["coreml_sha256"] as? String, hash != (try sha256(native)) {
            throw LoomError("The completed model's native export checksum changed.")
        }
    }

    static func optionalJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    static func settings(from values: [String: Any], descriptor: [String: Any]?) -> TrainingSettings? {
        let architecture = descriptor?["architecture_config"] as? [String: Any]
        let declared = descriptor?["architecture"] as? String
        let configured = architecture?["id"] as? String
        let recorded = values["model_architecture"] as? String ?? values["modelArchitecture"] as? String
        let identity = declared ?? configured ?? recorded ?? NativeArchitecture.identifier
        guard [declared, configured, values["model_architecture"] as? String,
               values["modelArchitecture"] as? String].compactMap({ $0 }).allSatisfy({ $0 == identity }),
              TrainingNetwork(rawValue: identity) != nil,
              descriptor == nil || (declared != nil && configured != nil) else { return nil }
        let width = architecture?["width"] as? Int ?? values["width"] as? Int
            ?? values["model_width"] as? Int ?? values["modelWidth"] as? Int
        let blocks = architecture?["blocks"] as? Int ?? values["blocks"] as? Int
            ?? values["model_blocks"] as? Int ?? values["modelBlocks"] as? Int
        let size = values["image_size"] as? Int ?? descriptor?["image_size"] as? Int
        guard let width, let blocks, let size, (8...256).contains(width), (1...12).contains(blocks), (32...8192).contains(size) else { return nil }
        guard ["width", "model_width", "modelWidth"].allSatisfy({ key in
            guard let value = values[key] else { return true }
            return value as? Int == width
        }), ["blocks", "model_blocks", "modelBlocks"].allSatisfy({ key in
            guard let value = values[key] else { return true }
            return value as? Int == blocks
        }) else { return nil }
        let aliases = ["width": "modelWidth", "blocks": "modelBlocks", "lr_schedule": "learningRateSchedule", "minimum_lr_ratio": "minimumLearningRateRatio"]
        var converted: [String: Any] = [:]
        for (key, value) in values {
            let parts = key.split(separator: "_")
            let camel = parts.first.map(String.init).map { first in first + parts.dropFirst().map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined() } ?? key
            converted[aliases[key] ?? camel] = value
        }
        converted["modelWidth"] = width; converted["modelBlocks"] = blocks; converted["imageSize"] = size
        converted["modelArchitecture"] = identity
        if let requestedEpochs = values["requested_epochs"] as? Int { converted["epochs"] = requestedEpochs }
        // Completed native descriptors use the backend's display name, while settings store its stable raw value.
        if converted["device"] as? String == "metal" { converted["device"] = TrainingDevice.metal.rawValue }
        guard let data = try? JSONSerialization.data(withJSONObject: converted),
              let settings = try? JSONDecoder().decode(TrainingSettings.self, from: data) else { return nil }
        return settings
    }

    /// Resolve only registered immutable architectures, checking the outer
    /// identity against its constructor settings before any saved model loads.
    static func nativeArchitecture(from descriptor: [String: Any], map: MapRole) throws -> NativeArchitecture {
        guard MapRole.targets.contains(map), let identity = descriptor["architecture"] as? String,
              let configuration = descriptor["architecture_config"] as? [String: Any],
              configuration["id"] as? String == identity,
              let width = configuration["width"] as? Int, let blocks = configuration["blocks"] as? Int else {
            throw LoomError("The saved model has an inconsistent native architecture identity.")
        }
        let architecture = try NativeArchitecture(mapKind: map.rawValue, width: width, blocks: blocks, identifier: identity)
        guard (configuration["input_channels"] as? Int).map({ $0 == 3 }) ?? true,
              (configuration["output_channels"] as? Int).map({ $0 == architecture.outputChannels }) ?? true else {
            throw LoomError("The saved model's channels do not match its material map.")
        }
        if identity == NativeArchitecture.pyramidIdentifier {
            guard configuration["scale_channels"] as? [Int] == [width, width * 2, width * 4],
                  configuration["residual_scale"] as? Int == 4,
                  configuration["upsampling"] as? String == "nearest-2x-trim-to-skip" else {
                throw LoomError("The saved pyramid model's feature scales are inconsistent.")
            }
        }
        return architecture
    }

    static func sha256(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
        var hasher = SHA256()
        while let block = try handle.read(upToCount: 1024 * 1024), !block.isEmpty { hasher.update(data: block) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
