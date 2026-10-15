import Foundation
import LoomNative

public struct TrainingCheckpoint: Identifiable, Sendable, Equatable {
    public var id: String { directory.path + (revisionID.map { "#" + $0 } ?? "") }
    public var runID: String { directory.lastPathComponent }
    public let directory: URL
    public let datasetID: String
    public let map: MapRole
    public let modelWidth: Int
    public let modelBlocks: Int
    public var modelArchitecture: String = NativeArchitecture.identifier
    public let imageSize: Int
    public let exaggerateFeatures: Bool
    public let epochsCompleted: Int
    public let status: String
    public let modifiedAt: Date
    public var updatesCompleted: Int = 0
    public var restoresOptimizer: Bool = false
    public var scalarLoss: NativeScalarLoss = .absoluteV1
    public var priorEpochsCompleted: Int = 0
    public var priorUpdatesCompleted: Int = 0
    public var trainingElapsedSeconds: Double? = nil
    /// Nil identifies a legacy run's current checkpoint; saved revisions use immutable IDs.
    public var revisionID: String? = nil
    public var totalEpochsCompleted: Int { total(epochsCompleted, priorEpochsCompleted) }
    public var totalUpdatesCompleted: Int { total(updatesCompleted, priorUpdatesCompleted) }
    public var parameterCount: Int { TrainingSettings(modelWidth: modelWidth, modelBlocks: modelBlocks,
        modelArchitecture: modelArchitecture).parameterCount(map: map) }
    private func total(_ current: Int, _ prior: Int) -> Int {
        let (result, overflow) = max(0, current).addingReportingOverflow(max(0, prior))
        return overflow ? Int.max : result
    }
}

extension LoomProject {
    public func checkpointVersions(runID: String) throws -> [TrainingCheckpoint] {
        let directory = try TrainingRunFiles.runDirectory(runID, projectFolder: projectFolder)
        return try loadCheckpointVersions(directory: directory)
    }

    func loadCheckpointVersions(directory: URL) throws -> [TrainingCheckpoint] {
        let files = try CheckpointArchive.descriptors(in: directory)
        var seen: Set<String> = []
        return files.compactMap { path in
            guard let checkpoint = loadCheckpoint(at: path, directory: directory) else { return nil }
            let identity = checkpoint.revisionID ?? "latest"
            return seen.insert(identity).inserted ? checkpoint : nil
        }.sorted {
            if $0.totalUpdatesCompleted != $1.totalUpdatesCompleted { return $0.totalUpdatesCompleted > $1.totalUpdatesCompleted }
            return $0.modifiedAt > $1.modifiedAt
        }
    }

    func loadCheckpoints() throws -> [TrainingCheckpoint] {
        let runs = projectFolder.appendingPathComponent("runs", isDirectory: true)
        guard FileManager.default.fileExists(atPath: runs.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: runs, includingPropertiesForKeys: nil).compactMap { directory in
            guard (try? TrainingRunFiles.ownedRun(directory, projectFolder: projectFolder)) != nil else { return nil }
            return loadCheckpoint(at: directory.appendingPathComponent("checkpoint.json"), directory: directory)
                ?? (try? loadCheckpointVersions(directory: directory).first)
        }.sorted { $0.modifiedAt > $1.modifiedAt }
    }

    private func loadCheckpoint(at path: URL, directory: URL) -> TrainingCheckpoint? {
            guard let file = try? TrainingRunFiles.containedFile(path.lastPathComponent, directory: directory),
                  let data = try? Data(contentsOf: file),
                  let descriptor = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  descriptor["schema"] as? String == "loom-factory-checkpoint-v1",
                  descriptor["project_id"] as? String == document.id,
                  descriptor["kind"] as? String == "map",
                  let datasetID = descriptor["dataset_id"] as? String,
                  let mapName = descriptor["map"] as? String, let map = MapRole(rawValue: mapName), MapRole.targets.contains(map),
                  let architecture = descriptor["architecture_config"] as? [String: Any],
                  let architectureID = architecture["id"] as? String,
                  descriptor["architecture"] as? String == architectureID,
                  let width = architecture["width"] as? Int, (8...256).contains(width),
                  let blocks = architecture["blocks"] as? Int, (1...12).contains(blocks),
                  let savedSettings = descriptor["settings"] as? [String: Any],
                  let scalarLoss = NativeScalarLoss(rawValue: savedSettings["scalar_loss"] as? String ?? NativeScalarLoss.absoluteV1.rawValue),
                  let size = savedSettings["image_size"] as? Int, (32...8192).contains(size),
                  let weights = try? TrainingRunFiles.containedFile(descriptor["weights_path"] as? String ?? "checkpoint.safetensors", directory: directory),
                  FileManager.default.fileExists(atPath: weights.path),
                  let native = try? TrainingRunFiles.nativeArchitecture(from: descriptor, map: map),
                  native.identifier == architectureID, native.width == width, native.blocks == blocks,
                  !native.isCompact || (scalarLoss.isCompactObjective && size % 8 == 0) else { return nil }
            if path.lastPathComponent != "checkpoint.json" {
                guard let id = descriptor["checkpoint_id"] as? String,
                      (try? CheckpointArchive.filename(id)) == path.lastPathComponent else { return nil }
            }
            let modified = (descriptor["saved_at"] as? String).flatMap { ISO8601DateFormatter().date(from: $0) }
                ?? (try? path.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return TrainingCheckpoint(directory: directory, datasetID: datasetID, map: map, modelWidth: width,
                modelBlocks: blocks, modelArchitecture: architectureID, imageSize: size, exaggerateFeatures: savedSettings["exaggerate_features"] as? Bool ?? false,
                epochsCompleted: descriptor["epochs_completed"] as? Int ?? 0, status: descriptor["status"] as? String ?? "saved", modifiedAt: modified,
                updatesCompleted: descriptor["updates"] as? Int ?? 0, restoresOptimizer: descriptor["optimizer_path"] as? String != nil,
                scalarLoss: scalarLoss,
                priorEpochsCompleted: descriptor["prior_epochs"] as? Int ?? 0,
                priorUpdatesCompleted: descriptor["prior_updates"] as? Int ?? 0,
                trainingElapsedSeconds: (descriptor["elapsed_seconds"] as? Double).map {
                    $0 + (descriptor["prior_duration_seconds"] as? Double ?? 0)
                }, revisionID: descriptor["checkpoint_id"] as? String)
    }
}

public struct TrainingCheckpointReceipt: Sendable, Equatable {
    public let runDirectory: URL
    public let completedUpdates: Int
    public let stopped: Bool
}

/// Requests never execute GPU work on the project actor. The serial training
/// owner consumes them only at a completed optimizer update.
final class TrainingRunControl: @unchecked Sendable {
    let runID: String?
    private let lock = NSLock()
    private var requests: [(stop: Bool, continuation: CheckedContinuation<TrainingCheckpointReceipt, Error>)] = []
    private var inFlight: [(stop: Bool, continuation: CheckedContinuation<TrainingCheckpointReceipt, Error>)] = []
    private var saving = false
    private var finished = false

    init(runID: String? = nil) { self.runID = runID }

    func request(stop: Bool) async throws -> TrainingCheckpointReceipt {
        try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            if finished {
                lock.unlock()
                continuation.resume(throwing: LoomError("This training run has finished. Choose its saved version in Model Versions."))
            } else {
                requests.append((stop, continuation))
                lock.unlock()
            }
        }
    }

    var hasPendingRequest: Bool { lock.withLock { !saving && !requests.isEmpty } }
    func beginRequest() -> Bool? {
        lock.withLock {
            guard !saving, !requests.isEmpty else { return nil }
            saving = true
            inFlight = requests; requests.removeAll()
            return inFlight.contains { $0.stop }
        }
    }

    func complete(_ result: Result<TrainingCheckpointReceipt, Error>) {
        let pending = lock.withLock {
            let pending = inFlight
            inFlight.removeAll(); saving = false
            return pending
        }
        for request in pending { request.continuation.resume(with: result) }
    }

    func finish() {
        let pending = lock.withLock {
            finished = true
            let pending = requests + inFlight
            requests.removeAll(); inFlight.removeAll(); saving = false
            return pending
        }
        for request in pending {
            request.continuation.resume(throwing: LoomError("Training finished before this checkpoint request could be saved. Check the latest saved version."))
        }
    }
}

public struct TrainingSavedStop: Error, Sendable {}

/// Every successful save keeps its immutable descriptor and tensor revisions.
/// The atomic checkpoint.json alias continues to identify the latest save.
enum TrainingCheckpointFiles {
    static func write(weights: [String: NativeTensor], optimizer: [String: NativeTensor],
                      bestWeights: [String: NativeTensor]? = nil,
                      descriptor: [String: Any], directory: URL) throws {
        let revision = UUID().uuidString.lowercased()
        let weightsName = "checkpoint-\(revision).safetensors"
        let optimizerName = "optimizer-\(revision).safetensors"
        let weightsURL = directory.appendingPathComponent(weightsName)
        let optimizerURL = directory.appendingPathComponent(optimizerName)
        let bestName = "best-\(revision).safetensors"
        let bestURL = directory.appendingPathComponent(bestName)
        var committed = false
        defer {
            if !committed {
                try? FileManager.default.removeItem(at: weightsURL)
                try? FileManager.default.removeItem(at: optimizerURL)
                try? FileManager.default.removeItem(at: bestURL)
            }
        }
        try NativeWeights.write(weights, to: weightsURL, metadata: ["schema": "loom-factory-checkpoint-v1"])
        try NativeWeights.write(optimizer, to: optimizerURL, metadata: ["schema": "loom-factory-optimizer-v1"])
        var value = descriptor
        value["checkpoint_id"] = revision
        value["weights_path"] = weightsName; value["optimizer_path"] = optimizerName
        value["weights_sha256"] = try TrainingRunFiles.sha256(weightsURL)
        value["optimizer_sha256"] = try TrainingRunFiles.sha256(optimizerURL)
        if let bestWeights {
            try NativeWeights.write(bestWeights, to: bestURL, metadata: ["schema": "loom-factory-best-weights-v1"])
            value["best_weights_path"] = bestName
            value["best_weights_sha256"] = try TrainingRunFiles.sha256(bestURL)
        }
        guard JSONSerialization.isValidJSONObject(value) else { throw LoomError("The checkpoint descriptor contains invalid values.") }
        try CheckpointArchive.preserveLatest(in: directory)
        try CheckpointArchive.publish(value, revisionID: revision, in: directory)
        committed = true
        try NativeFactoryTraining.writeJSON(value, to: directory.appendingPathComponent("checkpoint.json"))
    }
}

struct TrainingCheckpointCursor: Codable, Sendable {
    var epoch: Int
    var nextWindow: Int
    var order: [String]
    var crops: [NativeCropTransform]
    var epochLoss: Double
    var imageCount: Int
    var epochUpdates: Int
    var clippedUpdates: Int
    var gradientNormSum: Double
    var peakGradientNorm: Double
    var augmentation: NativeAugmentationPlan
    var randomState: UInt64
    var sourceFingerprint: String
    var bestLoss: Double?
    var bestEpoch: Int?
    var staleEpochs: Int
    var stoppedEarly: Bool? = nil

    static func compatibleSettings(_ saved: [String: Any], current: [String: Any], keys: [String]? = nil) -> Bool {
        let executionOnly: Set<String> = ["checkpoint_every_epochs", "requested_seed", "cache_images_in_memory", "image_cache_limit_mb"]
        let selected = keys ?? current.keys.filter { !executionOnly.contains($0) }
        var saved = saved
        if saved["scalar_loss"] == nil, current["scalar_loss"] != nil {
            saved["scalar_loss"] = NativeScalarLoss.absoluteV1.rawValue
        }
        let lhs = Dictionary(uniqueKeysWithValues: selected.compactMap { key in saved[key].map { (key, $0) } })
        let rhs = Dictionary(uniqueKeysWithValues: selected.compactMap { key in current[key].map { (key, $0) } })
        return NSDictionary(dictionary: lhs).isEqual(to: rhs)
    }
}
