import Foundation
import LoomNative

public struct TrainingCheckpoint: Identifiable, Sendable, Equatable {
    public var id: String { directory.path }
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
    public var parameterCount: Int { TrainingSettings(modelWidth: modelWidth, modelBlocks: modelBlocks,
        modelArchitecture: modelArchitecture).parameterCount(map: map) }
}

extension LoomProject {
    func loadCheckpoints() throws -> [TrainingCheckpoint] {
        let runs = projectFolder.appendingPathComponent("runs", isDirectory: true)
        guard FileManager.default.fileExists(atPath: runs.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: runs, includingPropertiesForKeys: nil).compactMap { directory in
            guard (try? TrainingRunFiles.ownedRun(directory, projectFolder: projectFolder)) != nil else { return nil }
            let path = directory.appendingPathComponent("checkpoint.json")
            guard let data = try? Data(contentsOf: path),
                  let descriptor = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  descriptor["schema"] as? String == "loom-factory-checkpoint-v1",
                  descriptor["project_id"] as? String == document.id,
                  descriptor["kind"] as? String == "map",
                  let datasetID = descriptor["dataset_id"] as? String,
                  let mapName = descriptor["map"] as? String, let map = MapRole(rawValue: mapName), MapRole.targets.contains(map),
                  let architecture = descriptor["architecture_config"] as? [String: Any],
                  let architectureID = architecture["id"] as? String,
                  descriptor["architecture"] as? String == architectureID,
                  [NativeArchitecture.identifier, NativeArchitecture.pyramidIdentifier].contains(architectureID),
                  let width = architecture["width"] as? Int, (8...256).contains(width),
                  let blocks = architecture["blocks"] as? Int, (1...12).contains(blocks),
                  let savedSettings = descriptor["settings"] as? [String: Any],
                  let scalarLoss = NativeScalarLoss(rawValue: savedSettings["scalar_loss"] as? String ?? NativeScalarLoss.absoluteV1.rawValue),
                  let size = savedSettings["image_size"] as? Int, (32...8192).contains(size),
                  let weights = try? TrainingRunFiles.containedFile(descriptor["weights_path"] as? String ?? "checkpoint.safetensors", directory: directory),
                  FileManager.default.fileExists(atPath: weights.path) else { return nil }
            let modified = (try? path.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return TrainingCheckpoint(directory: directory, datasetID: datasetID, map: map, modelWidth: width,
                modelBlocks: blocks, modelArchitecture: architectureID, imageSize: size, exaggerateFeatures: savedSettings["exaggerate_features"] as? Bool ?? false,
                epochsCompleted: descriptor["epochs_completed"] as? Int ?? 0, status: descriptor["status"] as? String ?? "saved", modifiedAt: modified,
                updatesCompleted: descriptor["updates"] as? Int ?? 0, restoresOptimizer: descriptor["optimizer_path"] as? String != nil,
                scalarLoss: scalarLoss)
        }.sorted { $0.modifiedAt > $1.modifiedAt }
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

/// Immutable tensor files are published by one atomic descriptor replacement.
/// A failed write leaves the last checkpoint descriptor and tensors intact.
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
        let old = TrainingRunFiles.optionalJSON(directory.appendingPathComponent("checkpoint.json"))
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
        value["weights_path"] = weightsName; value["optimizer_path"] = optimizerName
        value["weights_sha256"] = try TrainingRunFiles.sha256(weightsURL)
        value["optimizer_sha256"] = try TrainingRunFiles.sha256(optimizerURL)
        if let bestWeights {
            try NativeWeights.write(bestWeights, to: bestURL, metadata: ["schema": "loom-factory-best-weights-v1"])
            value["best_weights_path"] = bestName
            value["best_weights_sha256"] = try TrainingRunFiles.sha256(bestURL)
        }
        try NativeFactoryTraining.writeJSON(value, to: directory.appendingPathComponent("checkpoint.json"))
        committed = true
        // Only our immutable revisions may be reclaimed, after publication.
        for key in ["weights_path", "optimizer_path", "best_weights_path"] {
            guard let path = old?[key] as? String, !path.contains("/"),
                  (path.hasPrefix("checkpoint-") || path.hasPrefix("optimizer-") || path.hasPrefix("best-")),
                  let previous = try? TrainingRunFiles.containedFile(path, directory: directory) else { continue }
            try? FileManager.default.removeItem(at: previous)
        }
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
        let selected = keys ?? current.keys.filter { $0 != "checkpoint_every_epochs" }
        var saved = saved
        if saved["scalar_loss"] == nil, current["scalar_loss"] != nil {
            saved["scalar_loss"] = NativeScalarLoss.absoluteV1.rawValue
        }
        let lhs = Dictionary(uniqueKeysWithValues: selected.compactMap { key in saved[key].map { (key, $0) } })
        let rhs = Dictionary(uniqueKeysWithValues: selected.compactMap { key in current[key].map { (key, $0) } })
        return NSDictionary(dictionary: lhs).isEqual(to: rhs)
    }
}
