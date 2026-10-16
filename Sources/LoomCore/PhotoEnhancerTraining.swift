import CoreML
import Foundation
import LoomNative

public struct PhotoEnhancerSettings: Codable, Equatable, Sendable {
    public var kind: PhotoEnhancerKind = .repair
    public var width = 32
    public var epochs = 200
    public var minutes = 30
    public var learningRate = 0.0001
    public var seed: UInt64 = 9587
    public init() {}
    public var architecture: NativeArchitecture {
        get throws { try NativeArchitecture(mapKind: kind.product.rawValue, width: width, blocks: 1, identifier: NativeArchitecture.photoCompactIdentifier) }
    }
    public func validate() throws {
        guard [16, 22, 32, 46].contains(width), (1...10000).contains(epochs), (1...480).contains(minutes),
              learningRate.isFinite, (0.000001...0.01).contains(learningRate) else { throw LoomError("Choose a supported width, 1–480 minutes, 1–10000 epochs and a finite learning rate.") }
    }
}

/// A revision is immutable; checkpoint.json is the sole atomic latest pointer.
public struct PhotoEnhancerCheckpoint: Codable, Sendable {
    public var schema = "loom.photo-checkpoint.v1"
    public var runID: String
    public var parentRunID: String?
    public var settings: PhotoEnhancerSettings
    public var datasetFingerprint: String
    public var epoch: Int
    public var nextPair: Int
    public var updates: Int
    public var weights: String
    public var weightsSHA256: String
    public var optimizer: String
    public var optimizerSHA256: String
    public var bestWeights: String
    public var bestWeightsSHA256: String
    public var bestScore: Double
    public var bestUpdates: Int
    public var objective = "rgb-linear-detail-v1"
    public var status: String
}

public struct PhotoEnhancerModel: Codable, Sendable {
    public var schema = "loom.photo-enhancer.v1"
    public var name: String
    public var product: String
    public var architecture = NativeArchitecture.photoCompactIdentifier
    public var width: Int
    public var input = "rgb"
    public var output = "enhanced_rgb"
    public static let contract = "opaque sRGB numeric Float32 NCHW; input [0,1]; raw residual RGB output; no vector normalization"
    public var pixelContract = Self.contract
    public var modelFile = "Enhancer.mlmodel"
    public var modelSHA256: String
    public var weights = "weights.safetensors"
    public var weightsSHA256: String
    public var runID: String
    public var updates: Int
    public var qualified = false
}

extension LoomProject {
    public func trainPhotoEnhancer(dataset: URL, settings: PhotoEnhancerSettings, resume: URL? = nil,
                                   progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> URL {
        let lease = try TrainingQueueLease.requireAvailable(projectFolder: projectFolder)
        defer { withExtendedLifetime(lease) {} }
        guard !operationActive else { throw LoomError("Another operation is running in this project.") }
        try settings.validate()
        let id = UUID().uuidString.lowercased(), cancellation = NativeCancellation(), control = TrainingRunControl(runID: id)
        let output = projectFolder.appendingPathComponent("photo-runs/" + id)
        operationActive = true; activeTraining = cancellation; activeTrainingControl = control
        defer { control.finish(); operationActive = false; activeTraining = nil; activeTrainingControl = nil }
        return try await withTaskCancellationHandler {
            try await Task.detached(priority: .utility) {
                try PhotoEnhancerTrainer.run(dataset: dataset, output: output, settings: settings, resume: resume,
                    cancellation: cancellation, control: control, progress: progress)
            }.value
        } onCancel: { cancellation.cancel() }
    }
}

enum PhotoEnhancerTrainer {
    static func loadCheckpoint(_ file: URL) throws -> PhotoEnhancerCheckpoint {
        let value = try JSONDecoder().decode(PhotoEnhancerCheckpoint.self, from: Data(contentsOf: file))
        guard value.schema == "loom.photo-checkpoint.v1", value.objective == "rgb-linear-detail-v1",
              value.epoch >= 0, value.nextPair >= 0, value.updates >= 0, value.bestUpdates >= 0, value.bestUpdates <= value.updates, value.bestScore.isFinite else { throw LoomError("Unsupported photo checkpoint contract.") }
        return value
    }
    static func runRoot(_ file: URL) -> URL {
        let folder = file.deletingLastPathComponent()
        return folder.deletingLastPathComponent().lastPathComponent == "checkpoints" ? folder.deletingLastPathComponent().deletingLastPathComponent() : folder
    }
    static func checkedFile(_ name: String, hash: String, root: URL) throws -> URL {
        let file = try PhotoEnhancerData.contained(name, root: root)
        guard FileManager.default.fileExists(atPath: file.path) else { throw LoomError("Photo checkpoint tensor is missing: \(name). Choose an intact saved revision.") }
        guard try NativeFileHash.sha256(file) == hash else { throw LoomError("Photo checkpoint tensor checksum changed.") }
        return file
    }
    static func rgb(_ file: URL) throws -> [Float] {
        let image = try NativeImages.readRGB(from: file)
        guard image.width == 1024, image.height == 1024, image.channels == 3 else { throw LoomError("Photo training accepts only native 1K RGB pairs.") }
        let pixels = 1024*1024
        return (0..<3).flatMap { channel in (0..<pixels).map { image.values[$0*3 + channel] } }
    }
    static func checkCurrent(_ pair: PhotoTrainingPair) throws {
        guard try PhotoEnhancerData.read(pair.input) == pair.metadata,
              try NativeFileHash.sha256(pair.input) == pair.inputSHA256,
              try NativeFileHash.sha256(pair.target) == pair.targetSHA256 else {
            throw LoomError("Photo pair pixels or metadata changed during training. Latest completed update is preserved.")
        }
    }
    /// Checks the actual starting weights at the production grid before teaching.
    /// This is execution evidence, never learned-quality acceptance.
    static func preflight(_ session: NativeTrainingSession, pair: PhotoTrainingPair, output: URL) throws {
        let input = try rgb(pair.input), architecture = session.architecture
        let file = output.appendingPathComponent("preflight.mlmodel")
        let compiled = try NativeCoreML.export(weights: session.weights(), architecture: architecture, imageSize: 1024, to: file)
        defer { try? FileManager.default.removeItem(at: file); try? FileManager.default.removeItem(at: compiled) }
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        let array = try MLMultiArray(shape: [1,3,1024,1024], dataType: .float32)
        input.withUnsafeBufferPointer { array.dataPointer.assumingMemoryBound(to: Float.self).update(from: $0.baseAddress!, count: input.count) }
        let prediction = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)]))
        guard let tensor = prediction.featureValue(for: "enhanced_rgb")?.multiArrayValue,
              tensor.dataType == .float32, tensor.shape.map(\.intValue) == [1,3,1024,1024] else { throw LoomError("Photo preflight returned a different RGB interface.") }
        let native = try session.predict(input: input, batchSize: 1, height: 1024, width: 1024)
        let strides = tensor.strides.map(\.intValue), cpu = tensor.dataPointer.assumingMemoryBound(to: Float.self)
        var maximum: Float = 0
        for c in 0..<3 { for p in 0..<1024*1024 {
            let value = cpu[c*strides[1] + (p/1024)*strides[2] + (p%1024)*strides[3]]
            guard value.isFinite else { throw LoomError("Photo CPU preflight produced non-finite RGB.") }
            maximum = max(maximum, abs(value - native[c*1024*1024+p]))
        } }
        guard maximum <= 0.0002 else { throw LoomError("Native/Core ML photo preflight differs by \(maximum), above 0.0002. No training updates were applied.") }
        try JSONSerialization.data(withJSONObject: ["schema":"loom.photo-preflight.v1", "grid":1024, "product":architecture.mapKind,
            "starting_update":session.updateCount, "starting_coreml_sha256":try NativeFileHash.sha256(file),
            "maximum_absolute_cpu_gpu_error":maximum, "passed":true, "learned_quality_claim":false], options: [.prettyPrinted,.sortedKeys])
            .write(to: output.appendingPathComponent("preflight.json"), options: .atomic)
    }
    static func order(_ pairs: [PhotoTrainingPair], seed: UInt64, epoch: Int) -> [PhotoTrainingPair] {
        pairs.sorted { syntheticHash("\(seed):\(epoch):\($0.input.lastPathComponent)") < syntheticHash("\(seed):\(epoch):\($1.input.lastPathComponent)") }
    }
    static func run(dataset: URL, output: URL, settings: PhotoEnhancerSettings, resume: URL?,
                    cancellation: NativeCancellation, control: TrainingRunControl,
                    progress: @escaping @Sendable (TrainingEvent) -> Void) throws -> URL {
        guard !FileManager.default.fileExists(atPath: dataset.appendingPathComponent("IMPORT-INCOMPLETE").path) else { throw LoomError("This dataset import is incomplete; import into a new folder.") }
        let data = try PhotoEnhancerData.scan(dataset, kind: settings.kind, cancellation: cancellation)
        guard !data.training.isEmpty, !data.validation.isEmpty else { throw LoomError("This stage needs both training and whole-material validation pairs. Change the synthesis percentage/seed or import more physical surfaces; do not split crops of one surface.") }
        var previous: PhotoEnhancerCheckpoint?, starting: [String: NativeTensor]?, moments: [String: NativeTensor]?
        var best: [String: NativeTensor]?, bestScore = Double.greatestFiniteMagnitude, bestUpdates = 0
        if let resume {
            let saved = try loadCheckpoint(resume), root = runRoot(resume)
            // Continue exactly. Budget/total epochs may increase; teaching semantics cannot change.
            guard saved.settings.kind == settings.kind, saved.settings.width == settings.width,
                  saved.settings.learningRate == settings.learningRate, saved.settings.seed == settings.seed,
                  saved.datasetFingerprint == data.fingerprint else { throw LoomError("This continuation changes the product, data, seed, width or optimizer settings. Start a fresh model instead.") }
            starting = try NativeWeights.read(from: checkedFile(saved.weights, hash: saved.weightsSHA256, root: root))
            moments = try NativeWeights.read(from: checkedFile(saved.optimizer, hash: saved.optimizerSHA256, root: root))
            best = try NativeWeights.read(from: checkedFile(saved.bestWeights, hash: saved.bestWeightsSHA256, root: root))
            previous = saved; bestScore = saved.bestScore; bestUpdates = saved.bestUpdates
        }
        let architecture = try settings.architecture
        let session = try NativeTrainingSession(architecture: architecture,
            options: NativeTrainingOptions(weightDecay: 0, scalarLoss: .linearDetailV1), seed: settings.seed,
            weights: starting, optimizerState: moments, initialUpdateCount: previous?.updates ?? 0, cancellation: cancellation)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(settings).write(to: output.appendingPathComponent("settings.json"), options: .atomic)
        // A dataset location is a setting, never a frozen list of members.
        try Data(dataset.path.utf8).write(to: output.appendingPathComponent("dataset-location.txt"), options: .atomic)
        let runID = output.lastPathComponent
        var epoch = previous?.epoch ?? 0, nextPair = previous?.nextPair ?? 0
        guard nextPair <= data.training.count else { throw LoomError("Invalid saved photo sampler cursor.") }
        let began = ProcessInfo.processInfo.systemUptime
        var lastSave = began
        func evaluate() throws -> Double {
            var score = 0.0
            for pair in data.validation {
                try cancellation.check()
                try checkCurrent(pair)
                let prediction = try session.predict(input: rgb(pair.input), batchSize: 1, height: 1024, width: 1024)
                let target = try rgb(pair.target)
                score += zip(prediction, target).reduce(0.0) { $0 + abs(Double($1.0) - Double($1.1)) } / Double(target.count)
            }
            return score / Double(data.validation.count)
        }
        func save(_ status: String, preserve: Bool = false) throws {
            let revision = "checkpoints/" + UUID().uuidString.lowercased()
            let folder = output.appendingPathComponent(revision)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try NativeWeights.write(session.weights(), to: folder.appendingPathComponent("weights.safetensors"))
            try NativeWeights.write(session.optimizerState(), to: folder.appendingPathComponent("optimizer.safetensors"))
            try NativeWeights.write(best ?? session.weights(), to: folder.appendingPathComponent("best.safetensors"))
            let record = PhotoEnhancerCheckpoint(runID: runID, parentRunID: previous?.runID, settings: settings,
                datasetFingerprint: data.fingerprint, epoch: epoch, nextPair: nextPair, updates: session.updateCount,
                weights: revision + "/weights.safetensors", weightsSHA256: try NativeFileHash.sha256(folder.appendingPathComponent("weights.safetensors")),
                optimizer: revision + "/optimizer.safetensors", optimizerSHA256: try NativeFileHash.sha256(folder.appendingPathComponent("optimizer.safetensors")),
                bestWeights: revision + "/best.safetensors", bestWeightsSHA256: try NativeFileHash.sha256(folder.appendingPathComponent("best.safetensors")),
                bestScore: bestScore, bestUpdates: bestUpdates, status: status)
            let bytes = try encoder.encode(record)
            try bytes.write(to: folder.appendingPathComponent("checkpoint.json"), options: .atomic)
            if preserve { try Data().write(to: folder.appendingPathComponent("MANUAL"), options: .atomic) }
            try bytes.write(to: output.appendingPathComponent("checkpoint.json"), options: .atomic)
            // Retain latest+best together and explicit manual versions. Only
            // obsolete automatic revisions owned by THIS new run are recyclable.
            PhotoCheckpointRetention.prune(run: output, keeping: folder)
            lastSave = ProcessInfo.processInfo.systemUptime
        }
        var status = "completed"
        do {
            progress(.init(type: "status", message: "Checking actual starting weights: native 1K GPU versus exported Core ML CPU RGB…"))
            try preflight(session, pair: data.training[0], output: output)
            if best == nil { bestScore = try evaluate(); best = try session.weights() }
            try save("running")
            progress(.init(type: "status", message: "\(settings.kind.title) · \(architecture.parameterCount) parameters · \(data.summary) · RGB residual, native 1K"))
            training: while epoch < settings.epochs {
                let current = try PhotoEnhancerData.scan(dataset, kind: settings.kind, cancellation: cancellation)
                guard current.fingerprint == data.fingerprint else { throw LoomError("Photo data or metadata changed during training; checkpoint preserved. Restart with a deliberate fresh run.") }
                let pairs = order(data.training, seed: settings.seed, epoch: epoch)
                while nextPair < pairs.count {
                    if ProcessInfo.processInfo.systemUptime - began >= Double(settings.minutes)*60 { status = "timed_save_stop"; break training }
                    try cancellation.check()
                    let pair = pairs[nextPair], stepBegan = ProcessInfo.processInfo.systemUptime
                    try checkCurrent(pair)
                    let loss = try session.train(input: rgb(pair.input), target: rgb(pair.target), batchSize: 1, height: 1024, width: 1024)
                    _ = try session.applyGradients(learningRate: settings.learningRate)
                    nextPair += 1
                    progress(.init(type: "status", message: "\(settings.kind.title) · epoch \(epoch+1)/\(settings.epochs) · pair \(nextPair)/\(pairs.count) · update \(session.updateCount) · loss \(String(format: "%.5f", loss)) · \(String(format: "%.2f", ProcessInfo.processInfo.systemUptime-stepBegan)) s/update"))
                    if let stop = control.beginRequest() {
                        do {
                            try save(stop ? "saved_stopped" : "running", preserve: true)
                            control.complete(.success(.init(runDirectory: output, completedUpdates: session.updateCount, stopped: stop)))
                        } catch { control.complete(.failure(error)); throw error }
                        if stop { status = "saved_stopped"; break training }
                    }
                    if ProcessInfo.processInfo.systemUptime - lastSave >= 60 { try save("running") }
                }
                epoch += 1; nextPair = 0
                let score = try evaluate()
                if score < bestScore { bestScore = score; best = try session.weights(); bestUpdates = session.updateCount }
                try save("running")
                progress(.init(type: "status", message: "Validation RGB MAE \(String(format: "%.5f", score)); diagnostic only. Best export: update \(bestUpdates)."))
            }
        } catch {
            try? save(error is CancellationError ? "cancelled_saved" : "failed_saved")
            throw error
        }
        if session.updateCount > bestUpdates {
            let score = try evaluate()
            if score < bestScore { bestScore = score; best = try session.weights(); bestUpdates = session.updateCount }
        }
        try save(status)
        let exported = output.appendingPathComponent("export")
        try FileManager.default.createDirectory(at: exported, withIntermediateDirectories: true)
        let tensors = try best ?? session.weights()
        try NativeWeights.write(tensors, to: exported.appendingPathComponent("weights.safetensors"))
        let compiled = try NativeCoreML.export(weights: tensors, architecture: architecture, imageSize: 1024, to: exported.appendingPathComponent("Enhancer.mlmodel"))
        defer { try? FileManager.default.removeItem(at: compiled) }
        let descriptor = PhotoEnhancerModel(name: "\(settings.kind.title) · \(runID.prefix(8)) · best update \(bestUpdates)", product: settings.kind.product.rawValue,
            width: settings.width, modelSHA256: try NativeFileHash.sha256(exported.appendingPathComponent("Enhancer.mlmodel")),
            weightsSHA256: try NativeFileHash.sha256(exported.appendingPathComponent("weights.safetensors")), runID: runID, updates: bestUpdates)
        try encoder.encode(descriptor).write(to: exported.appendingPathComponent("enhancer.json"), options: .atomic)
        progress(.init(type: "summary", message: "Saved latest resumable checkpoint and exact best RGB export: \(exported.path). Unapproved candidate; inspect real photos and clean-input identity before use."))
        return output
    }
}

/// Only this run's unreferenced automatic revisions are disposable.
enum PhotoCheckpointRetention {
    static func prune(run: URL, keeping: URL) {
        let root = run.appendingPathComponent("checkpoints").standardizedFileURL
        guard keeping.deletingLastPathComponent().standardizedFileURL.path == root.path,
              (try? root.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == false,
              let entries = try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey]) else { return }
        for entry in entries where entry.standardizedFileURL.path != keeping.standardizedFileURL.path && UUID(uuidString: entry.lastPathComponent) != nil {
            let attributes = try? entry.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            if attributes?.isDirectory == true, attributes?.isSymbolicLink == false,
               !FileManager.default.fileExists(atPath: entry.appendingPathComponent("MANUAL").path) { try? FileManager.default.removeItem(at: entry) }
        }
    }
}
