import Foundation
import LoomNative

public struct ModelTestResult: Sendable {
    public let directory: URL
    /// The unchanged original selected by the user, including external files.
    public let inputURL: URL
    public let outputURL: URL
    public let runID: String
    public let source: TrainingWeightSource
    public let map: MapRole
    public let width: Int
    public let height: Int
    public let tileSize: Int
    public let tileCount: Int
    public let note: String

    public init(directory: URL, inputURL: URL, outputURL: URL, runID: String, source: TrainingWeightSource,
                map: MapRole, width: Int, height: Int, tileSize: Int, tileCount: Int, note: String) {
        self.directory = directory; self.inputURL = inputURL; self.outputURL = outputURL
        self.runID = runID; self.source = source; self.map = map
        self.width = width; self.height = height; self.tileSize = tileSize; self.tileCount = tileCount; self.note = note
    }
}

extension LoomProject {
    /// Tests either saved deployment weights or a checkpoint without training,
    /// changing the active export version, or importing/relabeling the image.
    public func testModel(datasetID: String, map: MapRole, runID: String, source: TrainingWeightSource,
                          inputURL: URL, progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> ModelTestResult {
        guard !operationActive else { throw LoomError("Stop the current operation before testing a model.") }
        let model = try ModelTestSelection.load(runID: runID, source: source, datasetID: datasetID,
            map: map, projectFolder: projectFolder, projectID: document.id)
        let output = projectFolder.appendingPathComponent("model-tests/" + UUID().uuidString.lowercased(), isDirectory: true)
        let cancellation = NativeCancellation()
        operationActive = true; activeTraining = cancellation
        defer { operationActive = false; activeTraining = nil }
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await Task.detached(priority: .userInitiated) {
                try ModelTestWorker.run(model: model, inputURL: inputURL, output: output,
                    cancellation: cancellation, progress: progress)
            }.value
        } onCancel: { cancellation.cancel() }
    }
}

struct ModelTestSelection: Sendable {
    let runID: String
    let source: TrainingWeightSource
    let datasetID: String
    let map: MapRole
    let architecture: NativeArchitecture
    let imageSize: Int
    let weightsURL: URL
    let weightsSHA256: String

    static func load(runID: String, source: TrainingWeightSource, datasetID: String, map: MapRole,
                     projectFolder: URL, projectID: String) throws -> Self {
        guard MapRole.targets.contains(map) else { throw LoomError("Choose a depth, roughness or normal model to test.") }
        let directory = try TrainingRunFiles.runDirectory(runID, projectFolder: projectFolder)
        let descriptor = try TrainingRunFiles.descriptor(at: directory, source: source)
        guard descriptor["project_id"] as? String == projectID,
              descriptor["dataset_id"] as? String == datasetID,
              descriptor["kind"] as? String == "map", descriptor["map"] as? String == map.rawValue else {
            throw LoomError("Choose saved weights from the selected dataset and map.")
        }
        let architecture = try TrainingRunFiles.nativeArchitecture(from: descriptor, map: map)
        let values = (source == .checkpoint ? descriptor["settings"] : descriptor["training"]) as? [String: Any] ?? [:]
        guard let settings = TrainingRunFiles.settings(from: values, descriptor: descriptor) else {
            throw LoomError("This saved version has no consistent native model settings.")
        }
        let path = descriptor["weights_path"] as? String ?? (source == .checkpoint ? "checkpoint.safetensors" : "")
        let weightsURL = try TrainingRunFiles.containedFile(path, directory: directory)
        let resume = try TrainingRunFiles.resume(weights: weightsURL, projectFolder: projectFolder,
            projectID: projectID, datasetID: datasetID, map: map, settings: settings)
        guard resume.source == source else { throw LoomError("The selected saved weights do not match their source.") }
        return Self(runID: runID, source: source, datasetID: datasetID, map: map,
            architecture: architecture, imageSize: settings.imageSize,
            weightsURL: resume.weights, weightsSHA256: resume.sha256)
    }
}

enum ModelTestWorker {
    typealias Predictor = (_ input: [Float], _ height: Int, _ width: Int) throws -> [Float]

    /// An injected predictor exercises complete image/file contracts without GPU
    /// work. Shipping callers always construct the registered native network.
    static func run(model: ModelTestSelection, inputURL: URL, output: URL,
                    cancellation: NativeCancellation, progress: @escaping @Sendable (TrainingEvent) -> Void,
                    predictor suppliedPredictor: Predictor? = nil) throws -> ModelTestResult {
        defer { NativeImages.clearDecodedPixelCache() }
        try cancellation.check()
        let original = inputURL.standardizedFileURL.resolvingSymlinksInPath()
        let properties = try original.resourceValues(forKeys: [.isRegularFileKey])
        guard properties.isRegularFile == true else { throw LoomError("Choose an existing diffuse image to test.") }
        progress(.init(type: "status", message: "Reading original diffuse image…"))
        let inputHash = try NativeFactoryTraining.sha(original, cancellation: cancellation)
        guard model.weightsSHA256 == (try NativeFactoryTraining.sha(model.weightsURL, cancellation: cancellation)) else {
            throw LoomError("The selected weights changed before testing started. Choose the saved version again.")
        }
        let image = try autoreleasepool { try NativeImages.readRGB(from: original, cancellation: cancellation) }
        NativeImages.clearDecodedPixelCache()
        try cancellation.check()
        let predictor: Predictor
        if let suppliedPredictor { predictor = suppliedPredictor }
        else {
            progress(.init(type: "status", message: "Loading saved \(model.source == .checkpoint ? "checkpoint" : "model")…"))
            let weights = try NativeWeights.read(from: model.weightsURL)
            let session = try NativeTrainingSession(architecture: model.architecture,
                options: NativeTrainingOptions(), seed: 0, weights: weights, cancellation: cancellation)
            predictor = { input, height, width in
                try session.predict(input: input, batchSize: 1, height: height, width: width)
            }
        }
        // Fixed tile dimensions bound GPU scratch use even for an 8K original.
        // Source pixels retain their native coordinates and numeric precision.
        let tileSize = min(1024, model.imageSize)
        let plan = try ModelTestTilePlan(width: image.width, height: image.height, tileSize: tileSize)
        let pixels = try ModelTestImage.predict(image: image, plan: plan, map: model.map,
            cancellation: cancellation, progress: progress, predictor: predictor)
        try cancellation.check()
        let parent = output.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(".pending-" + output.lastPathComponent, isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: staging) }
        let name = model.map.rawValue + ".png"
        progress(.init(type: "status", message: "Saving \(image.width) × \(image.height) \(model.map.rawValue) output…"))
        try ModelTestImage.writePNG(pixels, channels: model.architecture.outputChannels,
            width: image.width, height: image.height, to: staging.appendingPathComponent(name), cancellation: cancellation)
        try cancellation.check()
        guard inputHash == (try NativeFactoryTraining.sha(original, cancellation: cancellation)) else {
            throw LoomError("The diffuse image changed during testing. Run the test again on the completed image.")
        }
        guard model.weightsSHA256 == (try NativeFactoryTraining.sha(model.weightsURL, cancellation: cancellation)) else {
            throw LoomError("The saved weights changed during testing. Choose the version again.")
        }
        let note = "Original \(image.width) × \(image.height) resolution; \(plan.tiles.count) overlapping native tile\(plan.tiles.count == 1 ? "" : "s"). "
            + "Overlap blending can differ from whole-image inference. "
            + (model.map == .normal ? "OpenGL normal vectors are normalized after blending. " : "")
            + "PNG uses fixed [0,1] values quantized to 16 bits, without per-output normalization."
        try NativeFactoryTraining.writeJSON(["schema": "loom-factory-model-test-v1", "run_id": model.runID,
            "weight_source": model.source.rawValue, "dataset_id": model.datasetID, "map": model.map.rawValue,
            "architecture": model.architecture.identifier, "input_path": original.path, "input_sha256": inputHash,
            "weights_sha256": model.weightsSHA256, "output_path": name,
            "width": image.width, "height": image.height, "tile_size": tileSize, "tile_count": plan.tiles.count,
            "overlap": plan.overlap, "stitching": "linear-feather-overlap-v1",
            "normal_convention": model.map == .normal ? "OpenGL" as Any : NSNull(),
            "created_at": ISO8601DateFormatter().string(from: .now), "note": note],
            to: staging.appendingPathComponent("test.json"))
        try cancellation.check()
        try FileManager.default.moveItem(at: staging, to: output)
        let result = ModelTestResult(directory: output, inputURL: original, outputURL: output.appendingPathComponent(name),
            runID: model.runID, source: model.source, map: model.map, width: image.width, height: image.height,
            tileSize: tileSize, tileCount: plan.tiles.count, note: note)
        progress(.init(type: "completed", message: "Saved full-resolution \(model.map.rawValue) output", progress: 1))
        return result
    }
}
