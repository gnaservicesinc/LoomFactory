import Foundation
import ImageIO
import LoomNative
import Metal

public struct ModelTestResult: Sendable {
    public let directory: URL
    /// The unchanged original selected by the user, including external files.
    public let inputURL: URL
    /// Canonical model output: exact Float32 OpenEXR pixels.
    public let outputURL: URL
    /// A fixed-scale sixteen-bit PNG for ordinary image viewers.
    public let previewURL: URL
    public let runID: String
    public let source: TrainingWeightSource
    public let map: MapRole
    public let width: Int
    public let height: Int
    public let tileSize: Int
    public let tileCount: Int
    public let note: String
    public let checkpointID: String?
    public let inputSHA256: String?

    public init(directory: URL, inputURL: URL, outputURL: URL, runID: String, source: TrainingWeightSource,
                map: MapRole, width: Int, height: Int, tileSize: Int, tileCount: Int, note: String,
                previewURL: URL? = nil, checkpointID: String? = nil, inputSHA256: String? = nil) {
        self.directory = directory; self.inputURL = inputURL; self.outputURL = outputURL
        self.previewURL = previewURL ?? outputURL
        self.runID = runID; self.source = source; self.map = map
        self.width = width; self.height = height; self.tileSize = tileSize; self.tileCount = tileCount; self.note = note
        self.checkpointID = checkpointID
        self.inputSHA256 = inputSHA256
    }
}

extension LoomProject {
    /// Tests either saved deployment weights or a checkpoint without training,
    /// changing the active export version, or importing/relabeling the image.
    public func testModel(datasetID: String, map: MapRole, runID: String, source: TrainingWeightSource,
                          inputURL: URL, checkpointID: String? = nil,
                          progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> ModelTestResult {
        guard !operationActive else { throw LoomError("Stop the current operation before testing a model.") }
        let model = try ModelTestSelection.load(runID: runID, source: source, datasetID: datasetID,
            map: map, projectFolder: projectFolder, projectID: document.id, checkpointID: checkpointID)
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
    var checkpointID: String? = nil

    static func load(runID: String, source: TrainingWeightSource, datasetID: String, map: MapRole,
                     projectFolder: URL, projectID: String, checkpointID: String? = nil) throws -> Self {
        guard MapRole.targets.contains(map) else { throw LoomError("Choose a depth, roughness or normal model to test.") }
        let directory = try TrainingRunFiles.runDirectory(runID, projectFolder: projectFolder)
        let descriptor = try TrainingRunFiles.descriptor(at: directory, source: source, checkpointID: checkpointID)
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
            projectID: projectID, datasetID: datasetID, map: map, settings: settings, checkpointID: checkpointID)
        guard resume.source == source else { throw LoomError("The selected saved weights do not match their source.") }
        return Self(runID: runID, source: source, datasetID: datasetID, map: map,
            architecture: architecture, imageSize: settings.imageSize,
            weightsURL: resume.weights, weightsSHA256: resume.sha256, checkpointID: descriptor["checkpoint_id"] as? String)
    }
}

/// An advisory conservative forecast, not a promise about allocator behavior.
/// Fresh width-32 forward/evaluation probes used 5.06 GB at 1K and 20.17 GB at
/// 2K. 256 bytes per width-pixel leaves headroom above both measured slopes.
enum CompactModelMemoryAdmission {
    static func forecast(width: Int, height: Int, featureWidth: Int, parameterCount: Int) -> UInt64 {
        guard width > 0, height > 0, featureWidth > 0, parameterCount >= 0 else { return .max }
        var activation = UInt64(width)
        for factor in [UInt64(height), UInt64(featureWidth), 256] {
            let product = activation.multipliedReportingOverflow(by: factor)
            guard !product.overflow else { return .max }
            activation = product.partialValue
        }
        let weights = UInt64(parameterCount).multipliedReportingOverflow(by: 12)
        guard !weights.overflow else { return .max }
        let total = activation.addingReportingOverflow(weights.partialValue)
        return total.overflow ? .max : total.partialValue
    }

    static func budget(physical: UInt64, recommendedMetal: UInt64?, allocatedMetal: UInt64) -> UInt64 {
        let machine = physical - physical / 4
        guard let recommendedMetal else { return machine }
        return min(machine, recommendedMetal > allocatedMetal ? recommendedMetal - allocatedMetal : 0)
    }
}

enum ModelTestWorker {
    typealias Predictor = (_ input: [Float], _ height: Int, _ width: Int) throws -> [Float]

    /// An injected predictor exercises complete image/file contracts without GPU
    /// work. Shipping callers always construct the registered native network.
    static func run(model: ModelTestSelection, inputURL: URL, output: URL,
                    cancellation: NativeCancellation, progress: @escaping @Sendable (TrainingEvent) -> Void,
                    compactMemoryBudget suppliedCompactMemoryBudget: UInt64? = nil,
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
        var memoryAdmission: [String: Any]?
        var admittedDimensions: (width: Int, height: Int)?
        if model.architecture.isCompact {
            let options = [kCGImageSourceShouldCache: false] as CFDictionary
            guard let source = CGImageSourceCreateWithURL(original as CFURL, options),
                  let metadata = CGImageSourceCopyPropertiesAtIndex(source, 0, options) as? [CFString: Any],
                  let width = (metadata[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
                  let height = (metadata[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
                  model.architecture.supportsGrid(height: height, width: width) else {
                throw LoomError("Compact native model tests require native image dimensions of at least 32 pixels and multiples of eight. Choose a compatible original image.")
            }
            NativeImages.clearDecodedPixelCache()
            let metal = MTLCreateSystemDefaultDevice()
            let required = CompactModelMemoryAdmission.forecast(width: width, height: height,
                featureWidth: model.architecture.width, parameterCount: model.architecture.parameterCount)
            guard required != .max else {
                throw LoomError("The compact memory forecast overflowed for these native image dimensions.")
            }
            let budget = suppliedCompactMemoryBudget ?? CompactModelMemoryAdmission.budget(physical: ProcessInfo.processInfo.physicalMemory,
                recommendedMetal: metal?.recommendedMaxWorkingSetSize, allocatedMetal: UInt64(metal?.currentAllocatedSize ?? 0))
            let exceedsBudget = required > budget
            if exceedsBudget {
                let gib = Double(1 << 30)
                progress(.init(type: "status", message: String(format: "Attempting the complete native image field. Its conservative memory forecast is %.1f GiB, above the current %.1f GiB forecast budget.", Double(required) / gib, Double(budget) / gib)))
            }
            memoryAdmission = ["kind": "conservative-forecast-advisory-v1", "forecast_bytes": required,
                "budget_bytes": budget, "attempted_over_budget": exceedsBudget,
                "bytes_per_width_pixel": 256, "bytes_per_parameter": 12]
            admittedDimensions = (width, height)
        }
        let image = try autoreleasepool { try NativeImages.readRGB(from: original, cancellation: cancellation) }
        NativeImages.clearDecodedPixelCache()
        try cancellation.check()
        if model.architecture.isCompact {
            guard admittedDimensions?.width == image.width, admittedDimensions?.height == image.height else {
                throw LoomError("The diffuse image dimensions changed after its memory preflight. Run the test again on the completed image.")
            }
            guard model.architecture.supportsGrid(height: image.height, width: image.width) else {
                throw LoomError("Compact native model tests require native image dimensions of at least 32 pixels and multiples of eight. Choose a compatible original image.")
            }
        }
        let predictor: Predictor
        if let suppliedPredictor { predictor = suppliedPredictor }
        else {
            progress(.init(type: "status", message: "Loading saved \(model.source == .checkpoint ? "checkpoint" : "model")…"))
            let weights = try NativeWeights.read(from: model.weightsURL)
            let session = try NativeTrainingSession(architecture: model.architecture,
                options: NativeTrainingOptions(scalarLoss: model.architecture.isCompact ? .linearDetailV1 : .absoluteV1),
                seed: 0, weights: weights, cancellation: cancellation)
            predictor = { input, height, width in
                try session.predict(input: input, batchSize: 1, height: height, width: width)
            }
        }
        // Fixed tile dimensions bound GPU scratch use even for an 8K original.
        // Source pixels retain their native coordinates and numeric precision.
        let tileSize: Int, tileCount: Int, overlap: Int
        let pixels: [Float]
        if model.architecture.isCompact {
            // Compact channel attention uses the complete field's mean. Tiling
            // changes its math even when overlaps would hide a visible seam.
            tileSize = max(image.width, image.height); tileCount = 1; overlap = 0
            var input = [Float](repeating: 0, count: image.values.count)
            let plane = image.width * image.height
            for pixel in 0..<plane {
                if pixel % 65536 == 0 { try cancellation.check() }
                for channel in 0..<3 { input[channel * plane + pixel] = image.values[pixel * 3 + channel] }
            }
            progress(.init(type: "status", message: "Testing the complete native image field…"))
            pixels = try predictor(input, image.height, image.width)
            guard pixels.count == plane * model.architecture.outputChannels,
                  pixels.allSatisfy(\.isFinite) else { throw LoomError("The saved compact model produced invalid native output pixels.") }
        } else {
            tileSize = min(1024, model.imageSize)
            let plan = try ModelTestTilePlan(width: image.width, height: image.height, tileSize: tileSize)
            tileCount = plan.tiles.count; overlap = plan.overlap
            pixels = try ModelTestImage.predict(image: image, plan: plan, map: model.map,
                cancellation: cancellation, progress: progress, predictor: predictor)
        }
        let statistics = try ModelTestOutputStatistics(values: pixels, cancellation: cancellation)
        let saturationNote = statistics.saturationNote(map: model.map)
        try cancellation.check()
        let parent = output.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(".pending-" + output.lastPathComponent, isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: staging) }
        let name = model.map.rawValue + ".exr", previewName = model.map.rawValue + ".png"
        progress(.init(type: "status", message: "Saving \(image.width) × \(image.height) \(model.map.rawValue) output…"))
        try NativeFloatEXR.write(pixels, channels: model.architecture.outputChannels,
            width: image.width, height: image.height, to: staging.appendingPathComponent(name), cancellation: cancellation)
        try ModelTestImage.writePNG(pixels, channels: model.architecture.outputChannels,
            width: image.width, height: image.height, to: staging.appendingPathComponent(previewName),
            clampPreview: model.architecture.isCompact, cancellation: cancellation)
        try cancellation.check()
        guard inputHash == (try NativeFactoryTraining.sha(original, cancellation: cancellation)) else {
            throw LoomError("The diffuse image changed during testing. Run the test again on the completed image.")
        }
        guard model.weightsSHA256 == (try NativeFactoryTraining.sha(model.weightsURL, cancellation: cancellation)) else {
            throw LoomError("The saved weights changed during testing. Choose the version again.")
        }
        let note = (saturationNote.map { $0 + " " } ?? "")
            + (model.architecture.isCompact
                ? "Original \(image.width) × \(image.height) resolution; one complete native field, with raw linear output. "
                : "Original \(image.width) × \(image.height) resolution; \(tileCount) overlapping native tile\(tileCount == 1 ? "" : "s"). Overlap blending can differ from whole-image inference. ")
            + (model.map == .normal && !model.architecture.isCompact ? "OpenGL normal vectors are normalized after blending. " : "")
            + "The canonical EXR stores the Float32 prediction values without clamping or normalization. "
            + (model.architecture.isCompact ? "The PNG preview clamps to [0,1] before 16-bit quantization, without per-output normalization."
                : "The PNG preview uses fixed [0,1] values quantized to 16 bits, without per-output normalization.")
        try NativeFactoryTraining.writeJSON(["schema": "loom-factory-model-test-v1", "run_id": model.runID,
            "checkpoint_id": model.checkpointID as Any? ?? NSNull(),
            "weight_source": model.source.rawValue, "dataset_id": model.datasetID, "map": model.map.rawValue,
            "architecture": model.architecture.identifier, "input_path": original.path, "input_sha256": inputHash,
            "weights_sha256": model.weightsSHA256, "output_path": name,
            "output_format": "openexr", "output_component_type": "float32", "output_bits_per_channel": 32,
            "output_channels": model.map == .normal ? ["R", "G", "B"] : ["Y"],
            "output_value_transform": "none", "preview_path": previewName, "preview_format": "png",
            "preview_bits_per_channel": 16, "preview_value_range": [0, 1],
            "output_statistics": statistics.record,
            "output_diagnostic": saturationNote as Any? ?? NSNull(),
            "width": image.width, "height": image.height, "tile_size": tileSize, "tile_count": tileCount,
            "overlap": overlap, "stitching": model.architecture.isCompact ? "whole-field-v1" : "linear-feather-overlap-v1",
            "normal_output_transform": model.map == .normal && !model.architecture.isCompact ? "unit-normalize-after-blending" : "none",
            "head_activation": model.architecture.isCompact ? "linear" : "sigmoid",
            "memory_admission": memoryAdmission as Any? ?? NSNull(),
            "normal_convention": model.map == .normal ? "OpenGL" as Any : NSNull(),
            "created_at": ISO8601DateFormatter().string(from: .now), "note": note],
            to: staging.appendingPathComponent("test.json"))
        try cancellation.check()
        try FileManager.default.moveItem(at: staging, to: output)
        let result = ModelTestResult(directory: output, inputURL: original, outputURL: output.appendingPathComponent(name),
            runID: model.runID, source: model.source, map: model.map, width: image.width, height: image.height,
            tileSize: tileSize, tileCount: tileCount, note: note, previewURL: output.appendingPathComponent(previewName),
            checkpointID: model.checkpointID, inputSHA256: inputHash)
        progress(.init(type: "completed", message: "Saved full-resolution \(model.map.rawValue) output", progress: 1))
        return result
    }
}
