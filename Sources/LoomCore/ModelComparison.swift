import CoreGraphics
import CoreML
import Foundation
import ImageIO
import LoomNative
import UniformTypeIdentifiers

public enum ModelComparisonMembership: String, Sendable, Codable {
    case training, heldOut, unknown
    public var displayName: String {
        switch self {
        case .training: "Used for training"
        case .heldOut: "Held-out material"
        case .unknown: "Training membership unknown"
        }
    }
}

public struct ModelComparisonPair: Identifiable, Sendable {
    public let id: String
    public let inputURL: URL
    public let targetURL: URL
    public let label: String
    public let width: Int
    public let height: Int
    public let seenByFirst: ModelComparisonMembership
    public let seenBySecond: ModelComparisonMembership
}

public struct ModelComparisonResult: Sendable {
    public let directory: URL
    public let inputURL: URL
    public let targetURL: URL
    public let firstOutputURL: URL
    public let secondOutputURL: URL
    public let firstArtifactID: String
    public let secondArtifactID: String
    public let firstMAE: Double
    public let secondMAE: Double
    public let outputDifferenceMAE: Double
    public let cropSize: Int
    public let pair: ModelComparisonPair
    public let targetNote: String
}

extension LoomProject {
    public func comparePairOptions(datasetID: String, map: MapRole, artifactIDs: [String]) throws -> [ModelComparisonPair] {
        let current = try snapshot()
        let models = try ComparisonWorker.models(artifactIDs, artifacts: current.artifacts,
            projectFolder: projectFolder, projectID: document.id, datasetID: datasetID, map: map)
        guard let dataset = current.datasets.first(where: { $0.id == datasetID }) else { return [] }
        let size = ComparisonWorker.commonCropSize(models)
        return try TrainingPairSelector.select(images: current.images, dataset: dataset.name, map: map, size: size).map { pair in
            ComparisonWorker.option(pair, first: models[0], second: models[1], projectFolder: projectFolder)
        }
    }

    public func compareModels(firstArtifactID: String, secondArtifactID: String, pairID: String,
                              progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> ModelComparisonResult {
        guard !operationActive else { throw LoomError("Stop the current operation before comparing models.") }
        let current = try snapshot()
        guard let first = current.artifacts.first(where: { $0.id == firstArtifactID }),
              let datasetID = first.datasetID, let map = first.map,
              let dataset = current.datasets.first(where: { $0.id == datasetID }) else {
            throw LoomError("Choose two saved models from the current dataset and map.")
        }
        let models = try ComparisonWorker.models([firstArtifactID, secondArtifactID], artifacts: current.artifacts,
            projectFolder: projectFolder, projectID: document.id, datasetID: datasetID, map: map)
        let size = ComparisonWorker.commonCropSize(models)
        guard let pair = try TrainingPairSelector.select(images: current.images, dataset: dataset.name, map: map, size: size)
            .first(where: { $0.id == pairID }) else { throw LoomError("The selected diffuse and target pair is no longer available. Refresh the comparison.") }
        let option = ComparisonWorker.option(pair, first: models[0], second: models[1], projectFolder: projectFolder)
        let root = sourceFolder
        let output = projectFolder.appendingPathComponent("comparisons/" + UUID().uuidString.lowercased(), isDirectory: true)
        let cancellation = NativeCancellation()
        operationActive = true; activeTraining = cancellation
        defer { operationActive = false; activeTraining = nil }
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await Task.detached(priority: .userInitiated) {
                try ComparisonWorker.run(models: models, pair: pair, option: option, dataset: dataset.name, map: map,
                    root: root, output: output, size: size, cancellation: cancellation, progress: progress)
            }.value
        } onCancel: { cancellation.cancel() }
    }
}

private enum ComparisonWorker {
    struct Model: Sendable {
        let artifact: ModelArtifact
        let imageSize: Int
        let exaggerate: Bool
        let coreMLHash: String?
    }

    static func commonCropSize(_ models: [Model]) -> Int {
        let size = models.map(\.imageSize).min()!
        // Compare the same original pixels at a grid both constructors accept.
        // Reducing the crop extent never resamples or pads a source image.
        return models.contains { $0.artifact.architecture == NativeArchitecture.compactIdentifier }
            ? size - size % 8 : size
    }

    static func models(_ ids: [String], artifacts: [ModelArtifact], projectFolder: URL, projectID: String,
                       datasetID: String, map: MapRole) throws -> [Model] {
        guard ids.count == 2, ids[0] != ids[1] else { throw LoomError("Choose two different saved model versions.") }
        return try ids.map { id in
            guard let artifact = artifacts.first(where: { $0.id == id }), artifact.kind == "map",
                  artifact.datasetID == datasetID, artifact.map == map, let coreMLPath = artifact.coreMLPath,
                  coreMLPath == "model.mlmodel" else { throw LoomError("Both saved models must belong to this dataset and map.") }
            try TrainingRunFiles.ownedRun(artifact.directory, projectFolder: projectFolder)
            _ = try TrainingRunFiles.containedFile(coreMLPath, directory: artifact.directory)
            let descriptor = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("model.json"))
            guard descriptor["schema"] as? String == "loom-factory-model-v1",
                  descriptor["status"] as? String == "completed",
                  descriptor["project_id"] as? String == projectID,
                  descriptor["dataset_id"] as? String == datasetID,
                  descriptor["map"] as? String == map.rawValue,
                  descriptor["architecture"] as? String == artifact.architecture,
                  let size = descriptor["image_size"] as? Int, (32...8192).contains(size) else {
                throw LoomError("The saved model descriptor does not match this source project.")
            }
            if artifact.architecture == NativeArchitecture.compactIdentifier {
                _ = try TrainingRunFiles.nativeArchitecture(from: descriptor, map: map)
                guard size % 8 == 0 else {
                    throw LoomError("Compact native models require image dimensions that are multiples of eight.")
                }
            } else if !NativeArchitecture.supportedIdentifiers.contains(artifact.architecture) {
                throw LoomError("The saved model has an unsupported native architecture identity.")
            }
            let training = descriptor["training"] as? [String: Any]
            return Model(artifact: artifact, imageSize: size,
                exaggerate: training?["exaggerate_features"] as? Bool ?? false,
                coreMLHash: descriptor["coreml_sha256"] as? String)
        }
    }

    static func option(_ pair: TrainingPair, first: Model, second: Model, projectFolder: URL) -> ModelComparisonPair {
        ModelComparisonPair(id: pair.id, inputURL: pair.input.url, targetURL: pair.target.url,
            label: pair.input.displayName + " · " + pair.input.dimensions,
            width: pair.input.width, height: pair.input.height,
            seenByFirst: membership(pair, run: first.artifact.directory, projectFolder: projectFolder),
            seenBySecond: membership(pair, run: second.artifact.directory, projectFolder: projectFolder))
    }

    // A material is held out only when its entire folder was absent from every
    // ancestor's training split. A held-out crop from a trained folder is training.
    static func membership(_ pair: TrainingPair, run: URL, projectFolder: URL) -> ModelComparisonMembership {
        var directory = run, visited: Set<String> = [], knownHeldOut = false, unknown = false
        let material = pair.input.url.deletingLastPathComponent().standardizedFileURL.resolvingSymlinksInPath()
        while true {
            guard visited.insert(directory.lastPathComponent).inserted else { unknown = true; break }
            guard let request = try? NativeFactoryTraining.json(directory.appendingPathComponent("request.json")),
                  let samples = request["samples"] as? [[String: Any]] else { unknown = true; break }
            for sample in samples {
                guard let path = sample["input"] as? String else { unknown = true; continue }
                let folder = URL(fileURLWithPath: path).deletingLastPathComponent().standardizedFileURL.resolvingSymlinksInPath()
                if folder == material {
                    if sample["split"] as? String == "training" { return .training }
                    if sample["split"] as? String == "validation" { knownHeldOut = true }
                    else { unknown = true }
                }
            }
            do {
                if let metadata = try TrainingRunFiles.metadata(at: directory) {
                    guard let lineage = metadata.lineage else { break }
                    directory = try TrainingRunFiles.runDirectory(lineage.parentRunID, projectFolder: projectFolder)
                } else {
                    guard request["schema"] as? String == "loom-factory-training-v1" else { unknown = true; break }
                    if let path = request["resume_weights"] as? String {
                        directory = try TrainingRunFiles.ownedRun(URL(fileURLWithPath: path).deletingLastPathComponent(), projectFolder: projectFolder)
                    } else { break }
                }
            } catch { unknown = true; break }
        }
        // A new external material was not part of a recorded validation split.
        return knownHeldOut && !unknown ? .heldOut : .unknown
    }

    static func run(models: [Model], pair: TrainingPair, option: ModelComparisonPair, dataset: String, map: MapRole,
                    root: URL, output: URL, size: Int, cancellation: NativeCancellation,
                    progress: @escaping @Sendable (TrainingEvent) -> Void) throws -> ModelComparisonResult {
        defer { NativeImages.clearDecodedPixelCache() }
        try cancellation.check()
        try NativeFactoryTraining.checkTags(pair, dataset: dataset, map: map, root: root)
        let inputHash = try NativeFactoryTraining.sha(pair.input.url, cancellation: cancellation)
        let targetHash = try NativeFactoryTraining.sha(pair.target.url, cancellation: cancellation)
        let crop = NativeCropTransform(x: (pair.input.width - size) / 2, y: (pair.input.height - size) / 2, size: size)
        let prepared = try NativeImages.pair(inputURL: pair.input.url, targetURL: pair.target.url, transform: crop,
            mapKind: map.rawValue, normalConvention: pair.target.normalConvention ?? "OpenGL", exaggerate: false, cancellation: cancellation)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let input = output.appendingPathComponent("input.png"), target = output.appendingPathComponent("target.png")
        try writePreview(prepared.input.values, channels: 3, size: size, to: input)
        try writePreview(prepared.target.values, channels: map == .normal ? 3 : 1, size: size, to: target)
        var values: [[Float]] = [], modelHashes: [String] = []
        for (index, model) in models.enumerated() {
            try cancellation.check()
            progress(.init(type: "status", message: "Comparing model \(index + 1) of 2 on the same \(size) × \(size) crop…", progress: Double(index) / 2))
            let url = model.artifact.directory.appendingPathComponent("model.mlmodel")
            let hash = try NativeFactoryTraining.sha(url, cancellation: cancellation)
            guard model.coreMLHash == nil || model.coreMLHash == hash else { throw LoomError("Saved Core ML model checksum changed. Choose an intact version.") }
            modelHashes.append(hash)
            values.append(try predict(url, input: prepared.input.values, size: size,
                channels: map == .normal ? 3 : 1, cancellation: cancellation))
        }
        try cancellation.check()
        try NativeFactoryTraining.checkTags(pair, dataset: dataset, map: map, root: root)
        guard inputHash == (try NativeFactoryTraining.sha(pair.input.url, cancellation: cancellation)),
              targetHash == (try NativeFactoryTraining.sha(pair.target.url, cancellation: cancellation)) else {
            throw LoomError("The source pixels changed during comparison. Refresh and compare again.")
        }
        for (model, hash) in zip(models, modelHashes) {
            guard hash == (try NativeFactoryTraining.sha(model.artifact.directory.appendingPathComponent("model.mlmodel"), cancellation: cancellation)) else {
                throw LoomError("The saved model changed during comparison.")
            }
        }
        let first = output.appendingPathComponent("model-a.png"), second = output.appendingPathComponent("model-b.png")
        try writePreview(values[0], channels: map == .normal ? 3 : 1, size: size, to: first)
        try writePreview(values[1], channels: map == .normal ? 3 : 1, size: size, to: second)
        let a = meanAbsoluteError(values[0], prepared.target.values), b = meanAbsoluteError(values[1], prepared.target.values)
        let difference = meanAbsoluteError(values[0], values[1])
        let note = models.contains(where: \.exaggerate)
            ? "Both outputs are scored against the original target values. A model trained with exaggerated depth targets uses a different teaching transform."
            : "Both outputs are scored against the same original target values. MAE measures numeric agreement on this crop; lower error does not always look better."
        try NativeFactoryTraining.writeJSON(["schema": "loom-factory-comparison-v1", "model_a": models[0].artifact.id,
            "model_b": models[1].artifact.id, "model_a_sha256": modelHashes[0], "model_b_sha256": modelHashes[1],
            "model_a_architecture": models[0].artifact.architecture, "model_b_architecture": models[1].artifact.architecture,
            "pair_id": pair.id, "input_path": pair.input.url.path, "target_path": pair.target.url.path,
            "input_sha256": inputHash, "target_sha256": targetHash, "crop": ["x": crop.x, "y": crop.y, "size": size],
            "mae_a": a, "mae_b": b, "output_difference_mae": difference,
            "membership_a": option.seenByFirst.rawValue, "membership_b": option.seenBySecond.rawValue,
            "target_note": note, "output_scoring_transform": "none",
            "display": "16-bit PNG preview clamped to fixed [0,1] scale; no per-output normalization"],
            to: output.appendingPathComponent("comparison.json"))
        progress(.init(type: "completed", message: "Saved both model outputs and the common input and target crop", progress: 1))
        return ModelComparisonResult(directory: output, inputURL: input, targetURL: target, firstOutputURL: first,
            secondOutputURL: second, firstArtifactID: models[0].artifact.id, secondArtifactID: models[1].artifact.id,
            firstMAE: a, secondMAE: b, outputDifferenceMAE: difference, cropSize: size, pair: option, targetNote: note)
    }

    static func predict(_ url: URL, input: [Float], size: Int, channels: Int, cancellation: NativeCancellation) throws -> [Float] {
        let compiled = try MLModel.compileModel(at: url)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let configuration = MLModelConfiguration(); configuration.computeUnits = .all
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        try cancellation.check()
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        for index in input.indices { array[index] = NSNumber(value: input[index]) }
        let features = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["rgb": array]))
        guard let output = features.featureValue(for: "material_map")?.multiArrayValue,
              output.shape.map(\.intValue) == [1, channels, size, size] else { throw LoomError("Saved model output does not match the common crop dimensions.") }
        let result = (0..<output.count).map { output[$0].floatValue }
        guard result.allSatisfy(\.isFinite) else { throw LoomError("Saved model produced non-finite comparison pixels.") }
        try cancellation.check()
        return result
    }

    static func meanAbsoluteError(_ a: [Float], _ b: [Float]) -> Double {
        zip(a, b).reduce(0) { $0 + abs(Double($1.0) - Double($1.1)) } / Double(a.count)
    }

    static func writePreview(_ values: [Float], channels: Int, size: Int, to url: URL) throws {
        let count = size * size
        var pixels = [UInt16](repeating: 0, count: count * channels)
        for pixel in 0..<count {
            for channel in 0..<channels {
                pixels[pixel * channels + channel] = UInt16((min(1, max(0, values[channel * count + pixel])) * 65535).rounded()).bigEndian
            }
        }
        let data = pixels.withUnsafeBytes { Data($0) }
        let colorSpace = channels == 1 ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: data as CFData),
              let image = CGImage(width: size, height: size, bitsPerComponent: 16, bitsPerPixel: 16 * channels,
                bytesPerRow: size * channels * 2, space: colorSpace,
                bitmapInfo: [.byteOrder16Big, CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)],
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw LoomError("Could not create the comparison preview.")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw LoomError("Could not save the comparison preview.") }
    }
}
