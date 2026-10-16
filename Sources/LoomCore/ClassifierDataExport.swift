import CoreML
import CoreVideo
import Foundation

public struct ClassifierDataExportSummary: Sendable {
    public let directory: URL
    public let folderCount: Int
    public let imageCount: Int
    public let skippedFolders: [URL]
}

extension LoomProject {
    /// A disposable Create ML view over the current files. No source pixels are
    /// copied, resized or rewritten, and no membership inventory is saved.
    public func exportClassifierData(to destination: URL) throws -> ClassifierDataExportSummary {
        guard !operationActive else { throw LoomError("Finish or stop the current project operation before exporting classifier data.") }
        operationActive = true
        defer { operationActive = false }
        let images = try scanImages().filter { !$0.ignored }
        let folders = Dictionary(grouping: images) { $0.url.deletingLastPathComponent().standardizedFileURL }
            .sorted { $0.key.path.localizedStandardCompare($1.key.path) == .orderedAscending }
        var samples: [(folder: URL, image: SourceImage, category: String, categoryID: String)] = []
        var skipped: [URL] = []
        for folder in folders {
            let categories = Dictionary(grouping: folder.value.compactMap(\.category), by: ImageMetadataStore.categoryID)
            guard categories.count == 1, let category = categories.values.first?.sorted().first,
                  let image = folder.value.filter({ $0.role == .diffuse && $0.width == 1024 && $0.height == 1024 &&
                      $0.category.map(ImageMetadataStore.categoryID) == ImageMetadataStore.categoryID(category) })
                    .sorted(by: { $0.url.path.localizedStandardCompare($1.url.path) == .orderedAscending }).first else {
                skipped.append(folder.key)
                continue
            }
            samples.append((folder.key, image, category, ImageMetadataStore.categoryID(category)))
        }
        guard !samples.isEmpty else {
            throw LoomError("No material folders have a consistent dataset tag and a native 1,024 × 1,024 diffuse image. Tag the materials and add their 1K diffuse images before exporting.")
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: destination.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw LoomError("Choose an existing destination folder for the Create ML data export.")
        }
        let id = UUID().uuidString.lowercased()
        let stage = destination.appendingPathComponent(".create-ml-" + id, isDirectory: true)
        let output = destination.appendingPathComponent("Create ML Classifier Data-" + id.prefix(8), isDirectory: true)
        let categoryNames = samples.reduce(into: [String: String]()) { result, sample in
            if result[sample.categoryID] == nil { result[sample.categoryID] = sample.category }
        }
        let safeNames = categoryNames.mapValues(classifierSafeComponent)
        let foldedNames = safeNames.mapValues { $0.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX")) }
        var categoryLabels: [String: String] = [:]
        var usedFolderNames: Set<String> = []
        for id in categoryNames.keys.sorted() {
            let collides = foldedNames.values.filter { $0 == foldedNames[id] }.count > 1
            let base = safeNames[id]! + (collides ? "-" + id.suffix(8) : "")
            var candidate = base
            var suffix = 1
            while !usedFolderNames.insert(candidate.folding(options: [.caseInsensitive, .diacriticInsensitive],
                                                           locale: Locale(identifier: "en_US_POSIX"))).inserted {
                suffix += 1
                candidate = base + "-" + String(suffix)
            }
            categoryLabels[id] = candidate
        }
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
        do {
            var exported = 0
            var exportedCategoryIDs: Set<String> = []
            for sample in samples {
                try Task.checkCancellation()
                // Metadata and the actual file remain authoritative until the
                // export link is written; a concurrent edit is never ignored.
                guard FileManager.default.fileExists(atPath: sample.image.url.path),
                      let metadata = try ImageMetadataStore.read(sample.image.url), metadata.id == sample.image.id,
                      metadata.role == .diffuse, metadata.ignored != true,
                      metadata.category.map(ImageMetadataStore.categoryID) == sample.categoryID else {
                    skipped.append(sample.folder)
                    continue
                }
                let categoryFolder = stage.appendingPathComponent(categoryLabels[sample.categoryID]!, isDirectory: true)
                try FileManager.default.createDirectory(at: categoryFolder, withIntermediateDirectories: true)
                let filename = classifierSafeComponent(sample.folder.lastPathComponent) + "-" + classifierSafeComponent(sample.image.id) + "." + sample.image.url.pathExtension
                try FileManager.default.createSymbolicLink(at: categoryFolder.appendingPathComponent(filename),
                                                          withDestinationURL: sample.image.url.resolvingSymlinksInPath())
                exported += 1
                exportedCategoryIDs.insert(sample.categoryID)
            }
            guard exported > 0 else { throw LoomError("The source files or dataset tags changed during export. Refresh the source folder and try again.") }
            let catalog = samples.filter { exportedCategoryIDs.contains($0.categoryID) }
                .reduce(into: [String: [String: String]]()) { result, sample in
                    result[sample.categoryID] = ["id": sample.categoryID, "name": sample.category,
                                                 "sourceLabel": categoryLabels[sample.categoryID]!]
                }.values.sorted { $0["id"]! < $1["id"]! }
            let catalogs = projectFolder.appendingPathComponent("classifier-data-exports", isDirectory: true)
            try FileManager.default.createDirectory(at: catalogs, withIntermediateDirectories: true)
            let catalogURL = catalogs.appendingPathComponent(id + ".json")
            let descriptor: [String: Any] = ["schema": "loom-factory.classifier-data.v1", "project_id": document.id,
                "labels": catalog, "created_at": ISO8601DateFormatter().string(from: .now)]
            try JSONSerialization.data(withJSONObject: descriptor, options: [.prettyPrinted, .sortedKeys]).write(to: catalogURL, options: .atomic)
            do { try FileManager.default.moveItem(at: stage, to: output) }
            catch { try? FileManager.default.removeItem(at: catalogURL); throw error }
            return ClassifierDataExportSummary(directory: output, folderCount: exportedCategoryIDs.count,
                                               imageCount: exported, skippedFolders: skipped)
        } catch {
            try? FileManager.default.removeItem(at: stage)
            throw error
        }
    }

    /// Create ML owns classifier training. Factory validates the resulting
    /// native model and keeps it with this source folder's deployment models.
    public func importClassifier(from modelURL: URL) async throws -> ModelArtifact {
        guard !operationActive else { throw LoomError("Finish or stop the current project operation before importing a classifier.") }
        operationActive = true
        defer { operationActive = false }
        guard ["mlmodel", "mlpackage", "mlmodelc"].contains(modelURL.pathExtension.lowercased()),
              FileManager.default.fileExists(atPath: modelURL.path) else {
            throw LoomError("Choose a CoreML classifier exported by Create ML (.mlmodel, .mlpackage or .mlmodelc).")
        }
        let interface = try await nativeClassifierInterface(at: modelURL)
        let current = try snapshot()
        let labels = try classifierLabelCatalog(classes: interface.labels, datasets: current.datasets,
                                                exportedLabels: exportedClassifierLabels())
        let id = UUID().uuidString.lowercased()
        let runs = projectFolder.appendingPathComponent("runs", isDirectory: true)
        try FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        let stage = runs.appendingPathComponent(".import-" + id, isDirectory: true)
        let output = runs.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
        do {
            let nativePath = "model." + modelURL.pathExtension.lowercased()
            try FileManager.default.copyItem(at: modelURL, to: stage.appendingPathComponent(nativePath))
            var descriptor: [String: Any] = ["schema": "loom-factory-model-v1", "status": "completed",
                "kind": "classifier", "project_id": document.id, "architecture": "create-ml-image-classifier-v1",
                "coreml_path": nativePath, "input_name": interface.inputName, "output_name": interface.outputName,
                "labels": labels, "origin": "create-ml", "created_at": ISO8601DateFormatter().string(from: .now)]
            descriptor["input_width"] = interface.width
            descriptor["input_height"] = interface.height
            descriptor["input_color_space"] = "RGB"
            descriptor["input_layout"] = interface.tensorInput ? "NCHW" : "image"
            descriptor["input_type"] = interface.tensorInput ? "multiArray" : "image"
            if interface.tensorInput { descriptor["input_shape"] = [1, 3, interface.height, interface.width] }
            try JSONSerialization.data(withJSONObject: descriptor, options: [.prettyPrinted, .sortedKeys])
                .write(to: stage.appendingPathComponent("model.json"), options: .atomic)
            let artifact = ModelArtifact(id: id, kind: "classifier", datasetID: nil, datasetName: nil,
                                         map: nil, directory: output, epochs: 0, loss: nil,
                                         coreMLPath: nativePath, weightsPath: nil, architecture: "create-ml-image-classifier-v1")
            try JSONEncoder().encode(artifact).write(to: stage.appendingPathComponent("artifact.json"), options: .atomic)
            try Task.checkCancellation()
            try FileManager.default.moveItem(at: stage, to: output)
            return artifact
        } catch {
            try? FileManager.default.removeItem(at: stage)
            throw error
        }
    }

    private func exportedClassifierLabels() throws -> [[String: String]] {
        let folder = projectFolder.appendingPathComponent("classifier-data-exports", isDirectory: true)
        guard FileManager.default.fileExists(atPath: folder.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }.flatMap { file -> [[String: String]] in
                guard let data = try? Data(contentsOf: file),
                      let descriptor = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      descriptor["project_id"] as? String == document.id else { return [] }
                return descriptor["labels"] as? [[String: String]] ?? []
            }
    }
}

/// Original CoreML labels remain intact. The deployment catalog maps each one
/// onto the current source category's stable ID.
func classifierLabelCatalog(classes: [String], datasets: [DatasetCategory],
                            exportedLabels: [[String: String]] = []) throws -> [[String: String]] {
    guard !classes.isEmpty, classes.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
          Set(classes).count == classes.count else { throw LoomError("The classifier has missing or duplicate class labels.") }
    let categories = Dictionary(grouping: datasets, by: \.id)
    var usedIDs: Set<String> = []
    let result = try classes.map { label in
        let directIDs = Set([label, ImageMetadataStore.categoryID(label),
                             ImageMetadataStore.categoryID(ImageMetadataStore.friendlyName(label))].filter { categories[$0] != nil })
        let exportedIDs = Set(exportedLabels.compactMap { record -> String? in
            guard record["sourceLabel"] == label, let id = record["id"], categories[id] != nil else { return nil }
            return id
        })
        let possibleIDs = directIDs.union(exportedIDs)
        guard possibleIDs.count == 1, let id = possibleIDs.first, let category = categories[id]?.first else {
            throw LoomError("The classifier's label ‘\(label)’ does not match one current source dataset. Export the current category folders and train the classifier in Create ML.")
        }
        guard usedIDs.insert(id).inserted else {
            throw LoomError("The classifier has multiple labels for the same dataset: \(category.name). Train it with one folder per dataset.")
        }
        return ["id": id, "name": category.name, "sourceLabel": label]
    }
    guard usedIDs == Set(categories.keys) else {
        throw LoomError("The classifier does not cover every current source dataset. Export all current category folders and train the classifier in Create ML.")
    }
    return result
}

struct NativeClassifierInterface {
    let labels: [String]
    let inputName: String
    let outputName: String
    let width: Int
    let height: Int
    let tensorInput: Bool
}

/// Validate the actual native model at import and again when its deployment is
/// packaged. The saved descriptor alone cannot prove that routing is valid.
func nativeClassifierInterface(at modelURL: URL) async throws -> NativeClassifierInterface {
    let alreadyCompiled = modelURL.pathExtension.lowercased() == "mlmodelc"
    let compiled = alreadyCompiled ? modelURL : try await MLModel.compileModel(at: modelURL)
    defer {
        if !alreadyCompiled { try? FileManager.default.removeItem(at: compiled) }
    }
    try Task.checkCancellation()
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuOnly
    let description = try MLModel(contentsOf: compiled, configuration: configuration).modelDescription
    guard let classes = description.classLabels as? [String], !classes.isEmpty,
          Set(classes).count == classes.count,
          classes.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
          let outputName = description.predictedFeatureName,
          description.outputDescriptionsByName[outputName]?.type == .string else {
        throw LoomError("This model is not a string-label image classifier. Export an Image Classification model from Create ML.")
    }
    guard description.inputDescriptionsByName.count == 1,
          let input = description.inputDescriptionsByName.first else {
        throw LoomError("The classifier must accept one image or one RGB tensor input.")
    }
    if let constraint = input.value.imageConstraint {
        guard constraint.pixelsWide > 0, constraint.pixelsHigh > 0,
              [kCVPixelFormatType_32BGRA, kCVPixelFormatType_32ARGB].contains(constraint.pixelFormatType) else {
            throw LoomError("The classifier must accept a fixed RGB image input.")
        }
        return NativeClassifierInterface(labels: classes, inputName: input.key, outputName: outputName,
                                         width: constraint.pixelsWide, height: constraint.pixelsHigh, tensorInput: false)
    }
    if let constraint = input.value.multiArrayConstraint {
        let shape = constraint.shape.map(\.intValue)
        guard constraint.dataType == .float32, shape.count == 4, shape[0] == 1, shape[1] == 3,
              shape[2] > 0, shape[3] > 0 else {
            throw LoomError("The classifier must accept a fixed float32 RGB NCHW tensor input.")
        }
        return NativeClassifierInterface(labels: classes, inputName: input.key, outputName: outputName,
                                         width: shape[3], height: shape[2], tensorInput: true)
    }
    throw LoomError("The classifier has an unsupported image interface.")
}

private func classifierSafeComponent(_ value: String) -> String {
    let safe = String(value.map { character -> Character in
        if character == "/" || character == ":" || character.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains) { return " " }
        return character
    }).split(whereSeparator: \.isWhitespace).joined(separator: " ")
        .trimmingCharacters(in: CharacterSet(charactersIn: ". "))
    return safe.isEmpty ? "Dataset" : String(safe.prefix(120))
}
