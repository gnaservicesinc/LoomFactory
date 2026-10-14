import Foundation
import Darwin
import ImageIO
import LoomNative

struct ProjectDocument: Codable, Sendable {
    var schema = "loom-factory.project.v1"
    var id = UUID().uuidString.lowercased()
    var settings = ProjectSettings()
}

public actor LoomProject {
    public nonisolated let sourceFolder: URL
    public nonisolated let projectFolder: URL
    var document: ProjectDocument
    var activeProcess: Process?
    var activeTraining: NativeCancellation?
    var activeTrainingControl: TrainingRunControl?
    var operationActive = false
    public init(sourceFolder: URL) throws {
        // Foundation can mix /var and /private/var URLs. Establish one actual
        // project root, then keep every discovered membership path relative to it.
        guard let resolvedPath = realpath(sourceFolder.path, nil) else { throw LoomError("Choose an existing source folder.") }
        let root = URL(fileURLWithPath: String(cString: resolvedPath), isDirectory: true).standardizedFileURL
        free(resolvedPath)
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &directory), directory.boolValue else {
            throw LoomError("Choose an existing folder containing your material images.")
        }
        self.sourceFolder = root
        self.projectFolder = root.appendingPathComponent(".loom-factory", isDirectory: true)
        try FileManager.default.createDirectory(at: projectFolder, withIntermediateDirectories: true)
        let file = projectFolder.appendingPathComponent("project.json")
        if FileManager.default.fileExists(atPath: file.path) {
            document = try JSONDecoder().decode(ProjectDocument.self, from: Data(contentsOf: file))
            guard document.schema == "loom-factory.project.v1" else { throw LoomError("This folder belongs to a newer Loom Factory project format.") }
        } else {
            document = ProjectDocument()
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(document).write(to: file, options: .atomic)
        }
    }

    public func snapshot() throws -> ProjectSnapshot {
        let images = try scanImages()
        let categories = Set(images.filter { !$0.ignored }.compactMap(\.category)).sorted { $0.localizedStandardCompare($1) == .orderedAscending }
        let datasets = categories.map { category in
            let members = images.filter { $0.category == category && !$0.ignored }
            let families = Dictionary(grouping: members, by: \.pairingKey)
            let counts = Dictionary(uniqueKeysWithValues: MapRole.targets.map { map in
                (map.rawValue, families.values.reduce(0) { count, group in
                    count + ((try? TrainingPairSelector.select(images: group, dataset: category, map: map, size: 1).count) ?? 0)
                })
            })
            return DatasetCategory(id: ImageMetadataStore.categoryID(category), name: category,
                imageCount: members.count, verifiedCount: members.filter(\.verified).count, pairCounts: counts)
        }
        let artifacts = try loadArtifacts(), checkpoints = try loadCheckpoints()
        return ProjectSnapshot(sourceURL: sourceFolder, projectID: document.id, settings: document.settings,
            images: images, datasets: datasets, artifacts: artifacts, checkpoints: checkpoints,
            modelVersions: try loadModelVersions(artifacts: artifacts, checkpoints: checkpoints),
            modelCoverage: try loadModelCoverage(datasets: datasets, artifacts: artifacts, checkpoints: checkpoints))
    }

    public func updateSettings(_ settings: ProjectSettings) throws {
        guard !settings.modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw LoomError("Give Texture Loom a model name.") }
        document.settings = settings
        try saveDocument()
    }
    public func importImages(_ urls: [URL]) throws {
        let destination = sourceFolder.appendingPathComponent("Imported", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for original in urls {
            let target = original.resolvingSymlinksInPath().standardizedFileURL
            guard FileManager.default.fileExists(atPath: target.path) else { throw LoomError("\(original.lastPathComponent) was removed before import.") }
            // Imports reference the original, so image bytes are never duplicated.
            if target.path.hasPrefix(sourceFolder.path + "/") { continue }
            var link = destination.appendingPathComponent(original.lastPathComponent)
            if FileManager.default.fileExists(atPath: link.path) {
                if link.resolvingSymlinksInPath() == target { continue }
                link = destination.appendingPathComponent(UUID().uuidString.prefix(8) + "-" + original.lastPathComponent)
            }
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        }
    }
    public func updateImage(id: String, category: String?, verified: Bool, note: String) throws {
        let images = try scanImages()
        guard let image = images.first(where: { $0.id == id }) else { throw LoomError("This image was removed from the source folder. The viewer has been refreshed.") }
        let clean = category?.trimmingCharacters(in: .whitespacesAndNewlines)
        let label = clean?.isEmpty == false ? clean.map(ImageMetadataStore.friendlyName) : nil
        // A material folder has one category across all map resolutions. A
        // review also corrects earlier resolution-specific classifier tags.
        let folder = image.url.deletingLastPathComponent().standardizedFileURL
        for member in images where member.url.deletingLastPathComponent().standardizedFileURL == folder {
            guard var metadata = try ImageMetadataStore.read(member.url) else { throw LoomError("Image metadata changed. Refresh the folder and try again.") }
            metadata.category = label; metadata.verified = verified; metadata.note = note
            try ImageMetadataStore.write(metadata, to: member.url)
        }
    }
    public func setRole(id: String, role: MapRole) throws {
        guard let image = try scanImages().first(where: { $0.id == id }), var metadata = try ImageMetadataStore.read(image.url) else {
            throw LoomError("This image was removed from the source folder.")
        }
        metadata.role = role
        metadata.roleWasManuallySelected = true
        try ImageMetadataStore.write(metadata, to: image.url)
    }
    public func setIgnored(id: String, ignored: Bool) throws {
        guard let image = try scanImages().first(where: { $0.id == id }), var metadata = try ImageMetadataStore.read(image.url) else {
            throw LoomError("This image was removed from the source folder.")
        }
        metadata.ignored = ignored
        try ImageMetadataStore.write(metadata, to: image.url)
    }
    public func pairImages(ids: [String]) throws {
        let images = try scanImages().filter { ids.contains($0.id) }
        guard images.count == ids.count, !images.isEmpty else { throw LoomError("Choose images currently in this source folder.") }
        guard Set(images.map { "\($0.width)x\($0.height)" }).count == 1 else { throw LoomError("Paired maps must have the same pixel dimensions.") }
        if images.filter({ $0.role == .diffuse }).count > 1 {
            guard Set(images.map { $0.url.deletingLastPathComponent().standardizedFileURL }).count == 1 else {
                throw LoomError("Keep diffuse variants and their shared maps in one material folder so training and validation stay separate.")
            }
        }
        let targets = images.filter { $0.role != .diffuse }
        guard Set(targets.map(\.role)).count == targets.count else {
            throw LoomError("A material pair can contain several diffuse variants and one image of each other map type.")
        }
        let key = UUID().uuidString.lowercased()
        for image in images {
            guard var metadata = try ImageMetadataStore.read(image.url) else { throw LoomError("Refresh the source folder before pairing these images.") }
            metadata.pairingKey = key
            metadata.pairingWasManuallySelected = true
            try ImageMetadataStore.write(metadata, to: image.url)
        }
    }
    public func setNormalConvention(id: String, convention: String) throws {
        guard ["opengl", "directx"].contains(convention) else { throw LoomError("Choose OpenGL or DirectX normal format.") }
        guard let image = try scanImages().first(where: { $0.id == id }), image.role == .normal,
              var metadata = try ImageMetadataStore.read(image.url) else { throw LoomError("Choose a current normal map.") }
        metadata.normalConvention = convention
        try ImageMetadataStore.write(metadata, to: image.url)
    }
    public func removeImageMetadata(id: String) throws {
        guard let image = try scanImages().first(where: { $0.id == id }) else { return }
        try ImageMetadataStore.clear(image.url)
    }
    public func cancelTraining() {
        activeTraining?.cancel()
        if let activeProcess, activeProcess.isRunning { activeProcess.terminate() }
    }

    func saveDocument() throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(document).write(to: projectFolder.appendingPathComponent("project.json"), options: .atomic)
    }
    func loadArtifacts() throws -> [ModelArtifact] {
        let runs = projectFolder.appendingPathComponent("runs", isDirectory: true)
        guard FileManager.default.fileExists(atPath: runs.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: runs, includingPropertiesForKeys: nil)
            .compactMap { folder in
                let path = folder.appendingPathComponent("artifact.json")
                guard FileManager.default.fileExists(atPath: path.path) else { return nil }
                // One damaged run must not hide the other groups' saved models.
                // Explicit choices still surface as unavailable until replaced.
                guard let saved = try? JSONDecoder().decode(ModelArtifact.self, from: Data(contentsOf: path)) else { return nil }
                return ModelArtifact(id: saved.id, kind: saved.kind, datasetID: saved.datasetID,
                    datasetName: saved.datasetName, map: saved.map, directory: folder,
                    createdAt: saved.createdAt, epochs: saved.epochs, loss: saved.loss,
                    coreMLPath: saved.coreMLPath, weightsPath: saved.weightsPath, architecture: saved.architecture)
            }.sorted { $0.createdAt > $1.createdAt }
    }
    func scanImages() throws -> [SourceImage] {
        let supported: Set<String> = ["png", "jpg", "jpeg", "tif", "tiff", "heic", "heif", "webp", "bmp", "exr"]
        var result: [SourceImage] = []
        var visitedDirectories: Set<String> = []
        var visitedFiles: Set<String> = []
        var imageIDs: Set<String> = []
        func visit(_ folder: URL) throws {
            let resolved = folder.resolvingSymlinksInPath().standardizedFileURL
            guard visitedDirectories.insert(resolved.path).inserted else { return }
            for entry in try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) {
                // Retain the in-folder link path for imported external originals.
                let file = folder.appendingPathComponent(entry.lastPathComponent)
                let target = file.resolvingSymlinksInPath().standardizedFileURL
                var isDirectory: ObjCBool = false
                guard FileManager.default.fileExists(atPath: target.path, isDirectory: &isDirectory) else { continue }
                if isDirectory.boolValue { try visit(file); continue }
                guard supported.contains(file.pathExtension.lowercased()), visitedFiles.insert(target.path).inserted else { continue }
                guard let source = CGImageSourceCreateWithURL(target as CFURL, nil),
                      let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                      let width = props[kCGImagePropertyPixelWidth] as? Int,
                      let height = props[kCGImagePropertyPixelHeight] as? Int else {
                    throw LoomError("Could not read \(file.lastPathComponent). The file remains in the source folder; refresh after it finishes copying.")
                }
                var metadata = try ImageMetadataStore.read(target)
                if metadata == nil {
                    metadata = ImageMetadataStore.initialMetadata(file: file, sourceRoot: sourceFolder, width: width, height: height)
                    try ImageMetadataStore.write(metadata!, to: target)
                }
                var tag = metadata!
                let variantTag = ImageMetadataStore.repairingColorVariantInference(tag, file: file, sourceRoot: sourceFolder, width: width, height: height)
                let repaired = ImageMetadataStore.repairingLegacyInference(variantTag, file: file, sourceRoot: sourceFolder, width: width, height: height)
                if repaired != tag {
                    tag = repaired
                    try ImageMetadataStore.write(tag, to: target)
                }
                if tag.role == .normal, tag.normalConvention?.lowercased() == "directx", tag.ignored != true {
                    tag.ignored = true
                    try ImageMetadataStore.write(tag, to: target)
                }
                // Copying a file copies its metadata too. Each distinct image
                // needs its own selection identity while retaining its teaching tags.
                if !imageIDs.insert(tag.id).inserted {
                    tag.id = UUID().uuidString.lowercased()
                    imageIDs.insert(tag.id)
                    try ImageMetadataStore.write(tag, to: target)
                }
                result.append(SourceImage(id: tag.id, url: file, displayName: ImageMetadataStore.friendlyName(file.deletingPathExtension().lastPathComponent),
                    role: tag.role, category: tag.category, verified: tag.verified, note: tag.note,
                    pairingKey: tag.pairingKey, width: width, height: height, normalConvention: tag.normalConvention, ignored: tag.ignored == true))
            }
        }
        try visit(sourceFolder)
        return result.sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }
    }
}
