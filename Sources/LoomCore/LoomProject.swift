import Foundation
import Darwin
import ImageIO
import LoomNative

struct FileStamp: Equatable, Sendable {
    var dev: Int32
    var ino: UInt64
    var mode: UInt16
    var size: Int64
    var mtimeSec: Int
    var mtimeNsec: Int
    var ctimeSec: Int
    var ctimeNsec: Int

    func sameContent(as other: FileStamp) -> Bool {
        dev == other.dev && ino == other.ino && mode == other.mode && size == other.size
            && mtimeSec == other.mtimeSec && mtimeNsec == other.mtimeNsec
    }

    static func lstat(_ path: String) -> FileStamp? {
        var info = Darwin.stat()
        guard Darwin.lstat(path, &info) == 0 else { return nil }
        return FileStamp(info)
    }

    static func stat(_ path: String) -> FileStamp? {
        var info = Darwin.stat()
        guard fstatat(AT_FDCWD, path, &info, 0) == 0 else { return nil }
        return FileStamp(info)
    }

    private init(_ info: Darwin.stat) {
        self.dev = info.st_dev
        self.ino = info.st_ino
        self.mode = info.st_mode
        self.size = info.st_size
        self.mtimeSec = info.st_mtimespec.tv_sec
        self.mtimeNsec = info.st_mtimespec.tv_nsec
        self.ctimeSec = info.st_ctimespec.tv_sec
        self.ctimeNsec = info.st_ctimespec.tv_nsec
    }
}

private struct CachedScannedImage: Sendable {
    var linkStamp: FileStamp
    var targetStamp: FileStamp
    var xattrData: Data
    var image: SourceImage
}

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
    private var mapTypePredictor: (any MapTypePredictor)?
    private var mapTypeClassifierWasResolved = false
    private var failedMapTypePredictions: Set<String> = []
    private var scannedImageCache: [String: CachedScannedImage] = [:]
    private var cachedSnapshotImages: [SourceImage]?
    private var cachedDatasets: [DatasetCategory]?
    private var artifactFileCache: [String: (stamp: FileStamp, artifact: ModelArtifact)] = [:]
    var runVersionCache: [String: CachedRunVersionEntry] = [:]
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
        let datasets: [DatasetCategory]
        if let cachedSnapshotImages, let cachedDatasets, cachedSnapshotImages == images {
            datasets = cachedDatasets
        } else {
            let categories = Set(images.filter { !$0.ignored }.compactMap(\.category)).sorted { $0.localizedStandardCompare($1) == .orderedAscending }
            let computed = categories.map { category in
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
            cachedSnapshotImages = images
            cachedDatasets = computed
            datasets = computed
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
        metadata.mapTypeInference = nil
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
        guard FileManager.default.fileExists(atPath: runs.path) else {
            artifactFileCache.removeAll()
            return []
        }
        var nextCache: [String: (stamp: FileStamp, artifact: ModelArtifact)] = [:]
        let artifacts = try FileManager.default.contentsOfDirectory(at: runs, includingPropertiesForKeys: nil)
            .compactMap { folder -> ModelArtifact? in
                let path = folder.appendingPathComponent("artifact.json")
                guard let stamp = FileStamp.lstat(path.path) else { return nil }
                if let cached = artifactFileCache[path.path], cached.stamp == stamp {
                    let updated = ModelArtifact(id: cached.artifact.id, kind: cached.artifact.kind,
                        datasetID: cached.artifact.datasetID, datasetName: cached.artifact.datasetName,
                        map: cached.artifact.map, directory: folder, createdAt: cached.artifact.createdAt,
                        epochs: cached.artifact.epochs, loss: cached.artifact.loss,
                        coreMLPath: cached.artifact.coreMLPath, weightsPath: cached.artifact.weightsPath,
                        architecture: cached.artifact.architecture)
                    nextCache[path.path] = (stamp, updated)
                    return updated
                }
                // One damaged run must not hide the other groups' saved models.
                // Explicit choices still surface as unavailable until replaced.
                guard let saved = try? JSONDecoder().decode(ModelArtifact.self, from: Data(contentsOf: path)) else { return nil }
                let artifact = ModelArtifact(id: saved.id, kind: saved.kind, datasetID: saved.datasetID,
                    datasetName: saved.datasetName, map: saved.map, directory: folder,
                    createdAt: saved.createdAt, epochs: saved.epochs, loss: saved.loss,
                    coreMLPath: saved.coreMLPath, weightsPath: saved.weightsPath, architecture: saved.architecture)
                if let after = FileStamp.lstat(path.path), after == stamp {
                    nextCache[path.path] = (stamp, artifact)
                }
                return artifact
            }.sorted { $0.createdAt > $1.createdAt }
        artifactFileCache = nextCache
        return artifacts
    }
    func scanImages() throws -> [SourceImage] {
        let supported: Set<String> = ["png", "jpg", "jpeg", "tif", "tiff", "heic", "heif", "webp", "bmp", "exr"]
        var result: [SourceImage] = []
        var visitedDirectories: Set<String> = []
        var visitedFiles: Set<String> = []
        var imageIDs: Set<String> = []
        var nextCache: [String: CachedScannedImage] = [:]
        nextCache.reserveCapacity(scannedImageCache.count)
        func visit(_ folder: URL) throws {
            let resolved = folder.resolvingSymlinksInPath().standardizedFileURL
            guard visitedDirectories.insert(resolved.path).inserted else { return }
            for entry in try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) {
                try Task.checkCancellation()
                // Retain the in-folder link path for imported external originals.
                let file = folder.appendingPathComponent(entry.lastPathComponent)
                guard let linkStamp = FileStamp.lstat(file.path) else { continue }
                let target: URL
                let targetStamp: FileStamp
                if (linkStamp.mode & S_IFMT) == S_IFLNK {
                    target = file.resolvingSymlinksInPath().standardizedFileURL
                    guard let resolvedStamp = FileStamp.stat(target.path) else { continue }
                    targetStamp = resolvedStamp
                } else {
                    target = resolved.appendingPathComponent(entry.lastPathComponent).standardizedFileURL
                    targetStamp = linkStamp
                }
                if (targetStamp.mode & S_IFMT) == S_IFDIR { try visit(file); continue }
                guard supported.contains(file.pathExtension.lowercased()), visitedFiles.insert(target.path).inserted else { continue }
                let rawXattr = try ImageMetadataStore.rawBytes(target)
                let previous = scannedImageCache[file.path]
                if let rawXattr,
                   let previous,
                   previous.linkStamp == linkStamp,
                   previous.targetStamp == targetStamp,
                   previous.xattrData == rawXattr,
                   imageIDs.insert(previous.image.id).inserted {
                    nextCache[file.path] = previous
                    result.append(previous.image)
                    continue
                }
                let width: Int
                let height: Int
                if let previous, previous.targetStamp.sameContent(as: targetStamp) {
                    width = previous.image.width
                    height = previous.image.height
                } else {
                    guard let source = CGImageSourceCreateWithURL(target as CFURL, nil),
                          let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                          let w = props[kCGImagePropertyPixelWidth] as? Int,
                          let h = props[kCGImagePropertyPixelHeight] as? Int else {
                        throw LoomError("Could not read \(file.lastPathComponent). The file remains in the source folder; refresh after it finishes copying.")
                    }
                    width = w
                    height = h
                }
                var currentXattrData = rawXattr
                var metadata = try rawXattr.map { try ImageMetadataStore.decode($0, file: target) }
                if metadata == nil {
                    let initial = ImageMetadataStore.initialMetadata(file: file, sourceRoot: sourceFolder, width: width, height: height)
                    currentXattrData = try ImageMetadataStore.writeData(initial, to: target)
                    metadata = initial
                }
                var tag = metadata!
                let variantTag = ImageMetadataStore.repairingColorVariantInference(tag, file: file, sourceRoot: sourceFolder, width: width, height: height)
                let repaired = ImageMetadataStore.repairingLegacyInference(variantTag, file: file, sourceRoot: sourceFolder, width: width, height: height)
                if repaired != tag {
                    tag = repaired
                    currentXattrData = try ImageMetadataStore.writeData(tag, to: target)
                }
                let classified = resolveMapType(tag, file: file, width: width, height: height)
                try Task.checkCancellation()
                if classified != tag {
                    // A classifier call may outlive an external metadata edit.
                    // Human edits or metadata removal win over the stale guess.
                    if let latestRaw = try ImageMetadataStore.rawBytes(target) {
                        let current = try ImageMetadataStore.decode(latestRaw, file: target)
                        if current == tag {
                            tag = classified
                            currentXattrData = try ImageMetadataStore.writeData(tag, to: target)
                        } else {
                            tag = current
                            currentXattrData = latestRaw
                        }
                    }
                }
                if tag.role == .normal, tag.normalConvention?.lowercased() == "directx", tag.ignored != true {
                    tag.ignored = true
                    currentXattrData = try ImageMetadataStore.writeData(tag, to: target)
                }
                // Copying a file copies its metadata too. Each distinct image
                // needs its own selection identity while retaining its teaching tags.
                if !imageIDs.insert(tag.id).inserted {
                    tag.id = UUID().uuidString.lowercased()
                    imageIDs.insert(tag.id)
                    currentXattrData = try ImageMetadataStore.writeData(tag, to: target)
                }
                let sourceImage = SourceImage(id: tag.id, url: file, displayName: ImageMetadataStore.friendlyName(file.deletingPathExtension().lastPathComponent),
                    role: tag.role, category: tag.category, verified: tag.verified, note: tag.note,
                    pairingKey: tag.pairingKey, width: width, height: height, normalConvention: tag.normalConvention,
                    ignored: tag.ignored == true, mapTypeInference: tag.mapTypeInference)
                if let finalXattr = currentXattrData,
                   let postTargetStamp = FileStamp.stat(target.path),
                   postTargetStamp.sameContent(as: targetStamp),
                   let postLinkStamp = FileStamp.lstat(file.path) {
                    nextCache[file.path] = CachedScannedImage(linkStamp: postLinkStamp, targetStamp: postTargetStamp, xattrData: finalXattr, image: sourceImage)
                }
                result.append(sourceImage)
            }
        }
        try visit(sourceFolder)
        scannedImageCache = nextCache
        return result.sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }
    }

    // Internal injection keeps tests deterministic without loading a model for
    // every project. The shipping app resolves only its bundled native asset.
    func setMapTypePredictor(_ predictor: (any MapTypePredictor)?) {
        mapTypePredictor = predictor; mapTypeClassifierWasResolved = true
        failedMapTypePredictions.removeAll()
        scannedImageCache.removeAll()
        cachedSnapshotImages = nil
        cachedDatasets = nil
    }

    private func resolveMapType(_ metadata: ImageMetadata, file: URL, width: Int, height: Int) -> ImageMetadata {
        if metadata.roleWasManuallySelected == true {
            var chosen = metadata; chosen.mapTypeInference = nil
            return chosen
        }
        var tag = metadata
        let named = ImageMetadataStore.initialMetadata(file: file, sourceRoot: sourceFolder, width: width, height: height)
        let canClassify = ImageMetadataStore.canClassifyMapType(file: file)
        // An existing guess can be superseded by a newly meaningful filename.
        // Other persisted roles retain their authority across renames.
        if tag.mapTypeInference != nil, named.role != .unknown || !canClassify {
            tag.role = named.role; tag.mapTypeInference = nil
            tag.normalConvention = named.normalConvention
            if tag.pairingWasManuallySelected != true { tag.pairingKey = named.pairingKey }
            if named.ignored == true { tag.ignored = true }
            return tag
        }
        guard canClassify, named.role == .unknown, tag.ignored != true,
              tag.role == .unknown || tag.mapTypeInference != nil else { return tag }
        if !mapTypeClassifierWasResolved {
            mapTypeClassifierWasResolved = true
            mapTypePredictor = NativeMapTypeClassifier.bundled()
        }
        guard let signature = try? MapTypeInferencePolicy.sourceSignature(file: file, width: width, height: height),
              let predictor = mapTypePredictor else {
            if tag.mapTypeInference != nil { tag.role = .unknown; tag.mapTypeInference = nil }
            return tag
        }
        if let cached = tag.mapTypeInference, cached.modelID == predictor.modelID, cached.sourceSignature == signature { return tag }
        let attempt = file.resolvingSymlinksInPath().standardizedFileURL.path + ":" + predictor.modelID + ":" + signature
        guard !failedMapTypePredictions.contains(attempt) else { return tag }
        do {
            let probabilities = try predictor.probabilities(for: file)
            guard (try? MapTypeInferencePolicy.sourceSignature(file: file, width: width, height: height)) == signature else {
                if tag.mapTypeInference != nil { tag.role = .unknown; tag.mapTypeInference = nil }
                return tag
            }
            guard let inference = MapTypeInferencePolicy.evaluate(probabilities,
                                                                 modelID: predictor.modelID, sourceSignature: signature) else {
                failedMapTypePredictions.insert(attempt)
                if tag.mapTypeInference != nil { tag.role = .unknown; tag.mapTypeInference = nil }
                return tag
            }
            tag.role = inference.accepted ? inference.role : .unknown
            tag.mapTypeInference = inference
            return tag
        } catch {
            // A suggestion must never stop folder discovery or repeatedly retry
            // an unreadable image during polling. Editing the file permits retry.
            failedMapTypePredictions.insert(attempt)
            if tag.mapTypeInference != nil { tag.role = .unknown; tag.mapTypeInference = nil }
            return tag
        }
    }
}
