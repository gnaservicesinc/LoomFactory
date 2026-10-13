import AppKit
import CoreML
import CoreVideo
import Foundation
import ImageIO

public struct CategorizationSummary: Sendable {
    public let folderCount: Int
    public let predictionCount: Int
    public let reusedFolderCount: Int
    public let conflictingFolders: [URL]
}

extension LoomProject {
    @discardableResult
    public func categorize(using modelURL: URL, progress: @Sendable (Double, String) -> Void) async throws -> CategorizationSummary {
        guard !operationActive else { throw LoomError("An operation is already running in this source project. Stop it before categorizing images.") }
        operationActive = true
        defer { operationActive = false }
        let images = try scanImages()
        let compiled = modelURL.pathExtension == "mlmodelc" ? modelURL : try await MLModel.compileModel(at: modelURL)
        try Task.checkCancellation()
        var labelNames: [String: String] = [:]
        let descriptor = modelURL.deletingLastPathComponent().appendingPathComponent("model.json")
        if let data = try? Data(contentsOf: descriptor),
           let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
           let labels = json["labels"] as? [[String: String]] {
            for label in labels {
                if let id = label["id"], let name = label["name"] {
                    labelNames[id] = name
                    if let source = label["sourceLabel"] { labelNames[source] = name }
                }
            }
        }
        let classifier = try FolderClassifier(modelURL: compiled, labels: labelNames)
        return try await categorizeFolders(images: images, inputWidth: classifier.width, inputHeight: classifier.height, predict: { image in
            try await classifier.predict(image)
        }, progress: progress)
    }

    // The same folder pass serves CoreML inference and regression fixtures.
    // Its plan is temporary: every run derives membership from current files.
    func categorizeFolders(inputWidth: Int, inputHeight: Int,
                           predict: @Sendable (SourceImage) async throws -> String,
                           progress: @Sendable (Double, String) -> Void) async throws -> CategorizationSummary {
        guard !operationActive else { throw LoomError("An operation is already running in this source project. Stop it before categorizing images.") }
        operationActive = true
        defer { operationActive = false }
        return try await categorizeFolders(images: scanImages(), inputWidth: inputWidth, inputHeight: inputHeight,
                                           predict: predict, progress: progress)
    }

    private func categorizeFolders(images: [SourceImage], inputWidth: Int, inputHeight: Int,
                                   predict: @Sendable (SourceImage) async throws -> String,
                                   progress: @Sendable (Double, String) -> Void) async throws -> CategorizationSummary {
        let folders = Dictionary(grouping: images) { $0.url.deletingLastPathComponent().standardizedFileURL }
            .filter { _, members in
                members.contains { $0.role == .diffuse } &&
                (members.contains { $0.category == nil && !$0.verified } || categoryNames(members).count > 1)
            }
            .sorted { $0.key.path.localizedStandardCompare($1.key.path) == .orderedAscending }
        var completed = 0, predictions = 0, reused = 0
        var conflicts: [URL] = []
        for (index, folder) in folders.enumerated() {
            try Task.checkCancellation()
            let members = try folder.value.compactMap { image -> SourceImage? in
                guard let tag = try currentTag(image) else { return nil }
                return SourceImage(id: image.id, url: image.url, displayName: image.displayName, role: tag.role,
                    category: tag.category, verified: tag.verified, note: tag.note, pairingKey: tag.pairingKey,
                    width: image.width, height: image.height, normalConvention: tag.normalConvention)
            }
            let categories = categoryNames(members)
            if categories.count > 1 {
                // Existing image tags remain authoritative. Do not guess which
                // of two existing categories was a person's correction.
                conflicts.append(folder.key)
            } else if members.contains(where: { $0.category == nil && !$0.verified }),
                      let input = representative(members, width: inputWidth, height: inputHeight) {
                let category: String
                if let existing = categories.values.first {
                    category = existing
                } else {
                    category = try await predict(input)
                    predictions += 1
                    try Task.checkCancellation()
                }
                // A removed input or an edit made during inference invalidates
                // this prediction; never use it to overwrite the user's decision.
                let changedDuringPrediction = try members.contains { member in
                    guard let tag = try currentTag(member) else { return false }
                    return tag.category != member.category || tag.verified != member.verified
                }
                if !changedDuringPrediction, let tag = try currentTag(input),
                   tag.category == input.category, tag.verified == input.verified {
                    var saved = false
                    for member in members where member.category == nil && !member.verified {
                        try Task.checkCancellation()
                        guard var current = try currentTag(member), current.category == nil, !current.verified else { continue }
                        current.category = category
                        try ImageMetadataStore.write(current, to: member.url)
                        saved = true
                    }
                    if saved {
                        completed += 1
                        if !categories.isEmpty { reused += 1 }
                    }
                }
            }
            progress(Double(index + 1) / Double(folders.count), "Categorized folder \(index + 1) of \(folders.count)")
        }
        return CategorizationSummary(folderCount: completed, predictionCount: predictions,
                                     reusedFolderCount: reused, conflictingFolders: conflicts)
    }

    private func currentTag(_ image: SourceImage) throws -> ImageMetadata? {
        guard FileManager.default.fileExists(atPath: image.url.path) else { return nil }
        let tag: ImageMetadata?
        do { tag = try ImageMetadataStore.read(image.url) }
        catch {
            if !FileManager.default.fileExists(atPath: image.url.path) { return nil }
            throw error
        }
        guard let tag, tag.id == image.id, tag.pairingKey == image.pairingKey, tag.role == image.role else { return nil }
        return tag
    }
}

// CoreML objects stay on their own actor; only category strings cross back to
// the source project. Inference never runs on the application's UI actor.
private actor FolderClassifier {
    nonisolated let width: Int
    nonisolated let height: Int
    private let model: MLModel
    private let inputName: String
    private let imageFormat: OSType?
    private let outputName: String?
    private let labels: [String: String]

    init(modelURL: URL, labels: [String: String]) throws {
        let model = try MLModel(contentsOf: modelURL)
        guard let feature = model.modelDescription.inputDescriptionsByName.first(where: { [.image, .multiArray].contains($0.value.type) }) else {
            throw LoomError("The categorizer has an unsupported image interface.")
        }
        if let constraint = feature.value.imageConstraint {
            width = constraint.pixelsWide; height = constraint.pixelsHigh
            imageFormat = constraint.pixelFormatType
        } else if let constraint = feature.value.multiArrayConstraint {
            let shape = constraint.shape.map(\.intValue)
            guard shape.count == 4, shape[0] == 1, shape[1] == 3, shape[2] > 0, shape[3] > 0 else {
                throw LoomError("The trained classifier has an unsupported RGB tensor shape.")
            }
            width = shape[3]; height = shape[2]; imageFormat = nil
        } else { throw LoomError("The categorizer has an unsupported image interface.") }
        self.model = model; inputName = feature.key
        outputName = model.modelDescription.predictedFeatureName
        self.labels = labels
    }

    func predict(_ image: SourceImage) throws -> String {
        let value = try autoreleasepool {
            if let imageFormat {
                return MLFeatureValue(pixelBuffer: try classifierPixelBuffer(image.url, width: width, height: height, format: imageFormat))
            }
            return MLFeatureValue(multiArray: try classifierTensor(image.url, width: width, height: height))
        }
        let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: value]))
        let label = outputName.flatMap { result.featureValue(for: $0)?.stringValue }
            ?? result.featureNames.sorted().compactMap { name -> String? in
                guard let value = result.featureValue(for: name), value.type == .string else { return nil }
                return value.stringValue
            }.first
        guard let label, !label.isEmpty else { throw LoomError("The categorizer did not return a category for \(image.displayName).") }
        guard let category = labels[label] ?? (labels.isEmpty ? ImageMetadataStore.friendlyName(label) : nil) else {
            throw LoomError("The trained classifier returned a dataset outside its label catalog.")
        }
        return category
    }
}

private func categoryNames(_ images: [SourceImage]) -> [String: String] {
    var names: [String: String] = [:]
    for image in images {
        if let category = image.category { names[ImageMetadataStore.categoryID(category)] = category }
    }
    return names
}

private func representative(_ images: [SourceImage], width: Int, height: Int) -> SourceImage? {
    let diffuse = images.filter { $0.role == .diffuse }.sorted {
        let left = Int64($0.width) * Int64($0.height), right = Int64($1.width) * Int64($1.height)
        return left == right ? $0.url.path < $1.url.path : left < right
    }
    // One small diffuse view is sufficient for a folder's category. Prefer
    // native dimensions that cover the classifier grid without upsampling.
    return diffuse.first { $0.width >= width && $0.height >= height } ?? diffuse.last
}

private func classifierTensor(_ file: URL, width: Int, height: Int) throws -> MLMultiArray {
    let buffer = try classifierPixelBuffer(file, width: width, height: height, format: kCVPixelFormatType_32BGRA)
    let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw LoomError("Could not read the categorizer preview.") }
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    let destination = array.dataPointer.assumingMemoryBound(to: Float.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<height { for x in 0..<width {
        let offset = y * stride + x * 4, position = y * width + x
        destination[position] = Float(pixels[offset + 2]) / 255
        destination[height * width + position] = Float(pixels[offset + 1]) / 255
        destination[2 * height * width + position] = Float(pixels[offset]) / 255
    } }
    return array
}

private func classifierPixelBuffer(_ file: URL, width: Int, height: Int, format: OSType) throws -> CVPixelBuffer {
    guard width > 0, height > 0,
          let source = CGImageSourceCreateWithURL(file as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
          let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height),
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true
          ] as CFDictionary) else {
        throw LoomError("Could not open \(file.lastPathComponent) for categorization.")
    }
    // The classifier view is disposable; paired map training uses native pixels.
    var pixelBuffer: CVPixelBuffer?
    let attributes: [String: Any] = [kCVPixelBufferCGImageCompatibilityKey as String: true,
                                   kCVPixelBufferCGBitmapContextCompatibilityKey as String: true]
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attributes as CFDictionary, &pixelBuffer) == kCVReturnSuccess,
          let buffer = pixelBuffer else { throw LoomError("Could not allocate a categorizer preview.") }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let bitmapInfo = format == kCVPixelFormatType_32ARGB
        ? CGImageAlphaInfo.noneSkipFirst.rawValue
        : CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    guard let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: bitmapInfo) else {
        throw LoomError("Could not render the categorizer preview.")
    }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return buffer
}
