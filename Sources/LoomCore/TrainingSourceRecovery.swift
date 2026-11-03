import CryptoKit
import Foundation
import ImageIO

/// Resolves published originals directly from the provider. Current physical
/// files and image tags remain the only dataset membership authority.
enum TrainingSourceRecovery {
    struct Material: Sendable {
        let asset: String
        let folder: URL
        let category: String
        let tiers: [Int]
        var diffuseSuffix = "diff"
    }
    struct PublishedFile: Decodable, Sendable {
        let url: URL
        let md5: String
        let size: Int
    }
    static func materials(images: [SourceImage], category: String, settings: TrainingSettings) -> [Material] {
        let inputs = images.filter { $0.category == category && !$0.ignored && $0.role == .diffuse }
        // Match the category's existing tiers, with an automatic ceiling of 8K.
        // 16K originals are requested only with an explicit resolution setting.
        let categoryMaximum = min(8, images.filter {
            $0.category == category && !$0.ignored && $0.width == $0.height &&
            [1024, 2048, 4096, 8192, 16384].contains($0.width)
        }.map { $0.width / 1024 }.max() ?? 1)
        let folders = Dictionary(grouping: inputs) { $0.url.deletingLastPathComponent().standardizedFileURL }
        return folders.keys.sorted(by: { $0.path < $1.path }).flatMap { folder -> [Material] in
            let named = folders[folder]!.compactMap { image -> (String, Int, String)? in
                let name = image.url.deletingPathExtension().lastPathComponent
                guard let range = name.range(of: #"_(diff|coll?[0-9]+)_(1|2|4|8|16)k$"#, options: .regularExpression),
                      ["png", "jpg", "exr"].contains(image.url.pathExtension.lowercased()) else { return nil }
                let asset = String(name[..<range.lowerBound])
                guard asset.range(of: #"^[a-z0-9_]+$"#, options: .regularExpression) != nil else { return nil }
                let components = name[range].split(separator: "_")
                guard let tier = Int(components[1].dropLast()), image.width == tier * 1024, image.height == tier * 1024 else { return nil }
                return (asset, tier, String(components[0]))
            }
            return Dictionary(grouping: named, by: { $0.0 + "|" + $0.2 }).values.sorted(by: { $0[0].0 + $0[0].2 < $1[0].0 + $1[0].2 }).compactMap { group in
                let asset = group[0].0, suffix = group[0].2
                let present = group.map(\.1)
                let cropTier = [1, 2, 4, 8, 16].first { $0 * 1024 >= settings.imageSize } ?? 16
                let base = max(cropTier, present.min() ?? cropTier)
                let maximum = settings.sourceResolutionLevels == 0 ? min(8, max(base, categoryMaximum))
                    : min(16, base * (1 << (settings.sourceResolutionLevels - 1)))
                let tiers = [1, 2, 4, 8, 16].filter { $0 >= base && $0 <= maximum }
                return Material(asset: asset, folder: folder, category: category, tiers: tiers, diffuseSuffix: suffix)
            }
        }
    }

    static func published(_ data: Data, role: MapRole, tier: Int, diffuseSuffix: String = "diff") throws -> PublishedFile? {
        let catalog = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let key: String
        switch role { case .diffuse: key = diffuseSuffix == "diff" ? "Diffuse" : diffuseSuffix; case .depth: key = "Displacement"
        case .roughness: key = "Rough"; case .normal: key = "nor_gl"; default: return nil }
        guard let levels = catalog?[key] as? [String: Any], let formats = levels["\(tier)k"] as? [String: Any],
              let png = formats["png"] else { return nil }
        let file = try JSONDecoder().decode(PublishedFile.self, from: JSONSerialization.data(withJSONObject: png))
        guard file.url.scheme == "https", file.url.host == "dl.polyhaven.org",
              file.size > 0, file.md5.range(of: #"^[a-fA-F0-9]{32}$"#, options: .regularExpression) != nil else {
            throw LoomError("Poly Haven returned an invalid source file description.")
        }
        return file
    }

    static func filename(asset: String, role: MapRole, tier: Int, diffuseSuffix: String = "diff") -> String {
        let suffix: String
        switch role { case .diffuse: suffix = diffuseSuffix; case .depth: suffix = "disp"
        case .roughness: suffix = "rough"; default: suffix = "nor_gl" }
        return "\(asset)_\(suffix)_\(tier)k.png"
    }

    static func recover(images: [SourceImage], sourceRoot: URL, category: String, map: MapRole, settings: TrainingSettings,
                        explicit: Bool = true,
                        progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> Int {
        let materials = materials(images: images, category: category, settings: settings)
        let pending: [(material: Material, missing: [(Int, MapRole)])] = materials.compactMap { material in
            // Existing ignored or deliberately retagged files are not replaced.
            let missing = material.tiers.flatMap { tier in [MapRole.diffuse, map].compactMap { role -> (Int, MapRole)? in
                let file = material.folder.appendingPathComponent(filename(asset: material.asset, role: role, tier: tier, diffuseSuffix: material.diffuseSuffix))
                let existing = images.contains {
                    $0.url.deletingPathExtension().standardizedFileURL == file.deletingPathExtension().standardizedFileURL
                }
                let counterparts = images.filter {
                    $0.category == category && $0.role == (role == .diffuse ? map : .diffuse) &&
                    $0.width == tier * 1024 && $0.height == tier * 1024 &&
                    $0.url.deletingLastPathComponent().standardizedFileURL == material.folder &&
                    $0.url.deletingPathExtension().lastPathComponent == URL(fileURLWithPath: filename(asset: material.asset,
                        role: role == .diffuse ? map : .diffuse, tier: tier, diffuseSuffix: material.diffuseSuffix)).deletingPathExtension().lastPathComponent
                }
                let alreadyPaired = images.contains { image in
                    image.category == category && image.role == role && image.width == tier * 1024 && image.height == tier * 1024 &&
                    (image.role != .normal || image.normalConvention?.lowercased() != "directx") &&
                    image.url.deletingLastPathComponent().standardizedFileURL == material.folder &&
                    counterparts.contains { $0.pairingKey == image.pairingKey }
                }
                return existing || alreadyPaired || FileManager.default.fileExists(atPath: file.path) ? nil : (tier, role)
            } }
            return missing.isEmpty ? nil : (material, missing)
        }
        guard explicit || !pending.isEmpty else { return 0 }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 30 * 60
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let recognizedFolders = Set(materials.map(\.folder))
        let otherFolders = Set(images.filter { $0.category == category && !$0.ignored && $0.role == .diffuse }
            .map { $0.url.deletingLastPathComponent().standardizedFileURL }).subtracting(recognizedFolders)
        progress(.init(type: "source_recovery", message: "Checking source resolution levels for \(category)…"))
        if !otherFolders.isEmpty {
            progress(.init(type: "warning", message: "\(otherFolders.count) material folders have no recognized Poly Haven source filename. Their current files remain available; automatic downloads were not attempted for them."))
        }
        var downloads = 0
        for (material, missing) in pending {
            try Task.checkCancellation()
            progress(.init(type: "source_recovery", message: "Checking Poly Haven: \(material.asset)…"))
            var request = URLRequest(url: URL(string: "https://api.polyhaven.com/files/\(material.asset)")!)
            request.setValue("LoomFactory/1.0 (material training source recovery)", forHTTPHeaderField: "User-Agent")
            let (data, response) = try await session.data(for: request)
            if (response as? HTTPURLResponse)?.statusCode == 404 {
                progress(.init(type: "warning", message: "No Poly Haven source found for \(material.asset); using its current local files."))
                continue
            }
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                throw LoomError("Could not check Poly Haven sources for \(material.asset) (HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)). Training has not started; retry when the provider is available.")
            }
            for (tier, role) in missing {
                try Task.checkCancellation()
                let name = filename(asset: material.asset, role: role, tier: tier, diffuseSuffix: material.diffuseSuffix)
                guard let file = try published(data, role: role, tier: tier, diffuseSuffix: material.diffuseSuffix) else {
                    progress(.init(type: "warning", message: "Poly Haven does not publish \(name); using available native pairs."))
                    continue
                }
                guard file.url.lastPathComponent == name else {
                    throw LoomError("The published file does not match \(name). Originals were left unchanged.")
                }
                let destination = material.folder.appendingPathComponent(name)
                let delegate = SourceDownloadProgress(name: name, progress: progress)
                progress(.init(type: "source_recovery", message: "Downloading \(name) (\(ByteCountFormatter.string(fromByteCount: Int64(file.size), countStyle: .file)))…"))
                let (temporary, reply) = try await session.download(from: file.url, delegate: delegate)
                defer { try? FileManager.default.removeItem(at: temporary) }
                guard (reply as? HTTPURLResponse)?.statusCode == 200 else { throw LoomError("Could not download \(name). Retry source preparation.") }
                try validate(temporary, published: file, tier: tier)
                try Task.checkCancellation()
                var tag = ImageMetadataStore.initialMetadata(file: destination, sourceRoot: sourceRoot,
                    width: tier * 1024, height: tier * 1024)
                // Preserve deliberate pairing of a surviving counterpart.
                if let counterpart = images.first(where: {
                    $0.url.deletingLastPathComponent().standardizedFileURL == material.folder &&
                    $0.width == tier * 1024 && $0.height == tier * 1024 &&
                    $0.url.deletingPathExtension().lastPathComponent == URL(fileURLWithPath: filename(asset: material.asset, role: role == .diffuse ? map : .diffuse, tier: tier, diffuseSuffix: material.diffuseSuffix)).deletingPathExtension().lastPathComponent
                }) { tag.pairingKey = counterpart.pairingKey }
                tag.category = material.category; tag.role = role
                if role == .normal { tag.normalConvention = "opengl" }
                try ImageMetadataStore.write(tag, to: temporary)
                // moveItem refuses to overwrite a file created during download.
                try FileManager.default.moveItem(at: temporary, to: destination)
                downloads += 1
                progress(.init(type: "source_recovery", message: "Added verified original: \(name)"))
            }
        }
        progress(.init(type: "source_prepared", message: "Source preparation complete: \(downloads) downloaded originals. Rescanning current dataset…"))
        return downloads
    }

    static func validate(_ url: URL, published: PublishedFile, tier: Int) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = Insecure.MD5(), bytes = 0
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            try Task.checkCancellation()
            hash.update(data: data); bytes += data.count
        }
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        guard bytes == published.size, digest == published.md5.lowercased(),
              let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              properties[kCGImagePropertyPixelWidth] as? Int == tier * 1024,
              properties[kCGImagePropertyPixelHeight] as? Int == tier * 1024,
              CGImageSourceGetType(source) as String? == "public.png" else {
            throw LoomError("Downloaded source failed its byte count, checksum, PNG or native-dimension check. No original was replaced.")
        }
    }
}

final class SourceDownloadProgress: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let name: String
    let progress: @Sendable (TrainingEvent) -> Void
    private let lock = NSLock()
    private var lastPercent = -1
    private var lastBytes: Int64 = 0
    init(name: String, progress: @escaping @Sendable (TrainingEvent) -> Void) { self.name = name; self.progress = progress }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let percent = totalBytesExpectedToWrite > 0 ? Int(100 * totalBytesWritten / totalBytesExpectedToWrite) : -1
        let report = lock.withLock { () -> Bool in
            guard percent >= lastPercent + 5 || (percent < 0 && totalBytesWritten - lastBytes >= 1_048_576) else { return false }
            lastPercent = percent; lastBytes = totalBytesWritten; return true
        }
        if report {
            let amount = percent >= 0 ? "\(percent)%" : ByteCountFormatter.string(fromByteCount: totalBytesWritten, countStyle: .file)
            progress(.init(type: "source_recovery", message: "Downloading \(name): \(amount)"))
        }
    }
}

extension LoomProject {
    /// Also exposed separately so users can repair and inspect counts before a run.
    @discardableResult public func prepareTrainingSources(datasetID: String, map: MapRole, settings: TrainingSettings,
        progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> Int {
        let lease = try TrainingQueueLease.requireAvailable(projectFolder: projectFolder)
        defer { withExtendedLifetime(lease) {} }
        guard !operationActive else { throw LoomError("An operation is already running in this source project.") }
        guard MapRole.targets.contains(map) else { throw LoomError("Choose a material map to prepare.") }
        if let issue = settings.validationIssue { throw LoomError(issue) }
        operationActive = true
        defer { operationActive = false }
        return try await recoverTrainingSources(datasetID: datasetID, map: map, settings: settings, progress: progress)
    }
    func recoverTrainingSources(datasetID: String, map: MapRole, settings: TrainingSettings,
        explicit: Bool = true,
        progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> Int {
        let current = try snapshot()
        guard !current.closedCategories.contains(where: { $0.id == datasetID }) else {
            throw LoomError("Reopen this category from Models before preparing sources or training again.")
        }
        guard let dataset = current.datasets.first(where: { $0.id == datasetID }) else {
            throw LoomError("This dataset no longer has images in the source folder. Choose a current dataset.")
        }
        let log = try SourcePreparationLog(url: projectFolder.appendingPathComponent("source-preparation.log"))
        let report: @Sendable (TrainingEvent) -> Void = { event in log.append(event); progress(event) }
        do {
            return try await TrainingSourceRecovery.recover(images: current.images, sourceRoot: sourceFolder, category: dataset.name, map: map,
                                                           settings: settings, explicit: explicit, progress: report)
        } catch {
            report(.init(type: "source_recovery", message: "Source preparation stopped: \(error.localizedDescription)"))
            throw error
        }
    }
}

private final class SourcePreparationLog: @unchecked Sendable {
    private let lock = NSLock()
    private let handle: FileHandle
    init(url: URL) throws {
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }
    deinit { try? handle.close() }
    func append(_ event: TrainingEvent) {
        lock.withLock {
            if var data = try? JSONSerialization.data(withJSONObject: ["timestamp": ISO8601DateFormatter().string(from: .now),
                "event": event.type, "message": event.message], options: [.sortedKeys]) {
                data.append(10); try? handle.write(contentsOf: data)
            }
        }
    }
}
