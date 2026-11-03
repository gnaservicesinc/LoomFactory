import CryptoKit
import Darwin
import Foundation
import ImageIO
import LoomNative
import zlib

public struct ClosedCategory: Identifiable, Sendable, Equatable {
    public let id: String
    public let name: String
    public let closedAt: Date
    public let fileCount: Int
    public let materialCount: Int
    public let sourceBytes: Int64
    public let manifestURL: URL
}

public struct CategoryClosePreview: Identifiable, Sendable {
    public let id = UUID()
    public var categoryID: String { manifest.categoryID }
    public var name: String { manifest.name }
    public var fileCount: Int { manifest.files.count }
    public var materialCount: Int { Set(manifest.files.map { $0.pageURL }).count }
    public var preservedFileCount: Int { manifest.preservedFileCount }
    public var sourceBytes: Int64 { manifest.files.reduce(0) { $0 + $1.size } }
    public var formattedSize: String { ByteCountFormatter.string(fromByteCount: sourceBytes, countStyle: .file) }
    let manifest: CategoryArchiveManifest
    let stamps: [String: FileStamp]
}

/// A recovery recipe for explicitly closed categories, never an active image
/// inventory. Training membership still comes only from current physical files.
struct CategoryArchiveManifest: Codable, Sendable {
    var schema = "loom-factory-closed-category-v1"
    let projectID: String
    let categoryID: String
    let name: String
    let closedAt: Date
    let preservedFileCount: Int
    let preservedMaterials: [String]
    let files: [File]
    struct File: Codable, Sendable {
        let relativePath: String
        let provider: CategorySourceProvider
        let asset: String
        let pageURL: URL
        let downloadURL: URL
        let archiveEntry: String?
        let downloadSize: Int64
        let size: Int64
        let metadata: ImageMetadata
        let width: Int
        let height: Int
    }
    func summary(at url: URL) -> ClosedCategory {
        .init(id: categoryID, name: name, closedAt: closedAt, fileCount: files.count,
              materialCount: Set(files.map(\.pageURL)).count, sourceBytes: files.reduce(0) { $0 + $1.size }, manifestURL: url)
    }
}

extension ProjectSnapshot {
    public func canCloseCategory(_ datasetID: String) -> Bool {
        guard !closedCategories.contains(where: { $0.id == datasetID }) else { return false }
        return ModelProductID.categoryProducts.allSatisfy { product in
            guard let entry = modelCoverage.first(where: { $0.slot.datasetID == datasetID && $0.slot.product == product }),
                  let chosen = entry.candidates.first(where: { $0.id == entry.effectiveRunID }) else { return false }
            return chosen.isAvailable && chosen.isApproved && !chosen.approvalIsStale
        }
    }
}

extension LoomProject {
    func categoryArchiveURL(_ id: String) throws -> URL {
        guard id.range(of: #"^dataset-[a-f0-9]{16}$"#, options: .regularExpression) != nil else { throw LoomError("Invalid category identity.") }
        return projectFolder.appendingPathComponent("closed-categories/\(id).json")
    }
    func loadCategoryArchives() throws -> [(URL, CategoryArchiveManifest)] {
        let folder = projectFolder.appendingPathComponent("closed-categories")
        guard FileManager.default.fileExists(atPath: folder.path) else { return [] }
        try CategoryArchiveFiles.requireDirectory(folder)
        return try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }.map { url in
                guard (FileStamp.lstat(url.path)?.mode ?? 0) & UInt16(S_IFMT) == UInt16(S_IFREG) else { throw LoomError("Source recovery lists must be regular files.") }
                let manifest = try JSONDecoder().decode(CategoryArchiveManifest.self, from: Data(contentsOf: url))
                guard manifest.schema == "loom-factory-closed-category-v1", manifest.projectID == document.id,
                      manifest.categoryID == ImageMetadataStore.categoryID(manifest.name),
                      try categoryArchiveURL(manifest.categoryID).standardizedFileURL.path == url.standardizedFileURL.path,
                      Set(manifest.files.map(\.relativePath)).count == manifest.files.count else {
                    throw LoomError("A category recovery list does not match this project. Its files were left unchanged.")
                }
                for file in manifest.files {
                    // A conflicting file/link must not hide the closed category
                    // or its models. Enforce physical path checks on reopening.
                    _ = try CategoryArchiveFiles.destination(file.relativePath, root: sourceFolder, checkingExistingPaths: false)
                    guard file.metadata.category == manifest.name,
                          let identity = CategorySourceIdentity.recognize(URL(fileURLWithPath: file.relativePath)),
                          identity.provider == file.provider, identity.asset == file.asset,
                          file.pageURL == identity.pageURL, file.size > 0, file.downloadSize > 0,
                          file.width > 0, file.height > 0,
                          file.downloadURL.scheme == "https" else { throw LoomError("Invalid source recovery recipe.") }
                    switch file.provider {
                    case .polyHaven:
                        guard file.downloadURL.host == "dl.polyhaven.org", file.archiveEntry == nil,
                              file.downloadURL.lastPathComponent == URL(fileURLWithPath: file.relativePath).lastPathComponent else { throw LoomError("Invalid Poly Haven recovery URL.") }
                    case .ambientCG:
                        guard file.downloadURL.host?.lowercased() == "ambientcg.com", file.downloadURL.path == "/get",
                              URLComponents(url: file.downloadURL, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "file" })?.value == identity.package,
                              file.archiveEntry == URL(fileURLWithPath: file.relativePath).lastPathComponent else { throw LoomError("Invalid ambientCG recovery URL.") }
                    }
                }
                return (url, manifest)
            }
    }

    /// This is read-only apart from the existing scan/lease metadata. Nothing is
    /// deleted until the user confirms the concrete preview.
    public func previewCategoryClose(datasetID: String, progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> CategoryClosePreview {
        try await previewCategoryClose(datasetID: datasetID, session: CategorySourceCatalog.session(), progress: progress)
    }
    func previewCategoryClose(datasetID: String, session: URLSession, progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> CategoryClosePreview {
        defer { session.invalidateAndCancel() }
        guard !operationActive else { throw LoomError("Finish the current project operation before closing a category.") }
        let lease = try TrainingQueueLease.requireExclusive(projectFolder: projectFolder)
        defer { withExtendedLifetime(lease) {} }
        operationActive = true
        defer { operationActive = false }
        let current = try snapshot()
        guard current.canCloseCategory(datasetID), let dataset = current.datasets.first(where: { $0.id == datasetID }) else {
            throw LoomError("Approve the current depth, normal and roughness models before closing this category.")
        }
        var catalogs: [String: Data] = [:], directories: [URL: [CategorySourceCatalog.ZIPEntry]] = [:]
        var files: [CategoryArchiveManifest.File] = [], stamps: [String: FileStamp] = [:]
        let members = current.images.filter { $0.category == dataset.name }
        var kept: Set<String> = []
        for (index, image) in members.enumerated() {
            try Task.checkCancellation()
            let path = String(image.url.path.dropFirst(sourceFolder.path.count + 1))
            guard let identity = CategorySourceIdentity.recognize(image.url),
                  let destination = try? CategoryArchiveFiles.destination(path, root: sourceFolder),
                  let stamp = FileStamp.lstat(destination.path), stamp.mode & UInt16(S_IFMT) == UInt16(S_IFREG),
                  let metadata = try ImageMetadataStore.read(destination) else {
                kept.insert(image.url.deletingLastPathComponent().lastPathComponent); continue
            }
            progress(.init(type: "category_close", message: "Verifying source \(index + 1) of \(members.count): \(image.url.lastPathComponent)…"))
            let key = identity.provider.rawValue + ":" + identity.asset
            if catalogs[key] == nil { catalogs[key] = try await CategorySourceCatalog.catalog(identity, session: session, allowMissing: true) }
            let published: CategorySourceCatalog.Published?
            switch identity.provider {
            case .polyHaven: published = try CategorySourceCatalog.polyFile(catalogs[key]!, filename: image.url.lastPathComponent)
            case .ambientCG: published = try CategorySourceCatalog.ambientPackage(catalogs[key]!, identity: identity)
            }
            guard let published else { kept.insert(image.url.deletingLastPathComponent().lastPathComponent); continue }
            let digest = try CategoryArchiveFiles.digest(destination)
            let matches: Bool
            if identity.provider == .polyHaven {
                matches = stamp.size == published.size && digest.md5 == published.md5
            } else {
                if directories[published.url] == nil { directories[published.url] = try await CategorySourceCatalog.zipEntries(published, session: session) }
                matches = directories[published.url]!.contains { $0.name == image.url.lastPathComponent && $0.size == stamp.size && $0.crc32 == digest.crc32 }
            }
            guard stamp == FileStamp.lstat(destination.path) else { throw LoomError("\(image.url.lastPathComponent) changed during verification. Check the category again.") }
            guard matches else { kept.insert(image.url.deletingLastPathComponent().lastPathComponent); continue }
            files.append(.init(relativePath: path, provider: identity.provider, asset: identity.asset, pageURL: identity.pageURL,
                downloadURL: published.url, archiveEntry: identity.provider == .ambientCG ? image.url.lastPathComponent : nil,
                downloadSize: published.size, size: stamp.size, metadata: metadata, width: image.width, height: image.height))
            stamps[path] = stamp
        }
        let manifest = CategoryArchiveManifest(projectID: document.id, categoryID: dataset.id, name: dataset.name, closedAt: .now,
            preservedFileCount: members.count - files.count, preservedMaterials: kept.sorted(), files: files)
        return .init(manifest: manifest, stamps: stamps)
    }

    public func closeCategory(_ preview: CategoryClosePreview) throws {
        guard !operationActive else { throw LoomError("Finish the current project operation before closing a category.") }
        let lease = try TrainingQueueLease.requireExclusive(projectFolder: projectFolder)
        defer { withExtendedLifetime(lease) {} }
        guard preview.manifest.projectID == document.id, try snapshot().canCloseCategory(preview.categoryID) else {
            throw LoomError("This category or its approvals changed. Check the category again before closing.")
        }
        // Check every planned deletion before recording the recipe or deleting
        // anything. Generated, edited and shared/imported paths never enter it.
        for file in preview.manifest.files {
            let destination = try CategoryArchiveFiles.destination(file.relativePath, root: sourceFolder)
            guard let stamp = preview.stamps[file.relativePath], FileStamp.lstat(destination.path) == stamp,
                  try ImageMetadataStore.read(destination) == file.metadata else { throw LoomError("Source files changed after the preview. Check the category again.") }
        }
        try Task.checkCancellation()
        let url = try categoryArchiveURL(preview.categoryID)
        let folder = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try CategoryArchiveFiles.requireDirectory(folder)
        guard !FileManager.default.fileExists(atPath: url.path) else { throw LoomError("This category is already closed. Reopen it from Models.") }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(preview.manifest).write(to: url, options: .atomic)
        // Durable recovery instructions precede the first unlink. An interrupted
        // close remains closed and can restore all missing files on reopening.
        let handle = try FileHandle(forWritingTo: url); try handle.synchronize(); try handle.close()
        for file in preview.manifest.files {
            try Task.checkCancellation()
            let destination = try CategoryArchiveFiles.destination(file.relativePath, root: sourceFolder)
            guard FileStamp.lstat(destination.path) == preview.stamps[file.relativePath] else { throw LoomError("A source changed while closing. Remaining files were kept; reopen the category to restore removed files.") }
            try FileManager.default.removeItem(at: destination)
        }
    }

    public func reopenCategory(datasetID: String, progress: @escaping @Sendable (TrainingEvent) -> Void) async throws {
        try await reopenCategory(datasetID: datasetID, session: CategorySourceCatalog.session(), progress: progress)
    }
    func reopenCategory(datasetID: String, session: URLSession, progress: @escaping @Sendable (TrainingEvent) -> Void) async throws {
        defer { session.invalidateAndCancel() }
        guard !operationActive else { throw LoomError("Finish the current project operation before reopening a category.") }
        let lease = try TrainingQueueLease.requireExclusive(projectFolder: projectFolder)
        defer { withExtendedLifetime(lease) {} }
        operationActive = true
        defer { operationActive = false }
        guard let (manifestURL, manifest) = try loadCategoryArchives().first(where: { $0.1.categoryID == datasetID }) else { throw LoomError("Choose a closed category to reopen.") }
        var catalogs: [String: Data] = [:]
        let groups = Dictionary(grouping: manifest.files, by: \.downloadURL)
        for url in groups.keys.sorted(by: { $0.absoluteString < $1.absoluteString }) {
            try Task.checkCancellation()
            let files = groups[url]!
            var missing: [CategoryArchiveManifest.File] = []
            for file in files {
                let destination = try CategoryArchiveFiles.destination(file.relativePath, root: sourceFolder)
                if FileStamp.lstat(destination.path) != nil {
                    guard try CategoryArchiveFiles.isRestored(file, at: destination) else { throw LoomError("\(destination.lastPathComponent) already exists with different bytes or tags. It was kept; resolve the conflict before reopening again.") }
                } else { missing.append(file) }
            }
            guard !missing.isEmpty else { continue }
            // Resolve today's provider release. Saved byte hashes must never
            // prevent reopening after the provider fixes or republishes a map.
            let identity = CategorySourceIdentity.recognize(URL(fileURLWithPath: missing[0].relativePath))!
            let key = identity.provider.rawValue + ":" + identity.asset
            if catalogs[key] == nil { catalogs[key] = try await CategorySourceCatalog.catalog(identity, session: session) }
            let published: CategorySourceCatalog.Published?
            switch identity.provider {
            case .polyHaven: published = try CategorySourceCatalog.polyFile(catalogs[key]!, filename: URL(fileURLWithPath: missing[0].relativePath).lastPathComponent)
            case .ambientCG: published = try CategorySourceCatalog.ambientPackage(catalogs[key]!, identity: identity)
            }
            guard let published else { throw LoomError("The provider no longer publishes this source file or resolution. The category remains closed; restored and existing files were kept.") }
            // Budget the archive plus the uncompressed files and keep a reserve.
            // Download only one provider package at a time.
            let needed = published.size + missing.reduce(0) { $0 + $1.size } + 1_073_741_824
            let free = try sourceFolder.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage ?? 0
            guard free >= needed else { throw LoomError("Reopening needs at least \(ByteCountFormatter.string(fromByteCount: needed, countStyle: .file)) free for the next download and its restored files. Free some space and press Reopen Category again.") }
            progress(.init(type: "category_reopen", message: "Downloading \(missing[0].asset) from \(missing[0].provider == .polyHaven ? "Poly Haven" : "ambientCG")…"))
            let (temporary, response) = try await session.download(from: published.url, delegate: SourceDownloadProgress(name: missing[0].asset, progress: progress))
            defer { try? FileManager.default.removeItem(at: temporary) }
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  try temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize == Int(published.size) else { throw LoomError("The source download was incomplete. Reopen again to retry; completed files were kept.") }
            if let md5 = published.md5, try CategoryArchiveFiles.digest(temporary).md5 != md5 {
                throw LoomError("The download does not match the provider's current checksum. Reopen again to retry.")
            }
            for file in missing {
                try Task.checkCancellation()
                let destination = try CategoryArchiveFiles.destination(file.relativePath, root: sourceFolder)
                try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                let staged = destination.deletingLastPathComponent().appendingPathComponent(".loom-restore-\(UUID().uuidString).tmp")
                defer { try? FileManager.default.removeItem(at: staged) }
                if let entry = file.archiveEntry { try await CategoryArchiveFiles.extract(entry, from: temporary, to: staged) }
                else { try FileManager.default.copyItem(at: temporary, to: staged) }
                guard let source = CGImageSourceCreateWithURL(staged as CFURL, nil),
                      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                      properties[kCGImagePropertyPixelWidth] as? Int == file.width,
                      properties[kCGImagePropertyPixelHeight] as? Int == file.height else { throw LoomError("Restored \(destination.lastPathComponent) is not a valid image at the saved native resolution. The category remains closed; existing files were kept.") }
                try ImageMetadataStore.write(file.metadata, to: staged)
                try Task.checkCancellation()
                _ = try CategoryArchiveFiles.destination(file.relativePath, root: sourceFolder)
                try FileManager.default.moveItem(at: staged, to: destination)
                progress(.init(type: "category_reopen", message: "Restored verified original: \(destination.lastPathComponent)"))
            }
        }
        // Never remove the recipe on partial failure, cancellation or conflict.
        for file in manifest.files {
            guard try CategoryArchiveFiles.isRestored(file, at: CategoryArchiveFiles.destination(file.relativePath, root: sourceFolder)) else { throw LoomError("Reopening is incomplete. Retry to restore the remaining originals.") }
        }
        try Task.checkCancellation()
        try FileManager.default.removeItem(at: manifestURL)
        progress(.init(type: "category_reopen", message: "\(manifest.name) reopened using current provider releases. All saved source files and tags are restored."))
    }
}

enum CategoryArchiveFiles {
    static func requireDirectory(_ url: URL) throws {
        guard url.standardizedFileURL.path == url.resolvingSymlinksInPath().standardizedFileURL.path,
              (FileStamp.lstat(url.path)?.mode ?? 0) & UInt16(S_IFMT) == UInt16(S_IFDIR) else { throw LoomError("Category storage must be a local directory, without symbolic links.") }
    }
    static func destination(_ path: String, root: URL, checkingExistingPaths: Bool = true) throws -> URL {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !parts.isEmpty, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.hasPrefix(".") && !$0.contains("\0") }) else { throw LoomError("Invalid source recovery path.") }
        let url = root.appendingPathComponent(path).standardizedFileURL
        guard url.path.hasPrefix(root.path + "/") else {
            throw LoomError("Linked or external originals are preserved. Restore into the original local source folder.")
        }
        guard checkingExistingPaths else { return url }
        var ancestor = root
        for (index, part) in parts.enumerated() {
            ancestor.appendPathComponent(String(part))
            if let stamp = FileStamp.lstat(ancestor.path) {
                let kind = stamp.mode & UInt16(S_IFMT)
                guard kind != UInt16(S_IFLNK), index == parts.count - 1 || kind == UInt16(S_IFDIR) else {
                    throw LoomError("Linked or external originals are preserved. Restore into the original local source folder.")
                }
            }
        }
        return url
    }
    static func digest(_ url: URL) throws -> (md5: String, crc32: UInt32) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var md5 = Insecure.MD5(), crc: uLong = 0
        while try autoreleasepool(invoking: { () throws -> Bool in
            try Task.checkCancellation()
            guard let data = try handle.read(upToCount: 1_048_576), !data.isEmpty else { return false }
            md5.update(data: data)
            crc = data.withUnsafeBytes { zlib.crc32(crc, $0.bindMemory(to: Bytef.self).baseAddress, uInt(data.count)) }
            return true
        }) {}
        return (md5.finalize().map { String(format: "%02x", $0) }.joined(), UInt32(crc))
    }
    static func isRestored(_ file: CategoryArchiveManifest.File, at url: URL) throws -> Bool {
        guard (FileStamp.lstat(url.path)?.mode ?? 0) & UInt16(S_IFMT) == UInt16(S_IFREG),
              let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              properties[kCGImagePropertyPixelWidth] as? Int == file.width,
              properties[kCGImagePropertyPixelHeight] as? Int == file.height,
              try ImageMetadataStore.read(url) == file.metadata else { return false }
        return true
    }
    static func extract(_ entry: String, from archive: URL, to destination: URL) async throws {
        guard !entry.contains("/"), !entry.contains("\\"), !entry.contains("\0"),
              entry.range(of: #"^[A-Za-z0-9_.-]+$"#, options: .regularExpression) != nil else { throw LoomError("Invalid source archive filename.") }
        FileManager.default.createFile(atPath: destination.path, contents: nil)
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-p", archive.path, entry]
        process.standardOutput = output; process.standardError = FileHandle.nullDevice
        try process.run()
        do {
            while process.isRunning { try Task.checkCancellation(); try await Task.sleep(for: .milliseconds(100)) }
        } catch {
            if process.isRunning { process.terminate() }
            process.waitUntilExit()
            throw error
        }
        guard process.terminationStatus == 0 else { throw LoomError("Could not extract the saved source file from ambientCG's archive.") }
    }
}
