import Darwin
import Foundation
import ImageIO
import LoomNative

public enum PhotoEnhancerKind: String, Codable, CaseIterable, Sendable, Identifiable {
    case repair, lighting
    public var id: String { rawValue }
    public var product: NativeModelProduct { self == .repair ? .photoRepair : .photoLighting }
    public var title: String { product.title }
    public var recipe: SyntheticPhotoRecipe {
        var recipe = SyntheticPhotoRecipe()
        recipe.perspective = false
        recipe.blur = self == .repair; recipe.noise = self == .repair
        recipe.exposure = self == .lighting; recipe.lighting = self == .lighting
        return recipe
    }
}

/// Stored on imported derivative inputs, never on the original camera/PBR files.
public struct PhotoPairMetadata: Codable, Equatable, Sendable {
    public var schema = "loom.photo-pair.v1"
    public var kind: PhotoEnhancerKind
    public var target: String
    public var group: String
    public var split: String
    public var provenance: String
    public var ignored = false
    public init(kind: PhotoEnhancerKind, target: String, group: String, split: String, provenance: String) {
        self.kind = kind; self.target = target; self.group = group; self.split = split; self.provenance = provenance
    }
}

public struct PhotoTrainingPair: Sendable {
    public let input: URL
    public let target: URL
    public let metadata: PhotoPairMetadata
    public let inputSHA256: String
    public let targetSHA256: String
}

public struct PhotoTrainingData: Sendable {
    public let pairs: [PhotoTrainingPair]
    public let fingerprint: String
    public var training: [PhotoTrainingPair] { pairs.filter { $0.metadata.split == "train" } }
    public var validation: [PhotoTrainingPair] { pairs.filter { $0.metadata.split == "validation" } }
    public var test: [PhotoTrainingPair] { pairs.filter { $0.metadata.split == "test" } }
    public var summary: String {
        "\(training.count) train / \(validation.count) validation / \(test.count) test pairs · \(Set(pairs.map { $0.metadata.group }).count) physical groups"
    }
}

public enum PhotoEnhancerData {
    public static let attribute = "com.gnaservicesinc.loomfactory.photo-pair.v1"
    public static func write(_ metadata: PhotoPairMetadata, to file: URL) throws {
        let data = try JSONEncoder().encode(metadata)
        guard data.withUnsafeBytes({ setxattr(file.path, attribute, $0.baseAddress, data.count, 0, 0) }) == 0 else {
            throw LoomError("Could not save photo-pair metadata: \(String(cString: strerror(errno))).")
        }
    }
    public static func read(_ file: URL) throws -> PhotoPairMetadata? {
        let count = getxattr(file.path, attribute, nil, 0, 0, 0)
        if count < 0, errno == ENOATTR { return nil }
        guard count > 0, count < 65536 else { throw LoomError("Invalid photo-pair metadata on \(file.lastPathComponent).") }
        var data = Data(count: count)
        guard data.withUnsafeMutableBytes({ getxattr(file.path, attribute, $0.baseAddress, count, 0, 0) }) == count else {
            throw LoomError("Photo metadata changed during scanning.")
        }
        let value = try JSONDecoder().decode(PhotoPairMetadata.self, from: data)
        guard value.schema == "loom.photo-pair.v1" else { throw LoomError("Unsupported photo-pair metadata version.") }
        return value
    }
    static func native1K(_ url: URL) throws {
        guard url.pathExtension.lowercased() == "png",
              let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              props[kCGImagePropertyPixelWidth] as? Int == 1024,
              props[kCGImagePropertyPixelHeight] as? Int == 1024,
              (props[kCGImagePropertyDepth] as? Int ?? 8) <= 16,
              (props[kCGImagePropertyHasAlpha] as? Bool ?? false) == false else {
            throw LoomError("Photo training needs opaque native 1024 × 1024 PNG pairs: \(url.lastPathComponent).")
        }
        let profile = (props[kCGImagePropertyProfileName] as? String ?? "sRGB").lowercased()
        guard profile.contains("srgb") || profile.contains("gray") else { throw LoomError("Convert a derivative to sRGB before importing; originals stay untouched.") }
    }
    static func contained(_ relative: String, root: URL) throws -> URL {
        let url = root.appendingPathComponent(relative).resolvingSymlinksInPath().standardizedFileURL
        guard !relative.hasPrefix("/"), url.path.hasPrefix(root.resolvingSymlinksInPath().path + "/") else {
            throw LoomError("A photo pair points outside its dataset folder.")
        }
        return url
    }
    /// Enumerates live file metadata each time. A saved membership list is never authoritative.
    public static func scan(_ root: URL, kind: PhotoEnhancerKind, cancellation: NativeCancellation = .init()) throws -> PhotoTrainingData {
        let root = root.resolvingSymlinksInPath().standardizedFileURL
        guard !FileManager.default.fileExists(atPath: root.appendingPathComponent("IMPORT-INCOMPLETE").path) else { throw LoomError("This photo import is incomplete; import into a new folder.") }
        guard let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey]) else { throw LoomError("Choose an existing photo dataset folder.") }
        var pairs: [PhotoTrainingPair] = [], groups: [String: String] = [:], hashes: [String: String] = [:], identities: [String] = []
        for case let input as URL in files {
            try cancellation.check()
            guard input.pathExtension.lowercased() == "png", let metadata = try read(input), !metadata.ignored, metadata.kind == kind else { continue }
            guard ["train", "validation", "test"].contains(metadata.split), !metadata.group.isEmpty else { throw LoomError("A photo pair has an invalid group/split.") }
            guard input.resolvingSymlinksInPath().path.hasPrefix(root.path + "/") else { throw LoomError("An input points outside its photo dataset.") }
            let target = try contained(metadata.target, root: root)
            try native1K(input); try native1K(target)
            if let split = groups[metadata.group], split != metadata.split { throw LoomError("The same physical material appears in different splits.") }
            groups[metadata.group] = metadata.split
            var pairHashes: [String] = []
            for url in [input, target] {
                let hash = try NativeFileHash.sha256(url, checkCancellation: cancellation.check)
                if let split = hashes[hash], split != metadata.split { throw LoomError("Identical photo bytes appear in different splits. Keep that physical surface together.") }
                hashes[hash] = metadata.split; pairHashes.append(hash)
            }
            pairs.append(.init(input: input, target: target, metadata: metadata, inputSHA256: pairHashes[0], targetSHA256: pairHashes[1]))
            identities.append([input.path.replacingOccurrences(of: root.path + "/", with: ""), metadata.group, metadata.split, metadata.kind.rawValue] .joined(separator: ":") + pairHashes.joined())
        }
        return .init(pairs: pairs.sorted { $0.input.path < $1.input.path }, fingerprint: syntheticHash(identities.sorted().joined(separator: "\n")))
    }
    /// Imports derivative pairs into a NEW project dataset, adding metadata to copies only.
    /// Synthetic recipes are checked by task; perspective/mixed recipes cannot teach a lighting or repair stage accidentally.
    public static func importDataset(manifest: URL, kind: PhotoEnhancerKind, destination: URL, cancellation: NativeCancellation = .init()) throws -> PhotoTrainingData {
        let root = manifest.deletingLastPathComponent()
        guard let document = try JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as? [String: Any],
              let schema = document["schema"] as? String, let rows = document["pairs"] as? [[String: Any]],
              !FileManager.default.fileExists(atPath: destination.path) else { throw LoomError("Choose a supported pair manifest and a new destination.") }
        guard rows.count <= 100000 else { throw LoomError("Import fewer than 100,000 pairs at a time.") }
        let synthetic = schema == "loom.synthetic-photo-data.v1"
        guard synthetic || schema == "photo-capture-diagnostic-split-v1" || schema == "loom.photo-import.v1" else { throw LoomError("Unsupported photo dataset schema.") }
        if synthetic {
            guard document["status"] as? String == "completed" else { throw LoomError("Finish generating this dataset before importing it.") }
            let recipe = try JSONDecoder().decode(SyntheticPhotoRecipe.self, from: Data(contentsOf: root.appendingPathComponent("recipe.json")))
            guard kind == .repair ? !(recipe.exposure || recipe.lighting) : !(recipe.blur || recipe.noise) else {
                throw LoomError("This mixed recipe is not suitable for this stage. Use the Repair or Lighting preset in Synthetic Photos.")
            }
        } else if schema == "photo-capture-diagnostic-split-v1", kind != .repair { throw LoomError("Noisy/clean photographs teach repair, not neutral albedo.") }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let provenance = schema + ":" + (try NativeFileHash.sha256(manifest))
        var count = 0, copiedBytes: Int64 = 0
        try Data("Import in progress; do not train partial data.".utf8).write(to: destination.appendingPathComponent("IMPORT-INCOMPLETE"), options: .atomic)
        do {
            for row in rows {
                try cancellation.check()
                if synthetic, row["task"] as? String == "photo-reframe" { continue }
                guard let group = row["group"] as? String, let split = row["split"] as? String else { throw LoomError("Missing physical group or split.") }
                let input: URL, target: URL
                if let files = row["files"] as? [String: [String: Any]], let a = files["input"]?["native_1024"] as? String, let b = files["reference"]?["native_1024"] as? String {
                    input = URL(fileURLWithPath: a); target = URL(fileURLWithPath: b)
                    for (file, expected) in [(input, files["input"]?["crop_sha256"] as? String), (target, files["reference"]?["crop_sha256"] as? String)] {
                        if let expected, try NativeFileHash.sha256(file) != expected { throw LoomError("A real-photo derivative checksum changed.") }
                    }
                } else if let a = row["input"] as? String, let b = row["target"] as? String {
                    input = try contained(a, root: root); target = try contained(b, root: root)
                } else { throw LoomError("Missing native input/target pair.") }
                try native1K(input); try native1K(target)
                let targetHash = try NativeFileHash.sha256(target)
                let name = String(format: "%06d", count), a = name + "-input.png", b = "target-" + targetHash + ".png"
                let needsTarget = !FileManager.default.fileExists(atPath: destination.appendingPathComponent(b).path)
                let inputBytes = Int64(try input.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
                let targetBytes = needsTarget ? Int64(try target.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) : 0
                guard copiedBytes + inputBytes + targetBytes <= 8*1_073_741_824 else { throw LoomError("Photo-pair import reached its 8 GiB limit. Import a smaller selection into a new folder.") }
                copiedBytes += inputBytes + targetBytes
                try FileManager.default.copyItem(at: input, to: destination.appendingPathComponent(a))
                if needsTarget { try FileManager.default.copyItem(at: target, to: destination.appendingPathComponent(b)) }
                try write(.init(kind: kind, target: b, group: group, split: split, provenance: provenance), to: destination.appendingPathComponent(a))
                count += 1
            }
            guard count > 0 else { throw LoomError("No supported pairs found.") }
            try FileManager.default.removeItem(at: destination.appendingPathComponent("IMPORT-INCOMPLETE"))
            let data = try scan(destination, kind: kind, cancellation: cancellation)
            try Data("Imported \(count) native pairs. Membership and splits live in image metadata. Diagnostic data is not a quality-qualified corpus.\n".utf8).write(to: destination.appendingPathComponent("README.txt"), options: .atomic)
            return data
        } catch {
            // Preserve partial derivatives for inspection, clearly ineligible to train.
            try? Data(error.localizedDescription.utf8).write(to: destination.appendingPathComponent("IMPORT-INCOMPLETE"))
            throw error
        }
    }
}
