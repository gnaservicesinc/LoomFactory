import CryptoKit
import Darwin
import Foundation
import ImageIO

/// File metadata is the only category/review/pairing authority. No inventory is persisted.
public struct ImageMetadata: Codable, Sendable, Equatable {
    public var schema = "loom-factory.image.v1"
    public var id: String
    public var pairingKey: String
    public var role: MapRole
    public var category: String?
    public var verified: Bool
    public var note: String
    public var ignored: Bool?
    public var normalConvention: String?
    public var roleWasManuallySelected: Bool?
    public var pairingWasManuallySelected: Bool?
    public init(id: String = UUID().uuidString.lowercased(), pairingKey: String,
                role: MapRole = .unknown, category: String? = nil, verified: Bool = false,
                note: String = "", normalConvention: String? = nil,
                roleWasManuallySelected: Bool? = nil, pairingWasManuallySelected: Bool? = nil,
                ignored: Bool? = false) {
        self.id = id; self.pairingKey = pairingKey; self.role = role
        self.category = category; self.verified = verified; self.note = note
        self.normalConvention = normalConvention
        self.roleWasManuallySelected = roleWasManuallySelected
        self.pairingWasManuallySelected = pairingWasManuallySelected
        self.ignored = ignored
    }
}

public enum ImageMetadataStore {
    public static let attribute = "com.gnaservicesinc.loomfactory.image.v1"
    public static func read(_ file: URL) throws -> ImageMetadata? {
        let size = getxattr(file.path, attribute, nil, 0, 0, 0)
        if size < 0 {
            if errno == ENOATTR { return nil }
            throw LoomError("Could not read image metadata for \(file.lastPathComponent): \(String(cString: strerror(errno))).")
        }
        guard size > 0, size < 1024 * 1024 else { throw LoomError("Image metadata could not be read for \(file.lastPathComponent).") }
        var data = Data(count: size)
        let count = data.withUnsafeMutableBytes { getxattr(file.path, attribute, $0.baseAddress, size, 0, 0) }
        guard count == size else { throw LoomError("The image changed while its metadata was being read. Refresh the source folder.") }
        let value = try JSONDecoder().decode(ImageMetadata.self, from: data)
        guard value.schema == "loom-factory.image.v1" else { throw LoomError("This image has metadata from a newer Loom Factory version.") }
        return value
    }
    public static func write(_ metadata: ImageMetadata, to file: URL) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(metadata)
        let result = data.withUnsafeBytes { setxattr(file.path, attribute, $0.baseAddress, data.count, 0, 0) }
        guard result == 0 else { throw LoomError("Could not save image metadata for \(file.lastPathComponent): \(String(cString: strerror(errno))).") }
    }
    public static func clear(_ file: URL) throws {
        guard removexattr(file.path, attribute, 0) == 0 || errno == ENOATTR else {
            throw LoomError("Could not remove Loom metadata from \(file.lastPathComponent).")
        }
    }
    public static func categoryID(_ name: String) -> String {
        let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return "dataset-" + SHA256.hash(data: Data(key.utf8)).map { String(format: "%02x", $0) }.joined().prefix(16)
    }
    public static func friendlyName(_ name: String) -> String {
        name.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "-", with: " ")
            .split(whereSeparator: { $0.isWhitespace }).map { $0.capitalized }.joined(separator: " ")
    }

    static func initialMetadata(file: URL, sourceRoot: URL, width: Int, height: Int) -> ImageMetadata {
        inferredMetadata(file: file, sourceRoot: sourceRoot, width: width, height: height, currentNaming: true)
    }

    /// Repair only an exact filename inference made by earlier versions.
    /// File identity, human review and deliberate role/pair choices remain authoritative.
    static func repairingLegacyInference(_ metadata: ImageMetadata, file: URL, sourceRoot: URL,
                                         width: Int, height: Int) -> ImageMetadata {
        guard metadata.roleWasManuallySelected != true,
              metadata.pairingWasManuallySelected != true else { return metadata }
        let inferred = initialMetadata(file: file, sourceRoot: sourceRoot, width: width, height: height)
        guard inferred.role != .unknown else { return metadata }
        let legacy = inferredMetadata(file: file, sourceRoot: sourceRoot, width: width, height: height, currentNaming: false)
        guard metadata.role == legacy.role, metadata.pairingKey == legacy.pairingKey,
              metadata.role != inferred.role || metadata.pairingKey != inferred.pairingKey else { return metadata }
        var repaired = metadata
        repaired.role = inferred.role; repaired.pairingKey = inferred.pairingKey
        if metadata.role != .normal && inferred.role == .normal { repaired.normalConvention = inferred.normalConvention }
        repaired.roleWasManuallySelected = false; repaired.pairingWasManuallySelected = false
        return repaired
    }

    private static func inferredMetadata(file: URL, sourceRoot: URL, width: Int, height: Int,
                                         currentNaming: Bool) -> ImageMetadata {
        let stem = file.deletingPathExtension().lastPathComponent.lowercased()
        let tokens = stem.split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init)
        var roles: [(MapRole, Set<String>)] = [
            (.diffuse, currentNaming ? ["diff", "diffuse", "albedo", "basecolor", "basecolour", "basecol", "basecolo", "color", "colour", "col", "colo", "col1", "coll1", "coll"]
                                    : ["diff", "diffuse", "albedo", "basecolor", "color", "colour", "col"]),
            (.depth, ["disp", "displacement", "height", "depth"]),
            (.roughness, ["rough", "roughness"]),
            (.normal, ["nor", "normal", "normalgl", "normaldx"])
        ]
        if currentNaming {
            roles.insert((.bump, ["bump", "bumpmap"]), at: 2)
            roles.append((.unknown, ["ao", "ambientocclusion", "metal", "metalness", "metallic", "spec", "specular", "opacity", "alpha"]))
        }
        var role = MapRole.unknown
        var family = tokens
        let choices = roles.compactMap { candidate, words -> (MapRole, Int)? in
            guard let index = currentNaming ? tokens.lastIndex(where: words.contains) : tokens.firstIndex(where: words.contains) else { return nil }
            return (candidate, index)
        }
        let choice = currentNaming ? choices.max { $0.1 < $1.1 } : choices.first
        if let (candidate, index) = choice {
            role = candidate; family.remove(at: index)
            // Poly Haven's first colour variant is the primary paired input.
            // Strip its variant marker as well as the role, keeping resolution.
            if currentNaming, candidate == .diffuse, ["col", "coll"].contains(tokens[index]),
               index + 1 < tokens.count, tokens[index + 1] == "1" {
                family.remove(at: index)
            }
            if currentNaming, candidate == .diffuse, tokens[index] == "coll",
               !(index + 1 < tokens.count && tokens[index + 1] == "1") {
                role = .unknown; family = tokens
            }
            if currentNaming, candidate == .diffuse, index > 0, tokens[index - 1] == "base",
               ["color", "colour", "col", "colo"].contains(tokens[index]) {
                family.remove(at: index - 1)
            }
            // Normal convention is metadata, not part of a family's identity.
            if candidate == .normal { family.removeAll { ["gl", "dx", "opengl", "directx"].contains($0) } }
        }
        let parent = file.deletingLastPathComponent().standardizedFileURL.path
        let relative = parent.hasPrefix(sourceRoot.path) ? String(parent.dropFirst(sourceRoot.path.count)) : parent
        let familyName = family.joined(separator: "_")
        let identity = relative + "/" + (familyName.isEmpty ? "material" : familyName) + ":\(width)x\(height)"
        let pairing = SHA256.hash(data: Data(identity.utf8)).map { String(format: "%02x", $0) }.joined()
        let directX = role == .normal && tokens.contains(where: { ["dx", "directx", "normaldx"].contains($0) })
        return ImageMetadata(pairingKey: pairing, role: role,
            normalConvention: directX ? "directx" : nil,
            roleWasManuallySelected: false, pairingWasManuallySelected: false, ignored: directX)
    }
}
