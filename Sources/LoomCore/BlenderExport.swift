import CryptoKit
import Foundation
import ImageIO

/// A frozen test result, never an implicit lookup of currently selected weights.
public struct BlenderTestMap: Sendable {
    public let result: ModelTestResult
    public let modelName: String
    public init(result: ModelTestResult, modelName: String) {
        self.result = result; self.modelName = modelName
    }
}

public enum BlenderMaterialExport {
    public static var scriptURL: URL? {
        Bundle.main.url(forResource: "BlenderExport", withExtension: "py") ??
            Bundle.module.url(forResource: "BlenderExport", withExtension: "py")
    }

    public static func checksum(_ url: URL) throws -> String {
        SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe))
            .map { String(format: "%02x", $0) }.joined()
    }

    /// Writes a new portable export package. Images are copied byte-for-byte;
    /// no model inference, training, source mutation or assignment occurs here.
    public static func prepare(_ maps: [BlenderTestMap], sourceImages: [SourceImage],
                               includeSource: Bool, directory: URL) throws -> URL {
        guard let first = maps.first else { throw LoomError("Test a saved model before exporting to Blender.") }
        guard Set(maps.map { $0.result.map }).count == maps.count,
              maps.allSatisfy({ MapRole.targets.contains($0.result.map) }) else {
            throw LoomError("Choose one tested result per map type.")
        }
        let input = first.result.inputURL.standardizedFileURL.resolvingSymlinksInPath()
        let inputHash = try checksum(input)
        let size = try dimensions(input)
        for map in maps {
            let r = map.result
            guard r.inputURL.standardizedFileURL.resolvingSymlinksInPath() == input,
                  r.inputSHA256 == nil || r.inputSHA256 == inputHash,
                  r.width == size.0, r.height == size.1,
                  try dimensions(r.outputURL) == size else {
                throw LoomError("Use tested maps from the same unchanged input at its native dimensions. Run Model again if the input changed.")
            }
        }
        let fm = FileManager.default
        guard !fm.fileExists(atPath: directory.path) else { throw LoomError("Choose a new Blender export folder.") }
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let textures = directory.appendingPathComponent("textures", isDirectory: true)
        try fm.createDirectory(at: textures, withIntermediateDirectories: false)
        var generated: [String: Any] = [:], originals: [String: Any] = [:]
        func copy(_ url: URL, name: String) throws -> [String: Any] {
            let hash = try checksum(url)
            let relative = "textures/" + name + "." + url.pathExtension
            let destination = directory.appendingPathComponent(relative)
            try fm.copyItem(at: url, to: destination)
            guard try checksum(destination) == hash, try checksum(url) == hash else {
                throw LoomError("An image changed during Blender export. The original was left untouched.")
            }
            return ["file": relative, "sha256": hash]
        }
        let diffuse = try copy(input, name: "color")
        guard diffuse["sha256"] as? String == inputHash else { throw LoomError("The test input changed during export. Run Model again.") }
        for map in maps {
            let r = map.result
            var record = try copy(r.outputURL, name: "generated-" + r.map.rawValue)
            record["model_name"] = map.modelName
            record["run_id"] = r.runID
            record["weight_source"] = r.source.rawValue
            record["checkpoint_id"] = r.checkpointID ?? ""
            record["normal_convention"] = "OpenGL"
            let receipt = r.directory.appendingPathComponent("test.json")
            if FileManager.default.fileExists(atPath: receipt.path),
               let proof = try JSONSerialization.jsonObject(with: Data(contentsOf: receipt)) as? [String: Any] {
                guard proof["run_id"] as? String == r.runID,
                      proof["map"] as? String == r.map.rawValue,
                      proof["weight_source"] as? String == r.source.rawValue,
                      proof["checkpoint_id"] as? String == r.checkpointID,
                      proof["input_sha256"] as? String == inputHash else {
                    throw LoomError("The saved test receipt does not match this result. Run Model again.")
                }
                record["test_receipt"] = proof
            }
            generated[r.map.rawValue] = record
        }
        var sourceIssues: [String] = []
        if includeSource {
            for role in MapRole.targets {
                let match = ModelTestOriginalMap.resolve(inputURL: input, map: role, images: sourceImages)
                if let image = match.image {
                    var record = try copy(image.url, name: "source-" + role.rawValue)
                    record["normal_convention"] = image.normalConvention ?? "OpenGL"
                    originals[role.rawValue] = record
                } else if let issue = match.issue { sourceIssues.append(issue) }
            }
        }
        let manifest: [String: Any] = ["schema": "loom-blender-review-v1", "color": diffuse,
            "width": size.0, "height": size.1, "generated": generated,
            "include_source": includeSource, "source": originals, "source_issues": sourceIssues,
            "source_name": input.deletingPathExtension().lastPathComponent,
            "world_strength": 0.5, "displacement_scale": 0.025, "midpoint": 0.5]
        let url = directory.appendingPathComponent("review.json")
        try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
        guard let scriptURL else { throw LoomError("The Blender export resource is missing. Rebuild Loom Factory.") }
        try fm.copyItem(at: scriptURL, to: directory.appendingPathComponent("BlenderExport.py"))
        return url
    }

    private static func dimensions(_ url: URL) throws -> (Int, Int) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let p = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = p[kCGImagePropertyPixelWidth] as? Int, let h = p[kCGImagePropertyPixelHeight] as? Int,
              w > 0, h > 0 else { throw LoomError("Cannot read native image dimensions: \(url.lastPathComponent)") }
        return (w, h)
    }
}
