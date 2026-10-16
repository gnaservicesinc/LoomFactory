import CryptoKit
import Foundation
import ImageIO
import LoomNative

public struct SyntheticPhotoRecipe: Codable, Equatable, Sendable {
    public var schema = "loom.synthetic-photo-recipe.v1"
    public var seed: UInt64 = 20261016
    public var splitSeed: UInt64 = 9587
    public var categories: [String] = []
    public var materialPercentage = 25
    public var cropsPerResolution = 1
    public var variantsPerCrop = 2
    public var includeHigherResolutions = true
    public var includeIdentity = true
    public var exposure = true
    public var blur = true
    public var noise = true
    public var lighting = true
    public var perspective = true
    public var strength = 0.5
    public var maximumOutputGiB = 4
    public init() {}

    public func validate() throws {
        guard schema == "loom.synthetic-photo-recipe.v1", (1...100).contains(materialPercentage),
              (1...8).contains(cropsPerResolution), (1...16).contains(variantsPerCrop),
              strength.isFinite, (0.05...1).contains(strength), (1...64).contains(maximumOutputGiB),
              exposure || blur || noise || lighting || perspective else {
            throw LoomError("Choose at least one degradation, 1–100% of materials, 1–8 crops, 1–16 variants, and a valid output budget.")
        }
    }
    public var hasRestoration: Bool { exposure || blur || noise || lighting }
}

public struct SyntheticPhotoMaterial: Sendable {
    public let group: String
    public let name: String
    public let category: String
    public let images: [SourceImage]
}

public struct SyntheticPhotoPlan: Sendable {
    public let materials: [SyntheticPhotoMaterial]
    public let eligibleMaterialCount: Int
    public let skippedMaterialCount: Int
    public let cropCount: Int
    public let pairCount: Int
    public let estimatedMaximumBytes: Int64

    public init(images: [SourceImage], recipe: SyntheticPhotoRecipe, sourceRoot: URL? = nil) throws {
        try recipe.validate()
        let folders = Dictionary(grouping: images.filter { !$0.ignored }) { $0.url.deletingLastPathComponent().standardizedFileURL.path }
        var eligible: [SyntheticPhotoMaterial] = []
        for (folder, members) in folders.sorted(by: { $0.key < $1.key }) {
            let colors = members.filter { $0.role == .diffuse && $0.width >= 1024 && $0.height >= 1024 &&
                $0.width <= 16384 && $0.height <= 16384 && Int64($0.width)*Int64($0.height) <= 67_108_864 &&
                $0.url.pathExtension.lowercased() != "exr" &&
                (recipe.includeHigherResolutions || ($0.width == 1024 && $0.height == 1024)) }
            let categories = Set(members.compactMap(\.category))
            guard categories.count <= 1, !colors.isEmpty else { continue }
            let category = categories.first ?? "Uncategorized"
            guard recipe.categories.isEmpty || recipe.categories.contains(category) else { continue }
            // One color variant per native dimension level; alternatives remain untouched.
            let levels = Dictionary(grouping: colors) { "\($0.width)x\($0.height)" }.values.compactMap {
                $0.sorted { $0.url.path < $1.url.path }.first
            }.sorted { ($0.width*$0.height, $0.url.path) < ($1.width*$1.height, $1.url.path) }
            let materialName = URL(fileURLWithPath: folder).lastPathComponent
            let prefix = sourceRoot.map { $0.standardizedFileURL.path + "/" }
            let identity = prefix.map { folder.hasPrefix($0) ? String(folder.dropFirst($0.count)) : folder } ?? folder
            eligible.append(.init(group: "material-" + syntheticHash(identity), name: materialName,
                                  category: category, images: levels))
        }
        var selected: [SyntheticPhotoMaterial] = []
        for (_, group) in Dictionary(grouping: eligible, by: \.category).sorted(by: { $0.key < $1.key }) {
            let count = max(1, Int(ceil(Double(group.count)*Double(recipe.materialPercentage)/100)))
            selected += group.sorted { syntheticHash("\(recipe.seed):\($0.group)") < syntheticHash("\(recipe.seed):\($1.group)") }.prefix(count)
        }
        materials = selected
        eligibleMaterialCount = eligible.count
        skippedMaterialCount = folders.count - eligible.count
        cropCount = selected.reduce(0) { $0 + $1.images.reduce(0) { $0 + (($1.width == 1024 && $1.height == 1024) ? 1 : recipe.cropsPerResolution) } }
        let variants = recipe.variantsPerCrop * ((recipe.hasRestoration ? 1 : 0) + (recipe.perspective ? 1 : 0))
        pairCount = cropCount * (variants + (recipe.includeIdentity ? 1 : 0))
        // Conservative uncompressed 16-bit PNG allowance, including masks and clean targets.
        estimatedMaximumBytes = Int64(cropCount) * Int64(8 + recipe.variantsPerCrop * ((recipe.hasRestoration ? 8 : 0) + (recipe.perspective ? 12 : 0))) * 1_048_576
    }
}

func syntheticHash(_ text: String) -> String {
    SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
}

struct SyntheticRandom {
    var state: UInt64
    init(_ text: String) { state = UInt64(syntheticHash(text).prefix(16), radix: 16)! }
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
    mutating func unit() -> Double { Double(next() >> 11) / 9_007_199_254_740_992 }
    mutating func range(_ a: Double, _ b: Double) -> Double { a + (b-a)*unit() }
}

public struct SyntheticPhotoDataResult: Sendable {
    public let directory: URL
    public let materialCount: Int
    public let pairCount: Int
}

public enum SyntheticPhotoData {
    public static func generate(images: [SourceImage], recipe: SyntheticPhotoRecipe, destination: URL,
                                sourceRoot: URL? = nil,
                                cancellation: NativeCancellation = NativeCancellation(),
                                progress: @escaping @Sendable (Double, String) -> Void = { _, _ in }) throws -> SyntheticPhotoDataResult {
        let plan = try SyntheticPhotoPlan(images: images, recipe: recipe, sourceRoot: sourceRoot)
        guard !plan.materials.isEmpty else { throw LoomError("No selected materials have a native color/diffuse image at least 1K. Check tags, ignore state and category selection.") }
        let budget = Int64(recipe.maximumOutputGiB)*1_073_741_824
        guard plan.estimatedMaximumBytes <= budget else {
            throw LoomError("This recipe's conservative output forecast is \(String(format: "%.1f", Double(plan.estimatedMaximumBytes)/1_073_741_824)) GiB. Reduce the material percentage/crops/variants or raise the explicit disk budget.")
        }
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw LoomError("Preserving the existing dataset. Choose a new output folder.") }
        try cancellation.check()
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(recipe).write(to: destination.appendingPathComponent("recipe.json"), options: .atomic)
        var rows: [[String: Any]] = [], sources: [[String: Any]] = []
        var bytes: Int64 = 0
        let cache = NativeImageCache(policy: .stream)
        var sourceHashes: [String: String] = [:], contentGroups: [String: String] = [:]
        var aliases: [String: String] = [:]
        func root(_ group: String) -> String {
            var current = group
            while let parent = aliases[current] { current = parent }
            return current
        }
        func receipt(_ status: String) throws {
            let object: [String: Any] = ["schema": "loom.synthetic-photo-data.v1", "status": status,
                "recipe_file": "recipe.json", "synthetic": true, "trained": false,
                "ready_for_training": false, "target": "artist-authored color, not measured reflectance",
                "pixel_contract": "native 1024 crops; sRGB numeric RGB; 16-bit PNG derivatives; originals untouched",
                "sources": sources, "pairs": rows, "output_image_bytes": bytes,
                "grouping": "material group shared by all crops/levels; identical source bytes coalesced",
                "split_rule": "fixed splitSeed hash: train 80%, validation 10%, test 10%; verify coverage before training"]
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
                .write(to: destination.appendingPathComponent("manifest.json"), options: .atomic)
            func escape(_ value: String) -> String {
                value.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;")
                    .replacingOccurrences(of: ">", with: "&gt;").replacingOccurrences(of: "\"", with: "&quot;")
            }
            let samples = rows.prefix(60).map { row in
                let input = escape(row["input"] as? String ?? ""), target = escape(row["target"] as? String ?? "")
                return "<article><h2>\(escape(row["material"] as? String ?? "")) · \(escape(row["task"] as? String ?? ""))</h2><div><figure><a href='\(input)'><img src='\(input)'></a><figcaption>Synthetic input</figcaption></figure><figure><a href='\(target)'><img src='\(target)'></a><figcaption>Clean artist target</figcaption></figure></div></article>"
            }.joined(separator: "\n")
            let html = "<!doctype html><meta charset='utf-8'><title>Synthetic photo pairs</title><style>body{font:16px system-ui;background:#181b20;color:#eee;margin:32px}article{margin:32px 0}article div{display:flex;gap:16px}figure{margin:0;flex:1;min-width:0}img{width:100%;max-width:512px}a{color:#9bf}figcaption{margin-top:8px}</style><h1>Synthetic photo dataset</h1><p>Status: \(status) · \(rows.count) pairs. These are generated training examples, not model predictions. Showing the first 60 pairs. Click an image for its native 1K pixels.</p><p><a href='recipe.json'>Saved seed and recipe</a> · <a href='manifest.json'>Pair list</a></p>\(samples)"
            try Data(html.utf8).write(to: destination.appendingPathComponent("Review.html"), options: .atomic)
        }
        func save(_ values: [Float], channels: Int, relative: String) throws {
            try cancellation.check()
            let url = destination.appendingPathComponent(relative)
            try ModelTestImage.writePNG(values, channels: channels, width: 1024, height: 1024, to: url, cancellation: cancellation)
            bytes += Int64(try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
            guard bytes <= budget else { throw LoomError("Output reached the configured disk budget; completed pairs are preserved.") }
        }
        do {
            // Coalesce duplicate content before assigning any split. Otherwise
            // a duplicated low tier could move only part of a material's levels.
            for material in plan.materials {
                for image in material.images {
                    try cancellation.check()
                    try validateSource(image)
                    progress(0, "Checking source identity · \(image.displayName)")
                    let hash = try NativeFileHash.sha256(image.url, checkCancellation: cancellation.check)
                    sourceHashes[image.id] = hash
                    if let previous = contentGroups[hash] {
                        let a = root(previous), b = root(material.group)
                        if a != b { aliases[max(a,b)] = min(a,b) }
                    }
                    contentGroups[hash] = material.group
                }
            }
            for material in plan.materials {
                for image in material.images {
                    try cancellation.check()
                    try validateSource(image)
                    let hash = sourceHashes[image.id]!
                    let group = root(material.group)
                    let splitValue = UInt64(syntheticHash("\(recipe.splitSeed):\(group)").prefix(8), radix: 16)! % 100
                    let split = splitValue < 80 ? "train" : splitValue < 90 ? "validation" : "test"
                    sources.append(["id": image.id, "file": image.url.path, "sha256": hash, "group": group,
                                    "width": image.width, "height": image.height])
                    let cropTotal = image.width == 1024 && image.height == 1024 ? 1 : recipe.cropsPerResolution
                    for cropIndex in 0..<cropTotal {
                        try autoreleasepool {
                            var random = SyntheticRandom("\(recipe.seed):\(image.id):\(cropIndex)")
                            let x = Int(random.next() % UInt64(image.width-1024+1)), y = Int(random.next() % UInt64(image.height-1024+1))
                            let crop = NativeCropTransform(x: x, y: y, size: 1024)
                            let clean = try NativeImages.readRGBCrop(from: image.url, crop: crop, imageCache: cache, cancellation: cancellation)
                            let folder = "\(material.group.prefix(25))-\(syntheticHash(image.id).prefix(10))-\(cropIndex)"
                            try FileManager.default.createDirectory(at: destination.appendingPathComponent(folder), withIntermediateDirectories: false)
                            let target = folder + "/clean.png"
                            try save(SyntheticPhotoPixels.planar(clean.values), channels: 3, relative: target)
                            func append(task: String, input: String, masks: [String: String] = [:]) {
                                rows.append(["group": group, "category": material.category, "material": material.name,
                                    "source_id": image.id, "split": split, "task": task, "crop_index": cropIndex,
                                    "input": input, "target": target, "masks": masks])
                                progress(Double(rows.count)/Double(plan.pairCount), "Generated \(rows.count) of \(plan.pairCount) pairs · \(material.name)")
                            }
                            if recipe.includeIdentity { append(task: "photo-identity", input: target) }
                            for variant in 0..<recipe.variantsPerCrop {
                                if recipe.hasRestoration {
                                    let result = try SyntheticPhotoPixels.restore(clean.values, size: 1024, recipe: recipe, random: &random, cancellation: cancellation)
                                    let input = folder + "/restore-\(variant).png"
                                    try save(SyntheticPhotoPixels.planar(result.values), channels: 3, relative: input)
                                    append(task: "photo-restore-rgb", input: input)
                                }
                                if recipe.perspective {
                                    let pitch = random.range(-15, 15)*recipe.strength, yaw = random.range(-15, 15)*recipe.strength
                                    let result = try SyntheticPhotoPixels.perspective(clean.values, size: 1024, pitch: pitch, yaw: yaw, cancellation: cancellation)
                                    let input = folder + "/reframe-\(variant).png"
                                    let inputMask = folder + "/reframe-\(variant)-input-mask.png"
                                    let targetMask = folder + "/reframe-\(variant)-target-mask.png"
                                    try save(SyntheticPhotoPixels.planar(result.values), channels: 3, relative: input)
                                    try save(result.inputMask, channels: 1, relative: inputMask)
                                    try save(result.targetMask, channels: 1, relative: targetMask)
                                    append(task: "photo-reframe", input: input,
                                        masks: ["input_valid": inputMask, "target_visible": targetMask])
                                }
                            }
                        }
                        try receipt("generating")
                    }
                    try validateSource(image)
                    guard try NativeFileHash.sha256(image.url, checkCancellation: cancellation.check) == hash else {
                        throw LoomError("Source changed during generation; completed data is preserved for review.")
                    }
                }
            }
            try cancellation.check()
            try receipt("completed")
            return .init(directory: destination, materialCount: plan.materials.count, pairCount: rows.count)
        } catch {
            try? receipt(error is CancellationError ? "cancelled" : "failed")
            throw error
        }
    }

    private static func validateSource(_ image: SourceImage) throws {
        guard let metadata = try ImageMetadataStore.read(image.url), metadata.id == image.id,
              metadata.role == .diffuse, metadata.ignored != true, metadata.category == image.category,
              let source = CGImageSourceCreateWithURL(image.url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue == image.width,
              (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue == image.height,
              ((properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1) == 1 else {
            throw LoomError("Source pixels/tags changed or require orientation correction: \(image.displayName). Refresh first.")
        }
        guard let decoded = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary),
              decoded.bitsPerComponent <= 16, [.none, .noneSkipFirst, .noneSkipLast].contains(decoded.alphaInfo),
              decoded.colorSpace?.name == CGColorSpace.sRGB || decoded.colorSpace?.model == .monochrome else {
            throw LoomError("Synthetic v1 requires opaque sRGB/gray color sources up to 16-bit. Alpha, HDR and other color profiles need a separate conversion derivative: \(image.displayName).")
        }
    }
}
