import CoreGraphics
import Foundation
import ImageIO
import LoomNative
import Testing
@testable import LoomCore

@Suite(.serialized)
struct SyntheticPhotoDataTests {
    @Test func selectionIsMaterialStratifiedDeterministicAndIncludesNativeLevels() throws {
        var images: [SourceImage] = []
        for category in ["Wood", "Paper"] {
            for material in 0..<10 {
                for size in [1024, 2048, 4096] {
                    images.append(fake("\(category)/\(material)", size: size, category: category))
                }
                images.append(fake("\(category)/\(material)", size: 1024, category: category, role: .roughness))
            }
        }
        var recipe = SyntheticPhotoRecipe(); recipe.materialPercentage = 30; recipe.cropsPerResolution = 2
        let a = try SyntheticPhotoPlan(images: images, recipe: recipe)
        let b = try SyntheticPhotoPlan(images: images.reversed(), recipe: recipe)
        #expect(a.materials.map(\.group) == b.materials.map(\.group))
        #expect(a.eligibleMaterialCount == 20)
        #expect(a.materials.count == 6)
        #expect(Dictionary(grouping: a.materials, by: \.category).values.allSatisfy { $0.count == 3 })
        #expect(a.materials.allSatisfy { $0.images.count == 3 && $0.images.allSatisfy { $0.role == .diffuse } })
        #expect(a.cropCount == 30)
        recipe.includeHigherResolutions = false
        #expect(try SyntheticPhotoPlan(images: images, recipe: recipe).cropCount == 6)
        recipe.materialPercentage = 100
        #expect(try SyntheticPhotoPlan(images: images, recipe: recipe).materials.count == 20)
        recipe.categories = ["Paper"]
        #expect(try SyntheticPhotoPlan(images: images, recipe: recipe).materials.allSatisfy { $0.category == "Paper" })
        let moved = images.map { image in
            SourceImage(id: image.id, url: URL(fileURLWithPath: "/restored" + image.url.path), displayName: image.displayName,
                role: image.role, category: image.category, verified: image.verified, note: image.note,
                pairingKey: image.pairingKey, width: image.width, height: image.height)
        }
        #expect(try SyntheticPhotoPlan(images: images, recipe: recipe, sourceRoot: URL(fileURLWithPath: "/test")).materials.map(\.group) ==
            SyntheticPhotoPlan(images: moved, recipe: recipe, sourceRoot: URL(fileURLWithPath: "/restored/test")).materials.map(\.group))
    }

    @Test func cpuEffectsAreSeededBoundedAndIdentityGeometryIsExact() throws {
        let size = 32
        let values = (0..<(size*size*3)).map { Float($0%197)/196 }
        var recipe = SyntheticPhotoRecipe()
        var a = SyntheticRandom("same-seed"), b = SyntheticRandom("same-seed")
        let first = try SyntheticPhotoPixels.restore(values, size: size, recipe: recipe, random: &a, cancellation: .init())
        let second = try SyntheticPhotoPixels.restore(values, size: size, recipe: recipe, random: &b, cancellation: .init())
        #expect(first.values == second.values)
        #expect(first.values != values)
        #expect(first.values.allSatisfy { $0.isFinite && (0...1).contains($0) })
        let identity = try SyntheticPhotoPixels.perspective(values, size: size, pitch: 0, yaw: 0, cancellation: .init())
        #expect(identity.values == values)
        #expect(identity.inputMask.allSatisfy { $0 == 1 })
        #expect(identity.targetMask.allSatisfy { $0 == 1 })
        let h = SyntheticPhotoPixels.homography(size: size, pitch: 12, yaw: -9)
        let projected = SyntheticPhotoPixels.project(h, x: 8, y: 11)
        let restored = SyntheticPhotoPixels.project(SyntheticPhotoPixels.inverse(h), x: projected.0, y: projected.1)
        #expect(abs(restored.0-8) < 1e-10 && abs(restored.1-11) < 1e-10)
        let tilted = try SyntheticPhotoPixels.perspective(values, size: size, pitch: 12, yaw: -9, cancellation: .init())
        #expect(tilted.inputMask.contains(0))
        #expect(tilted.values.allSatisfy { $0.isFinite && (0...1).contains($0) })
        let constant = [Float](repeating: 0.4, count: values.count)
        let blurred = try SyntheticPhotoPixels.gaussian(constant, size: size, sigma: 1.7, cancellation: .init())
        #expect(blurred.allSatisfy { abs($0-0.4) < 1e-6 })
        recipe.strength = .nan
        #expect(throws: (any Error).self) { try recipe.validate() }
    }

    @Test func nativeCropExportPreservesOriginalsAndReplaysExactly() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = try fixture(root, name: "Paper", width: 1032, height: 1040)
        let original = try Data(contentsOf: source.url), metadata = try ImageMetadataStore.read(source.url)
        var recipe = SyntheticPhotoRecipe(); recipe.materialPercentage = 100; recipe.variantsPerCrop = 1
        recipe.blur = false; recipe.noise = false; recipe.lighting = false
        let a = try SyntheticPhotoData.generate(images: [source], recipe: recipe, destination: root.appendingPathComponent("a"))
        let b = try SyntheticPhotoData.generate(images: [source], recipe: recipe, destination: root.appendingPathComponent("b"))
        #expect(a.pairCount == 3 && b.pairCount == 3)
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: a.directory.appendingPathComponent("manifest.json"))) as! [String: Any]
        let rows = manifest["pairs"] as! [[String: Any]]
        #expect(Set(rows.compactMap { $0["group"] as? String }).count == 1)
        #expect(Set(rows.compactMap { $0["split"] as? String }).count == 1)
        #expect(rows.allSatisfy { $0["parameters"] == nil })
        for row in rows {
            for key in ["input", "target"] {
                let name = row[key] as! String
                #expect(try Data(contentsOf: a.directory.appendingPathComponent(name)) == Data(contentsOf: b.directory.appendingPathComponent(name)))
            }
        }
        var rng = SyntheticRandom("\(recipe.seed):\(source.id):0")
        let x = Int(rng.next()%UInt64(source.width-1024+1)), y = Int(rng.next()%UInt64(source.height-1024+1))
        let exact = try NativeImages.readRGBCrop(from: source.url, crop: .init(x: x, y: y, size: 1024), imageCache: .init(policy: .stream))
        let saved = try NativeImages.readRGB(from: a.directory.appendingPathComponent(rows[0]["target"] as! String))
        #expect(zip(exact.values, saved.values).allSatisfy { abs($0-$1) < 1/65535.0 })
        #expect(try Data(contentsOf: source.url) == original)
        #expect(try ImageMetadataStore.read(source.url) == metadata)
        #expect(throws: (any Error).self) { try SyntheticPhotoData.generate(images: [source], recipe: recipe, destination: a.directory) }
        let cancelled = NativeCancellation(); cancelled.cancel()
        #expect(throws: CancellationError.self) {
            try SyntheticPhotoData.generate(images: [source], recipe: recipe, destination: root.appendingPathComponent("cancelled"), cancellation: cancelled)
        }
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("cancelled").path))
    }

    @Test func budgetRejectsBeforeWritingAndTagChangesAreNotSilentlyAccepted() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = try fixture(root, name: "Wood", width: 1024, height: 1024)
        var recipe = SyntheticPhotoRecipe(); recipe.materialPercentage = 100
        recipe.cropsPerResolution = 8; recipe.variantsPerCrop = 16; recipe.maximumOutputGiB = 1
        let many = (0..<4).map { fake("Group\($0)", size: 2048, category: "Wood") }
        #expect(throws: (any Error).self) { try SyntheticPhotoData.generate(images: many, recipe: recipe, destination: root.appendingPathComponent("oversize")) }
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("oversize").path))
        var changed = try #require(try ImageMetadataStore.read(source.url)); changed.role = .roughness
        try ImageMetadataStore.write(changed, to: source.url)
        recipe.variantsPerCrop = 1
        #expect(throws: (any Error).self) { try SyntheticPhotoData.generate(images: [source], recipe: recipe, destination: root.appendingPathComponent("changed")) }
        let data = try Data(contentsOf: root.appendingPathComponent("changed/manifest.json"))
        #expect(String(decoding: data, as: UTF8.self).contains("failed"))
    }

    @Test func cancellationKeepsCompletedPairsAndDuplicateLevelsShareOneSplit() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let first = try fixture(root, name: "First", width: 1024, height: 1024)
        let duplicate = try fixture(root, name: "Duplicate", width: 1024, height: 1024)
        let higher = try fixture(root, name: "DuplicateLevel", width: 1025, height: 1025)
        let levelURL = duplicate.url.deletingLastPathComponent().appendingPathComponent("higher.png")
        try FileManager.default.moveItem(at: higher.url, to: levelURL)
        var levelMetadata = try #require(try ImageMetadataStore.read(levelURL)); levelMetadata.category = duplicate.category
        try ImageMetadataStore.write(levelMetadata, to: levelURL)
        let level = SourceImage(id: higher.id, url: levelURL, displayName: higher.displayName,
            role: .diffuse, category: duplicate.category, verified: false, note: "", pairingKey: higher.pairingKey,
            width: higher.width, height: higher.height)
        var recipe = SyntheticPhotoRecipe(); recipe.materialPercentage = 100; recipe.variantsPerCrop = 1
        recipe.blur = false; recipe.noise = false; recipe.lighting = false; recipe.perspective = false
        // Exact duplicates from different folders must be coalesced, including
        // all levels of the folder whose representative was duplicated.
        let result = try SyntheticPhotoData.generate(images: [first, duplicate, level], recipe: recipe, destination: root.appendingPathComponent("duplicates"))
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: result.directory.appendingPathComponent("manifest.json"))) as! [String: Any]
        let sources = object["sources"] as! [[String: Any]]
        #expect(Set(sources.compactMap { $0["group"] as? String }).count == 1)
        let cancellation = NativeCancellation()
        let output = root.appendingPathComponent("partial")
        #expect(throws: CancellationError.self) {
            try SyntheticPhotoData.generate(images: [first], recipe: recipe, destination: output,
                cancellation: cancellation, progress: { fraction, _ in if fraction > 0 { cancellation.cancel() } })
        }
        let partial = try JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("manifest.json"))) as! [String: Any]
        #expect(partial["status"] as? String == "cancelled")
        #expect((partial["pairs"] as? [[String: Any]])?.count == 1)
        #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("recipe.json").path))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["LOOM_SYNTHETIC_REAL_OUTPUT"] != nil))
    func actualNativeColorSourcesGenerateReviewWithoutChangingTheirBytes() throws {
        let output = URL(fileURLWithPath: ProcessInfo.processInfo.environment["LOOM_SYNTHETIC_REAL_OUTPUT"]!)
        let paths = ["red_brick/red_brick_diff_1k.png", "Cardboard_02/Cardboard002_1K-PNG_Color.png",
                     "Cardboard_02/Cardboard002_2K-PNG_Color.png", "Fabric080/Fabric080_2K-PNG_Color.png"]
        let images = try paths.map { relative -> SourceImage in
            let url = URL(fileURLWithPath: "/opt/ipde/sources_mats").appendingPathComponent(relative)
            let metadata = try #require(try ImageMetadataStore.read(url))
            let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
            let properties = try #require(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
            return .init(id: metadata.id, url: url, displayName: relative, role: metadata.role,
                category: metadata.category, verified: metadata.verified, note: metadata.note, pairingKey: metadata.pairingKey,
                width: (properties[kCGImagePropertyPixelWidth] as! NSNumber).intValue,
                height: (properties[kCGImagePropertyPixelHeight] as! NSNumber).intValue, ignored: metadata.ignored ?? false)
        }
        let before = try images.map { try NativeFileHash.sha256($0.url) }
        let tags = try images.map { try ImageMetadataStore.read($0.url) }
        var recipe = SyntheticPhotoRecipe(); recipe.materialPercentage = 100; recipe.variantsPerCrop = 1
        let result = try SyntheticPhotoData.generate(images: images, recipe: recipe, destination: output)
        #expect(result.materialCount == 3 && result.pairCount == 12)
        #expect(try images.map { try NativeFileHash.sha256($0.url) } == before)
        #expect(try images.map { try ImageMetadataStore.read($0.url) } == tags)
    }

    private func fake(_ folder: String, size: Int, category: String, role: MapRole = .diffuse) -> SourceImage {
        .init(id: folder+"-\(size)-\(role)", url: URL(fileURLWithPath: "/test/"+folder+"/\(size)-\(role).png"),
            displayName: folder, role: role, category: category, verified: false, note: "", pairingKey: folder, width: size, height: size)
    }
    private func fixture(_ root: URL, name: String, width: Int, height: Int) throws -> SourceImage {
        let folder = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("color.png")
        let count = width*height
        let values = (0..<(count*3)).map { Float(($0%count)%101)/100 }
        try ModelTestImage.writePNG(values, channels: 3, width: width, height: height, to: url)
        try ImageMetadataStore.write(.init(id: name, pairingKey: name, role: .diffuse, category: name), to: url)
        return .init(id: name, url: url, displayName: name, role: .diffuse, category: name, verified: false,
                     note: "", pairingKey: name, width: width, height: height)
    }
}
