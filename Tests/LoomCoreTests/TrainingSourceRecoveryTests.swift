import CryptoKit
import Foundation
import ImageIO
import LoomNative
import XCTest
@testable import LoomCore

final class TrainingSourceRecoveryTests: XCTestCase {
    private let root = URL(fileURLWithPath: "/sources")
    private func image(_ tier: Int, asset: String = "stone", folder: String = "Stone", category: String = "Tiles") -> SourceImage {
        .init(id: "\(folder)-\(tier)", url: root.appendingPathComponent("\(folder)/\(asset)_diff_\(tier)k.png"),
              displayName: asset, role: .diffuse, category: category, verified: false, note: "",
              pairingKey: "\(asset)-\(tier)", width: tier * 1024, height: tier * 1024)
    }
    func testRequestedLevelsFillMissingTiersAndAllDoesNotExpandToProviderMaximum() {
        let images = [image(1), image(4), image(1, folder: "Other"), image(8, folder: "Other"), image(1, category: "Elsewhere")]
        var settings = TrainingSettings(sourceResolutionLevels: 3)
        XCTAssertEqual(TrainingSourceRecovery.materials(images: images, category: "Tiles", settings: settings).map(\.tiers), [[1, 2, 4], [1, 2, 4]])
        settings.sourceResolutionLevels = 4
        XCTAssertEqual(TrainingSourceRecovery.materials(images: images, category: "Tiles", settings: settings).map(\.tiers), [[1, 2, 4, 8], [1, 2, 4, 8]])
        settings.sourceResolutionLevels = 5
        XCTAssertEqual(TrainingSourceRecovery.materials(images: images, category: "Tiles", settings: settings).map(\.tiers), [[1, 2, 4, 8, 16], [1, 2, 4, 8, 16]])
        settings.sourceResolutionLevels = 0
        XCTAssertEqual(TrainingSourceRecovery.materials(images: images, category: "Tiles", settings: settings).map(\.tiers), [[1, 2, 4, 8], [1, 2, 4, 8]])
        settings.sourceResolutionLevels = 1
        settings.imageSize = 512
        XCTAssertEqual(TrainingSourceRecovery.materials(images: images, category: "Tiles", settings: settings).map(\.tiers), [[1], [1]])
        settings.imageSize = 2048
        XCTAssertEqual(TrainingSourceRecovery.materials(images: images, category: "Tiles", settings: settings).map(\.tiers), [[2], [2]])
        var ignored = image(1); ignored.ignored = true
        XCTAssertTrue(TrainingSourceRecovery.materials(images: [ignored], category: "Tiles", settings: settings).isEmpty)
    }
    func testAutomaticCategoryMatchingNeverRequests16KEvenWhenSomeLocalSourcesAre16K() {
        let images = [image(1), image(4), image(16, folder: "Other")]
        let materials = TrainingSourceRecovery.materials(images: images, category: "Tiles", settings: .init(sourceResolutionLevels: 0))
        XCTAssertTrue(materials.allSatisfy { !$0.tiers.contains(16) })
        XCTAssertEqual(materials.first { $0.folder.lastPathComponent == "Stone" }?.tiers, [1, 2, 4, 8])
        let explicit = TrainingSourceRecovery.materials(images: [image(1)], category: "Tiles", settings: .init(sourceResolutionLevels: 5))
        XCTAssertEqual(explicit.first?.tiers, [1, 2, 4, 8, 16])
    }
    func testCatalogUsesPublishedPNGAndOpenGLOnlyAndRejectsInvalidDescriptions() throws {
        let file: [String: Any] = ["url": "https://dl.polyhaven.org/file/ph-assets/stone_nor_gl_2k.png", "size": 123, "md5": String(repeating: "a", count: 32)]
        let data = try JSONSerialization.data(withJSONObject: ["nor_gl": ["2k": ["png": file]], "nor_dx": ["4k": ["png": file]]])
        let normal = try XCTUnwrap(TrainingSourceRecovery.published(data, role: .normal, tier: 2))
        XCTAssertTrue(normal.url.lastPathComponent.contains("nor_gl"))
        XCTAssertNil(try TrainingSourceRecovery.published(data, role: .normal, tier: 4))
        XCTAssertNil(try TrainingSourceRecovery.published(data, role: .depth, tier: 2))
        var invalid = file; invalid["url"] = "https://example.com/stone.png"
        XCTAssertThrowsError(try TrainingSourceRecovery.published(JSONSerialization.data(withJSONObject: ["Diffuse": ["1k": ["png": invalid]]]), role: .diffuse, tier: 1))
    }
    func testColorVariantsKeepTheirOwnPublishedDiffuseAndSharedTargets() throws {
        let input = SourceImage(id: "leather", url: root.appendingPathComponent("Leather/leather_red_03_coll2_1k.png"),
            displayName: "Leather", role: .diffuse, category: "Tiles", verified: false, note: "", pairingKey: "leather",
            width: 1024, height: 1024)
        let material = try XCTUnwrap(TrainingSourceRecovery.materials(images: [input], category: "Tiles",
            settings: .init(sourceResolutionLevels: 2)).first)
        XCTAssertEqual(material.asset, "leather_red_03")
        XCTAssertEqual(material.diffuseSuffix, "coll2")
        XCTAssertEqual(material.tiers, [1, 2])
        XCTAssertEqual(TrainingSourceRecovery.filename(asset: material.asset, role: .diffuse, tier: 2, diffuseSuffix: material.diffuseSuffix), "leather_red_03_coll2_2k.png")
        let file: [String: Any] = ["url": "https://dl.polyhaven.org/file/ph-assets/leather_red_03_coll2_2k.png", "size": 123, "md5": String(repeating: "b", count: 32)]
        let data = try JSONSerialization.data(withJSONObject: ["coll2": ["2k": ["png": file]]])
        XCTAssertNotNil(try TrainingSourceRecovery.published(data, role: .diffuse, tier: 2, diffuseSuffix: material.diffuseSuffix))
        XCTAssertNil(try TrainingSourceRecovery.published(data, role: .diffuse, tier: 2))
    }
    func testDownloadedOriginalMustMatchPublishedBytesChecksumAndDimensions() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        defer { try? FileManager.default.removeItem(at: url) }
        let bytes = Data(repeating: 128, count: 1024 * 1024)
        let provider = try XCTUnwrap(CGDataProvider(data: bytes as CFData))
        let image = try XCTUnwrap(CGImage(width: 1024, height: 1024, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: 1024,
            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: [], provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil); XCTAssertTrue(CGImageDestinationFinalize(destination))
        let data = try Data(contentsOf: url)
        let digest = Insecure.MD5.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let published = TrainingSourceRecovery.PublishedFile(url: url, md5: digest, size: data.count)
        XCTAssertNoThrow(try TrainingSourceRecovery.validate(url, published: published, tier: 1))
        XCTAssertThrowsError(try TrainingSourceRecovery.validate(url, published: published, tier: 2))
        XCTAssertThrowsError(try TrainingSourceRecovery.validate(url, published: .init(url: url, md5: String(repeating: "0", count: 32), size: data.count), tier: 1))
        XCTAssertThrowsError(try TrainingSourceRecovery.validate(url, published: .init(url: url, md5: digest, size: data.count + 1), tier: 1))
    }

    // Explicitly enabled for the development corpus; never trains or changes
    // existing image bytes, recipes, runs, checkpoints or model assignments.
    func testLiveTilesSourceRecovery() async throws {
        guard let path = ProcessInfo.processInfo.environment["LOOM_TEST_SOURCE_RECOVERY"] else { throw XCTSkip("Set LOOM_TEST_SOURCE_RECOVERY to a source folder for the provider integration check") }
        let project = try LoomProject(sourceFolder: URL(fileURLWithPath: path))
        let before = try await project.snapshot()
        let dataset = try XCTUnwrap(before.datasets.first { $0.name == "Tiles And Pavers" })
        let originals = try before.images.filter { $0.category == dataset.name }.map { ($0.url, try NativeFileHash.sha256($0.url)) }
        let settings = TrainingSettings(sourceResolutionLevels: 3, validationFraction: 0)
        let log = RecoveryTestLog()
        let count = try await project.prepareTrainingSources(datasetID: dataset.id, map: .normal, settings: settings) { log.append($0.message) }
        let after = try await project.snapshot()
        let plan = try XCTUnwrap(after.trainingPlan(datasetID: dataset.id, map: .normal, settings: settings))
        for tier in [1024, 2048, 4096] {
            XCTAssertEqual(plan.resolutions.first { $0.width == tier }?.trainingPairCount, 18)
        }
        for (url, hash) in originals { XCTAssertEqual(try NativeFileHash.sha256(url), hash) }
        let repeated = try await project.prepareTrainingSources(datasetID: dataset.id, map: .normal, settings: settings) { log.append($0.message) }
        XCTAssertEqual(repeated, 0)
        print("SOURCE_RECOVERY_DOWNLOADS=\(count)\n" + log.messages.joined(separator: "\n"))
    }
}

private final class RecoveryTestLog: @unchecked Sendable {
    let lock = NSLock()
    private var values: [String] = []
    func append(_ value: String) { lock.withLock { values.append(value) } }
    var messages: [String] { lock.withLock { values } }
}
