import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class BumpMapTests: XCTestCase {
    func testPrimaryProviderColorVariantsPairAndRepairWithoutHumanReview() throws {
        let root = URL(fileURLWithPath: "/source")
        let target = ImageMetadataStore.initialMetadata(file: root.appendingPathComponent("leather_red_02_disp_1k.png"),
                                                       sourceRoot: root, width: 1024, height: 1024)
        for alias in ["col1", "coll1", "col_1", "coll_1"] {
            let file = root.appendingPathComponent("leather_red_02_\(alias)_1k.png")
            let tag = ImageMetadataStore.initialMetadata(file: file, sourceRoot: root, width: 1024, height: 1024)
            XCTAssertEqual(tag.role, .diffuse, alias)
            XCTAssertEqual(tag.pairingKey, target.pairingKey, alias)
        }
        let file = root.appendingPathComponent("leather_red_02_col_1_1k.png")
        let oldKey = SHA256.hash(data: Data("/leather_red_02_1_1k:1024x1024".utf8)).map { String(format: "%02x", $0) }.joined()
        let old = ImageMetadata(id: "same-file", pairingKey: oldKey, role: .diffuse, category: "Leather")
        let repaired = ImageMetadataStore.repairingLegacyInference(old, file: file, sourceRoot: root, width: 1024, height: 1024)
        XCTAssertEqual(repaired.pairingKey, target.pairingKey)
        XCTAssertEqual(repaired.id, old.id); XCTAssertFalse(repaired.verified)
        var manual = old; manual.pairingWasManuallySelected = true
        XCTAssertEqual(ImageMetadataStore.repairingLegacyInference(manual, file: file, sourceRoot: root, width: 1024, height: 1024), manual)
        let alternate = ImageMetadataStore.initialMetadata(file: root.appendingPathComponent("leather_red_02_coll2_1k.png"),
                                                          sourceRoot: root, width: 1024, height: 1024)
        XCTAssertEqual(alternate.role, .diffuse)
        XCTAssertEqual(alternate.pairingKey, target.pairingKey)
    }

    func testMaterialNameRoleWordCannotKeepAnOldNormalMapTaggedAsRoughness() throws {
        let root = URL(fileURLWithPath: "/source")
        let file = root.appendingPathComponent("white_rough_plaster_nor_gl_1k.png")
        let oldKey = SHA256.hash(data: Data("/white_plaster_nor_gl_1k:64x64".utf8)).map { String(format: "%02x", $0) }.joined()
        let saved = ImageMetadata(id: "same-image", pairingKey: oldKey, role: .roughness,
                                  category: "Plaster", verified: true, note: "Keep this review")
        let repaired = ImageMetadataStore.repairingLegacyInference(saved, file: file, sourceRoot: root, width: 64, height: 64)
        let diffuse = ImageMetadataStore.initialMetadata(file: root.appendingPathComponent("white_rough_plaster_diff_1k.png"),
                                                        sourceRoot: root, width: 64, height: 64)
        XCTAssertEqual(repaired.role, .normal); XCTAssertEqual(repaired.pairingKey, diffuse.pairingKey)
        XCTAssertEqual(repaired.id, saved.id); XCTAssertEqual(repaired.category, saved.category)
        XCTAssertEqual(repaired.verified, saved.verified); XCTAssertEqual(repaired.note, saved.note)
        var manual = saved; manual.roleWasManuallySelected = true
        XCTAssertEqual(ImageMetadataStore.repairingLegacyInference(manual, file: file, sourceRoot: root, width: 64, height: 64), manual)
    }

    func testProviderNormalNamesAndDirectXExclusion() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try image(root, "Bark_1K-PNG_Color.png")
        _ = try image(root, "Bark_1K-PNG_NormalGL.png")
        let dx = try image(root, "Bark_1K-PNG_NormalDX.png")
        let project = try LoomProject(sourceFolder: root)
        let initial = try await project.snapshot()
        try await project.updateImage(id: initial.images[0].id, category: "Bark", verified: true, note: "")
        let current = try await project.snapshot()
        XCTAssertEqual(current.images.first { $0.url == dx }?.ignored, true)
        let pairs = try TrainingPairSelector.select(images: current.images, dataset: "Bark", map: .normal, size: 32)
        XCTAssertEqual(pairs.count, 1)
        XCTAssertEqual(pairs.first?.target.url.lastPathComponent, "Bark_1K-PNG_NormalGL.png")
        let directXOnly = current.images.filter { !$0.url.lastPathComponent.contains("NormalGL") }.map { image in
            var restored = image; restored.ignored = false; return restored
        }
        XCTAssertTrue(try TrainingPairSelector.select(images: directXOnly, dataset: "Bark", map: .normal, size: 32).isEmpty)
        let auxiliary = ImageMetadataStore.initialMetadata(file: root.appendingPathComponent("color_bark_ao_1k.png"), sourceRoot: root, width: 64, height: 64)
        XCTAssertEqual(auxiliary.role, .unknown)
    }

    func testBumpHasItsOwnRoleAndPairsWithItsMatchingDiffuseResolution() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try image(root, "bark_diff_1k.png")
        _ = try image(root, "bark_bump_1k.png")
        _ = try image(root, "bark_disp_1k.png")
        _ = try image(root, "bark_bumpmap_2k.png", size: 128)
        _ = try image(root, "bark_diff_2k.png", size: 128)
        let project = try LoomProject(sourceFolder: root)
        let snapshot = try await project.snapshot()
        XCTAssertEqual(snapshot.images.filter { $0.role == .diffuse }.count, 2)
        XCTAssertEqual(snapshot.images.filter { $0.role == .bump }.count, 2)
        XCTAssertEqual(snapshot.images.filter { $0.role == .depth }.count, 1)
        let one = snapshot.images.filter { $0.width == 64 }
        XCTAssertEqual(Set(one.map(\.pairingKey)).count, 1)
        XCTAssertEqual(Set(snapshot.images.filter { $0.width == 128 }.map(\.pairingKey)).count, 1)
        XCTAssertNotEqual(one.first?.pairingKey, snapshot.images.first { $0.width == 128 }?.pairingKey)
        XCTAssertFalse(MapRole.targets.contains(.bump))
    }

    func testDepthUsesDisplacementAndNeverSubstitutesBump() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try image(root, "bark_diff.png"); _ = try image(root, "bark_bump.png")
        let displacement = try image(root, "bark_disp.png")
        let project = try LoomProject(sourceFolder: root)
        let initial = try await project.snapshot()
        try await project.updateImage(id: initial.images[0].id, category: "Bark", verified: true, note: "")
        let reviewed = try await project.snapshot()
        let preferred = try TrainingPairSelector.select(images: reviewed.images, dataset: "Bark", map: .depth, size: 32)
        XCTAssertEqual(preferred.count, 1); XCTAssertEqual(preferred.first?.target.role, .depth)
        XCTAssertEqual(preferred.first?.target.url.lastPathComponent, displacement.lastPathComponent)
        XCTAssertTrue(try TrainingPairSelector.select(images: reviewed.images, dataset: "Bark", map: .roughness, size: 32).isEmpty)
        try FileManager.default.removeItem(at: displacement)
        let remaining = try await project.snapshot()
        XCTAssertTrue(try TrainingPairSelector.select(images: remaining.images, dataset: "Bark", map: .depth, size: 32).isEmpty)
        XCTAssertEqual(remaining.datasets.first?.pairCounts["depth"], 0)
        XCTAssertEqual(remaining.eligiblePairCount(datasetID: ImageMetadataStore.categoryID("Bark"), map: .depth, imageSize: 32), 0)
    }

    func testTaggedDepthCanTrainWithoutReviewAndNeverSubstitutesBump() throws {
        let folder = URL(fileURLWithPath: "/test/material")
        let diffuse = member(folder, role: .diffuse)
        let bump = member(folder, role: .bump)
        let displacement = member(folder, role: .depth, verified: false)
        let pairs = try TrainingPairSelector.select(images: [diffuse, bump, displacement], dataset: "Bark", map: .depth, size: 32)
        XCTAssertEqual(pairs.count, 1); XCTAssertEqual(pairs.first?.target.id, displacement.id)
    }

    func testIgnoredFilesStayVisibleButCannotFormADatasetOrTrainingPair() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try image(root, "bark_diff.png"); _ = try image(root, "bark_disp.png")
        let project = try LoomProject(sourceFolder: root)
        let initial = try await project.snapshot()
        let input = try XCTUnwrap(initial.images.first { $0.role == .diffuse })
        let depth = try XCTUnwrap(initial.images.first { $0.role == .depth })
        try await project.updateImage(id: input.id, category: "Bark", verified: true, note: "Preserve review")
        try await project.setIgnored(id: depth.id, ignored: true)
        let ignoredTarget = try await project.snapshot()
        XCTAssertEqual(ignoredTarget.images.count, 2)
        XCTAssertEqual(ignoredTarget.images.first { $0.id == depth.id }?.ignored, true)
        XCTAssertEqual(ignoredTarget.datasets.first?.imageCount, 1)
        XCTAssertEqual(ignoredTarget.datasets.first?.pairCounts["depth"], 0)
        XCTAssertTrue(try TrainingPairSelector.select(images: ignoredTarget.images, dataset: "Bark", map: .depth, size: 32).isEmpty)
        try await project.setIgnored(id: input.id, ignored: true)
        let allIgnored = try await project.snapshot()
        XCTAssertEqual(allIgnored.images.count, 2); XCTAssertTrue(allIgnored.datasets.isEmpty)
        try await project.setIgnored(id: input.id, ignored: false)
        try await project.setIgnored(id: depth.id, ignored: false)
        let restored = try await project.snapshot()
        XCTAssertEqual(restored.datasets.first?.imageCount, 2)
        XCTAssertTrue(restored.images.allSatisfy { $0.verified && $0.note == "Preserve review" && !$0.ignored })
        XCTAssertEqual(restored.eligiblePairCount(datasetID: ImageMetadataStore.categoryID("Bark"), map: .depth, imageSize: 32), 1)
    }

    func testDiffuseAliasesStripTheCompleteMapNameAndPairWithTargets() throws {
        let root = URL(fileURLWithPath: "/source")
        let target = ImageMetadataStore.initialMetadata(file: root.appendingPathComponent("bark_disp_1k.png"),
                                                       sourceRoot: root, width: 1024, height: 1024)
        for alias in ["diff", "diffuse", "color", "colour", "albedo", "basecolor", "basecolour", "base_color", "base_colour", "base_col", "base_colo"] {
            let input = ImageMetadataStore.initialMetadata(file: root.appendingPathComponent("bark_\(alias)_1k.png"),
                                                          sourceRoot: root, width: 1024, height: 1024)
            XCTAssertEqual(input.role, .diffuse, alias); XCTAssertEqual(input.pairingKey, target.pairingKey, alias)
        }
        let bump = ImageMetadataStore.initialMetadata(file: root.appendingPathComponent("color_bark_bump_1k.png"),
                                                      sourceRoot: root, width: 1024, height: 1024)
        XCTAssertEqual(bump.role, .bump)
    }

    func testLegacyCompoundColorInferenceRepairsPairingOnly() throws {
        let root = URL(fileURLWithPath: "/source"), file = URL(fileURLWithPath: "/source/bark_base_color_1k.png")
        let oldIdentity = "/bark_base_1k:1024x1024"
        let oldKey = SHA256.hash(data: Data(oldIdentity.utf8)).map { String(format: "%02x", $0) }.joined()
        let old = ImageMetadata(id: "same-image", pairingKey: oldKey, role: .diffuse, category: "Bark", verified: true, note: "Reviewed")
        let repaired = ImageMetadataStore.repairingLegacyInference(old, file: file, sourceRoot: root, width: 1024, height: 1024)
        let target = ImageMetadataStore.initialMetadata(file: root.appendingPathComponent("bark_height_1k.png"), sourceRoot: root, width: 1024, height: 1024)
        XCTAssertEqual(repaired.id, old.id); XCTAssertEqual(repaired.category, old.category)
        XCTAssertEqual(repaired.note, old.note); XCTAssertEqual(repaired.verified, old.verified)
        XCTAssertEqual(repaired.pairingKey, target.pairingKey)
    }

    func testLegacyBumpInferenceRepairsRoleAndPairingWithoutLosingImageReview() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let material = root.appendingPathComponent("Bark Brown 01", isDirectory: true)
        try FileManager.default.createDirectory(at: material, withIntermediateDirectories: true)
        let bump = try image(material, "bark_brown_01_bump_1k.png")
        _ = try image(material, "bark_brown_01_diff_1k.png")
        let bytes = try Data(contentsOf: bump)
        let identity = "/Bark Brown 01/bark_brown_01_bump_1k:64x64"
        let legacyKey = SHA256.hash(data: Data(identity.utf8)).map { String(format: "%02x", $0) }.joined()
        let metadata = ImageMetadata(id: "preserved-image-id", pairingKey: legacyKey, role: .unknown,
                                     category: "Bark", verified: true, note: "Keep the bark details")
        try ImageMetadataStore.write(metadata, to: bump)
        let project = try LoomProject(sourceFolder: root)
        let snapshot = try await project.snapshot()
        let repaired = try XCTUnwrap(ImageMetadataStore.read(bump))
        XCTAssertEqual(repaired.id, metadata.id); XCTAssertEqual(repaired.role, .bump)
        XCTAssertEqual(repaired.category, metadata.category); XCTAssertEqual(repaired.verified, metadata.verified)
        XCTAssertEqual(repaired.note, metadata.note)
        XCTAssertEqual(repaired.pairingKey, snapshot.images.first { $0.role == .diffuse }?.pairingKey)
        XCTAssertEqual(try Data(contentsOf: bump), bytes)
        let renamed = material.appendingPathComponent("renamed teaching map.png")
        try FileManager.default.moveItem(at: bump, to: renamed)
        let after = try await project.snapshot()
        XCTAssertEqual(after.images.first { $0.id == metadata.id }?.role, .bump)
        XCTAssertEqual(after.images.first { $0.id == metadata.id }?.pairingKey, repaired.pairingKey)
    }

    func testLegacyRepairPreservesCustomPairAndManuallySelectedRoles() throws {
        let root = URL(fileURLWithPath: "/source")
        let bump = root.appendingPathComponent("bark_bump_1k.png")
        let identity = "/bark_bump_1k:64x64"
        let legacyKey = SHA256.hash(data: Data(identity.utf8)).map { String(format: "%02x", $0) }.joined()
        let variants = [
            ImageMetadata(pairingKey: "custom-pair", role: .unknown, category: "Bark", verified: true, note: "Deliberate pairing"),
            ImageMetadata(pairingKey: legacyKey, role: .unknown, roleWasManuallySelected: true),
            ImageMetadata(pairingKey: legacyKey, role: .unknown, pairingWasManuallySelected: true),
            ImageMetadata(pairingKey: legacyKey, role: .diffuse)
        ]
        for tag in variants {
            XCTAssertEqual(ImageMetadataStore.repairingLegacyInference(tag, file: bump, sourceRoot: root, width: 64, height: 64), tag)
        }
    }

    func testManualUnknownRoleStaysManualAcrossLaterScans() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try image(root, "bark_bump.png")
        let project = try LoomProject(sourceFolder: root)
        let initial = try await project.snapshot()
        try await project.setRole(id: initial.images[0].id, role: .unknown)
        let after = try await project.snapshot()
        XCTAssertEqual(after.images[0].role, .unknown)
        XCTAssertTrue(try XCTUnwrap(ImageMetadataStore.read(after.images[0].url)).roleWasManuallySelected == true)
    }

    private func member(_ folder: URL, role: MapRole, verified: Bool = true) -> SourceImage {
        SourceImage(id: role.rawValue, url: folder.appendingPathComponent(role.rawValue + ".png"), displayName: role.displayName,
                    role: role, category: "Bark", verified: verified, note: "", pairingKey: "pair", width: 64, height: 64)
    }
    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-bump-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.resolvingSymlinksInPath()
    }
    @discardableResult private func image(_ folder: URL, _ name: String, size: Int = 64) throws -> URL {
        let file = folder.appendingPathComponent(name)
        let bytes = Data(repeating: 120, count: size * size * 3)
        let provider = CGDataProvider(data: bytes as CFData)!
        let cg = CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: size * 3,
                         space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [], provider: provider,
                         decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let destination = CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, cg, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return file
    }
}
