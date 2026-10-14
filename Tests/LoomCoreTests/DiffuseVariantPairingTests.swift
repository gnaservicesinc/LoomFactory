import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class DiffuseVariantPairingTests: XCTestCase {
    func testProviderColorVariantsShareTheMatchingTargetFamily() throws {
        let root = URL(fileURLWithPath: "/source")
        let aliases = ["col1", "coll1", "col_1", "coll_1", "col2", "coll2", "col_2", "coll_2", "col12", "coll12"]
        for size in [1024, 2048, 4096] {
            let suffix = "\(size / 1024)k"
            let targets = ["disp", "rough", "nor_gl"].map { map in
                ImageMetadataStore.initialMetadata(file: root.appendingPathComponent("leather_red_03_\(map)_\(suffix).png"),
                                                   sourceRoot: root, width: size, height: size)
            }
            XCTAssertEqual(Set(targets.map(\.pairingKey)).count, 1)
            for alias in aliases {
                let file = root.appendingPathComponent("leather_red_03_\(alias)_\(suffix).png")
                let metadata = ImageMetadataStore.initialMetadata(file: file, sourceRoot: root, width: size, height: size)
                XCTAssertEqual(metadata.role, .diffuse, alias)
                XCTAssertEqual(metadata.pairingKey, targets[0].pairingKey, alias)
                XCTAssertFalse(metadata.ignored == true, alias)
            }
        }
    }

    func testLegacyAlternateRepairsPairingWhilePreservingReviewAndManualDiffuseRole() throws {
        let root = URL(fileURLWithPath: "/source")
        let file = root.appendingPathComponent("leather_red_03_coll2_4k.png")
        let legacyKey = digest("/leather_red_03_coll2_4k:4096x4096")
        let target = ImageMetadataStore.initialMetadata(file: root.appendingPathComponent("leather_red_03_disp_4k.png"),
                                                       sourceRoot: root, width: 4096, height: 4096)
        for manuallyChosenDiffuse in [false, true] {
            for ignored in [false, true] {
                let old = ImageMetadata(id: "same-original-image", pairingKey: legacyKey,
                                        role: manuallyChosenDiffuse ? .diffuse : .unknown,
                                        category: "Leather", verified: true, note: "Both colors share these maps",
                                        roleWasManuallySelected: manuallyChosenDiffuse,
                                        pairingWasManuallySelected: false, ignored: ignored)
                let repaired = ImageMetadataStore.repairingColorVariantInference(old, file: file, sourceRoot: root,
                                                                                width: 4096, height: 4096)
                XCTAssertEqual(repaired.role, .diffuse)
                XCTAssertEqual(repaired.pairingKey, target.pairingKey)
                XCTAssertEqual(repaired.id, old.id)
                XCTAssertEqual(repaired.category, old.category)
                XCTAssertEqual(repaired.verified, old.verified)
                XCTAssertEqual(repaired.note, old.note)
                XCTAssertEqual(repaired.ignored, old.ignored)
                XCTAssertEqual(repaired.roleWasManuallySelected, old.roleWasManuallySelected)
                XCTAssertFalse(repaired.pairingWasManuallySelected == true)
            }
        }
    }

    func testColorVariantMigrationPreservesManualAndCustomPairing() throws {
        let root = URL(fileURLWithPath: "/source")
        let file = root.appendingPathComponent("leather_red_03_coll2_4k.png")
        let legacyKey = digest("/leather_red_03_coll2_4k:4096x4096")
        let choices = [
            ImageMetadata(pairingKey: legacyKey, role: .diffuse, category: "Leather", verified: true,
                          note: "Keep this chosen pairing", pairingWasManuallySelected: true),
            ImageMetadata(pairingKey: "custom-pairing-key", role: .diffuse, category: "Leather", verified: true,
                          note: "Keep this custom pairing", pairingWasManuallySelected: false)
        ]
        for metadata in choices {
            XCTAssertEqual(ImageMetadataStore.repairingColorVariantInference(metadata, file: file, sourceRoot: root,
                                                                            width: 4096, height: 4096), metadata)
        }
    }

    func testSharedTargetsSupportManualColorVariantPairingAndCurrentFileCounts() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let names = ["coll1", "coll2", "disp", "rough", "nor_gl"]
        let files = try names.map { try image(root, "leather_red_03_\($0)_1k.png") }
        let originals = try Dictionary(uniqueKeysWithValues: files.map { ($0, try Data(contentsOf: $0)) })
        let project = try LoomProject(sourceFolder: root)
        let initial = try await project.snapshot()
        XCTAssertEqual(initial.images.filter { $0.role == .diffuse }.count, 2)
        XCTAssertEqual(Set(initial.images.map(\.pairingKey)).count, 1)
        let selected = try XCTUnwrap(initial.images.first { $0.role == .diffuse })
        try await project.updateImage(id: selected.id, category: "Leather", verified: true, note: "Color variants")
        let tagged = try await project.snapshot()
        assertPairCounts(tagged, expected: 2)

        try await project.pairImages(ids: tagged.images.map(\.id))
        let paired = try await project.snapshot()
        XCTAssertEqual(Set(paired.images.map(\.pairingKey)).count, 1)
        XCTAssertTrue(try paired.images.allSatisfy { try ImageMetadataStore.read($0.url)?.pairingWasManuallySelected == true })
        assertPairCounts(paired, expected: 2)
        for map in MapRole.targets {
            let pairs = try TrainingPairSelector.select(images: paired.images, dataset: "Leather", map: map, size: 32)
            XCTAssertEqual(pairs.count, 2)
            XCTAssertEqual(Set(pairs.map(\.id)).count, 2)
            XCTAssertEqual(Set(pairs.map { $0.target.id }).count, 1)
        }

        let duplicate = try image(root, "leather_red_03_height_1k.png")
        let withDuplicate = try await project.snapshot()
        let originalPairing = Dictionary(uniqueKeysWithValues: withDuplicate.images.map { ($0.id, $0.pairingKey) })
        do {
            try await project.pairImages(ids: withDuplicate.images.map(\.id))
            XCTFail("Manual pairing must reject two depth targets")
        } catch {
            // The rejection must leave every existing pairing intact.
        }
        let rejected = try await project.snapshot()
        XCTAssertEqual(Dictionary(uniqueKeysWithValues: rejected.images.map { ($0.id, $0.pairingKey) }), originalPairing)
        for (file, bytes) in originals { XCTAssertEqual(try Data(contentsOf: file), bytes) }
        try FileManager.default.removeItem(at: duplicate)
        let alternate = try XCTUnwrap(files.first { $0.lastPathComponent.contains("coll2") })
        try FileManager.default.removeItem(at: alternate)
        let remaining = try await project.snapshot()
        XCTAssertEqual(remaining.images.count, 4)
        assertPairCounts(remaining, expected: 1)
        for (file, bytes) in originals where file != alternate { XCTAssertEqual(try Data(contentsOf: file), bytes) }
    }

    private func assertPairCounts(_ snapshot: ProjectSnapshot, expected: Int, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(snapshot.datasets.count, 1, file: file, line: line)
        for map in MapRole.targets {
            XCTAssertEqual(snapshot.datasets.first?.pairCounts[map.rawValue], expected, map.rawValue, file: file, line: line)
        }
    }

    private func digest(_ identity: String) -> String {
        SHA256.hash(data: Data(identity.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-diffuse-variants-" + UUID().uuidString,
                                                                                 isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.resolvingSymlinksInPath()
    }

    private func image(_ folder: URL, _ name: String) throws -> URL {
        let file = folder.appendingPathComponent(name)
        let size = 64
        let bytes = Data(repeating: 120, count: size * size * 3)
        let provider = try XCTUnwrap(CGDataProvider(data: bytes as CFData))
        let image = try XCTUnwrap(CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 24,
                                        bytesPerRow: size * 3, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [],
                                        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return file
    }
}
