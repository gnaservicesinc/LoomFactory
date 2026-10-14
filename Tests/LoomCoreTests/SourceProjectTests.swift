import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class SourceProjectTests: XCTestCase {
    func testOneFolderReviewCorrectsEveryMapResolutionWithoutChangingPairingOrPixels() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let material = root.appendingPathComponent("Cloth", isDirectory: true)
        let separate = root.appendingPathComponent("Stone", isDirectory: true)
        try FileManager.default.createDirectory(at: material, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: separate, withIntermediateDirectories: true)
        let first = try image(material, "cloth_diff_1k.png")
        let second = try image(material, "cloth_diff_2k.png")
        _ = try image(material, "cloth_disp_1k.png")
        _ = try image(material, "cloth_disp_2k.png")
        _ = try image(separate, "stone_diff_1k.png")
        let project = try LoomProject(sourceFolder: root)
        let before = try await project.snapshot()
        let hashes = try before.images.map { SHA256.hash(data: try Data(contentsOf: $0.url)) }
        var wrong = try XCTUnwrap(ImageMetadataStore.read(second)); wrong.category = "Ground"
        try ImageMetadataStore.write(wrong, to: second)
        let selected = try XCTUnwrap(before.images.first { $0.url.lastPathComponent == first.lastPathComponent })
        try await project.updateImage(id: selected.id, category: "Fabric", verified: true, note: "Corrected the material folder")
        let after = try await project.snapshot()
        let members = after.images.filter { $0.url.deletingLastPathComponent().lastPathComponent == "Cloth" }
        XCTAssertEqual(members.count, 4)
        XCTAssertTrue(members.allSatisfy { $0.category == "Fabric" && $0.verified && $0.note == "Corrected the material folder" })
        XCTAssertNil(after.images.first { $0.url.deletingLastPathComponent().lastPathComponent == "Stone" }?.category)
        XCTAssertEqual(before.images.map(\.pairingKey), after.images.map(\.pairingKey))
        XCTAssertEqual(hashes, try after.images.map { SHA256.hash(data: try Data(contentsOf: $0.url)) })
    }

    func testPhysicalFilesAndImageMetadataAreTheOnlyDatasetAuthority() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let diffuse = try image(root, "brick_diff_2k.png")
        let target = try image(root, "brick_disp_2k.png")
        try Data("{\"samples\":[],\"training_size\":4096}".utf8).write(to: root.appendingPathComponent("dataset.json"))
        let project = try LoomProject(sourceFolder: root)
        var current = try await project.snapshot()
        XCTAssertEqual(current.images.count, 2)
        XCTAssertEqual(current.settings.training.imageSize, 1024)
        XCTAssertEqual(current.datasets.count, 0)
        let input = try XCTUnwrap(current.images.first { $0.url.resolvingSymlinksInPath() == diffuse.resolvingSymlinksInPath() })
        try await project.updateImage(id: input.id, category: "Bricks", verified: true, note: "Reviewed")
        current = try await project.snapshot()
        XCTAssertEqual(current.datasets.first?.name, "Bricks")
        XCTAssertEqual(current.datasets.first?.pairCounts["depth"], 1)
        XCTAssertTrue(current.images.allSatisfy(\.verified))
        let settings = TrainingSettings(imageSize: 64, epochs: 3, validationFraction: 0)
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        let presentPlan = try XCTUnwrap(current.trainingPlan(datasetID: datasetID, map: .depth, settings: settings))
        XCTAssertEqual(presentPlan.sourceImageCount, 2)
        XCTAssertEqual(presentPlan.sourceMaterialCount, 1)
        XCTAssertEqual(presentPlan.totalPairCount, 1)
        XCTAssertEqual(presentPlan.trainingSampleVisits, 3)
        try FileManager.default.removeItem(at: target)
        current = try await project.snapshot()
        XCTAssertEqual(current.images.count, 1)
        XCTAssertEqual(current.datasets.first?.pairCounts["depth"], 0)
        let removedPlan = try XCTUnwrap(current.trainingPlan(datasetID: datasetID, map: .depth, settings: settings))
        XCTAssertEqual(removedPlan.sourceImageCount, 1)
        XCTAssertEqual(removedPlan.sourceMaterialCount, 1)
        XCTAssertEqual(removedPlan.totalPairCount, 0)
        XCTAssertEqual(removedPlan.trainingSampleVisits, 0)
        try FileManager.default.removeItem(at: diffuse)
        let empty = try await project.snapshot()
        XCTAssertTrue(empty.datasets.isEmpty)
        XCTAssertNil(try empty.trainingPlan(datasetID: datasetID, map: .depth, settings: settings))
    }

    func testRenameKeepsIdentityPairingAndHumanReviewWithoutChangingPixels() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let diffuse = try image(root, "cloth_diff.png")
        let target = try image(root, "cloth_height.png")
        let hashes = try [diffuse, target].map { SHA256.hash(data: try Data(contentsOf: $0)) }
        let project = try LoomProject(sourceFolder: root)
        let before = try await project.snapshot()
        let selected = try XCTUnwrap(before.images.first { $0.role == .diffuse })
        try await project.updateImage(id: selected.id, category: "Fabric", verified: true, note: "Keep this weave")
        let renamedDiffuse = root.appendingPathComponent("a completely new name.png")
        let renamedTarget = root.appendingPathComponent("another name.png")
        try FileManager.default.moveItem(at: diffuse, to: renamedDiffuse)
        try FileManager.default.moveItem(at: target, to: renamedTarget)
        let after = try await project.snapshot()
        XCTAssertEqual(Set(before.images.map(\.id)), Set(after.images.map(\.id)))
        XCTAssertEqual(after.datasets.first?.pairCounts["depth"], 1)
        XCTAssertTrue(after.images.allSatisfy { $0.category == "Fabric" && $0.verified && $0.note == "Keep this weave" })
        XCTAssertEqual(hashes, try [renamedDiffuse, renamedTarget].map { SHA256.hash(data: try Data(contentsOf: $0)) })
        try await project.updateImage(id: selected.id, category: "Wood", verified: true, note: "Corrected")
        let corrected = try await project.snapshot()
        XCTAssertEqual(corrected.datasets.map(\.name), ["Wood"])
    }

    func testSwitchingFoldersResetsEverythingAndReopeningRestoresThatFoldersProject() async throws {
        let first = try fixture(), second = try fixture()
        defer { try? FileManager.default.removeItem(at: first); try? FileManager.default.removeItem(at: second) }
        _ = try image(first, "brick_diff.png"); _ = try image(second, "wood_diff.png")
        let a = try LoomProject(sourceFolder: first)
        var settings = ProjectSettings(); settings.modelName = "My Bricks"; settings.training.epochs = 37
        try await a.updateSettings(settings)
        let one = try await a.snapshot()
        try await a.updateImage(id: one.images[0].id, category: "Bricks", verified: true, note: "")
        let b = try LoomProject(sourceFolder: second)
        let two = try await b.snapshot()
        XCTAssertNotEqual(one.projectID, two.projectID)
        XCTAssertEqual(two.settings.modelName, "Texture Loom")
        XCTAssertEqual(two.settings.training.epochs, 10)
        XCTAssertTrue(two.datasets.isEmpty)
        let reopened = try LoomProject(sourceFolder: first)
        let restored = try await reopened.snapshot()
        XCTAssertEqual(restored.projectID, one.projectID)
        XCTAssertEqual(restored.settings.modelName, "My Bricks")
        XCTAssertEqual(restored.settings.training.epochs, 37)
        XCTAssertEqual(restored.datasets.map(\.name), ["Bricks"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.appendingPathComponent(".loom-factory/project.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: first.appendingPathComponent(".loom-factory/inventory.json").path))
    }

    func testImportLinksOriginalsAndDeletingTheLinkRemovesMembership() async throws {
        let root = try fixture(), originalFolder = try fixture()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: originalFolder) }
        let original = try image(originalFolder, "stone_diff.png")
        let bytes = try Data(contentsOf: original)
        let project = try LoomProject(sourceFolder: root)
        try await project.importImages([original])
        let snapshot = try await project.snapshot()
        XCTAssertEqual(snapshot.images.count, 1)
        let link = try XCTUnwrap(snapshot.images.first?.url)
        XCTAssertTrue(try link.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true)
        XCTAssertEqual(link.resolvingSymlinksInPath(), original.resolvingSymlinksInPath())
        XCTAssertEqual(try Data(contentsOf: original), bytes)
        try FileManager.default.removeItem(at: link)
        let empty = try await project.snapshot()
        XCTAssertTrue(empty.images.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: original.path))
    }

    func testTaggedPairsDoNotRequireReviewAndWrongDatasetsCannotTrain() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try image(root, "brick_diff.png"); _ = try image(root, "brick_disp.png")
        let project = try LoomProject(sourceFolder: root)
        let current = try await project.snapshot()
        let input = try XCTUnwrap(current.images.first { $0.role == .diffuse })
        try await project.updateImage(id: input.id, category: "Bricks", verified: false, note: "")
        let tagged = try await project.snapshot()
        XCTAssertTrue(tagged.images.allSatisfy { !$0.verified })
        XCTAssertEqual(tagged.eligiblePairCount(datasetID: ImageMetadataStore.categoryID("Bricks"), map: .depth, imageSize: 64), 1)
        XCTAssertEqual(try tagged.trainingPlan(datasetID: ImageMetadataStore.categoryID("Bricks"), map: .depth, settings: .init(imageSize: 64))?.trainingPairCount, 1)
        do {
            _ = try await project.trainMap(datasetID: ImageMetadataStore.categoryID("Wood"), map: .depth,
                                           settings: .init(imageSize: 64), progress: { _ in })
            XCTFail("An absent category entered training")
        } catch { XCTAssertTrue(error.localizedDescription.contains("no longer")) }
        do {
            _ = try await project.exportPackage(name: "Texture Loom", version: "1.0.0")
            XCTFail("An untrained project exported a model package")
        } catch { XCTAssertTrue(error.localizedDescription.contains("classifier")) }
    }

    func testCopyingTaggedImagesKeepsTagsAndAssignsDistinctViewerIdentities() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let first = try image(root, "brick_diff.png")
        let project = try LoomProject(sourceFolder: root)
        let before = try await project.snapshot()
        try await project.updateImage(id: before.images[0].id, category: "Bricks", verified: true, note: "Keep this tag")
        let second = try image(root, "copy_diff.png")
        let metadata = try XCTUnwrap(ImageMetadataStore.read(first))
        try ImageMetadataStore.write(metadata, to: second)
        let after = try await project.snapshot()
        XCTAssertEqual(after.images.count, 2)
        XCTAssertEqual(Set(after.images.map(\.id)).count, 2)
        XCTAssertTrue(after.images.allSatisfy { $0.category == "Bricks" && $0.verified && $0.note == "Keep this tag" })
        let refreshed = try await project.snapshot()
        XCTAssertEqual(Set(after.images.map(\.id)), Set(refreshed.images.map(\.id)))
    }

    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-project-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    @discardableResult private func image(_ root: URL, _ filename: String) throws -> URL {
        let file = root.appendingPathComponent(filename)
        let width = 64, height = 64
        let pixels = [UInt8](repeating: 120, count: width * height * 3)
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                         bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [],
                         provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let destination = CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, cg, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return file
    }
}
