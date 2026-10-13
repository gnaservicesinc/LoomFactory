import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class CategorizerTests: XCTestCase {
    func testOnePredictionPerMaterialFolderSelectsSmallestSuitableDiffuse() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let small = try image(root, "Oak/oak_diff_small.png", width: 32, height: 32)
        let fitting = try image(root, "Oak/oak_diff_medium.png", width: 64, height: 64)
        _ = try image(root, "Oak/oak_diff_large.png", width: 128, height: 128)
        _ = try image(root, "Oak/oak_disp_small.png", width: 32, height: 32)
        _ = try image(root, "Oak/oak_rough_large.png", width: 128, height: 128)
        _ = try image(root, "Pine/pine_diff.png", width: 64, height: 64)
        _ = try image(root, "Pine/pine_disp.png", width: 64, height: 64)
        let originalPixels = try Data(contentsOf: small)
        let project = try LoomProject(sourceFolder: root)
        let predictions = PredictionRecorder()

        let summary = try await project.categorizeFolders(inputWidth: 64, inputHeight: 64, predict: {
            await predictions.record($0)
            return "Wood"
        }, progress: { _, _ in })

        let selected = await predictions.images
        XCTAssertEqual(selected.count, 2)
        XCTAssertEqual(selected.filter { $0.url.deletingLastPathComponent().lastPathComponent == "Oak" }.map(\.url),
                       [fitting.resolvingSymlinksInPath()])
        XCTAssertEqual(summary.predictionCount, 2)
        XCTAssertEqual(summary.folderCount, 2)
        let snapshot = try await project.snapshot()
        XCTAssertTrue(snapshot.images.allSatisfy { $0.category == "Wood" && !$0.verified })
        XCTAssertEqual(try Data(contentsOf: small), originalPixels)
    }

    func testNoSuitableDiffuseUsesLargestAvailableAndKeepsNestedFoldersSeparate() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try image(root, "Stone/stone_diff_small.png", width: 32, height: 32)
        let largest = try image(root, "Stone/stone_diff_large.png", width: 48, height: 48)
        _ = try image(root, "Stone/Nested/other_diff.png", width: 40, height: 40)
        let project = try LoomProject(sourceFolder: root)
        let predictions = PredictionRecorder()
        let summary = try await project.categorizeFolders(inputWidth: 64, inputHeight: 64, predict: {
            await predictions.record($0)
            return "Rock"
        }, progress: { _, _ in })

        let selected = await predictions.images
        XCTAssertEqual(summary.predictionCount, 2)
        XCTAssertEqual(selected.filter { $0.url.deletingLastPathComponent().lastPathComponent == "Stone" }.map(\.url),
                       [largest.resolvingSymlinksInPath()])
    }

    func testResumeReusesExistingCategoryPreservesReviewNotesAndPixels() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let reviewed = try image(root, "Cloth/cloth_diff_64.png", width: 64, height: 64)
        let remaining = try image(root, "Cloth/cloth_diff_128.png", width: 128, height: 128)
        let target = try image(root, "Cloth/cloth_disp_128.png", width: 128, height: 128)
        let protected = try image(root, "Cloth/cloth_rough_128.png", width: 128, height: 128)
        let originalPixels = try [reviewed, remaining, target, protected].map { try Data(contentsOf: $0) }
        let project = try LoomProject(sourceFolder: root)
        _ = try await project.snapshot()
        var reviewedTag = try XCTUnwrap(ImageMetadataStore.read(reviewed))
        reviewedTag.category = "Fabric"
        reviewedTag.verified = true
        reviewedTag.note = "Keep this manual review"
        try ImageMetadataStore.write(reviewedTag, to: reviewed)
        var protectedTag = try XCTUnwrap(ImageMetadataStore.read(protected))
        protectedTag.verified = true
        protectedTag.note = "Reviewed without a category"
        try ImageMetadataStore.write(protectedTag, to: protected)
        let predictions = PredictionRecorder()

        let summary = try await project.categorizeFolders(inputWidth: 64, inputHeight: 64, predict: {
            await predictions.record($0)
            return "Wrong Prediction"
        }, progress: { _, _ in })

        let selected = await predictions.images
        XCTAssertTrue(selected.isEmpty)
        XCTAssertEqual(summary.predictionCount, 0)
        XCTAssertEqual(summary.reusedFolderCount, 1)
        XCTAssertEqual(try ImageMetadataStore.read(reviewed), reviewedTag)
        XCTAssertEqual(try ImageMetadataStore.read(protected), protectedTag)
        XCTAssertEqual(try ImageMetadataStore.read(remaining)?.category, "Fabric")
        XCTAssertEqual(try ImageMetadataStore.read(target)?.category, "Fabric")
        XCTAssertEqual(try [reviewed, remaining, target, protected].map { try Data(contentsOf: $0) }, originalPixels)
    }

    func testConflictingExistingCategoriesAreReportedWithoutGuessingOrOverwriting() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = try image(root, "Mixed/material_diff_64.png", width: 64, height: 64)
        let second = try image(root, "Mixed/material_diff_128.png", width: 128, height: 128)
        let untagged = try image(root, "Mixed/material_disp_128.png", width: 128, height: 128)
        let project = try LoomProject(sourceFolder: root)
        _ = try await project.snapshot()
        var firstTag = try XCTUnwrap(ImageMetadataStore.read(first))
        firstTag.category = "Fabric"
        firstTag.note = "Manual edit"
        try ImageMetadataStore.write(firstTag, to: first)
        var secondTag = try XCTUnwrap(ImageMetadataStore.read(second))
        secondTag.category = "Ground"
        try ImageMetadataStore.write(secondTag, to: second)
        let predictions = PredictionRecorder()

        let summary = try await project.categorizeFolders(inputWidth: 64, inputHeight: 64, predict: {
            await predictions.record($0)
            return "Unexpected"
        }, progress: { _, _ in })

        let selected = await predictions.images
        XCTAssertTrue(selected.isEmpty)
        XCTAssertEqual(summary.conflictingFolders.map(\.lastPathComponent), ["Mixed"])
        XCTAssertEqual(try ImageMetadataStore.read(first), firstTag)
        XCTAssertEqual(try ImageMetadataStore.read(second), secondTag)
        XCTAssertNil(try ImageMetadataStore.read(untagged)?.category)
    }

    func testCancelledRunResumesWithoutPredictingCompletedFolderAgain() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try image(root, "First/material_diff.png", width: 64, height: 64)
        _ = try image(root, "First/material_disp.png", width: 64, height: 64)
        _ = try image(root, "Second/material_diff.png", width: 64, height: 64)
        _ = try image(root, "Second/material_disp.png", width: 64, height: 64)
        let project = try LoomProject(sourceFolder: root)
        let predictions = PredictionRecorder()
        do {
            _ = try await project.categorizeFolders(inputWidth: 64, inputHeight: 64, predict: {
                let count = await predictions.record($0)
                if count == 2 { throw CancellationError() }
                return "Bricks"
            }, progress: { _, _ in })
            XCTFail("Prediction cancellation should stop the operation")
        } catch is CancellationError {
        }
        let interrupted = try await project.snapshot()
        XCTAssertEqual(interrupted.images.filter { $0.category == "Bricks" }.count, 2)
        let summary = try await project.categorizeFolders(inputWidth: 64, inputHeight: 64, predict: {
            await predictions.record($0)
            return "Bricks"
        }, progress: { _, _ in })

        XCTAssertEqual(summary.predictionCount, 1)
        let selected = await predictions.images
        let completed = selected[0].url.deletingLastPathComponent()
        XCTAssertEqual(selected.filter { $0.url.deletingLastPathComponent() == completed }.count, 1)
        let resumed = try await project.snapshot()
        XCTAssertTrue(resumed.images.allSatisfy { $0.category == "Bricks" })
    }

    func testDeletedOrRetaggedRepresentativeCannotApplyStalePrediction() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let removed = try image(root, "Deleted/gone_diff.png", width: 64, height: 64)
        let removedTarget = try image(root, "Deleted/gone_disp.png", width: 64, height: 64)
        let retagged = try image(root, "Changed/changed_diff.png", width: 64, height: 64)
        let retaggedTarget = try image(root, "Changed/changed_disp.png", width: 64, height: 64)
        let project = try LoomProject(sourceFolder: root)

        _ = try await project.categorizeFolders(inputWidth: 64, inputHeight: 64, predict: { image in
            if image.url.lastPathComponent == "gone_diff.png" {
                try FileManager.default.removeItem(at: image.url)
            } else {
                var tag = try XCTUnwrap(ImageMetadataStore.read(image.url))
                tag.category = "Manual Correction"
                tag.verified = true
                tag.note = "Changed during prediction"
                try ImageMetadataStore.write(tag, to: image.url)
            }
            return "Stale Prediction"
        }, progress: { _, _ in })

        XCTAssertFalse(FileManager.default.fileExists(atPath: removed.path))
        XCTAssertNil(try ImageMetadataStore.read(removedTarget)?.category)
        XCTAssertEqual(try ImageMetadataStore.read(retagged)?.category, "Manual Correction")
        XCTAssertEqual(try ImageMetadataStore.read(retagged)?.note, "Changed during prediction")
        XCTAssertTrue(try ImageMetadataStore.read(retagged)?.verified == true)
        XCTAssertNotEqual(try ImageMetadataStore.read(retaggedTarget)?.category, "Stale Prediction")
        let current = try await project.snapshot()
        XCTAssertEqual(current.images.count, 3)
    }

    func testImportedSymlinksGroupByProjectFolderAndRemovedLinkLeavesNoMembership() async throws {
        let root = try fixture(), originals = try fixture()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: originals) }
        let first = try image(originals, "One/first_diff.png", width: 64, height: 64)
        let second = try image(originals, "Two/second_diff.png", width: 128, height: 128)
        let folder = root.appendingPathComponent("ImportedMaterial", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let firstLink = folder.appendingPathComponent(first.lastPathComponent)
        let secondLink = folder.appendingPathComponent(second.lastPathComponent)
        try FileManager.default.createSymbolicLink(at: firstLink, withDestinationURL: first)
        try FileManager.default.createSymbolicLink(at: secondLink, withDestinationURL: second)
        let originalPixels = try [first, second].map { try Data(contentsOf: $0) }
        let project = try LoomProject(sourceFolder: root)
        let predictions = PredictionRecorder()

        let summary = try await project.categorizeFolders(inputWidth: 64, inputHeight: 64, predict: {
            await predictions.record($0)
            return "Stone"
        }, progress: { _, _ in })

        XCTAssertEqual(summary.predictionCount, 1)
        let selected = await predictions.images
        XCTAssertEqual(selected.first?.url.deletingLastPathComponent().lastPathComponent, "ImportedMaterial")
        XCTAssertEqual(try ImageMetadataStore.read(first)?.category, "Stone")
        XCTAssertEqual(try ImageMetadataStore.read(second)?.category, "Stone")
        XCTAssertEqual(try [first, second].map { try Data(contentsOf: $0) }, originalPixels)
        try FileManager.default.removeItem(at: firstLink)
        let remaining = try await project.snapshot()
        XCTAssertEqual(remaining.images.count, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.path))
    }

    func testEditingAnotherMapDuringPredictionPreservesTheWholeFolderDecision() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let input = try image(root, "Material/cloth_diff.png", width: 64, height: 64)
        let target = try image(root, "Material/cloth_disp.png", width: 64, height: 64)
        let project = try LoomProject(sourceFolder: root)
        let summary = try await project.categorizeFolders(inputWidth: 64, inputHeight: 64, predict: { _ in
            var tag = try XCTUnwrap(ImageMetadataStore.read(target))
            tag.category = "Manual Correction"; tag.verified = true
            try ImageMetadataStore.write(tag, to: target)
            return "Stale Prediction"
        }, progress: { _, _ in })
        XCTAssertEqual(summary.predictionCount, 1)
        XCTAssertEqual(summary.folderCount, 0)
        XCTAssertNil(try ImageMetadataStore.read(input)?.category)
        XCTAssertEqual(try ImageMetadataStore.read(target)?.category, "Manual Correction")
        XCTAssertTrue(try ImageMetadataStore.read(target)?.verified == true)
    }

    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-categorizer-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @discardableResult private func image(_ root: URL, _ name: String, width: Int, height: Int) throws -> URL {
        let file = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let pixels = [UInt8](repeating: 120, count: width * height * 3)
        let provider = try XCTUnwrap(CGDataProvider(data: Data(pixels) as CFData))
        let image = try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                                        bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [],
                                        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return file
    }
}

private actor PredictionRecorder {
    private(set) var images: [SourceImage] = []
    @discardableResult func record(_ image: SourceImage) -> Int {
        images.append(image)
        return images.count
    }
}
