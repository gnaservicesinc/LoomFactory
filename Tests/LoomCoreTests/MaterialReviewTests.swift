import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
@testable import LoomCore

struct MaterialReviewTests {
    @Test
    func reviewingOneVisibleMapMarksItsEntirePhysicalMaterialAndPreservesEveryOtherFieldAndBytes() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-material-review-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        var urls: [URL] = []
        for (folder, name, role) in [("Cloth", "diff_1k", MapRole.diffuse), ("Cloth", "depth_1k", .depth),
                                     ("Cloth", "diff_2k", .diffuse), ("Stone", "diff_1k", .diffuse)] {
            let url = root.appendingPathComponent(folder).appendingPathComponent(name + ".png")
            try writeReviewImage(url)
            try ImageMetadataStore.write(.init(pairingKey: folder + name, role: role, category: folder,
                verified: false, note: "Keep " + name, roleWasManuallySelected: true, ignored: name == "depth_1k"), to: url)
            urls.append(url)
        }
        let originals = try urls.map { try #require(try ImageMetadataStore.read($0)) }
        let bytes = try urls.map { try Data(contentsOf: $0) }
        let project = try LoomProject(sourceFolder: root)
        try await project.setMaterialReviewed(imageIDs: [originals[0].id], verified: true)
        for index in urls.indices {
            var expected = originals[index]
            expected.verified = index < 3
            #expect(try ImageMetadataStore.read(urls[index]) == expected)
            #expect(try Data(contentsOf: urls[index]) == bytes[index])
        }
        let snapshot = try await project.snapshot()
        #expect(snapshot.images.filter(\.verified).count == 3)
        #expect(snapshot.images.filter(\.ignored).count == 1)
        try await project.setMaterialReviewed(imageIDs: [originals[1].id], verified: false)
        #expect(try urls.map { try #require(try ImageMetadataStore.read($0)) } == originals)
    }

    @Test
    func staleReviewSelectionCannotPartiallyWriteOtherMaterials() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-material-review-stale-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("material_diff.png")
        try writeReviewImage(url)
        let original = ImageMetadata(pairingKey: "material", role: .diffuse, category: "Cloth", note: "unchanged")
        try ImageMetadataStore.write(original, to: url)
        let project = try LoomProject(sourceFolder: root)
        await #expect(throws: LoomError.self) {
            try await project.setMaterialReviewed(imageIDs: [original.id, "removed-id"], verified: true)
        }
        #expect(try ImageMetadataStore.read(url) == original)
    }
}

private func writeReviewImage(_ url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let provider = try #require(CGDataProvider(data: Data(repeating: 128, count: 16 * 16 * 4) as CFData))
    let image = try #require(CGImage(width: 16, height: 16, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 16 * 4,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let writer = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(writer, image, nil)
    #expect(CGImageDestinationFinalize(writer))
}
