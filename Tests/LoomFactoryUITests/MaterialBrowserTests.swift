import Foundation
import LoomCore
import Testing
@testable import LoomFactory

struct MaterialBrowserTests {
    @Test
    func selectionExpandsOnlyItsMaterialAndResolutionAndKeepsUserExpansion() throws {
        let source = URL(fileURLWithPath: "/Example Source", isDirectory: true)
        let images = [
            image(id: "first-low", source: source, material: "bark_brown_01", size: 1024),
            image(id: "first-high", source: source, material: "bark_brown_01", size: 2048),
            image(id: "second-low", source: source, material: "bark_willow", size: 1024)
        ]
        let folders = SourceMaterialFolder.group(images: images, sourceFolder: source)
        #expect(folders.count == 2)
        let first = try #require(folders.first { $0.displayName == "Bark Brown 01" })
        let second = try #require(folders.first { $0.displayName == "Bark Willow" })
        let high = try #require(first.resolutions.first { $0.width == 2048 })
        var expansion = MaterialBrowserExpansion()
        expansion.reveal(selectedImageID: "first-high", in: folders)
        #expect(expansion.materials == [first.id])
        #expect(expansion.resolutions == [high.id])
        expansion.reveal(selectedImageID: "second-low", in: folders)
        #expect(expansion.materials == [first.id, second.id])
        expansion.materials.remove(first.id)
        expansion.retainExisting(in: SourceMaterialFolder.group(images: images, sourceFolder: source))
        #expect(expansion.materials == [second.id])
        #expect(expansion.resolutions.contains(high.id))
    }

    @Test
    func pollingRemovesDeletedExpansionWithoutOpeningOtherMaterials() throws {
        let source = URL(fileURLWithPath: "/Example Source", isDirectory: true)
        let first = image(id: "first", source: source, material: "bark_brown_01", size: 1024)
        let second = image(id: "second", source: source, material: "bark_willow", size: 1024)
        let folders = SourceMaterialFolder.group(images: [first, second], sourceFolder: source)
        var expansion = MaterialBrowserExpansion()
        expansion.reveal(selectedImageID: first.id, in: folders)
        expansion.retainExisting(in: SourceMaterialFolder.group(images: [second], sourceFolder: source))
        #expect(expansion.materials.isEmpty)
        #expect(expansion.resolutions.isEmpty)
        expansion.reveal(selectedImageID: "deleted", in: folders)
        #expect(expansion.materials.isEmpty)
    }

    private func image(id: String, source: URL, material: String, size: Int) -> SourceImage {
        SourceImage(id: id, url: source.appendingPathComponent(material, isDirectory: true).appendingPathComponent("\(id).png"),
                    displayName: id, role: .diffuse, category: "Bark", verified: true, note: "",
                    pairingKey: "\(material)/\(size)", width: size, height: size)
    }
}
