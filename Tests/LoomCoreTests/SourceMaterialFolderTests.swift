import Foundation
import XCTest
@testable import LoomCore

final class SourceMaterialFolderTests: XCTestCase {
    func testMapsAndNativeResolutionsBelongToOnePhysicalMaterialFolder() {
        let root = URL(fileURLWithPath: "/source")
        let folder = root.appendingPathComponent("bark_brown_01")
        let images = [member(folder, .bump, 1024), member(folder, .diffuse, 4096),
                      member(folder, .depth, 1024), member(folder, .diffuse, 1024), member(folder, .diffuse, 2048)]
        let groups = SourceMaterialFolder.group(images: images, sourceFolder: root)
        XCTAssertEqual(groups.count, 1); XCTAssertEqual(groups.first?.displayName, "Bark Brown 01")
        XCTAssertEqual(groups.first?.imageCount, 5)
        XCTAssertEqual(groups.first?.resolutions.map(\.width), [1024, 2048, 4096])
        XCTAssertEqual(groups.first?.resolutions.first?.images.map(\.role), [.diffuse, .depth, .bump])
    }

    func testNestedFoldersKeepDistinctIdentitiesAndDeletedMembersDisappear() {
        let root = URL(fileURLWithPath: "/source")
        let first = member(root.appendingPathComponent("Collection A/bark"), .diffuse, 1024)
        let second = member(root.appendingPathComponent("Collection B/bark"), .diffuse, 1024)
        let before = SourceMaterialFolder.group(images: [first, second], sourceFolder: root)
        XCTAssertEqual(before.count, 2); XCTAssertEqual(Set(before.map(\.id)).count, 2)
        XCTAssertEqual(before.map(\.displayName), ["Collection A / Bark", "Collection B / Bark"])
        let after = SourceMaterialFolder.group(images: [second], sourceFolder: root)
        XCTAssertEqual(after.count, 1); XCTAssertEqual(after.first?.id, before[1].id)
    }

    private func member(_ folder: URL, _ role: MapRole, _ size: Int) -> SourceImage {
        let name = "\(role.rawValue)_\(size).png"
        return SourceImage(id: folder.path + name, url: folder.appendingPathComponent(name), displayName: name,
                           role: role, category: "Bark", verified: false, note: "", pairingKey: "pair-\(size)", width: size, height: size)
    }
}
