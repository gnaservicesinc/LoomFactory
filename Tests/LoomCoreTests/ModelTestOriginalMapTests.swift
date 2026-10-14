import Foundation
import XCTest
@testable import LoomCore

final class ModelTestOriginalMapTests: XCTestCase {
    func testOriginalPairIsShownWithoutTrainingApprovalReviewOrCropRestrictions() {
        let input = image("diffuse", role: .diffuse, size: 96)
        var target = image("depth", role: .depth, size: 96)
        target.ignored = true; target.category = nil
        let lowerResolution = image("depth-small", role: .depth, size: 48)
        let otherMaterial = image("other", role: .depth, size: 96, pairingKey: "other")
        let result = ModelTestOriginalMap.resolve(inputURL: input.url, map: .depth,
                                                images: [input, target, lowerResolution, otherMaterial])
        XCTAssertEqual(result.image, target)
        XCTAssertNil(result.issue)
    }

    func testAmbiguousOrMissingTargetsRemainOptionalAndNeverGuessAnotherMaterial() {
        let input = image("diffuse", role: .diffuse, size: 256)
        let first = image("first-depth", role: .depth, size: 256)
        let second = image("second-depth", role: .depth, size: 256)
        let ambiguous = ModelTestOriginalMap.resolve(inputURL: input.url, map: .depth, images: [input, first, second])
        XCTAssertNil(ambiguous.image); XCTAssertTrue(ambiguous.issue?.contains("Several original") == true)
        let missing = ModelTestOriginalMap.resolve(inputURL: input.url, map: .roughness, images: [input, first])
        XCTAssertNil(missing.image); XCTAssertTrue(missing.issue?.contains("still test") == true)
        let external = ModelTestOriginalMap.resolve(inputURL: URL(fileURLWithPath: "/outside/image.png"), map: .depth,
                                                   images: [input, first])
        XCTAssertNil(external.image); XCTAssertNil(external.issue)
    }

    func testOriginalNormalConventionIsPreservedForDisplay() {
        let input = image("diffuse", role: .diffuse, size: 256)
        let normal = SourceImage(id: "normal", url: URL(fileURLWithPath: "/material/normal.png"), displayName: "Normal",
            role: .normal, category: "Bricks", verified: false, note: "", pairingKey: "material", width: 256, height: 256,
            normalConvention: "DirectX")
        let result = ModelTestOriginalMap.resolve(inputURL: input.url, map: .normal, images: [input, normal])
        XCTAssertEqual(result.image?.normalConvention, "DirectX")
    }

    private func image(_ name: String, role: MapRole, size: Int, pairingKey: String = "material") -> SourceImage {
        .init(id: name, url: URL(fileURLWithPath: "/material/\(name).png"), displayName: name, role: role,
              category: "Bricks", verified: false, note: "", pairingKey: pairingKey, width: size, height: size)
    }
}
