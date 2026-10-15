import Foundation
import LoomNative
import XCTest
@testable import LoomCore

final class PyramidV2ContractTests: XCTestCase {
    func testDescriptorBindsBothFeatureScalesAndActivation() throws {
        let architecture = try NativeArchitecture(mapKind: "depth", width: 38, blocks: 4, identifier: NativeArchitecture.pyramidV2Identifier)
        let descriptor: [String: Any] = ["architecture": architecture.identifier, "architecture_config": architecture.descriptor]
        XCTAssertEqual(try TrainingRunFiles.nativeArchitecture(from: descriptor, map: .depth), architecture)
        for key in ["hidden_activation", "negative_slope", "head_activation", "scale_channels", "residual_scale", "upsampling"] {
            var missing = architecture.descriptor; missing.removeValue(forKey: key)
            XCTAssertThrowsError(try TrainingRunFiles.nativeArchitecture(from: ["architecture": architecture.identifier,
                "architecture_config": missing], map: .depth), key)
        }
        var inconsistent = architecture.descriptor; inconsistent["negative_slope"] = 0
        XCTAssertThrowsError(try TrainingRunFiles.nativeArchitecture(from: ["architecture": architecture.identifier,
            "architecture_config": inconsistent], map: .depth))
    }
}
