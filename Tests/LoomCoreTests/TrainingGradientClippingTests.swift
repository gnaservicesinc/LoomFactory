import Foundation
import XCTest
@testable import LoomCore

final class TrainingGradientClippingTests: XCTestCase {
    func testExistingSettingsKeepFixedClippingAndZeroRemainsDisabled() throws {
        for norm in [0.0, 0.5, 1.0] {
            let data = Data("{\"gradientClipNorm\":\(norm)}".utf8)
            let settings = try JSONDecoder().decode(TrainingSettings.self, from: data)
            XCTAssertEqual(settings.gradientClipPolicy, .fixed)
            XCTAssertEqual(settings.gradientClipNorm, norm)
            XCTAssertNil(settings.validationIssue)
        }
        XCTAssertEqual(TrainingSettings().gradientClipPolicy, .warmup)
    }

    func testEveryClippingPolicyRoundTripsWithSavedSettings() throws {
        for policy in TrainingGradientClipping.allCases {
            let settings = TrainingSettings(gradientClipNorm: 0.8, gradientClipPolicy: policy,
                gradientClipWarmupStartNorm: 0.1, gradientClipWarmupUpdates: 37)
            let decoded = try JSONDecoder().decode(TrainingSettings.self, from: JSONEncoder().encode(settings))
            XCTAssertEqual(decoded, settings)
        }
    }

    func testActiveWarmupControlsRejectInvalidValues() {
        var settings = TrainingSettings(gradientClipNorm: 1, gradientClipPolicy: .warmup)
        for norm in [0.0, -1, 1.1, .infinity, .nan] {
            settings.gradientClipWarmupStartNorm = norm
            XCTAssertNotNil(settings.validationIssue)
        }
        settings.gradientClipWarmupStartNorm = 1
        for updates in [0, -1, 1_000_001, Int.max] {
            settings.gradientClipWarmupUpdates = updates
            XCTAssertNotNil(settings.validationIssue)
        }
        settings.gradientClipWarmupUpdates = 1
        XCTAssertNil(settings.validationIssue)
    }

    func testInactiveClippingControlsDoNotDisableTraining() {
        var settings = TrainingSettings(gradientClipPolicy: .off,
            gradientClipWarmupStartNorm: -1, gradientClipWarmupUpdates: -1)
        settings.gradientClipNorm = -1
        XCTAssertNil(settings.validationIssue)
        settings.gradientClipPolicy = .fixed
        XCTAssertNotNil(settings.validationIssue)
        settings.gradientClipNorm = 0.1
        XCTAssertNil(settings.validationIssue)
        settings.gradientClipPolicy = .warmup
        XCTAssertNotNil(settings.validationIssue)
        settings.gradientClipNorm = 0
        XCTAssertNil(settings.validationIssue, "Legacy zero disables clipping even when the selected policy is warmup")
    }
}
