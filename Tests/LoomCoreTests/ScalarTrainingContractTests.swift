import Foundation
import LoomNative
import XCTest
@testable import LoomCore

final class ScalarTrainingContractTests: XCTestCase {
    func testFreshScalarRunsUpgradeAnOldProjectWithoutChangingNormalTraining() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("scalar-contract-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let project = try LoomProject(sourceFolder: root)
        let legacy = try JSONDecoder().decode(TrainingSettings.self, from: Data("{}".utf8))
        XCTAssertEqual(legacy.scalarLoss, .absoluteV1)
        for map in [MapRole.depth, .roughness] {
            let prepared = try await project.prepareTrainingStart(.fresh, datasetID: "fixture", map: map, settings: legacy)
            XCTAssertEqual(prepared.settings.scalarLoss, .logitCrossEntropyV2)
            XCTAssertEqual(prepared.settings.modelArchitecture, legacy.modelArchitecture)
            XCTAssertNil(prepared.resume)
        }
        let normal = try await project.prepareTrainingStart(.fresh, datasetID: "fixture", map: .normal, settings: legacy)
        XCTAssertEqual(normal.settings, legacy)
    }

    func testSavedObjectiveIDsRoundTripAndUnknownIDsAreRejected() throws {
        let architecture = try NativeArchitecture(mapKind: "roughness", width: 8, blocks: 1)
        let descriptor: [String: Any] = ["architecture": architecture.identifier, "architecture_config": architecture.descriptor]
        for objective in NativeScalarLoss.allCases where objective != .linearDetailV1 {
            let settings = TrainingSettings(imageSize: 32, modelWidth: 8, modelBlocks: 1,
                modelArchitecture: architecture.identifier, scalarLoss: objective)
            let record = NativeFactoryTraining.settingsJSON(settings)
            XCTAssertEqual(TrainingRunFiles.settings(from: record, descriptor: descriptor)?.scalarLoss, objective)
        }
        var unsupported = NativeFactoryTraining.settingsJSON(TrainingSettings(imageSize: 32, modelWidth: 8, modelBlocks: 1,
            modelArchitecture: architecture.identifier))
        unsupported["scalar_loss"] = "unknown-objective"
        XCTAssertNil(TrainingRunFiles.settings(from: unsupported, descriptor: descriptor))
    }
}
