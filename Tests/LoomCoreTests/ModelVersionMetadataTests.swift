import Foundation
import XCTest
@testable import LoomCore

final class ModelVersionMetadataTests: XCTestCase {
    func testNameAndNotesPersistSeparatelyFromRunArtifactsAndSettings() async throws {
        let source = FileManager.default.temporaryDirectory.appendingPathComponent("loom-version-metadata-" + UUID().uuidString)
            .resolvingSymlinksInPath().standardizedFileURL
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: source) }
        let project = try LoomProject(sourceFolder: source)
        let snapshot = try await project.snapshot()
        let run = project.projectFolder.appendingPathComponent("runs/example")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        var settings = TrainingSettings()
        settings.optimizer = .adamW; settings.learningRate = 0.00042; settings.seed = 91
        try TrainingRunFiles.writeMetadata(.init(projectID: snapshot.projectID, runID: "example", datasetID: "bricks",
            map: .depth, label: "Original name", createdAt: .now, settings: settings, lineage: nil, status: "cancelled"), at: run)
        let weights = run.appendingPathComponent("checkpoint.safetensors")
        try Data("unchanged weights".utf8).write(to: weights)
        let originals = [run.appendingPathComponent("run-version.json"), weights]
        let originalBytes = try originals.map { try Data(contentsOf: $0) }

        try await project.updateModelVersionMetadata(runID: "example", name: "  Brick detail study  ", notes: "Check mortar on held-out materials.")
        let versions = try await project.modelVersions()
        let version = try XCTUnwrap(versions.first)
        XCTAssertEqual(version.label, "Brick detail study")
        XCTAssertEqual(version.notes, "Check mortar on held-out materials.")
        XCTAssertEqual(version.savedSettings, settings)
        for (index, file) in originals.enumerated() { XCTAssertEqual(try Data(contentsOf: file), originalBytes[index]) }
        let reopened = try LoomProject(sourceFolder: source)
        let reloadedVersions = try await reopened.modelVersions()
        let reloaded = try XCTUnwrap(reloadedVersions.first)
        XCTAssertEqual(reloaded.label, version.label); XCTAssertEqual(reloaded.notes, version.notes)
        let persistenceURLs = await project.managementPersistenceURLs
        XCTAssertTrue(persistenceURLs.contains(project.projectFolder.appendingPathComponent("model-version-metadata.json")))
        do {
            try await project.updateModelVersionMetadata(runID: "../example", name: "Wrong run", notes: "")
            XCTFail("A model name must not address another directory")
        } catch {}
        do {
            try await project.updateModelVersionMetadata(runID: "example", name: "  ", notes: "")
            XCTFail("An empty model name must not be saved")
        } catch {}
        let remainingVersions = try await reopened.modelVersions()
        let remaining = try XCTUnwrap(remainingVersions.first)
        XCTAssertEqual(remaining.label, version.label)
    }
}
