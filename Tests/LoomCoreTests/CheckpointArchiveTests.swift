import Foundation
import LoomNative
import XCTest
@testable import LoomCore

@MainActor
final class CheckpointArchiveTests: XCTestCase {
    func testEveryIntervalIsListedAndExactEarlierWeightsCanBeSelectedAfterReopening() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        var saved: [(id: String, bytes: Data)] = []
        for (epoch, value) in [(25, Float(0.1)), (50, Float(0.2)), (75, Float(0.3))] {
            try fixture.save(epoch: epoch, value: value)
            let descriptor = try NativeFactoryTraining.json(fixture.directory.appendingPathComponent("checkpoint.json"))
            let id = try XCTUnwrap(descriptor["checkpoint_id"] as? String)
            let path = try XCTUnwrap(descriptor["weights_path"] as? String)
            saved.append((id, try Data(contentsOf: fixture.directory.appendingPathComponent(path))))
        }
        let reopened = try LoomProject(sourceFolder: fixture.root)
        let checkpoints = try await reopened.checkpointVersions(runID: fixture.runID)
        XCTAssertEqual(checkpoints.count, 3)
        XCTAssertEqual(Set(checkpoints.map(\.epochsCompleted)), [25, 50, 75])
        let versions = try await reopened.modelVersions()
        let version = try XCTUnwrap(versions.first)
        XCTAssertEqual(versions.count, 1, "Checkpoint intervals belong to their original run")
        XCTAssertEqual(version.checkpointVersions.count, 3)
        XCTAssertEqual(version.checkpoint?.epochsCompleted, 75)
        let snapshot = try await reopened.snapshot()
        XCTAssertEqual(snapshot.checkpoints.count, 1, "Coverage counts runs rather than interval saves")
        for saved in saved {
            let choice = try ModelTestSelection.load(runID: fixture.runID, source: .checkpoint,
                datasetID: fixture.datasetID, map: .depth, projectFolder: reopened.projectFolder,
                projectID: fixture.projectID, checkpointID: saved.id)
            XCTAssertEqual(try Data(contentsOf: choice.weightsURL), saved.bytes)
            XCTAssertEqual(choice.checkpointID, saved.id)
        }
        let first = try XCTUnwrap(saved.first)
        let promoted = try await reopened.selectCheckpointModelVersion(runID: fixture.runID,
            datasetID: fixture.datasetID, map: .depth, checkpointID: first.id)
        XCTAssertEqual(try Data(contentsOf: promoted.directory.appendingPathComponent("model.safetensors")), first.bytes)
        let descriptor = try NativeFactoryTraining.json(promoted.directory.appendingPathComponent("model.json"))
        XCTAssertEqual(descriptor["checkpoint_id"] as? String, first.id)
        XCTAssertEqual(descriptor["checkpoint_updates"] as? Int, 250)
        let preserved = try await reopened.checkpointVersions(runID: fixture.runID)
        XCTAssertEqual(preserved.count, 3)
    }

    func testFirstNewSavePreservesLegacyLatestAndRejectsMissingOrEscapedRevision() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        try fixture.save(epoch: 25, value: 0.1)
        let latest = fixture.directory.appendingPathComponent("checkpoint.json")
        var legacy = try NativeFactoryTraining.json(latest)
        let oldID = try XCTUnwrap(legacy.removeValue(forKey: "checkpoint_id") as? String)
        try FileManager.default.removeItem(at: fixture.directory.appendingPathComponent(try CheckpointArchive.filename(oldID)))
        try NativeFactoryTraining.writeJSON(legacy, to: latest)
        let oldPath = try XCTUnwrap(legacy["weights_path"] as? String)
        let oldBytes = try Data(contentsOf: fixture.directory.appendingPathComponent(oldPath))
        try fixture.save(epoch: 50, value: 0.2)
        let checkpoints = try await fixture.project.checkpointVersions(runID: fixture.runID)
        XCTAssertEqual(checkpoints.count, 2)
        let archivedLegacy = try XCTUnwrap(checkpoints.first { $0.revisionID?.hasPrefix("legacy-") == true })
        let choice = try ModelTestSelection.load(runID: fixture.runID, source: .checkpoint,
            datasetID: fixture.datasetID, map: .depth, projectFolder: fixture.project.projectFolder,
            projectID: fixture.projectID, checkpointID: archivedLegacy.revisionID)
        XCTAssertEqual(try Data(contentsOf: choice.weightsURL), oldBytes)
        for id in ["missing", "../checkpoint", "/tmp/outside"] {
            XCTAssertThrowsError(try ModelTestSelection.load(runID: fixture.runID, source: .checkpoint,
                datasetID: fixture.datasetID, map: .depth, projectFolder: fixture.project.projectFolder,
                projectID: fixture.projectID, checkpointID: id))
        }
    }

    private struct Fixture {
        let root: URL
        let project: LoomProject
        let projectID: String
        let runID = "checkpoint-history-fixture"
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        let directory: URL
        init() async throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-checkpoint-history-" + UUID().uuidString)
                .resolvingSymlinksInPath().standardizedFileURL
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            project = try LoomProject(sourceFolder: root)
            projectID = try await project.snapshot().projectID
            directory = project.projectFolder.appendingPathComponent("runs/" + runID)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
        func save(epoch: Int, value: Float) throws {
            let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 1)
            let weights = architecture.parameterShapes.mapValues { shape in
                NativeTensor(shape: shape, values: [Float](repeating: value, count: shape.reduce(1, *)))
            }
            let descriptor: [String: Any] = ["schema": "loom-factory-checkpoint-v1", "status": "saved",
                "project_id": projectID, "kind": "map", "dataset_id": datasetID, "dataset_name": "Bricks", "map": "depth",
                "architecture": architecture.identifier, "architecture_config": architecture.descriptor,
                "settings": NativeFactoryTraining.settingsJSON(TrainingSettings(imageSize: 32, epochs: 100,
                    modelWidth: 8, modelBlocks: 1, modelArchitecture: architecture.identifier)),
                "epochs_completed": epoch, "updates": epoch * 10,
                "history": [["epoch": epoch, "loss": 0.1]], "saved_at": ISO8601DateFormatter().string(from: .now)]
            try TrainingCheckpointFiles.write(weights: weights, optimizer: [:], descriptor: descriptor, directory: directory)
        }
    }
}
