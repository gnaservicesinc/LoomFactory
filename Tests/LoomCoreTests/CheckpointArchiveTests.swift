import Foundation
import LoomNative
import XCTest
@testable import LoomCore

@MainActor
final class CheckpointArchiveTests: XCTestCase {
    func testCompletedRunRetainsEveryCheckpointAndSweepsNativeOutputsInTrainingOrder() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        for (epoch, value) in [(25, Float(0.001)), (50, Float(0.002)), (75, Float(0.003))] {
            try fixture.save(epoch: epoch, value: value)
        }
        try fixture.complete()
        let input = fixture.root.appendingPathComponent("input.png")
        try ModelTestImage.writePNG([Float](repeating: 0.4, count: 32 * 32 * 3), channels: 3, width: 32, height: 32, to: input)
        let originals = try FileManager.default.contentsOfDirectory(at: fixture.directory, includingPropertiesForKeys: nil)
            .reduce(into: [URL: Data]()) { $0[$1] = try Data(contentsOf: $1) }
        let source = try Data(contentsOf: input)
        let versions = try await fixture.project.modelVersions()
        let version = try XCTUnwrap(versions.first)
        XCTAssertEqual(version.status, "completed")
        XCTAssertNotNil(version.artifact)
        XCTAssertEqual(version.checkpointVersions.count, 3)
        let result = try await fixture.project.testAllCheckpoints(datasetID: fixture.datasetID, map: .depth,
            runID: fixture.runID, inputURL: input, progress: { _ in })
        XCTAssertEqual(result.outputs.map { $0.checkpoint.totalEpochsCompleted }, [25, 50, 75])
        XCTAssertEqual(result.results.count, 3)
        XCTAssertFalse(result.cancelled)
        XCTAssertTrue(result.outputs.allSatisfy { $0.issue == nil })
        XCTAssertEqual(result.results.map(\.epochsCompleted), [25, 50, 75])
        XCTAssertEqual(Set(result.results.map(\.inputSHA256)).count, 1)
        XCTAssertEqual(Set(try result.results.map { try Data(contentsOf: $0.outputURL) }).count, 3)
        for output in result.results {
            XCTAssertEqual(output.width, 32); XCTAssertEqual(output.height, 32)
            XCTAssertTrue(FileManager.default.fileExists(atPath: output.previewURL.path))
            XCTAssertTrue(output.imageCaption.contains("Epoch"))
        }
        let manifest = try NativeFactoryTraining.json(result.directory.appendingPathComponent("progression.json"))
        XCTAssertEqual(manifest["status"] as? String, "completed")
        XCTAssertEqual(manifest["completed_count"] as? Int, 3)
        let reopened = try LoomProject(sourceFolder: fixture.root)
        let saved = try await reopened.latestCheckpointProgression(runID: fixture.runID, inputURL: input)
        XCTAssertEqual(saved?.results.map(\.checkpointID), result.results.map(\.checkpointID))
        XCTAssertEqual(saved?.results.map(\.outputURL), result.results.map(\.outputURL))
        XCTAssertEqual(saved?.results.map(\.epochsCompleted), [25, 50, 75])
        XCTAssertEqual(try Data(contentsOf: input), source)
        for (url, data) in originals { XCTAssertEqual(try Data(contentsOf: url), data) }
    }

    func testSweepKeepsSuccessfulOutputsWhenCancelledAndRecordsBadCheckpointWithoutDiscardingOthers() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        for epoch in [25, 50, 75] { try fixture.save(epoch: epoch, value: Float(epoch) / 10000) }
        let input = fixture.root.appendingPathComponent("input.png")
        try ModelTestImage.writePNG([Float](repeating: 0.4, count: 32 * 32 * 3), channels: 3, width: 32, height: 32, to: input)
        let checkpoints = try await fixture.project.checkpointVersions(runID: fixture.runID)
        for stop in [false, true] {
            let cancellation = NativeCancellation()
            let output = fixture.root.appendingPathComponent(stop ? "stopped" : "failed")
            let value = try CheckpointProgressionWorker.run(checkpoints: checkpoints, inputURL: input, output: output,
                cancellation: cancellation, boundaryHandling: .context, progress: { _ in }, load: { checkpoint in
                    if !stop && checkpoint.epochsCompleted == 50 { throw LoomError("Broken checkpoint fixture") }
                    return try ModelTestSelection.load(runID: fixture.runID, source: .checkpoint, datasetID: fixture.datasetID,
                        map: .depth, projectFolder: fixture.project.projectFolder, projectID: fixture.projectID, checkpointID: checkpoint.revisionID)
                }, runner: { model, input, output, cancellation, boundary, progress in
                    let result = try ModelTestWorker.run(model: model, inputURL: input, output: output,
                        cancellation: cancellation, progress: progress, boundaryHandling: boundary,
                        predictor: { _, h, w in [Float](repeating: 0.25, count: h * w) })
                    if stop { cancellation.cancel() }
                    return result
                })
            XCTAssertEqual(value.cancelled, stop)
            XCTAssertEqual(value.results.count, stop ? 1 : 2)
            XCTAssertEqual(value.outputs.compactMap(\.issue).count, stop ? 0 : 1)
            let manifest = try NativeFactoryTraining.json(output.appendingPathComponent("progression.json"))
            XCTAssertEqual(manifest["status"] as? String, stop ? "cancelled" : "completed-with-errors")
            XCTAssertTrue(value.results.allSatisfy { FileManager.default.fileExists(atPath: $0.outputURL.path) })
        }
    }

    func testOptInReopenProductionProgressionWithoutInference() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let source = environment["LOOM_CHECKPOINT_SWEEP_SOURCE"], let run = environment["LOOM_CHECKPOINT_SWEEP_RUN"],
              let input = environment["LOOM_CHECKPOINT_SWEEP_INPUT"] else { throw XCTSkip("Opt-in saved production progression") }
        let project = try LoomProject(sourceFolder: URL(fileURLWithPath: source))
        let value = try await project.latestCheckpointProgression(runID: run, inputURL: URL(fileURLWithPath: input))
        let saved = try XCTUnwrap(value)
        XCTAssertEqual(saved.results.count, saved.checkpointCount)
        XCTAssertTrue(saved.results.allSatisfy { $0.width == 1024 && $0.height == 1024 })
        print("REOPENED_CHECKPOINT_OUTPUTS=\(saved.results.count)")
    }

    func testSweepStopsIfInputBytesChangeBetweenCheckpoints() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        for epoch in [25, 50] { try fixture.save(epoch: epoch, value: 0.001) }
        let input = fixture.root.appendingPathComponent("input.png")
        try ModelTestImage.writePNG([Float](repeating: 0.4, count: 32 * 32 * 3), channels: 3, width: 32, height: 32, to: input)
        let checkpoints = try await fixture.project.checkpointVersions(runID: fixture.runID)
        let value = try CheckpointProgressionWorker.run(checkpoints: checkpoints, inputURL: input,
            output: fixture.root.appendingPathComponent("changed-input"), cancellation: NativeCancellation(),
            boundaryHandling: .context, progress: { _ in }, load: { checkpoint in
                try ModelTestSelection.load(runID: fixture.runID, source: .checkpoint, datasetID: fixture.datasetID,
                    map: .depth, projectFolder: fixture.project.projectFolder, projectID: fixture.projectID, checkpointID: checkpoint.revisionID)
            }, runner: { model, input, output, cancellation, boundary, progress in
                let result = try ModelTestWorker.run(model: model, inputURL: input, output: output,
                    cancellation: cancellation, progress: progress, boundaryHandling: boundary,
                    predictor: { _, h, w in [Float](repeating: 0.25, count: h * w) })
                try ModelTestImage.writePNG([Float](repeating: 0.8, count: 32 * 32 * 3), channels: 3, width: 32, height: 32, to: input)
                return result
            })
        XCTAssertEqual(value.results.count, 1)
        XCTAssertEqual(value.outputs.compactMap(\.issue).count, 1)
        let record = try NativeFactoryTraining.json(value.directory.appendingPathComponent("progression.json"))
        XCTAssertEqual(record["status"] as? String, "input-changed")
        let stale = try await fixture.project.latestCheckpointProgression(runID: fixture.runID, inputURL: input)
        XCTAssertNil(stale)
    }

    func testOptInCompletedProductionRunCheckpointProgression() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let source = environment["LOOM_CHECKPOINT_SWEEP_SOURCE"], let runID = environment["LOOM_CHECKPOINT_SWEEP_RUN"],
              let input = environment["LOOM_CHECKPOINT_SWEEP_INPUT"] else { throw XCTSkip("Opt-in saved production run sweep") }
        let project = try LoomProject(sourceFolder: URL(fileURLWithPath: source))
        let versions = try await project.modelVersions()
        let version = try XCTUnwrap(versions.first { $0.id == runID })
        XCTAssertEqual(version.status, "completed")
        XCTAssertGreaterThan(version.checkpointVersions.count, 1)
        let result = try await project.testAllCheckpoints(datasetID: version.datasetID, map: version.map, runID: runID,
            inputURL: URL(fileURLWithPath: input), progress: { print($0.message) })
        XCTAssertEqual(result.results.count, version.checkpointVersions.count)
        XCTAssertTrue(result.outputs.allSatisfy { $0.issue == nil })
        print("CHECKPOINT_PROGRESSION_OUTPUT=\(result.directory.path)")
    }

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
        func complete() throws {
            var descriptor = try NativeFactoryTraining.json(directory.appendingPathComponent("checkpoint.json"))
            let weights = try XCTUnwrap(descriptor["weights_path"] as? String)
            try FileManager.default.copyItem(at: directory.appendingPathComponent(weights), to: directory.appendingPathComponent("model.safetensors"))
            descriptor["schema"] = "loom-factory-model-v1"; descriptor["status"] = "completed"
            descriptor["weights_path"] = "model.safetensors"; descriptor["image_size"] = 32
            descriptor["training"] = descriptor["settings"]
            try NativeFactoryTraining.writeJSON(descriptor, to: directory.appendingPathComponent("model.json"))
            let artifact = ModelArtifact(id: runID, kind: "map", datasetID: datasetID, datasetName: "Bricks", map: .depth,
                directory: directory, epochs: 75, loss: nil, coreMLPath: nil, weightsPath: "model.safetensors",
                architecture: "loom-factory-material-residual-v1")
            try JSONEncoder().encode(artifact).write(to: directory.appendingPathComponent("artifact.json"))
        }
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
