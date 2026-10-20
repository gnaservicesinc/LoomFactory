import Foundation
import LoomNative
import XCTest
@testable import LoomCore

@MainActor
final class CheckpointCleanupTests: XCTestCase {
    func testApprovalKeepsExactLatestResumeStateExportHistoryAndSavedTests() async throws {
        let f = try await Fixture(); defer { f.remove() }
        let artifact = try f.run("approved")
        let untouched = try f.run("unapproved")
        let latest = try TrainingRunFiles.descriptor(at: artifact.directory, source: .checkpoint)
        let retained = try f.retainedFiles(artifact, latest: latest)
        let unrelated = try f.bytes(in: untouched.directory)
        let summary = try await f.project.setModelApproval(runID: artifact.id, approved: true)
        XCTAssertEqual(summary.runCount, 1)
        XCTAssertEqual(summary.checkpointCount, 2)
        XCTAssertGreaterThan(summary.byteCount, 0)
        let checkpoints = try await f.project.checkpointVersions(runID: artifact.id)
        XCTAssertEqual(checkpoints.count, 1)
        XCTAssertEqual(checkpoints.first?.revisionID, latest["checkpoint_id"] as? String)
        XCTAssertTrue(checkpoints.first?.restoresOptimizer == true)
        for (url, bytes) in retained { XCTAssertEqual(try Data(contentsOf: url), bytes) }
        XCTAssertEqual(try f.bytes(in: untouched.directory), unrelated)
        XCTAssertEqual(try Data(contentsOf: f.original), Data("original image bytes".utf8))
        XCTAssertEqual(try Data(contentsOf: f.savedTest), Data("saved A/B output".utf8))
        let management = try await f.project.loadModelManagement()
        let approval = await f.project.modelApprovalState(for: artifact, management: management)
        XCTAssertTrue(approval.approved)
        let settings = try XCTUnwrap(TrainingRunFiles.settings(from: latest["settings"] as? [String: Any] ?? [:], descriptor: latest))
        let resume = try TrainingRunFiles.resume(weights: artifact.directory.appendingPathComponent(latest["weights_path"] as! String),
            projectFolder: f.project.projectFolder, projectID: f.projectID, datasetID: f.datasetID, map: .depth, settings: settings)
        XCTAssertEqual(resume.source, .checkpoint)
        let optimizer = try NativeWeights.read(from: artifact.directory.appendingPathComponent(latest["optimizer_path"] as! String))
        XCTAssertFalse(optimizer.isEmpty)
        let repeated = try await f.project.cleanupApprovedCheckpoints()
        XCTAssertEqual(repeated.fileCount, 0)
    }

    func testExistingApprovalPreviewIsReadOnlyAndStaleApprovalIsSkipped() async throws {
        let f = try await Fixture(); defer { f.remove() }
        let approved = try f.run("approved"), stale = try f.run("stale"), unapproved = try f.run("unapproved")
        try await f.approveWithoutCleanup([approved, stale])
        try Data("changed export".utf8).write(to: stale.directory.appendingPathComponent("model.mlmodel"))
        let before = try f.bytes(in: approved.directory), staleBefore = try f.bytes(in: stale.directory)
        let otherBefore = try f.bytes(in: unapproved.directory)
        let preview = try await f.project.approvedCheckpointCleanupPreview()
        XCTAssertEqual(preview.runCount, 1); XCTAssertEqual(preview.checkpointCount, 2)
        XCTAssertEqual(try f.bytes(in: approved.directory), before)
        let result = try await f.project.cleanupApprovedCheckpoints()
        XCTAssertEqual(result, preview)
        XCTAssertEqual(try f.bytes(in: stale.directory), staleBefore)
        XCTAssertEqual(try f.bytes(in: unapproved.directory), otherBefore)
    }

    func testBadLatestOptimizerPreventsApprovalAndAllDeletion() async throws {
        let f = try await Fixture(); defer { f.remove() }
        let artifact = try f.run("bad-optimizer")
        let latest = try TrainingRunFiles.descriptor(at: artifact.directory, source: .checkpoint)
        try Data("corrupt optimizer".utf8).write(to: artifact.directory.appendingPathComponent(latest["optimizer_path"] as! String))
        let before = try f.bytes(in: artifact.directory)
        do { try await f.project.setModelApproval(runID: artifact.id, approved: true); XCTFail("Corrupt recovery state must block deletion") }
        catch { XCTAssertTrue(error.localizedDescription.contains("checksum")) }
        XCTAssertEqual(try f.bytes(in: artifact.directory), before)
        let management = try await f.project.loadModelManagement()
        XCTAssertTrue(management.approvals.isEmpty)
    }

    func testEntireBatchIsValidatedBeforeDeletingAnyCheckpoint() async throws {
        let f = try await Fixture(); defer { f.remove() }
        let good = try f.run("good"), bad = try f.run("bad")
        try await f.approveWithoutCleanup([good, bad])
        let archive = try CheckpointArchive.descriptors(in: bad.directory).first { $0.lastPathComponent != "checkpoint.json" }!
        var descriptor = try NativeFactoryTraining.json(archive)
        descriptor["project_id"] = "foreign-project"
        try NativeFactoryTraining.writeJSON(descriptor, to: archive)
        let before = try f.bytes(in: good.directory)
        do { _ = try await f.project.cleanupApprovedCheckpoints(); XCTFail("Foreign checkpoint must block entire batch") }
        catch {}
        XCTAssertEqual(try f.bytes(in: good.directory), before)
    }

    func testSavedProgressionReopensEveryOutputAfterCheckpointCleanup() async throws {
        let f = try await Fixture(); defer { f.remove() }
        let artifact = try f.run("with-progression")
        let checkpoints = try await f.project.checkpointVersions(runID: artifact.id).reversed()
        let folder = f.project.projectFolder.appendingPathComponent("model-tests/progression-fixture")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let hash = try TrainingRunFiles.sha256(f.original)
        var rows: [[String: Any]] = []
        for checkpoint in checkpoints {
            let id = try XCTUnwrap(checkpoint.revisionID)
            let output = folder.appendingPathComponent(id)
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            for name in ["output.exr", "preview.png"] { try Data(id.utf8).write(to: output.appendingPathComponent(name)) }
            try NativeFactoryTraining.writeJSON(["schema": "loom-factory-model-test-v1", "run_id": artifact.id,
                "input_sha256": hash, "checkpoint_id": id, "weight_source": "checkpoint", "map": "depth",
                "width": 32, "height": 32, "tile_size": 32, "tile_count": 1], to: output.appendingPathComponent("test.json"))
            rows.append(["checkpoint_id": id, "epochs_completed": checkpoint.totalEpochsCompleted,
                "updates_completed": checkpoint.totalUpdatesCompleted, "output_path": id + "/output.exr",
                "preview_path": id + "/preview.png", "test_record_path": id + "/test.json"])
        }
        try NativeFactoryTraining.writeJSON(["schema": "loom-factory-checkpoint-progression-v1", "run_id": artifact.id,
            "input_sha256": hash, "input_path": f.original.path, "checkpoint_count": 3, "outputs": rows,
            "status": "completed"], to: folder.appendingPathComponent("progression.json"))
        let before = try await f.project.latestCheckpointProgression(runID: artifact.id, inputURL: f.original)
        XCTAssertEqual(before?.results.count, 3)
        try await f.project.setModelApproval(runID: artifact.id, approved: true)
        let reopened = try LoomProject(sourceFolder: f.root)
        let after = try await reopened.latestCheckpointProgression(runID: artifact.id, inputURL: f.original)
        XCTAssertEqual(after?.results.map(\.checkpointID), before?.results.map(\.checkpointID))
        XCTAssertEqual(after?.outputs.map { $0.checkpoint.totalUpdatesCompleted }, [10, 20, 30])
        let remaining = try await reopened.checkpointVersions(runID: artifact.id)
        XCTAssertEqual(remaining.count, 1, "Display metadata must never appear as usable checkpoints")
    }

    func testCleanupRetriesAfterMissingOldTensorAndPreservesSharedBestWeights() async throws {
        let f = try await Fixture(); defer { f.remove() }
        let artifact = try f.run("retry")
        try await f.approveWithoutCleanup([artifact])
        let descriptors = try CheckpointArchive.descriptors(in: artifact.directory)
        let latestURL = artifact.directory.appendingPathComponent("checkpoint.json")
        var latest = try NativeFactoryTraining.json(latestURL)
        let oldURL = try XCTUnwrap(descriptors.first {
            $0.lastPathComponent != "checkpoint.json" && (try? NativeFactoryTraining.json($0)["updates"] as? Int) == 10
        })
        let old = try NativeFactoryTraining.json(oldURL)
        let sharedBest = artifact.directory.appendingPathComponent(old["best_weights_path"] as! String)
        let bestBytes = try Data(contentsOf: sharedBest)
        latest["best_weights_path"] = old["best_weights_path"]
        latest["best_weights_sha256"] = old["best_weights_sha256"]
        try NativeFactoryTraining.writeJSON(latest, to: latestURL)
        try NativeFactoryTraining.writeJSON(latest, to: artifact.directory.appendingPathComponent(try CheckpointArchive.filename(latest["checkpoint_id"] as! String)))
        try FileManager.default.removeItem(at: artifact.directory.appendingPathComponent(old["weights_path"] as! String))
        let summary = try await f.project.cleanupApprovedCheckpoints()
        XCTAssertEqual(summary.checkpointCount, 2)
        XCTAssertEqual(try Data(contentsOf: sharedBest), bestBytes)
        let repeated = try await f.project.cleanupApprovedCheckpoints()
        XCTAssertEqual(repeated.fileCount, 0)
    }

    func testActiveLeaseRefusesApprovalAndCleanupEvenInSameProcess() async throws {
        let f = try await Fixture(); defer { f.remove() }
        let artifact = try f.run("approved")
        try await f.approveWithoutCleanup([artifact])
        let before = try f.bytes(in: artifact.directory)
        let lease = try TrainingQueueLease(projectFolder: f.project.projectFolder, controlFolder: f.root)
        defer { withExtendedLifetime(lease) {} }
        let preview = try await f.project.approvedCheckpointCleanupPreview()
        XCTAssertEqual(preview.checkpointCount, 2)
        for action in [false, true] {
            do {
                if action { try await f.project.setModelApproval(runID: artifact.id, approved: true) }
                else { _ = try await f.project.cleanupApprovedCheckpoints() }
                XCTFail("An active training/testing lease must block cleanup")
            } catch { XCTAssertTrue(error.localizedDescription.contains("Another model operation")) }
        }
        XCTAssertEqual(try f.bytes(in: artifact.directory), before)
    }

    func testLinkedTensorAndEscapedPathNeverTouchOriginal() async throws {
        for linked in [true, false] {
            let f = try await Fixture(); defer { f.remove() }
            let artifact = try f.run("linked")
            try await f.approveWithoutCleanup([artifact])
            let archives = try CheckpointArchive.descriptors(in: artifact.directory)
            let latestID = try TrainingRunFiles.descriptor(at: artifact.directory, source: .checkpoint)["checkpoint_id"] as? String
            let old = try XCTUnwrap(archives.first { url in
                url.lastPathComponent != "checkpoint.json" && (try? NativeFactoryTraining.json(url)["checkpoint_id"] as? String) != latestID
            })
            var descriptor = try NativeFactoryTraining.json(old)
            if linked {
                let tensor = artifact.directory.appendingPathComponent(descriptor["weights_path"] as! String)
                try FileManager.default.removeItem(at: tensor)
                try FileManager.default.createSymbolicLink(at: tensor, withDestinationURL: f.original)
            } else {
                descriptor["weights_path"] = "../../source.png"
                try NativeFactoryTraining.writeJSON(descriptor, to: old)
            }
            do { _ = try await f.project.cleanupApprovedCheckpoints(); XCTFail("Linked or escaped files must block cleanup") }
            catch {}
            XCTAssertEqual(try Data(contentsOf: f.original), Data("original image bytes".utf8))
            XCTAssertTrue(FileManager.default.fileExists(atPath: old.path))
        }
    }

    private struct Fixture {
        let root: URL, project: LoomProject, projectID: String
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        var original: URL { root.appendingPathComponent("source.png") }
        var savedTest: URL { project.projectFolder.appendingPathComponent("model-tests/saved-output.dat") }
        init() async throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-cleanup-" + UUID().uuidString)
                .resolvingSymlinksInPath().standardizedFileURL
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            project = try LoomProject(sourceFolder: root)
            projectID = try await project.snapshot().projectID
            try Data("original image bytes".utf8).write(to: original)
            try FileManager.default.createDirectory(at: savedTest.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("saved A/B output".utf8).write(to: savedTest)
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
        func run(_ id: String) throws -> ModelArtifact {
            let directory = project.projectFolder.appendingPathComponent("runs/" + id, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 1)
            for epoch in 1...3 {
                let tensor = NativeTensor(shape: [1], values: [Float(epoch)])
                let descriptor: [String: Any] = ["schema": "loom-factory-checkpoint-v1", "project_id": projectID,
                    "kind": "map", "run_id": id, "dataset_id": datasetID, "map": "depth",
                    "architecture": architecture.identifier, "architecture_config": architecture.descriptor,
                    "settings": NativeFactoryTraining.settingsJSON(TrainingSettings(imageSize: 32, modelWidth: 8, modelBlocks: 1)),
                    "epochs_completed": epoch, "updates": epoch * 10, "history": [["epoch": epoch, "loss": 0.1]]]
                try TrainingCheckpointFiles.write(weights: ["weight": tensor], optimizer: ["moment": tensor],
                    bestWeights: ["weight": tensor], descriptor: descriptor, directory: directory)
            }
            try Data("best export weights".utf8).write(to: directory.appendingPathComponent("model.safetensors"))
            try Data("native model export".utf8).write(to: directory.appendingPathComponent("model.mlmodel"))
            let descriptor: [String: Any] = ["schema": "loom-factory-model-v1", "status": "completed", "project_id": projectID,
                "kind": "map", "dataset_id": datasetID, "map": "depth", "architecture": architecture.identifier,
                "architecture_config": architecture.descriptor, "image_size": 32,
                "weights_path": "model.safetensors", "coreml_path": "model.mlmodel", "input_name": "rgb", "output_name": "material_map"]
            try NativeFactoryTraining.writeJSON(descriptor, to: directory.appendingPathComponent("model.json"))
            let artifact = ModelArtifact(id: id, kind: "map", datasetID: datasetID, datasetName: "Bricks", map: .depth,
                directory: directory, epochs: 3, loss: 0.1, coreMLPath: "model.mlmodel", weightsPath: "model.safetensors", architecture: architecture.identifier)
            try JSONEncoder().encode(artifact).write(to: directory.appendingPathComponent("artifact.json"))
            return artifact
        }
        func approveWithoutCleanup(_ artifacts: [ModelArtifact]) async throws {
            for artifact in artifacts {
                let originals = try bytes(in: artifact.directory)
                _ = try await project.setModelApproval(runID: artifact.id, approved: true)
                // Restore old revisions to emulate an approval saved by an older app.
                for (name, data) in originals {
                    let file = artifact.directory.appendingPathComponent(name)
                    if !FileManager.default.fileExists(atPath: file.path) { try data.write(to: file) }
                }
            }
        }
        func bytes(in directory: URL) throws -> [String: Data] {
            try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .reduce(into: [:]) { $0[$1.lastPathComponent] = try Data(contentsOf: $1) }
        }
        func retainedFiles(_ artifact: ModelArtifact, latest: [String: Any]) throws -> [URL: Data] {
            let names = ["checkpoint.json", "model.json", "model.safetensors", "model.mlmodel", "artifact.json",
                try CheckpointArchive.filename(latest["checkpoint_id"] as! String)]
                + ["weights_path", "optimizer_path", "best_weights_path"].compactMap { latest[$0] as? String }
            return try names.reduce(into: [:]) { result, name in
                let url = artifact.directory.appendingPathComponent(name); result[url] = try Data(contentsOf: url)
            }
        }
    }
}
