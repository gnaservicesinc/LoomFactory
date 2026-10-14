import Foundation
import XCTest
@testable import LoomCore

@MainActor
final class RunDeletionTests: XCTestCase {
    func testDeletingSelectedRunPreservesExplicitChoiceOtherRunsOriginalsAndPackages() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let old = try fixture.run("old"), keep = try fixture.run("keep")
        let original = fixture.root.appendingPathComponent("original.png")
        let package = fixture.project.projectFolder.appendingPathComponent("exports/test.textureloom")
        try Data("original image bytes".utf8).write(to: original)
        try FileManager.default.createDirectory(at: package.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("package bytes".utf8).write(to: package)
        try fixture.selection(runIDs: ["old"])
        let trash = fixture.trash
        let result = try await fixture.project.deleteTrainingRuns(runIDs: ["old", "old"], moveToTrash: { folder in
            let destination = trash.appendingPathComponent(folder.lastPathComponent)
            try FileManager.default.moveItem(at: folder, to: destination)
            return destination
        })
        XCTAssertEqual(result.runIDs, ["old"])
        XCTAssertEqual(result.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: old.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: keep.path))
        XCTAssertEqual(try Data(contentsOf: original), Data("original image bytes".utf8))
        XCTAssertEqual(try Data(contentsOf: package), Data("package bytes".utf8))
        let selection = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.selectionFile)) as? [String: Any]
        XCTAssertEqual((selection?["models"] as? [[String: Any]])?.compactMap { $0["runID"] as? String }, ["old"])
        do {
            _ = try await fixture.project.selectedModelArtifact(datasetID: "bricks-0", map: .depth)
            XCTFail("Deleting the selected model must not silently choose another version")
        } catch { XCTAssertTrue(error.localizedDescription.contains("selected depth version is missing")) }
        let versions = try await fixture.project.modelVersions()
        XCTAssertEqual(versions.map(\.id), ["keep"])
    }

    func testBulkInventoryIncludesStoppedAndFailedMapRunsAndPreservesClassifier() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        _ = try fixture.run("stopped", status: "cancelled")
        _ = try fixture.run("failed", status: "failed")
        let classifier = fixture.project.projectFolder.appendingPathComponent("runs/classifier")
        try FileManager.default.createDirectory(at: classifier, withIntermediateDirectories: true)
        try json(["project_id": fixture.projectID, "kind": "classifier"], to: classifier.appendingPathComponent("request.json"))
        let ids = try await fixture.project.modelVersions().map(\.id)
        XCTAssertEqual(Set(ids), ["stopped", "failed"])
        let trash = fixture.trash
        let result = try await fixture.project.deleteTrainingRuns(runIDs: ids, moveToTrash: { folder in
            let destination = trash.appendingPathComponent(folder.lastPathComponent)
            try FileManager.default.moveItem(at: folder, to: destination)
            return destination
        })
        XCTAssertEqual(result.count, 2)
        XCTAssertTrue(FileManager.default.fileExists(atPath: classifier.path))
        let versions = try await fixture.project.modelVersions()
        XCTAssertTrue(versions.isEmpty)
    }

    func testForeignRunTraversalAndDirectoryLinkRefuseEntireRequestBeforeMovingAnything() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let good = try fixture.run("a-good")
        _ = try fixture.run("foreign", projectID: "other-project")
        let linked = fixture.project.projectFolder.appendingPathComponent("runs/linked")
        try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: good)
        for invalid in ["../a-good", "foreign", "linked"] {
            do {
                _ = try await fixture.project.deleteTrainingRuns(runIDs: ["a-good", invalid], moveToTrash: { _ in
                    XCTFail("The whole request must be checked before any move")
                    return nil
                })
                XCTFail("An unowned or substituted run must be refused")
            } catch {}
            XCTAssertTrue(FileManager.default.fileExists(atPath: good.path))
        }
    }

    func testLinksInsideRunAndReplacedRunsRootNeverTouchOriginalTarget() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let run = try fixture.run("linked-image")
        let target = fixture.root.appendingPathComponent("source.dat")
        let bytes = Data("unchanged original".utf8)
        try bytes.write(to: target)
        try FileManager.default.createSymbolicLink(at: run.appendingPathComponent("image.png"), withDestinationURL: target)
        do {
            _ = try await fixture.project.deleteTrainingRuns(runIDs: ["linked-image"], moveToTrash: { _ in
                XCTFail("A linked run must not be moved"); return nil
            })
            XCTFail("A linked source reference must be refused")
        } catch { XCTAssertTrue(error.localizedDescription.contains("contains a link")) }
        XCTAssertEqual(try Data(contentsOf: target), bytes)
        let runs = fixture.project.projectFolder.appendingPathComponent("runs")
        let moved = fixture.root.appendingPathComponent("external-runs")
        try FileManager.default.moveItem(at: runs, to: moved)
        try FileManager.default.createSymbolicLink(at: runs, withDestinationURL: moved)
        do {
            _ = try await fixture.project.deleteTrainingRuns(runIDs: ["linked-image"], moveToTrash: { _ in
                XCTFail("A substituted root must not be moved"); return nil
            })
            XCTFail("The runs root cannot be a link")
        } catch { XCTAssertTrue(error.localizedDescription.contains("replaced by a link")) }
        XCTAssertEqual(try Data(contentsOf: target), bytes)
    }

    func testActiveOperationAndForeignSelectionRefuseDeletion() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let run = try fixture.run("protected")
        await fixture.project.setRunDeletionTestOperationActive(true)
        do {
            _ = try await fixture.project.deleteTrainingRuns(runIDs: ["protected"], moveToTrash: { _ in
                XCTFail("Active operations must block deletion"); return nil
            })
            XCTFail("A training run cannot be deleted during an active operation")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Finish or stop")) }
        await fixture.project.setRunDeletionTestOperationActive(false)
        try fixture.selection(runIDs: ["protected"], projectID: "foreign")
        do {
            _ = try await fixture.project.deleteTrainingRuns(runIDs: ["protected"], moveToTrash: { _ in
                XCTFail("A foreign selection must block deletion"); return nil
            })
            XCTFail("A foreign selection must not be rewritten")
        } catch { XCTAssertTrue(error.localizedDescription.contains("does not belong")) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: run.path))
    }

    func testPartialTrashFailurePreservesEveryExplicitSelection() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        _ = try fixture.run("a-delete"); let retained = try fixture.run("b-fail")
        try fixture.selection(runIDs: ["a-delete", "b-fail"])
        let trash = fixture.trash
        do {
            _ = try await fixture.project.deleteTrainingRuns(runIDs: ["a-delete", "b-fail"], moveToTrash: { folder in
                if folder.lastPathComponent == "b-fail" { throw LoomError("Trash is unavailable") }
                let destination = trash.appendingPathComponent(folder.lastPathComponent)
                try FileManager.default.moveItem(at: folder, to: destination)
                return destination
            })
            XCTFail("The failed move must be reported")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Trash is unavailable")) }
        let selection = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.selectionFile)) as? [String: Any]
        XCTAssertEqual((selection?["models"] as? [[String: Any]])?.compactMap { $0["runID"] as? String }, ["a-delete", "b-fail"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: retained.path))
    }

    func testInvalidSelectionEntriesAreRejectedBeforeAnyRunMoves() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let run = try fixture.run("keep")
        let invalidEntries = [
            [["datasetID": "bricks", "map": "diffuse", "runID": "keep"]],
            [["datasetID": "bricks", "map": "depth", "runID": "../keep"]],
            [["datasetID": "bricks", "map": "depth", "runID": "keep"],
             ["datasetID": "bricks", "map": "depth", "runID": "other"]]
        ]
        for models in invalidEntries {
            try json(["schema": "loom-factory-model-selection-v1", "projectID": fixture.projectID,
                "models": models], to: fixture.selectionFile)
            do {
                _ = try await fixture.project.deleteTrainingRuns(runIDs: ["keep"], moveToTrash: { _ in
                    XCTFail("An invalid saved selection must block deletion"); return nil
                })
                XCTFail("The existing selection contract must be checked before deletion")
            } catch { XCTAssertTrue(error.localizedDescription.contains("saved model selection")) }
            XCTAssertTrue(FileManager.default.fileExists(atPath: run.path))
        }
    }

    private struct Fixture {
        let root: URL
        let project: LoomProject
        let projectID: String
        let trash: URL
        var selectionFile: URL { project.projectFolder.appendingPathComponent("model-selection.json") }
        init() async throws {
            let folder = FileManager.default.temporaryDirectory.appendingPathComponent("loom-run-deletion-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            root = folder.resolvingSymlinksInPath().standardizedFileURL
            project = try LoomProject(sourceFolder: root)
            projectID = try await project.snapshot().projectID
            trash = root.appendingPathComponent("isolated-trash")
            try FileManager.default.createDirectory(at: trash, withIntermediateDirectories: true)
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
        func run(_ id: String, status: String = "completed", projectID requestedProjectID: String? = nil) throws -> URL {
            let directory = project.projectFolder.appendingPathComponent("runs/" + id)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try TrainingRunFiles.writeMetadata(TrainingRunMetadata(projectID: requestedProjectID ?? projectID,
                runID: id, datasetID: "bricks", map: .depth, label: "", createdAt: .now,
                settings: TrainingSettings(imageSize: 32, epochs: 1, modelWidth: 8, modelBlocks: 1), lineage: nil,
                status: status), at: directory)
            try Data("checkpoint fixture".utf8).write(to: directory.appendingPathComponent("checkpoint.safetensors"))
            return directory
        }
        func selection(runIDs: [String], projectID requestedProjectID: String? = nil) throws {
            try json(["schema": "loom-factory-model-selection-v1", "projectID": requestedProjectID ?? projectID,
                "models": runIDs.enumerated().map { index, id in
                    ["datasetID": "bricks-\(index)", "map": "depth", "runID": id]
                }], to: selectionFile)
        }
    }
}

private func json(_ value: [String: Any], to file: URL) throws {
    try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: file)
}

private extension LoomProject {
    func setRunDeletionTestOperationActive(_ active: Bool) { operationActive = active }
}
