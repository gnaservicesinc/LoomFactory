import Foundation
import LoomNative
import XCTest
@testable import LoomCore

@MainActor
final class ModelCoverageTests: XCTestCase {
    func testCoverageShowsEveryCurrentGroupAndDistinctDefaultSelection() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        var entries = try await fixture.coverage()
        XCTAssertEqual(entries.count, 4)
        XCTAssertTrue(entries.allSatisfy(\.isRequired))
        XCTAssertTrue(entries.allSatisfy { $0.status == .missing })
        XCTAssertEqual(entries.first { $0.slot.product == .depth }?.pairCount, 2)
        _ = try fixture.artifact("older", date: Date(timeIntervalSince1970: 10))
        _ = try fixture.artifact("newer", date: Date(timeIntervalSince1970: 20))
        entries = try await fixture.coverage()
        let depth = try XCTUnwrap(entries.first { $0.slot.product == .depth })
        XCTAssertEqual(depth.status, .available)
        XCTAssertNil(depth.selectedRunID)
        XCTAssertEqual(depth.effectiveRunID, "newer")
        XCTAssertTrue(depth.usesLatestByDefault)
        XCTAssertEqual(depth.availableCount, 2)
        XCTAssertEqual(depth.approvedCount, 0)
        XCTAssertTrue(depth.candidates.allSatisfy { !$0.isSelected })
    }

    func testCoverageCountsUnfinishedCheckpointsWithoutDoubleCountingCompletedVersions() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        func checkpoint(_ runID: String, map: MapRole = .depth) -> TrainingCheckpoint {
            TrainingCheckpoint(directory: fixture.project.projectFolder.appendingPathComponent("runs/" + runID),
                datasetID: fixture.dataset.id, map: map, modelWidth: 8, modelBlocks: 1,
                imageSize: 32, exaggerateFeatures: false, epochsCompleted: 19, status: "cancelled", modifiedAt: .now)
        }
        let unfinished = [checkpoint("cancelled-depth"), checkpoint("interrupted-depth")]
        let entries = try await fixture.project.loadModelCoverage(datasets: [fixture.dataset], artifacts: [],
            checkpoints: unfinished + [checkpoint("normal-checkpoint", map: .normal)])
        let depth = try XCTUnwrap(entries.first { $0.slot.product == .depth })
        XCTAssertEqual(depth.availableCount, 0, "Checkpoints cannot satisfy deployment model coverage")
        XCTAssertEqual(depth.checkpointCount, 2)
        XCTAssertNil(depth.effectiveRunID)
        XCTAssertEqual(entries.first { $0.slot.product == .normal }?.checkpointCount, 1)

        let artifact = try fixture.artifact("completed-depth", date: .now)
        let continuation = checkpoint("completed-depth")
        XCTAssertNotEqual(artifact.directory.absoluteString, continuation.directory.absoluteString,
            "This fixture covers equivalent directory URLs with different trailing-slash representations")
        let completed = try await fixture.project.loadModelCoverage(datasets: [fixture.dataset], artifacts: [artifact],
            checkpoints: unfinished + [continuation])
        let completedDepth = try XCTUnwrap(completed.first { $0.slot.product == .depth })
        XCTAssertEqual(completedDepth.availableCount, 1)
        XCTAssertEqual(completedDepth.checkpointCount, 2, "A completed model's continuation checkpoint is the same saved version")
        XCTAssertEqual(completedDepth.candidates.map(\.id), ["completed-depth"])
    }

    func testLegacyMapSelectionAndIndependentApprovalsRestoreWithoutChangingRuns() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let old = try fixture.artifact("old", date: Date(timeIntervalSince1970: 10))
        _ = try fixture.artifact("new", date: Date(timeIntervalSince1970: 20))
        let selectionFile = fixture.project.projectFolder.appendingPathComponent("model-selection.json")
        try writeJSON(["schema": "loom-factory-model-selection-v1", "projectID": fixture.projectID,
            "models": [["datasetID": fixture.dataset.id, "map": "depth", "runID": "old"]]], to: selectionFile)
        let selectionBytes = try Data(contentsOf: selectionFile)
        let oldDescriptor = try Data(contentsOf: old.directory.appendingPathComponent("model.json"))
        try await fixture.project.setModelApproval(runID: "new", approved: true)
        var depth = try await fixture.depth()
        XCTAssertEqual(depth.status, .selected)
        XCTAssertEqual(depth.selectedRunID, "old")
        XCTAssertEqual(depth.approvedCount, 1)
        XCTAssertFalse(try XCTUnwrap(depth.candidates.first { $0.id == "old" }).isApproved)
        XCTAssertFalse(depth.usesLatestByDefault)
        XCTAssertEqual(try Data(contentsOf: selectionFile), selectionBytes)
        XCTAssertEqual(try Data(contentsOf: old.directory.appendingPathComponent("model.json")), oldDescriptor)
        try await fixture.project.setModelApproval(runID: "old", approved: true)
        let reopened = try LoomProject(sourceFolder: fixture.root)
        depth = try await fixture.depth(project: reopened)
        XCTAssertEqual(depth.status, .approved)
        XCTAssertEqual(depth.selectedRunID, "old")
        XCTAssertEqual(depth.approvedCount, 2)
        let reopenedVersions = try await reopened.modelVersions()
        XCTAssertTrue(try XCTUnwrap(reopenedVersions.first { $0.id == "old" }).isApproved)
        try await reopened.setModelApproval(runID: "old", approved: false)
        depth = try await fixture.depth(project: reopened)
        XCTAssertEqual(depth.status, .selected)
        XCTAssertEqual(depth.selectedRunID, "old")
        XCTAssertEqual(depth.approvedCount, 1)
    }

    func testApprovalDetectsSameSizeNativeMutationWithCachedFingerprintAndRequiresFreshDecision() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let artifact = try fixture.artifact("chosen", date: .now)
        try await fixture.project.selectModelVersion(runID: artifact.id, datasetID: fixture.dataset.id, map: .depth)
        try await fixture.project.setModelApproval(runID: artifact.id, approved: true)
        var depth = try await fixture.depth()
        XCTAssertEqual(depth.status, .approved)
        let native = artifact.directory.appendingPathComponent("model.mlmodel")
        let original = try Data(contentsOf: native)
        let savedDate = try native.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate!
        // A legacy descriptor without a native checksum is still a valid saved
        // version, but an approval of its previous bytes must become stale.
        try Data(repeating: 88, count: original.count).write(to: native)
        try FileManager.default.setAttributes([.modificationDate: savedDate], ofItemAtPath: native.path)
        depth = try await fixture.depth()
        XCTAssertEqual(depth.status, .stale)
        XCTAssertEqual(depth.availableCount, 1)
        XCTAssertEqual(depth.approvedCount, 0)
        XCTAssertTrue(try XCTUnwrap(depth.candidates.first).approvalIsStale)
        XCTAssertEqual(depth.selectedRunID, "chosen")
        try await fixture.project.setModelApproval(runID: artifact.id, approved: true)
        depth = try await fixture.depth()
        XCTAssertEqual(depth.status, .approved)
    }

    func testInvalidAndRemovedChosenArtifactNeverPromotesAnotherModel() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let old = try fixture.artifact("old", date: Date(timeIntervalSince1970: 10))
        let newer = try fixture.artifact("newer", date: Date(timeIntervalSince1970: 20))
        try await fixture.project.selectModelVersion(runID: old.id, datasetID: fixture.dataset.id, map: .depth)
        try Data("changed weights".utf8).write(to: newer.directory.appendingPathComponent("model.safetensors"))
        do {
            try await fixture.project.setModelApproval(runID: newer.id, approved: true)
            XCTFail("Changed completed weights cannot be approved")
        } catch { XCTAssertTrue(error.localizedDescription.contains("checksum")) }
        do {
            try await fixture.project.selectModelVersion(runID: newer.id, datasetID: fixture.dataset.id, map: .depth)
            XCTFail("Changed completed weights cannot replace a chosen version")
        } catch { XCTAssertTrue(error.localizedDescription.contains("checksum")) }
        try FileManager.default.removeItem(at: old.directory.appendingPathComponent("artifact.json"))
        let depth = try await fixture.depth()
        XCTAssertEqual(depth.status, .unavailable)
        XCTAssertEqual(depth.selectedRunID, "old")
        XCTAssertNil(depth.selectedArtifact)
        do {
            _ = try await fixture.project.selectedModelArtifact(datasetID: fixture.dataset.id, map: .depth)
            XCTFail("A removed choice must not silently use another candidate")
        } catch { XCTAssertTrue(error.localizedDescription.contains("selected depth version is missing")) }
    }

    func testMalformedArtifactRecordDoesNotHideAnotherCandidateOrReplaceTheChoice() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let old = try fixture.artifact("old", date: Date(timeIntervalSince1970: 10))
        _ = try fixture.artifact("newer", date: Date(timeIntervalSince1970: 20))
        try await fixture.project.selectModelVersion(runID: old.id, datasetID: fixture.dataset.id, map: .depth)
        try Data("damaged artifact record".utf8).write(to: old.directory.appendingPathComponent("artifact.json"))
        let reopened = try LoomProject(sourceFolder: fixture.root)
        let current = try await reopened.snapshot()
        XCTAssertEqual(current.artifacts.map(\.id), ["newer"])
        let depth = try await fixture.depth(project: reopened)
        XCTAssertEqual(depth.status, .unavailable)
        XCTAssertEqual(depth.selectedRunID, "old")
        XCTAssertEqual(depth.availableCount, 1)
        XCTAssertEqual(depth.candidates.map(\.id), ["newer"])
        do {
            _ = try await reopened.selectedModelArtifact(datasetID: fixture.dataset.id, map: .depth)
            XCTFail("A damaged chosen artifact record must not promote the remaining candidate")
        } catch { XCTAssertTrue(error.localizedDescription.contains("selected depth version is missing")) }
    }

    func testOptionalRequirementPersistsAndClassifierCannotBeOptional() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let normal = ModelSlot(group: .dataset(fixture.dataset.id), product: .normal)
        try await fixture.project.setModelRequirement(slot: normal, required: false)
        let reopened = try LoomProject(sourceFolder: fixture.root)
        var entries = try await fixture.coverage(project: reopened)
        XCTAssertEqual(entries.first { $0.slot == normal }?.isRequired, false)
        XCTAssertTrue(entries.filter { $0.slot != normal }.allSatisfy(\.isRequired))
        do {
            try await reopened.setModelRequirement(slot: .init(group: .project, product: .classifier), required: false)
            XCTFail("A Texture Loom deployment must retain its classifier requirement")
        } catch { XCTAssertTrue(error.localizedDescription.contains("always requires")) }
        try await reopened.setModelRequirement(slot: normal, required: true)
        entries = try await fixture.coverage(project: reopened)
        XCTAssertTrue(entries.allSatisfy(\.isRequired))
    }

    func testClassifierCoverageIncludesRouteDriftAndHistoricalChoice() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        // Classifier assembly fixtures verify saved catalog/file management,
        // never execution or model quality.
        _ = try fixture.classifier("old-classifier", date: Date(timeIntervalSince1970: 10), datasets: [])
        _ = try fixture.classifier("new-classifier", date: Date(timeIntervalSince1970: 20), datasets: [fixture.dataset])
        let management = ModelManagement(projectID: fixture.projectID, classifierRunID: "old-classifier")
        try JSONEncoder().encode(management).write(to: fixture.project.projectFolder.appendingPathComponent("model-management.json"))
        var entries = try await fixture.coverage()
        var entry = try XCTUnwrap(entries.first { $0.slot.product == .classifier })
        XCTAssertEqual(entry.status, .stale)
        XCTAssertEqual(entry.selectedRunID, "old-classifier")
        XCTAssertEqual(entry.availableCount, 1)
        XCTAssertTrue(entry.detail?.contains("dataset") == true)
        let chosen = try await fixture.project.selectedClassifierArtifact()
        XCTAssertEqual(chosen?.id, "old-classifier", "Route drift must be fixed explicitly, never by silently switching classifiers")
        try FileManager.default.removeItem(at: chosen!.directory.appendingPathComponent("artifact.json"))
        entries = try await fixture.coverage()
        entry = try XCTUnwrap(entries.first { $0.slot.product == .classifier })
        XCTAssertEqual(entry.status, .unavailable)
        do {
            _ = try await fixture.project.selectedClassifierArtifact()
            XCTFail("The missing classifier must not promote the latest import")
        } catch { XCTAssertTrue(error.localizedDescription.contains("selected classifier version is missing")) }
    }

    func testMalformedClassifierCatalogAndInterfaceCannotAppearAvailableOrBeApproved() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let artifact = try fixture.classifier("malformed", date: .now, datasets: [fixture.dataset])
        let descriptorFile = artifact.directory.appendingPathComponent("model.json")
        let original = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: descriptorFile)) as? [String: Any])
        for corruption in ["sourceLabel", "input_name"] {
            var descriptor = original
            if corruption == "sourceLabel" {
                descriptor["labels"] = [["id": fixture.dataset.id, "name": fixture.dataset.name, "sourceLabel": " "]]
            } else { descriptor["input_name"] = "" }
            try writeJSON(descriptor, to: descriptorFile)
            let entries = try await fixture.coverage()
            let classifier = try XCTUnwrap(entries.first { $0.slot.product == .classifier })
            XCTAssertEqual(classifier.availableCount, 0)
            XCTAssertEqual(classifier.status, .stale)
            XCTAssertFalse(try XCTUnwrap(classifier.candidates.first).canChoose)
            do {
                try await fixture.project.setModelApproval(runID: artifact.id, approved: true)
                XCTFail("An export-incompatible descriptor cannot be approved")
            } catch {
                let message = error.localizedDescription
                XCTAssertTrue(message.contains(corruption == "sourceLabel" ? "catalog" : "native input or output"))
            }
        }
    }

    func testBroadClassifierCanBeChosenForCategorizationBeforeDeploymentRoutesAreComplete() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let wood = DatasetCategory(id: ImageMetadataStore.categoryID("Wood"), name: "Wood", imageCount: 0,
            verifiedCount: 0, pairCounts: [:])
        let classifier = try fixture.classifier("broad-classifier", date: .now, datasets: [fixture.dataset, wood])
        try await fixture.project.selectClassifierVersion(runID: classifier.id)
        let selected = try await fixture.project.selectedClassifierArtifact()
        XCTAssertEqual(selected?.id, classifier.id)
        let entries = try await fixture.coverage()
        let coverage = try XCTUnwrap(entries.first { $0.slot.product == .classifier })
        let candidate = try XCTUnwrap(coverage.candidates.first)
        XCTAssertTrue(candidate.canChoose, "Valid broad classifiers can establish groups by categorizing the folder")
        XCTAssertFalse(candidate.isAvailable, "Deployment still needs a complete exact category catalog")
        XCTAssertEqual(coverage.status, .stale)
        XCTAssertEqual(coverage.selectedRunID, classifier.id)
        do {
            try await fixture.project.setModelApproval(runID: classifier.id, approved: true)
            XCTFail("A deployment-incomplete classifier cannot receive effective approval")
        } catch { XCTAssertTrue(error.localizedDescription.contains("does not match every dataset")) }
        // Public categorization lookup also refuses damaged native files, even
        // when an explicit choice is still recorded.
        try FileManager.default.removeItem(at: classifier.directory.appendingPathComponent("model.mlmodel"))
        do {
            _ = try await fixture.project.selectedClassifierArtifact()
            XCTFail("A chosen classifier with removed native files cannot be used for categorization")
        } catch { XCTAssertFalse(error.localizedDescription.isEmpty) }
    }

    private struct Fixture {
        let root: URL
        let project: LoomProject
        let projectID: String
        let dataset = DatasetCategory(id: ImageMetadataStore.categoryID("Bricks"), name: "Bricks", imageCount: 4,
            verifiedCount: 0, pairCounts: ["depth": 2, "roughness": 0, "normal": 0])
        init() async throws {
            let folder = FileManager.default.temporaryDirectory.appendingPathComponent("loom-model-coverage-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            root = folder.resolvingSymlinksInPath().standardizedFileURL
            project = try LoomProject(sourceFolder: root)
            projectID = try await project.snapshot().projectID
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
        func coverage(project selectedProject: LoomProject? = nil) async throws -> [ModelCoverageEntry] {
            let active = selectedProject ?? project
            return try await active.loadModelCoverage(datasets: [dataset], artifacts: active.loadArtifacts(), checkpoints: [])
        }
        func depth(project selectedProject: LoomProject? = nil) async throws -> ModelCoverageEntry {
            let entries = try await coverage(project: selectedProject)
            return try XCTUnwrap(entries.first { $0.slot.product == .depth })
        }
        func artifact(_ id: String, date: Date) throws -> ModelArtifact {
            let directory = project.projectFolder.appendingPathComponent("runs/" + id)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data((id + " best weights").utf8).write(to: directory.appendingPathComponent("model.safetensors"))
            try Data("Native assembly fixture".utf8).write(to: directory.appendingPathComponent("model.mlmodel"))
            try writeJSON(["schema": "loom-factory-model-v1", "status": "completed", "project_id": projectID,
                "kind": "map", "dataset_id": dataset.id, "dataset_name": dataset.name, "map": "depth",
                "architecture": NativeArchitecture.identifier,
                "architecture_config": ["id": NativeArchitecture.identifier, "width": 8, "blocks": 1],
                "image_size": 32, "weights_path": "model.safetensors", "coreml_path": "model.mlmodel",
                "weights_sha256": try TrainingRunFiles.sha256(directory.appendingPathComponent("model.safetensors")),
                "input_name": "rgb", "output_name": "material_map"], to: directory.appendingPathComponent("model.json"))
            let artifact = ModelArtifact(id: id, kind: "map", datasetID: dataset.id, datasetName: dataset.name,
                map: .depth, directory: directory, createdAt: date, epochs: 1, loss: 0.4,
                coreMLPath: "model.mlmodel", weightsPath: "model.safetensors", architecture: NativeArchitecture.identifier)
            try JSONEncoder().encode(artifact).write(to: directory.appendingPathComponent("artifact.json"))
            return artifact
        }
        func classifier(_ id: String, date: Date, datasets: [DatasetCategory]) throws -> ModelArtifact {
            let directory = project.projectFolder.appendingPathComponent("runs/" + id)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data("Classifier assembly fixture".utf8).write(to: directory.appendingPathComponent("model.mlmodel"))
            let labels = datasets.isEmpty ? [["id": ImageMetadataStore.categoryID("Obsolete"), "name": "Obsolete"]] : datasets.map { ["id": $0.id, "name": $0.name] }
            try writeJSON(["schema": "loom-factory-model-v1", "status": "completed", "project_id": projectID,
                "kind": "classifier", "architecture": "create-ml-image-classifier-v1", "coreml_path": "model.mlmodel",
                "input_name": "image", "output_name": "classLabel", "labels": labels], to: directory.appendingPathComponent("model.json"))
            let artifact = ModelArtifact(id: id, kind: "classifier", datasetID: nil, datasetName: nil, map: nil,
                directory: directory, createdAt: date, epochs: 0, loss: nil, coreMLPath: "model.mlmodel", weightsPath: nil,
                architecture: "create-ml-image-classifier-v1")
            try JSONEncoder().encode(artifact).write(to: directory.appendingPathComponent("artifact.json"))
            return artifact
        }
    }
}

private func writeJSON(_ value: [String: Any], to file: URL) throws {
    try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: file, options: .atomic)
}
