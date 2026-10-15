import CoreML
import Foundation
import ImageIO
import LoomNative
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

@MainActor
final class ModelVersionLifecycleTests: XCTestCase {
    func testBestValidationExportUsesSavedBestBytesAndPreservesLatestAndSelection() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let selected = try fixture.artifact("selected", date: Date(timeIntervalSince1970: 100))
        let parent = try fixture.artifact("candidate", date: .now, checkpoint: true)
        try fixture.nativeCheckpoint(parent)
        try await fixture.project.selectModelVersion(runID: selected.id, datasetID: fixture.datasetID, map: .depth)
        let best = parent.directory.appendingPathComponent("best-immutable.safetensors")
        let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 1)
        try NativeWeights.write(architecture.parameterShapes.mapValues {
            NativeTensor(shape: $0, values: Array(repeating: 0.007, count: $0.reduce(1, *)))
        }, to: best)
        let file = parent.directory.appendingPathComponent("checkpoint.json")
        var descriptor = try NativeFactoryTraining.json(file)
        descriptor["best_weights_path"] = best.lastPathComponent
        descriptor["best_weights_sha256"] = try TrainingRunFiles.sha256(best)
        descriptor["best_epoch"] = 1
        descriptor["history"] = [["epoch": 1, "loss": 0.5, "validation_loss": 0.1],
                                 ["epoch": 2, "loss": 0.3, "validation_loss": 0.2]]
        try NativeFactoryTraining.writeJSON(descriptor, to: file)
        let original = try fixture.savedFiles()
        let selectionURL = fixture.project.projectFolder.appendingPathComponent("model-selection.json")
        let selection = try Data(contentsOf: selectionURL)
        let candidate = try await fixture.project.exportCheckpointModelVersion(runID: parent.id,
            datasetID: fixture.datasetID, map: .depth, preferBestValidation: true)
        XCTAssertEqual(try Data(contentsOf: candidate.directory.appendingPathComponent("model.safetensors")), try Data(contentsOf: best))
        XCTAssertEqual(try Data(contentsOf: selectionURL), selection)
        let after = try fixture.savedFiles()
        for (path, bytes) in original { XCTAssertEqual(after[path], bytes) }
        let model = try NativeFactoryTraining.json(candidate.directory.appendingPathComponent("model.json"))
        let training = try XCTUnwrap(model["training"] as? [String: Any])
        XCTAssertEqual(training["export_epoch"] as? Int, 1)
        XCTAssertEqual(training["epochs"] as? Int, 2, "Retain the actual parent training history")
        XCTAssertEqual(training["export_weight_policy"] as? String, "best-validation")
        XCTAssertEqual(model["final_loss"] as? Double, 0.5)
        XCTAssertEqual(model["final_validation_loss"] as? Double, 0.1)
        let lineage = try XCTUnwrap(model["lineage"] as? [String: Any])
        XCTAssertEqual(lineage["weights_sha256"] as? String, try TrainingRunFiles.sha256(best))
        try TrainingRunFiles.validateArtifact(candidate, projectFolder: fixture.project.projectFolder, projectID: fixture.projectID)
        let versions = try await fixture.project.modelVersions()
        XCTAssertFalse(try XCTUnwrap(versions.first { $0.id == candidate.id }).isApproved)
        let exact = try await fixture.project.exportCheckpointModelVersion(runID: parent.id,
            datasetID: fixture.datasetID, map: .depth)
        XCTAssertEqual(try Data(contentsOf: exact.directory.appendingPathComponent("model.safetensors")),
            try Data(contentsOf: parent.directory.appendingPathComponent("checkpoint.safetensors")),
            "Manual exact checkpoint exports retain their original behavior even when a best version exists")

        try Data("corrupted best weights".utf8).write(to: best)
        let beforeFailure = try fixture.savedFiles()
        do {
            _ = try await fixture.project.exportCheckpointModelVersion(runID: parent.id,
                datasetID: fixture.datasetID, map: .depth, preferBestValidation: true)
            XCTFail("A corrupt saved best must not silently export latest weights")
        } catch { XCTAssertTrue(error.localizedDescription.contains("best-validation")) }
        XCTAssertEqual(try fixture.savedFiles(), beforeFailure)
        XCTAssertEqual(try Data(contentsOf: selectionURL), selection)
    }

    func testCheckpointCandidateExportPreservesExplicitSelectionAndSourceWeights() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let selected = try fixture.artifact("selected", date: Date(timeIntervalSince1970: 100))
        let parent = try fixture.artifact("candidate", date: .now, checkpoint: true)
        try fixture.nativeCheckpoint(parent)
        try await fixture.project.selectModelVersion(runID: selected.id, datasetID: fixture.datasetID, map: .depth)
        let selectionURL = fixture.project.projectFolder.appendingPathComponent("model-selection.json")
        let selection = try Data(contentsOf: selectionURL)
        let checkpoint = try Data(contentsOf: parent.directory.appendingPathComponent("checkpoint.safetensors"))
        let candidate = try await fixture.project.exportCheckpointModelVersion(runID: parent.id,
            datasetID: fixture.datasetID, map: .depth, preferBestValidation: true)
        XCTAssertNotEqual(candidate.id, parent.id)
        XCTAssertEqual(try Data(contentsOf: candidate.directory.appendingPathComponent("model.safetensors")), checkpoint)
        XCTAssertEqual(try Data(contentsOf: selectionURL), selection)
        XCTAssertEqual(try Data(contentsOf: parent.directory.appendingPathComponent("checkpoint.safetensors")), checkpoint)
        let active = try await fixture.project.selectedModelArtifact(datasetID: fixture.datasetID, map: .depth)
        XCTAssertEqual(active?.id, selected.id)
    }

    func testCheckpointCanBecomeSelectedModelWithoutChangingStoppedRunOrItsExport() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let parent = try fixture.artifact("stopped", date: Date(timeIntervalSince1970: 100), checkpoint: true)
        try fixture.nativeCheckpoint(parent)
        let sourceDescriptorURL = parent.directory.appendingPathComponent("checkpoint.json")
        var sourceDescriptor = try NativeFactoryTraining.json(sourceDescriptorURL)
        sourceDescriptor["prior_history"] = [["epoch": 1, "loss": 0.8], ["epoch": 2, "loss": 0.7], ["epoch": 3, "loss": 0.6]]
        sourceDescriptor["prior_epochs"] = 3; sourceDescriptor["prior_updates"] = 6
        sourceDescriptor["prior_images"] = 12; sourceDescriptor["completed_images"] = 8
        sourceDescriptor["prior_duration_seconds"] = 9.0; sourceDescriptor["elapsed_seconds"] = 1.0
        try NativeFactoryTraining.writeJSON(sourceDescriptor, to: sourceDescriptorURL)
        try await fixture.project.updateModelVersionMetadata(runID: parent.id, name: "Darker checkpoint", notes: "Compare detail")
        try await fixture.project.selectModelVersion(runID: parent.id, datasetID: fixture.datasetID, map: .depth)
        let original = try fixture.savedFiles()
        let checkpointBytes = try Data(contentsOf: parent.directory.appendingPathComponent("checkpoint.safetensors"))
        let promoted = try await fixture.project.selectCheckpointModelVersion(runID: parent.id, datasetID: fixture.datasetID, map: .depth)
        XCTAssertNotEqual(promoted.id, parent.id)
        XCTAssertEqual(try Data(contentsOf: promoted.directory.appendingPathComponent("model.safetensors")), checkpointBytes)
        let after = try fixture.savedFiles()
        for (path, bytes) in original {
            XCTAssertEqual(after[path], bytes)
        }
        let reopened = try LoomProject(sourceFolder: fixture.root)
        let selected = try await reopened.selectedModelArtifact(datasetID: fixture.datasetID, map: .depth)
        XCTAssertEqual(selected?.id, promoted.id)
        let versions = try await reopened.modelVersions()
        let derived = try XCTUnwrap(versions.first { $0.id == promoted.id })
        XCTAssertEqual(derived.label, "Darker checkpoint · checkpoint")
        XCTAssertEqual(derived.lineage?.parentRunID, parent.id)
        XCTAssertEqual(derived.lineage?.source, .checkpoint)
        XCTAssertTrue(derived.isActive)
        XCTAssertFalse(derived.isApproved, "Selection does not invent a human quality approval")
        XCTAssertEqual(derived.exportedEpochsCompleted, 5)
        let exportedDescriptor = try NativeFactoryTraining.json(promoted.directory.appendingPathComponent("model.json"))
        let continuation = TrainingContinuationHistory(descriptor: exportedDescriptor, source: .model,
            runID: promoted.id, resumingCursor: false, trainingImages: 0)
        XCTAssertEqual(continuation.epochs, 5)
        XCTAssertEqual(continuation.updates, 8)
        XCTAssertEqual(continuation.images, 20)
        XCTAssertEqual(continuation.elapsedSeconds, 10)
        let tested = try ModelTestSelection.load(runID: promoted.id, source: .model, datasetID: fixture.datasetID,
            map: .depth, projectFolder: fixture.project.projectFolder, projectID: fixture.projectID)
        XCTAssertEqual(try Data(contentsOf: tested.weightsURL), checkpointBytes)
        try TrainingRunFiles.validateArtifact(promoted, projectFolder: fixture.project.projectFolder, projectID: fixture.projectID)
        try await reopened.setModelApproval(runID: promoted.id, approved: true)
        let approved = try await reopened.modelVersions().first { $0.id == promoted.id }
        XCTAssertTrue(try XCTUnwrap(approved).isApproved)
    }

    func testCheckpointOnlyRunCanBeSelectedAndCorruptCheckpointLeavesChoiceUntouched() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let parent = try fixture.artifact("unfinished", date: .now, checkpoint: true)
        try fixture.nativeCheckpoint(parent)
        for name in ["artifact.json", "model.json", "model.safetensors", "model.mlmodel"] {
            try FileManager.default.removeItem(at: parent.directory.appendingPathComponent(name))
        }
        let promoted = try await fixture.project.selectCheckpointModelVersion(runID: parent.id, datasetID: fixture.datasetID, map: .depth)
        let originals = try fixture.savedFiles()
        try Data("corrupt".utf8).write(to: parent.directory.appendingPathComponent("checkpoint.safetensors"))
        do {
            _ = try await fixture.project.selectCheckpointModelVersion(runID: parent.id, datasetID: fixture.datasetID, map: .depth)
            XCTFail("Changed checkpoint weights must not replace the selected version")
        } catch { XCTAssertTrue(error.localizedDescription.contains("checksum")) }
        let chosen = try await fixture.project.selectedModelArtifact(datasetID: fixture.datasetID, map: .depth)
        XCTAssertEqual(chosen?.id, promoted.id)
        let versions = try await fixture.project.modelVersions()
        XCTAssertEqual(versions.count, 2)
        XCTAssertEqual(Set(try fixture.savedFiles().keys), Set(originals.keys))
    }

    func testContinuationUsesSelectedRunAndDistinguishesCheckpointFromExportedBestWeights() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let old = try fixture.artifact("old", date: Date(timeIntervalSince1970: 100), size: 32, width: 8, checkpoint: true)
        _ = try fixture.artifact("latest", date: Date(timeIntervalSince1970: 200), size: 64, width: 16, checkpoint: true)
        var changed = TrainingSettings(imageSize: 128, epochs: 4, modelWidth: 32, modelBlocks: 5)
        changed.learningRate = 0.0002; changed.optimizer = .adam

        let continued = try await fixture.project.prepareTrainingStart(.checkpoint(runID: "old"),
            datasetID: fixture.datasetID, map: .depth, settings: changed)
        XCTAssertEqual(continued.resume?.weights, old.directory.appendingPathComponent("checkpoint.safetensors"))
        XCTAssertEqual(continued.resume?.source, .checkpoint)
        XCTAssertEqual(continued.settings.imageSize, 128, "A requested native crop grid does not change the weight shapes")
        XCTAssertEqual(continued.settings.modelWidth, 8)
        XCTAssertEqual(continued.settings.modelBlocks, 1)
        XCTAssertEqual(continued.settings.modelArchitecture, NativeArchitecture.identifier)
        XCTAssertEqual(continued.settings.scalarLoss, .absoluteV1, "Legacy checkpoint continuation preserves its objective")
        let detail = try await fixture.project.prepareTrainingStart(.checkpoint(runID: "old"),
            datasetID: fixture.datasetID, map: .depth, settings: changed, scalarLossOverride: .nativeDetailV3)
        XCTAssertEqual(detail.settings.scalarLoss, .nativeDetailV3)
        XCTAssertEqual(detail.resume?.weights, continued.resume?.weights)
        let freshDetail = try await fixture.project.prepareTrainingStart(.fresh,
            datasetID: fixture.datasetID, map: .depth, settings: changed, scalarLossOverride: .nativeDetailV3)
        XCTAssertEqual(freshDetail.settings.scalarLoss, .nativeDetailV3)
        do {
            _ = try await fixture.project.prepareTrainingStart(.fresh, datasetID: fixture.datasetID,
                map: .normal, settings: changed, scalarLossOverride: .nativeDetailV3)
            XCTFail("Scalar objective overrides cannot change normal training")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Normal maps")) }
        XCTAssertEqual(continued.settings.learningRate, changed.learningRate)
        XCTAssertEqual(continued.settings.optimizer, .adam)
        XCTAssertEqual(changed.seed, -1)
        XCTAssertEqual(continued.settings.seed, -1, "Continuation preserves an explicit system-random preference")

        let best = try await fixture.project.prepareTrainingStart(.model(runID: "old"),
            datasetID: fixture.datasetID, map: .depth, settings: changed)
        XCTAssertEqual(best.resume?.weights, old.directory.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(best.resume?.source, .model)
        XCTAssertEqual(best.settings.modelArchitecture, NativeArchitecture.identifier)
        XCTAssertEqual(best.settings.scalarLoss, .absoluteV1)
        XCTAssertEqual(best.settings.seed, -1)
        XCTAssertNotEqual(best.resume?.sha256, continued.resume?.sha256)
        changed.seed = 91
        let override = try await fixture.project.prepareTrainingStart(.checkpoint(runID: "old"),
            datasetID: fixture.datasetID, map: .depth, settings: changed)
        XCTAssertEqual(override.settings.seed, 91, "Explicit continuation seed overrides remain available")
        changed.exaggerateFeatures = true
        let changedTarget = try await fixture.project.prepareTrainingStart(.checkpoint(runID: "old"),
            datasetID: fixture.datasetID, map: .depth, settings: changed)
        XCTAssertTrue(changedTarget.settings.exaggerateFeatures, "A deliberate target-preparation edit must reach the worker")
        let fresh = try await fixture.project.prepareTrainingStart(.fresh, datasetID: fixture.datasetID, map: .depth, settings: changed)
        XCTAssertNil(fresh.resume)
        XCTAssertEqual(fresh.settings, changed)
    }

    func testFailedFreshReplacementLeavesSelectionAndAllSavedRunsUntouched() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let old = try fixture.artifact("old", date: Date(timeIntervalSince1970: 100), checkpoint: true)
        _ = try fixture.artifact("latest", date: Date(timeIntervalSince1970: 200), checkpoint: true)
        try await fixture.project.selectModelVersion(runID: "old", datasetID: fixture.datasetID, map: .depth)
        let originals = try fixture.savedFiles()
        do {
            _ = try await fixture.project.trainMap(datasetID: fixture.datasetID, map: .depth,
                settings: TrainingSettings(imageSize: 32, epochs: 1, modelWidth: 8, modelBlocks: 1), start: .fresh, progress: { _ in })
            XCTFail("A project without source teaching pairs cannot start replacement training")
        } catch { XCTAssertTrue(error.localizedDescription.contains("no longer has images")) }
        let selected = try await fixture.project.selectedModelArtifact(datasetID: fixture.datasetID, map: .depth)
        XCTAssertEqual(selected?.id, old.id)
        XCTAssertEqual(try fixture.savedFiles(), originals)
    }

    func testSelectionPersistsAcrossReopeningAndMissingSelectionNeverSilentlyUsesLatest() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let old = try fixture.artifact("old", date: Date(timeIntervalSince1970: 100), checkpoint: true)
        let latest = try fixture.artifact("latest", date: Date(timeIntervalSince1970: 200), checkpoint: true)
        let fallback = try await fixture.project.selectedModelArtifact(datasetID: fixture.datasetID, map: .depth)
        XCTAssertEqual(fallback?.id, latest.id)
        try await fixture.project.selectModelVersion(runID: "old", datasetID: fixture.datasetID, map: .depth)
        let reopened = try LoomProject(sourceFolder: fixture.root)
        let restored = try await reopened.selectedModelArtifact(datasetID: fixture.datasetID, map: .depth)
        XCTAssertEqual(restored?.id, old.id)
        let versions = try await reopened.modelVersions(datasetID: fixture.datasetID, map: .depth)
        XCTAssertEqual(Set(versions.map(\.id)), ["old", "latest"])
        XCTAssertEqual(versions.filter(\.isActive).map(\.id), ["old"])
        try FileManager.default.removeItem(at: old.directory.appendingPathComponent("artifact.json"))
        do {
            _ = try await reopened.selectedModelArtifact(datasetID: fixture.datasetID, map: .depth)
            XCTFail("A missing selected version must not silently export another run")
        } catch { XCTAssertTrue(error.localizedDescription.contains("selected depth version is missing")) }
    }

    func testArtifactOnlyLegacyHistoryRestoresSettingsMetricsAndExactModelSelection() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let artifact = try fixture.artifact("legacy-export", date: Date(timeIntervalSince1970: 100), size: 64, width: 16)
        let versions = try await fixture.project.modelVersions()
        let saved = try XCTUnwrap(versions.first)
        XCTAssertNil(saved.checkpoint)
        XCTAssertEqual(saved.artifact?.id, artifact.id)
        XCTAssertEqual(saved.settings?.imageSize, 64)
        XCTAssertEqual(saved.settings?.modelWidth, 16)
        XCTAssertEqual(saved.settings?.gradientAccumulationSteps, 3)
        XCTAssertEqual(saved.settings?.learningRate, 0.0003)
        XCTAssertEqual(saved.history.map(\.validationLoss), [0.4, 0.2])
        try await fixture.project.selectModelVersion(runID: artifact.id, datasetID: fixture.datasetID, map: .depth)
        let resumed = try await fixture.project.prepareTrainingStart(.model(runID: artifact.id),
            datasetID: fixture.datasetID, map: .depth, settings: TrainingSettings(epochs: 2))
        XCTAssertEqual(resumed.resume?.source, .model)
        XCTAssertEqual(resumed.settings.imageSize, 1024, "Continuation keeps the requested crop size")
    }

    func testLineageAndRunLabelsAreReadWithoutChangingHistoricalDescriptors() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let parent = try fixture.artifact("parent", date: Date(timeIntervalSince1970: 100), checkpoint: true)
        let child = try fixture.artifact("child", date: Date(timeIntervalSince1970: 200), checkpoint: true)
        let lineage = TrainingLineage(parentRunID: parent.id, source: .model,
            weightsSHA256: try TrainingRunFiles.sha256(parent.directory.appendingPathComponent("model.safetensors")))
        let settings = TrainingSettings(imageSize: 32, epochs: 2, modelWidth: 8, modelBlocks: 1,
            modelArchitecture: NativeArchitecture.identifier)
        try TrainingRunFiles.writeMetadata(TrainingRunMetadata(projectID: fixture.projectID, runID: child.id,
            datasetID: fixture.datasetID, map: .depth, label: "Lower learning rate", createdAt: child.createdAt,
            settings: settings, lineage: lineage, status: "completed"), at: child.directory)
        let originals = try fixture.savedFiles()
        let versions = try await fixture.project.modelVersions()
        let saved = try XCTUnwrap(versions.first { $0.id == child.id })
        XCTAssertEqual(saved.label, "Lower learning rate")
        XCTAssertEqual(saved.lineage, lineage)
        XCTAssertEqual(saved.settings, settings)
        XCTAssertEqual(try fixture.savedFiles(), originals)
    }

    func testResumeRejectsChecksumProjectIdentityAndSymlinkChanges() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let artifact = try fixture.artifact("protected", date: .now, checkpoint: true)
        let settings = TrainingSettings(imageSize: 32, epochs: 1, modelWidth: 8, modelBlocks: 1)
        let weights = artifact.directory.appendingPathComponent("checkpoint.safetensors")
        let original = try Data(contentsOf: weights)
        try Data("changed".utf8).write(to: weights)
        do {
            _ = try await fixture.project.prepareTrainingStart(.checkpoint(runID: artifact.id), datasetID: fixture.datasetID, map: .depth, settings: settings)
            XCTFail("Changed checkpoint weights must fail before training")
        } catch { XCTAssertTrue(error.localizedDescription.contains("checksum")) }
        try original.write(to: weights)
        do {
            _ = try await fixture.project.prepareTrainingStart(.checkpoint(runID: artifact.id), datasetID: "another-dataset", map: .depth, settings: settings)
            XCTFail("Weights from another dataset must fail before training")
        } catch { XCTAssertTrue(error.localizedDescription.contains("project, dataset, map")) }
        let linked = fixture.project.projectFolder.appendingPathComponent("runs/linked")
        try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: artifact.directory)
        do {
            _ = try await fixture.project.prepareTrainingStart(.checkpoint(runID: "linked"), datasetID: fixture.datasetID, map: .depth, settings: settings)
            XCTFail("A substituted run directory must fail before training")
        } catch { XCTAssertTrue(error.localizedDescription.contains("unchanged run")) }
        do {
            _ = try await fixture.project.prepareTrainingStart(.checkpoint(runID: "../protected"), datasetID: fixture.datasetID, map: .depth, settings: settings)
            XCTFail("Run selection must not escape its project")
        } catch { XCTAssertTrue(error.localizedDescription.contains("saved run")) }
    }

    func testNativeChecksumDriftDoesNotReplaceAnExistingExportSelection() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let old = try fixture.artifact("old", date: Date(timeIntervalSince1970: 100))
        let replacement = try fixture.artifact("replacement", date: Date(timeIntervalSince1970: 200))
        // Legacy exports without a native checksum remain selectable.
        try await fixture.project.selectModelVersion(runID: old.id, datasetID: fixture.datasetID, map: .depth)
        let native = replacement.directory.appendingPathComponent("model.mlmodel")
        let original = try Data(contentsOf: native)
        let descriptorFile = replacement.directory.appendingPathComponent("model.json")
        var descriptor = try NativeFactoryTraining.json(descriptorFile)
        descriptor["coreml_sha256"] = try TrainingRunFiles.sha256(native)
        try NativeFactoryTraining.writeJSON(descriptor, to: descriptorFile)
        try Data("Changed native export".utf8).write(to: native)
        do {
            try await fixture.project.selectModelVersion(runID: replacement.id, datasetID: fixture.datasetID, map: .depth)
            XCTFail("A changed native model must not become the package's selected version")
        } catch { XCTAssertTrue(error.localizedDescription.contains("native export checksum changed")) }
        let unchanged = try await fixture.project.selectedModelArtifact(datasetID: fixture.datasetID, map: .depth)
        XCTAssertEqual(unchanged?.id, old.id)
        try original.write(to: native)
        try await fixture.project.selectModelVersion(runID: replacement.id, datasetID: fixture.datasetID, map: .depth)
        let restored = try await fixture.project.selectedModelArtifact(datasetID: fixture.datasetID, map: .depth)
        XCTAssertEqual(restored?.id, replacement.id)
    }

    func testPackageExportsSelectedHistoricalAndCheckpointWeightsAfterReopening() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let classifierURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Resources/DTTC3.mlmodel")
        let interface = try await nativeClassifierInterface(at: classifierURL)
        for (index, label) in interface.labels.enumerated() {
            try fixture.image(name: "Material-\(index)/sample_diff.png", category: ImageMetadataStore.friendlyName(label))
        }
        _ = try await fixture.project.importClassifier(from: classifierURL)
        var historical: [String: Data] = [:]
        for label in interface.labels {
            let name = ImageMetadataStore.friendlyName(label), id = ImageMetadataStore.categoryID(name)
            let old = try fixture.artifact("old-\(id)", date: Date(timeIntervalSince1970: 100), checkpoint: historical.isEmpty, datasetID: id, datasetName: name)
            _ = try fixture.artifact("new-\(id)", date: Date(timeIntervalSince1970: 200), datasetID: id, datasetName: name)
            if historical.isEmpty {
                try fixture.nativeCheckpoint(old)
                _ = try await fixture.project.selectCheckpointModelVersion(runID: old.id, datasetID: id, map: .depth)
                historical[id] = try Data(contentsOf: old.directory.appendingPathComponent("checkpoint.safetensors"))
            } else {
                try await fixture.project.selectModelVersion(runID: old.id, datasetID: id, map: .depth)
                historical[id] = try Data(contentsOf: old.directory.appendingPathComponent("model.safetensors"))
            }
        }
        let originals = try fixture.savedFiles()
        let reopened = try LoomProject(sourceFolder: fixture.root)
        let archive = try await reopened.exportPackage(name: "Selected Versions", version: "1.0.0")
        let extracted = fixture.project.projectFolder.appendingPathComponent("unpacked-selection")
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = ["-x", "-k", archive.path, extracted.path]
        try process.run(); process.waitUntilExit(); XCTAssertEqual(process.terminationStatus, 0)
        for (id, bytes) in historical {
            XCTAssertEqual(try Data(contentsOf: extracted.appendingPathComponent("models/\(id)/depth/model.safetensors")), bytes)
        }
        XCTAssertEqual(try fixture.savedFiles(), originals, "Exporting must preserve every checkpoint and historical run descriptor")
    }

    private struct Fixture {
        let root: URL
        let project: LoomProject
        let projectID: String
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        init() async throws {
            let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("loom-versions-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
            root = temporary.resolvingSymlinksInPath().standardizedFileURL
            project = try LoomProject(sourceFolder: root)
            projectID = try await project.snapshot().projectID
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
        func nativeCheckpoint(_ artifact: ModelArtifact) throws {
            let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 1)
            let weights = architecture.parameterShapes.mapValues { shape in
                NativeTensor(shape: shape, values: [Float](repeating: 0.013, count: shape.reduce(1, *)))
            }
            let file = artifact.directory.appendingPathComponent("checkpoint.safetensors")
            try NativeWeights.write(weights, to: file, metadata: ["checkpoint_test_fixture": "not-trained"])
            var descriptor = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("checkpoint.json"))
            descriptor["weights_sha256"] = try TrainingRunFiles.sha256(file)
            descriptor["updates"] = 2
            try NativeFactoryTraining.writeJSON(descriptor, to: artifact.directory.appendingPathComponent("checkpoint.json"))
        }
        func artifact(_ id: String, date: Date, size: Int = 32, width: Int = 8, checkpoint: Bool = false,
                      datasetID selectedID: String? = nil, datasetName: String = "Bricks") throws -> ModelArtifact {
            let datasetID = selectedID ?? self.datasetID
            let directory = project.projectFolder.appendingPathComponent("runs/" + id, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            var settings = NativeFactoryTraining.settingsJSON(TrainingSettings(imageSize: size, epochs: 2,
                learningRate: 0.0003, modelWidth: width, modelBlocks: 1, gradientAccumulationSteps: 3,
                modelArchitecture: NativeArchitecture.identifier))
            settings.removeValue(forKey: "scalar_loss") // Historical runs predate the objective ID.
            settings.removeValue(forKey: "seed") // Historical omission must retain seed zero.
            let rows: [[String: Any]] = [["epoch": 1, "loss": 0.5, "validation_loss": 0.4, "learning_rate": 0.0003, "updates": 1],
                                        ["epoch": 2, "loss": 0.3, "validation_loss": 0.2, "learning_rate": 0.0003, "updates": 2]]
            settings["history"] = rows
            settings["device"] = "metal"
            let configuration: [String: Any] = ["id": "loom-factory-material-residual-v1", "width": width, "blocks": 1]
            let bytes = Data((id + "-best-export").utf8)
            try bytes.write(to: directory.appendingPathComponent("model.safetensors"))
            // Assembly fixtures validate routing and file preservation without claiming prediction quality.
            try Data("Native map export fixture".utf8).write(to: directory.appendingPathComponent("model.mlmodel"))
            var descriptor: [String: Any] = ["schema": "loom-factory-model-v1", "status": "completed", "project_id": projectID,
                "kind": "map", "dataset_id": datasetID, "dataset_name": datasetName, "map": "depth",
                "architecture": "loom-factory-material-residual-v1", "architecture_config": configuration,
                "image_size": size, "weights_path": "model.safetensors", "coreml_path": "model.mlmodel",
                "input_name": "rgb", "output_name": "material_map", "training": settings]
            descriptor["weights_sha256"] = try TrainingRunFiles.sha256(directory.appendingPathComponent("model.safetensors"))
            try NativeFactoryTraining.writeJSON(descriptor, to: directory.appendingPathComponent("model.json"))
            let artifact = ModelArtifact(id: id, kind: "map", datasetID: datasetID, datasetName: datasetName, map: .depth,
                directory: directory, createdAt: date, epochs: 2, loss: 0.3, coreMLPath: "model.mlmodel",
                weightsPath: "model.safetensors", architecture: "loom-factory-material-residual-v1")
            try JSONEncoder().encode(artifact).write(to: directory.appendingPathComponent("artifact.json"))
            if checkpoint {
                try Data((id + "-final-checkpoint").utf8).write(to: directory.appendingPathComponent("checkpoint.safetensors"))
                descriptor["schema"] = "loom-factory-checkpoint-v1"; descriptor["weights_path"] = "checkpoint.safetensors"
                descriptor["settings"] = settings; descriptor["epochs_completed"] = 2; descriptor["history"] = rows
                descriptor["weights_sha256"] = try TrainingRunFiles.sha256(directory.appendingPathComponent("checkpoint.safetensors"))
                try NativeFactoryTraining.writeJSON(descriptor, to: directory.appendingPathComponent("checkpoint.json"))
            }
            return artifact
        }
        func savedFiles() throws -> [String: Data] {
            let runs = project.projectFolder.appendingPathComponent("runs")
            let files = try XCTUnwrap(FileManager.default.enumerator(at: runs, includingPropertiesForKeys: [.isRegularFileKey]))
            var result: [String: Data] = [:]
            for case let file as URL in files where try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                result[String(file.path.dropFirst(runs.path.count + 1))] = try Data(contentsOf: file)
            }
            return result
        }
        func image(name: String, category: String) throws {
            let file = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            let pixels = Data(repeating: 120, count: 32 * 32 * 3)
            let provider = try XCTUnwrap(CGDataProvider(data: pixels as CFData))
            let image = try XCTUnwrap(CGImage(width: 32, height: 32, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: 96,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [], provider: provider, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent))
            let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil); XCTAssertTrue(CGImageDestinationFinalize(destination))
            try ImageMetadataStore.write(ImageMetadata(pairingKey: name, role: .diffuse, category: category), to: file)
        }
    }
}
