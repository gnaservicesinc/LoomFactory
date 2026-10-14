import AppKit
import ImageIO
import LoomCore
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

@Suite(.serialized)
struct ModelCoverageControlsTests {
    @Test @MainActor
    func approveAndUseIsOneActionAndMissingPackageExportDoesNotHideTesting() async throws {
        let fixture = try await CoverageControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        let initial = try #require(model.modelCoverage.first { $0.slot == fixture.depthSlot })
        let candidate = try #require(ModelCoveragePresentation(initial).actionCandidate)
        #expect(candidate.id == initial.effectiveRunID)
        model.approveAndUseCoverageModel(artifact: candidate.artifact)
        try await wait(model)
        let chosen = try #require(model.modelCoverage.first { $0.slot == fixture.depthSlot })
        #expect(chosen.selectedRunID == candidate.id)
        #expect(ModelCoveragePresentation(chosen).isReady)

        let version = try #require(model.snapshot?.modelVersions.first { $0.id == candidate.id })
        model.updateModelVersionMetadata(version, name: "Brick approval candidate", notes: "Check mortar detail.")
        try await wait(model)
        let renamed = try #require(model.snapshot?.modelVersions.first { $0.id == candidate.id })
        #expect(renamed.label == "Brick approval candidate")
        #expect(renamed.notes == "Check mortar detail.")
        #expect(renamed.isApproved)
        if let coreML = candidate.artifact.coreMLPath {
            try FileManager.default.removeItem(at: candidate.artifact.directory.appendingPathComponent(coreML))
        }
        await model.pollFolder()
        let unavailable = try #require(model.modelCoverage.first { $0.slot == fixture.depthSlot })
        #expect(unavailable.candidates.first { $0.id == candidate.id }?.isAvailable == false)
        model.testCoverageModel(artifact: candidate.artifact)
        #expect(model.workspaceSection == .testing)
        #expect(model.selectedTestVersionID == candidate.id)
        #expect(model.availableTestWeightSources.contains(.model))
        try await render(LoomRoomView(model: model), size: NSSize(width: 1120, height: 760), name: "test-missing-package-export.png")
    }

    @Test @MainActor
    func actualSourceCoverageRendersSavedDepthVersionsWhenRequested() async throws {
        guard let path = ProcessInfo.processInfo.environment["LOOM_FACTORY_COVERAGE_SOURCE_ROOT"] else { return }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(URL(fileURLWithPath: path, isDirectory: true))
        try await wait(model)
        let snapshot = try #require(model.snapshot)
        let bricks = try #require(snapshot.datasets.first { $0.name == "Bricks" })
        let depth = try #require(snapshot.modelCoverage.first { $0.slot.datasetID == bricks.id && $0.slot.product == .depth })
        let runs = snapshot.sourceURL.appendingPathComponent(".loom-factory/runs", isDirectory: true)
        let physicalRunIDs = try FileManager.default.contentsOfDirectory(at: runs, includingPropertiesForKeys: [.isDirectoryKey])
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }.map(\.lastPathComponent)
        #expect(Set(snapshot.modelVersions.map(\.id)) == Set(physicalRunIDs))
        let depthVersions = snapshot.modelVersions.filter { $0.datasetID == bricks.id && $0.map == .depth }
        #expect(!depthVersions.isEmpty)
        #expect(depth.checkpointCount == depthVersions.filter { $0.artifact == nil && $0.checkpoint != nil }.count)
        model.selectedDatasetID = bricks.id
        model.selectedMap = .depth
        model.workspaceSection = .models
        try await render(ModelCoverageView(model: model).padding(24).preferredColorScheme(.dark),
            size: NSSize(width: 1380, height: 1250), name: "actual-bricks-depth-coverage-dark.png")
    }

    @Test @MainActor
    func savedCheckpointsRemainVisibleWithoutSatisfyingModelApprovalOrPackageCoverage() async throws {
        let fixture = try await CoverageControlsFixture()
        defer { fixture.remove() }
        for id in ["older-depth", "latest-depth"] {
            try FileManager.default.removeItem(at: fixture.source.appendingPathComponent(".loom-factory/runs/\(id)"))
        }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        let depth = try #require(model.modelCoverage.first { $0.slot == fixture.depthSlot })
        let state = ModelCoveragePresentation(depth)
        #expect(depth.availableCount == 0)
        #expect(depth.checkpointCount == 2)
        #expect(state.modelCounts == "0 completed models · 0 approved")
        #expect(state.checkpointCounts == "2 saved checkpoints")
        #expect(state.title == "Training unfinished")
        #expect(ModelCoverageFilter.missing.includes(depth))
        #expect(!ModelCoverageFilter.needsApproval.includes(depth))
        #expect(!state.isReady)
        let unfinished = try #require(model.snapshot).modelVersions.filter { $0.artifact == nil }
        #expect(Set(unfinished.map(\.id)) == Set(["cancelled-depth", "interrupted-depth"]))
        #expect(unfinished.allSatisfy { $0.checkpoint != nil && !$0.isApproved && !$0.isActive })
        let version = try #require(unfinished.first { $0.id == "cancelled-depth" })
        model.showModelTesting(version)
        #expect(model.workspaceSection == .testing)
        #expect(model.selectedTestVersionID == version.id)
        ModelCoverageCheckpointRow(model: model, version: version).continueTraining()
        #expect(model.selectedDatasetID == fixture.depthSlot.datasetID)
        #expect(model.selectedMap == .depth)
        #expect(model.trainingStartMode == .checkpoint)
        #expect(model.selectedTrainingVersionID == version.id)
        for scheme in [ColorScheme.light, .dark] {
            try await render(ModelCoverageView(model: model).padding(24).preferredColorScheme(scheme),
                size: NSSize(width: 1120, height: 1150), name: "coverage-checkpoints-visible-\(scheme == .light ? "light" : "dark").png")
            try await render(ModelCoverageCheckpointRow(model: model, version: version).padding(24).preferredColorScheme(scheme),
                size: NSSize(width: 1060, height: 220), name: "coverage-saved-checkpoint-\(scheme == .light ? "light" : "dark").png")
        }
    }

    @Test @MainActor
    func approvalAndChosenModelAreIndependentAndPersistWithTheSourceFolder() async throws {
        let fixture = try await CoverageControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        let initial = try #require(model.modelCoverage.first { $0.slot == fixture.depthSlot })
        #expect(initial.availableCount == 2)
        #expect(initial.selectedRunID == nil)
        #expect(initial.usesLatestByDefault)
        #expect(!ModelCoveragePresentation(initial).isReady)
        #expect(ModelCoverageFilter.needsSelection.includes(initial))
        #expect(ModelCoverageFilter.needsApproval.includes(initial))

        model.setModelApproval(runID: "older-depth", approved: true)
        try await wait(model)
        let approved = try #require(model.modelCoverage.first { $0.slot == fixture.depthSlot })
        #expect(approved.approvedCount == 1)
        #expect(approved.selectedRunID == nil)
        #expect(approved.effectiveRunID == "latest-depth")
        #expect(!ModelCoverageFilter.needsApproval.includes(approved))
        #expect(ModelCoverageFilter.needsSelection.includes(approved))
        #expect(ModelCoveragePresentation(approved).title == "Choose approved model")

        let older = try #require(approved.candidates.first { $0.id == "older-depth" })
        model.useCoverageModel(artifact: older.artifact)
        try await wait(model)
        let chosen = try #require(model.modelCoverage.first { $0.slot == fixture.depthSlot })
        #expect(chosen.selectedRunID == "older-depth")
        #expect(ModelCoveragePresentation(chosen).isReady)
        #expect(ModelCoverageFilter.ready.includes(chosen))
        #expect(!chosen.usesLatestByDefault)
        #expect(chosen.candidates.first { $0.id == "latest-depth" }?.isApproved == false)

        model.setModelApproval(runID: "older-depth", approved: false)
        try await wait(model)
        let unapproved = try #require(model.modelCoverage.first { $0.slot == fixture.depthSlot })
        #expect(unapproved.selectedRunID == "older-depth")
        #expect(!ModelCoveragePresentation(unapproved).isReady)
        #expect(ModelCoverageFilter.needsApproval.includes(unapproved))
        #expect(ModelCoveragePresentation(unapproved).title == "Needs approval")

        let missing = try #require(model.modelCoverage.first { $0.slot == fixture.normalSlot })
        #expect(ModelCoverageFilter.missing.includes(missing))
        model.setModelRequirement(slot: fixture.normalSlot, required: false)
        try await wait(model)
        let optional = try #require(model.modelCoverage.first { $0.slot == fixture.normalSlot })
        #expect(!optional.isRequired)
        #expect(!ModelCoverageFilter.missing.includes(optional))
        #expect(ModelCoverageFilter.all.includes(optional))

        let reopened = FactoryModel(rememberSourceFolder: false)
        reopened.openFolder(fixture.source)
        try await wait(reopened)
        #expect(reopened.modelCoverage.first { $0.slot == fixture.depthSlot }?.selectedRunID == "older-depth")
        #expect(reopened.modelCoverage.first { $0.slot == fixture.normalSlot }?.isRequired == false)

        reopened.setModelApproval(runID: "latest-depth", approved: true)
        try await wait(reopened)
        try FileManager.default.removeItem(at: older.artifact.directory.appendingPathComponent("weights.safetensors"))
        await reopened.pollFolder()
        let unavailable = try #require(reopened.modelCoverage.first { $0.slot == fixture.depthSlot })
        #expect(unavailable.selectedRunID == "older-depth")
        #expect(unavailable.availableCount == 1)
        #expect(unavailable.approvedCount == 1)
        #expect(ModelCoveragePresentation(unavailable).title == "Chosen model unavailable")
        #expect(ModelCoverageFilter.needsSelection.includes(unavailable))
        #expect(!ModelCoverageFilter.ready.includes(unavailable))
    }

    @Test @MainActor
    func maintenanceActionsOpenTheMatchingTrainingAndTestingWorkspaces() async throws {
        let fixture = try await CoverageControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        model.trainCoverageSlot(slot: fixture.normalSlot)
        #expect(model.workspaceSection == .training)
        #expect(model.selectedDatasetID == fixture.normalSlot.datasetID)
        #expect(model.selectedMap == .normal)
        let depth = try #require(model.modelCoverage.first { $0.slot == fixture.depthSlot })
        let candidate = try #require(depth.candidates.first)
        model.testCoverageModel(artifact: candidate.artifact)
        #expect(model.workspaceSection == .testing)
        #expect(model.selectedTestVersionID == candidate.id)
        #expect(model.selectedTestVersion?.map == .depth)
    }

    @Test @MainActor
    func coverageAndCandidateActionsRenderAtMinimumWindowSizeInBothAppearances() async throws {
        let fixture = try await CoverageControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        model.selectedDatasetID = fixture.depthSlot.datasetID ?? ""
        model.selectedMap = .depth
        model.setModelApproval(runID: "older-depth", approved: true)
        try await wait(model)
        let entry = try #require(model.modelCoverage.first { $0.slot == fixture.depthSlot })
        let candidate = try #require(entry.candidates.first { $0.id == "older-depth" })
        for scheme in [ColorScheme.light, .dark] {
            let suffix = scheme == .light ? "light" : "dark"
            try await render(ScrollView { ModelCoverageView(model: model).padding(24) }.preferredColorScheme(scheme),
                size: NSSize(width: 1120, height: 760), name: "model-coverage-minimum-\(suffix).png")
            try await render(ModelCoverageView(model: model).padding(24).preferredColorScheme(scheme),
                size: NSSize(width: 1200, height: 1450), name: "model-coverage-full-\(suffix).png")
            try await render(FactoryPanel("Model actions") {
                ModelCoverageCandidateRow(model: model, entry: entry, candidate: candidate)
            }.padding(24).preferredColorScheme(scheme),
                size: NSSize(width: 1060, height: 290), name: "model-coverage-actions-\(suffix).png")
        }
    }

    @Test @MainActor
    func modelsWorkspaceRendersInsideLoomRoomAtMinimumAndDefaultWindowSizes() async throws {
        let fixture = try await CoverageControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        model.selectedDatasetID = fixture.depthSlot.datasetID ?? ""
        model.workspaceSection = .models
        #expect(LoomWorkspaceSection.allCases.contains(.models))
        #expect(model.modelCoverage.count == 7)
        try await render(LoomRoomView(model: model).preferredColorScheme(.light),
            size: NSSize(width: 1120, height: 760), name: "loom-room-models-minimum-light.png")
        try await render(LoomRoomView(model: model).preferredColorScheme(.dark),
            size: NSSize(width: 1380, height: 980), name: "loom-room-models-default-dark.png")
        #expect(model.workspaceSection == .models)
    }

    @MainActor private func wait(_ model: FactoryModel) async throws {
        for _ in 0..<500 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!model.busy)
        #expect(model.errorMessage == nil)
        _ = try #require(model.snapshot)
    }

    @MainActor private func render<Content: View>(_ content: Content, size: NSSize, name: String) async throws {
        _ = NSApplication.shared
        let view = NSHostingView(rootView: content.frame(width: size.width))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFrontRegardless()
        defer { window.close() }
        view.frame = NSRect(origin: .zero, size: size)
        try await Task.sleep(for: .milliseconds(300))
        view.layoutSubtreeIfNeeded()
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1000)
        if let path = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let directory = URL(fileURLWithPath: path, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try png.write(to: directory.appendingPathComponent(name))
        }
    }
}

/// Saved-file fixtures exercise management and layout only; they are never used for inference.
private struct CoverageControlsFixture {
    let source: URL
    let depthSlot: ModelSlot
    let normalSlot: ModelSlot

    init() async throws {
        source = FileManager.default.temporaryDirectory.appendingPathComponent("loom-coverage-controls-\(UUID().uuidString)")
            .resolvingSymlinksInPath().standardizedFileURL
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        for category in ["Bricks", "Wood"] {
            for role in [MapRole.diffuse, .depth, .roughness] {
                let url = source.appendingPathComponent("\(category)_\(role.rawValue).png")
                try Self.image(url)
                try ImageMetadataStore.write(.init(pairingKey: category, role: role, category: category), to: url)
            }
        }
        let project = try LoomProject(sourceFolder: source)
        let snapshot = try await project.snapshot()
        let bricks = try #require(snapshot.datasets.first { $0.name == "Bricks" })
        let wood = try #require(snapshot.datasets.first { $0.name == "Wood" })
        depthSlot = ModelSlot(group: .dataset(bricks.id), product: .depth)
        normalSlot = ModelSlot(group: .dataset(wood.id), product: .normal)
        for (id, dataset, map, createdAt) in [
            ("older-depth", bricks, MapRole.depth, 1_700_000_001.0),
            ("latest-depth", bricks, MapRole.depth, 1_700_000_002.0),
            ("wood-roughness", wood, MapRole.roughness, 1_700_000_003.0)
        ] {
            let directory = project.projectFolder.appendingPathComponent("runs/\(id)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let architecture = "loom-factory-material-residual-v1"
            let descriptor: [String: Any] = [
                "schema": "loom-factory-model-v1", "project_id": snapshot.projectID, "kind": "map",
                "dataset_id": dataset.id, "dataset_name": dataset.name, "map": map.rawValue, "status": "completed",
                "architecture": architecture, "architecture_config": ["id": architecture, "width": 8, "blocks": 1],
                "input_name": "rgb", "output_name": "material_map",
                "image_size": 256, "training": ["image_size": 256, "exaggerate_features": false],
                "weights_path": "weights.safetensors", "coreml_path": "model.mlmodel"
            ]
            try JSONSerialization.data(withJSONObject: descriptor).write(to: directory.appendingPathComponent("model.json"))
            try Data([1, 2, 3]).write(to: directory.appendingPathComponent("weights.safetensors"))
            try Data([4, 5, 6]).write(to: directory.appendingPathComponent("model.mlmodel"))
            let artifact = ModelArtifact(id: id, kind: "map", datasetID: dataset.id, datasetName: dataset.name, map: map,
                directory: directory, createdAt: Date(timeIntervalSince1970: createdAt), epochs: 20, loss: 0.05,
                coreMLPath: "model.mlmodel", weightsPath: "weights.safetensors", architecture: architecture)
            try JSONEncoder().encode(artifact).write(to: directory.appendingPathComponent("artifact.json"))
        }
        for (id, status, epochs) in [("cancelled-depth", "cancelled", 19), ("interrupted-depth", "training", 0)] {
            let directory = project.projectFolder.appendingPathComponent("runs/\(id)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let descriptor: [String: Any] = [
                "schema": "loom-factory-checkpoint-v1", "project_id": snapshot.projectID, "kind": "map",
                "dataset_id": bricks.id, "dataset_name": bricks.name, "map": "depth", "status": status,
                "architecture": "loom-factory-material-residual-v1",
                "architecture_config": ["id": "loom-factory-material-residual-v1", "width": 8, "blocks": 1],
                "settings": ["image_size": 32, "epochs": 100], "epochs_completed": epochs,
                "weights_path": "checkpoint.safetensors"
            ]
            try JSONSerialization.data(withJSONObject: descriptor).write(to: directory.appendingPathComponent("checkpoint.json"))
            try Data([7, 8, 9]).write(to: directory.appendingPathComponent("checkpoint.safetensors"))
        }
    }

    func remove() { try? FileManager.default.removeItem(at: source) }

    private static func image(_ url: URL) throws {
        let provider = try #require(CGDataProvider(data: Data(repeating: 140, count: 32 * 32 * 4) as CFData))
        let image = try #require(CGImage(width: 32, height: 32, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: 32 * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }
}
