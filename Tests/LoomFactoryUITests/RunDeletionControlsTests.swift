import AppKit
import Foundation
import LoomCore
import SwiftUI
import Testing
@testable import LoomFactory

@Suite(.serialized)
struct RunDeletionControlsTests {
    @Test @MainActor
    func failedAndStoppedRunsExposeIndividualAndWholeProjectDeletionControls() async throws {
        let source = FileManager.default.temporaryDirectory.appendingPathComponent("loom-run-controls-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: source) }
        let project = try LoomProject(sourceFolder: source)
        let projectID = try await project.snapshot().projectID
        for (id, map, status) in [("failed-depth", "depth", "failed"), ("stopped-normal", "normal", "cancelled")] {
            let run = source.appendingPathComponent(".loom-factory/runs/" + id)
            try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
            let request: [String: Any] = ["schema": "loom-factory-training-v1", "project_id": projectID,
                "dataset_id": "bricks", "display_name": "Bricks", "map_kind": map,
                "run_label": id, "image_size": 32, "width": 8, "blocks": 1]
            try JSONSerialization.data(withJSONObject: request).write(to: run.appendingPathComponent("request.json"))
            try JSONSerialization.data(withJSONObject: ["status": status]).write(to: run.appendingPathComponent("progress.json"))
        }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(source)
        for _ in 0..<250 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!model.busy)
        #expect(model.errorMessage == nil)
        #expect(model.hasTrainingRuns)
        #expect(model.trainingRunCount == 2)
        let versions = try #require(model.snapshot).modelVersions
        #expect(Set(versions.map(\.status)) == ["failed", "cancelled"])
        let failed = try #require(versions.first { $0.map == .depth })
        model.selectedMap = .roughness
        #expect(model.selectedModelVersions.isEmpty)

        _ = NSApplication.shared
        let content = VStack(alignment: .leading, spacing: 16) {
            TrainingRunDeletionControl(model: model)
            ModelVersionRow(model: model, version: failed)
        }.padding(20).frame(width: 760)
        let view = NSHostingView(rootView: content)
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 760, height: 320),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        view.frame = window.contentView!.bounds
        window.orderFrontRegardless()
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(200))
        view.layoutSubtreeIfNeeded()
        #expect(model.hasTrainingRuns, "Bulk cleanup remains available when the selected map has no runs")
        #expect(failed.checkpoint == nil, "Failed runs expose individual cleanup without a checkpoint")
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1000)
        if let path = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let directory = URL(fileURLWithPath: path, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try png.write(to: directory.appendingPathComponent("training-run-deletion-controls.png"))
        }
    }

}
