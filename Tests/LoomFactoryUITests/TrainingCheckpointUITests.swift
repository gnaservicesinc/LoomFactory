import AppKit
import CoreGraphics
import ImageIO
import LoomCore
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

@Suite(.serialized)
struct TrainingCheckpointUITests {
    @Test @MainActor
    func saveControlsReportQueuedAndSavedStateAndSourceSwitchPreservesRun() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("checkpoint-ui-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try createPair(root)
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(root)
        for _ in 0..<1_000 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.errorMessage == nil)
        model.selectedDatasetID = try #require(model.snapshot?.datasets.first?.id)
        model.setTrainingStartMode(.fresh)
        model.settings.training = TrainingSettings(imageSize: 32, epochs: 1_000, shuffleEachEpoch: false,
            modelWidth: 8, modelBlocks: 1, validationFraction: 0, checkpointEveryEpochs: 100)
        #expect(model.canTrain)
        model.trainMap()
        #expect(model.busy && model.isTraining)
        try render(model, name: "checkpoint-controls-running.png")
        model.saveCheckpoint()
        #expect(model.checkpointRequestPending)
        #expect(model.checkpointStatus?.contains("Checkpoint queued") == true)
        try render(model, name: "checkpoint-controls-queued.png")
        for _ in 0..<1_000 {
            if !model.checkpointRequestPending { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.busy && model.isTraining)
        #expect(model.checkpointStatus?.contains("Saved step") == true)
        #expect(model.lastCheckpointDirectory != nil)
        try render(model, name: "checkpoint-controls-saved.png")
        model.saveAndStopTraining()
        #expect(model.checkpointRequestPending)
        #expect(model.checkpointStatus?.contains("Save & Stop queued") == true)
        let originalSource = model.sourceFolder
        model.openFolder(root.appendingPathComponent("another-source"))
        #expect(model.sourceFolder == originalSource)
        #expect(model.isTraining && model.busy)
        #expect(model.errorMessage?.contains("Save & Stop") == true)
        try render(model, name: "checkpoint-controls-save-stop-queued.png")
        #expect(await model.saveTrainingAndPrepareForTermination())
        #expect(!model.isTraining && !model.busy)
        #expect(!model.checkpointRequestPending)
        #expect(model.activityTitle.contains("training stopped"))
        let directory = try #require(model.lastCheckpointDirectory)
        let data = try Data(contentsOf: directory.appendingPathComponent("checkpoint.json"))
        let descriptor = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect((descriptor["updates"] as? Int ?? 0) > 0)
        #expect(descriptor["optimizer_path"] as? String != nil)
        #expect(descriptor["resume_cursor"] != nil)
        #expect(descriptor["status"] as? String == "saved_stopped")
        try render(model, name: "checkpoint-controls-stopped.png")
    }

    @MainActor private func render(_ model: FactoryModel, name: String) throws {
        let size = NSSize(width: 460, height: 920)
        let view = NSHostingView(rootView: RunPanel(model: model).frame(width: size.width).padding(14))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        view.frame = NSRect(origin: .zero, size: size)
        view.layoutSubtreeIfNeeded()
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let data = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(data.count > 1_000)
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] ?? "/tmp/loom-checkpoint-controls-ui")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try data.write(to: directory.appendingPathComponent(name))
        window.close()
    }

    private func createPair(_ root: URL) throws {
        for role in [MapRole.diffuse, .depth] {
            let pixels = Data(repeating: 127, count: 64 * 64 * 3)
            let provider = try #require(CGDataProvider(data: pixels as CFData))
            let image = try #require(CGImage(width: 64, height: 64, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: 64 * 3,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
            let file = root.appendingPathComponent("material_\(role.rawValue).png")
            let destination = try #require(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            #expect(CGImageDestinationFinalize(destination))
            var metadata = ImageMetadata(pairingKey: "checkpoint-ui", role: role)
            metadata.category = "Bricks"; metadata.roleWasManuallySelected = true; metadata.pairingWasManuallySelected = true
            try ImageMetadataStore.write(metadata, to: file)
        }
    }
}
