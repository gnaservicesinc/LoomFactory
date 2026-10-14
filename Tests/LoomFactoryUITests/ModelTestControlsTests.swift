import AppKit
import ImageIO
import LoomCore
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

@Suite(.serialized)
struct ModelTestControlsTests {
    @Test @MainActor
    func oneSavedRunAndAnExternalDiffuseNeedNoTargetOrSelectedTrainingDataset() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        try await wait(model)
        #expect(model.selectedDatasetID.isEmpty)
        #expect(model.snapshot?.images.count == 1)
        #expect(model.snapshot?.images.allSatisfy { $0.role == .diffuse } == true)
        #expect(model.testableModelVersions.count == 1)
        #expect(!model.canCompareModels)
        let version = try #require(model.testableModelVersions.first)
        model.selectedMap = .normal
        #expect(!model.canTestModel)
        model.showModelTesting(version)
        #expect(model.workspaceSection == .testing)
        #expect(model.selectedMap == .normal)
        #expect(model.selectedTestVersion?.map == .depth)
        #expect(model.availableTestWeightSources == [.model, .checkpoint])
        #expect(model.testInputURL == fixture.sourceInput)
        #expect(model.canTestModel)
        let original = try Data(contentsOf: fixture.externalInput)
        model.setModelTestInput(fixture.externalInput)
        #expect(model.canTestModel)
        #expect(model.modelTestIssue == nil)
        #expect(model.snapshot?.images.contains { $0.url == fixture.externalInput } == false)
        #expect(model.selectedDatasetID.isEmpty)
        #expect(try Data(contentsOf: fixture.externalInput) == original)
        try await render(ModelTestView(model: model), size: NSSize(width: 1140, height: 930), name: "model-test-controls.png")

        model.testWeightSource = .checkpoint
        try FileManager.default.removeItem(at: version.directory.appendingPathComponent("checkpoint.safetensors"))
        await model.pollFolder()
        #expect(!model.canTestModel)
        #expect(model.modelTestIssue?.contains("no longer available") == true)
        model.testWeightSource = .model
        #expect(model.canTestModel)
        try FileManager.default.removeItem(at: version.directory)
        await model.pollFolder()
        #expect(model.selectedTestVersion == nil)
        #expect(!model.canTestModel)
        #expect(model.modelTestIssue?.contains("selected run") == true)

        let empty = fixture.root.appendingPathComponent("Another Source")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        model.openFolder(empty)
        try await wait(model)
        #expect(model.testInputURL == nil)
        #expect(model.modelTestResult == nil)
        #expect(model.selectedTestVersionID.isEmpty)
        #expect(model.workspaceSection == .materials)
    }

    @Test @MainActor
    func resultRendersInputAndOutputAtFitAndNativeZoom() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let result = fixture.result
        for zoom in ComparisonImageZoom.allCases {
            let content = ModelTestResultView(result: result, imageZoom: .constant(zoom),
                                             saveOutput: {}, showOutput: {}, showFolder: {})
            try await render(content, size: NSSize(width: 1140, height: 740), name: "model-test-result-\(zoom.rawValue).png")
        }
        let image = try #require(CGImageSourceCreateWithURL(result.outputURL as CFURL, nil))
        let properties = try #require(CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [CFString: Any])
        #expect(properties[kCGImagePropertyPixelWidth] as? Int == result.width)
        #expect(properties[kCGImagePropertyPixelHeight] as? Int == result.height)
    }

    @Test
    func savingCopiesOutputBytesAndProtectsOriginalImagesThroughAliases() async throws {
        let fixture = try await ModelTestControlsFixture()
        defer { fixture.remove() }
        let result = fixture.result
        let originals = [fixture.externalInput, fixture.sourceInput]
        let before = try originals.map { try Data(contentsOf: $0) }
        let alias = fixture.root.appendingPathComponent("original-alias.png")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: fixture.externalInput)
        for path in originals + [alias] {
            #expect(throws: (any Error).self) {
                try ModelTestOutputExport.save(result, to: path, sourceImageURLs: [fixture.sourceInput])
            }
        }
        for (index, url) in originals.enumerated() { #expect(try Data(contentsOf: url) == before[index]) }
        let saved = fixture.root.appendingPathComponent("saved-depth.png")
        try ModelTestOutputExport.save(result, to: saved, sourceImageURLs: [fixture.sourceInput])
        #expect(try Data(contentsOf: saved) == Data(contentsOf: result.outputURL))
        try ModelTestOutputExport.save(result, to: result.outputURL, sourceImageURLs: [fixture.sourceInput])
    }

    @MainActor private func wait(_ model: FactoryModel) async throws {
        for _ in 0..<250 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!model.busy)
        #expect(model.errorMessage == nil)
        _ = try #require(model.snapshot)
    }

    @MainActor private func render<Content: View>(_ content: Content, size: NSSize, name: String) async throws {
        _ = NSApplication.shared
        let view = NSHostingView(rootView: content.padding(20).frame(width: size.width))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        view.frame = NSRect(origin: .zero, size: size)
        window.orderFrontRegardless()
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(800))
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

private struct ModelTestControlsFixture {
    let root: URL
    let source: URL
    let sourceInput: URL
    let externalInput: URL
    let output: URL

    var result: ModelTestResult {
        .init(directory: output.deletingLastPathComponent(), inputURL: externalInput, outputURL: output,
              runID: "one-saved-run", source: .checkpoint, map: .depth, width: 384, height: 256,
              tileSize: 256, tileCount: 2,
              note: "Preview layout fixture. The original diffuse and generated map retain their native dimensions.")
    }

    init() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-model-test-ui-" + UUID().uuidString)
            .resolvingSymlinksInPath().standardizedFileURL
        source = root.appendingPathComponent("Source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        sourceInput = source.appendingPathComponent("brick_diffuse.png")
        externalInput = root.appendingPathComponent("unseen_diffuse.png")
        output = root.appendingPathComponent("depth.png")
        try Self.writeImage(to: sourceInput, color: true)
        try Self.writeImage(to: externalInput, color: true)
        try Self.writeImage(to: output, color: false)
        try ImageMetadataStore.write(.init(pairingKey: "brick", role: .diffuse, category: "Bricks"), to: sourceInput)
        let project = try LoomProject(sourceFolder: source)
        let snapshot = try await project.snapshot()
        let dataset = try #require(snapshot.datasets.first)
        let run = project.projectFolder.appendingPathComponent("runs/one-saved-run")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let architecture: [String: Any] = ["id": "loom-factory-material-residual-v1", "width": 8, "blocks": 1]
        let saved: [String: Any] = ["image_size": 256, "exaggerate_features": false]
        let descriptor: [String: Any] = ["schema": "loom-factory-model-v1", "project_id": snapshot.projectID,
            "kind": "map", "dataset_id": dataset.id, "dataset_name": dataset.name, "map": "depth",
            "architecture": "loom-factory-material-residual-v1", "architecture_config": architecture,
            "image_size": 256, "training": saved, "weights_path": "weights.safetensors"]
        try JSONSerialization.data(withJSONObject: descriptor).write(to: run.appendingPathComponent("model.json"))
        var checkpoint = descriptor
        checkpoint["schema"] = "loom-factory-checkpoint-v1"
        checkpoint["settings"] = saved
        checkpoint["epochs_completed"] = 20
        try JSONSerialization.data(withJSONObject: checkpoint).write(to: run.appendingPathComponent("checkpoint.json"))
        let artifact = ModelArtifact(id: "one-saved-run", kind: "map", datasetID: dataset.id, datasetName: dataset.name,
                                     map: .depth, directory: run, epochs: 20, loss: nil, coreMLPath: "model.mlmodel",
                                     weightsPath: "weights.safetensors", architecture: "loom-factory-material-residual-v1")
        try JSONEncoder().encode(artifact).write(to: run.appendingPathComponent("artifact.json"))
        // Selection-only fixtures do not execute these opaque saved files.
        for name in ["checkpoint.safetensors", "weights.safetensors", "model.mlmodel"] {
            try Data([0]).write(to: run.appendingPathComponent(name))
        }
    }

    private static func writeImage(to url: URL, color: Bool) throws {
        let width = 384, height = 256
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let mortar = y % 44 < 5 || (x + (y / 44 % 2) * 45) % 90 < 5
                let i = (y * width + x) * 4
                let level: UInt8 = mortar ? 60 : UInt8(175 + (x + y) % 45)
                pixels[i] = color ? (mortar ? 80 : 170) : level
                pixels[i + 1] = color ? (mortar ? 80 : 85) : level
                pixels[i + 2] = color ? (mortar ? 80 : 60) : level
            }
        }
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let bitmap = try #require(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let writer = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(writer, bitmap, nil)
        #expect(CGImageDestinationFinalize(writer))
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}
