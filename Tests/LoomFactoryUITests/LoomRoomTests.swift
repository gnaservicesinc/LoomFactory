import AppKit
import ImageIO
import LoomCore
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

@Suite(.serialized)
struct LoomRoomTests {
    @Test @MainActor
    func nativeRoomRendersAtMinimumAndDefaultSizesWithRealProjectState() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, fixture.source)
        #expect(model.snapshot?.images.count == 2)
        #expect(model.snapshot?.datasets.count == 1)
        model.selectedDatasetID = try #require(model.snapshot?.datasets.first?.id)
        model.settings.training.imageSize = 256
        #expect(model.canTrain)
        #expect(model.eligiblePairCount == 1)
        for size in [NSSize(width: 1200, height: 800), NSSize(width: 1380, height: 980)] {
            try await render(model: model, size: size, name: "loom-room-\(Int(size.width))x\(Int(size.height))-project.png")
        }
        let empty = FactoryModel(rememberSourceFolder: false)
        try await render(model: empty, size: NSSize(width: 1200, height: 800), name: "loom-room-1200x800-start.png")
    }

    @Test @MainActor
    func changingSourceFolderClearsDatasetSelectionAndRestoresFolderSettings() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let second = fixture.root.appendingPathComponent("Second Source", isDirectory: true)
        try FileManager.default.createDirectory(at: second, withIntermediateDirectories: true)
        let firstProject = try LoomProject(sourceFolder: fixture.source)
        try await firstProject.updateSettings(ProjectSettings(modelName: "Bricks Loom", training: .init(imageSize: 256, shuffleEachEpoch: false)))
        let secondProject = try LoomProject(sourceFolder: second)
        try await secondProject.updateSettings(ProjectSettings(modelName: "Wood Loom", training: .init(imageSize: 1024, shuffleEachEpoch: true)))
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, fixture.source)
        model.selectedDatasetID = try #require(model.snapshot?.datasets.first?.id)
        #expect(model.settings.modelName == "Bricks Loom")
        #expect(model.settings.training.imageSize == 256)
        model.openFolder(second)
        #expect(model.snapshot == nil)
        #expect(model.selectedDatasetID.isEmpty)
        #expect(model.selectedImageID == nil)
        #expect(!model.canTrain)
        try await finishLoading(model)
        #expect(model.settings.modelName == "Wood Loom")
        #expect(model.settings.training.imageSize == 1024)
        #expect(model.snapshot?.images.isEmpty == true)
        try await load(model, fixture.source)
        #expect(model.settings.modelName == "Bricks Loom")
        #expect(model.settings.training.imageSize == 256)
        #expect(model.settings.training.shuffleEachEpoch == false)
    }

    @Test @MainActor
    func idleFolderPollRemovesDeletedImagesAndUpdatesCurrentTags() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, fixture.source)
        model.selectedDatasetID = try #require(model.snapshot?.datasets.first?.id)
        model.settings.training.imageSize = 256
        #expect(model.eligiblePairCount == 1)
        let image = try #require(model.snapshot?.images.first { $0.role == .depth })
        var tag = try #require(try ImageMetadataStore.read(image.url))
        tag.category = "Wood"
        try ImageMetadataStore.write(tag, to: image.url)
        await model.pollFolder()
        #expect(model.eligiblePairCount == 0)
        #expect(!model.canTrain)
        #expect(model.snapshot?.datasets.count == 2)
        try FileManager.default.removeItem(at: image.url)
        await model.pollFolder()
        #expect(model.snapshot?.images.count == 1)
        #expect(model.snapshot?.images.contains { $0.id == image.id } == false)
    }

    @Test @MainActor
    func unavailableFolderClearsTrainingMembershipAndRestoresItWhenTheFolderReturns() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, fixture.source)
        model.selectedDatasetID = try #require(model.snapshot?.datasets.first?.id)
        model.settings.training.imageSize = 256
        #expect(model.canTrain)
        let moved = fixture.root.appendingPathComponent("Temporarily Moved", isDirectory: true)
        try FileManager.default.moveItem(at: fixture.source, to: moved)
        await model.pollFolder()
        #expect(model.snapshot == nil)
        #expect(!model.canTrain)
        #expect(model.sourceIssue != nil)
        #expect(model.errorMessage == nil)
        await model.pollFolder()
        #expect(model.errorMessage == nil)
        try FileManager.default.moveItem(at: moved, to: fixture.source)
        await model.pollFolder()
        #expect(model.sourceIssue == nil)
        #expect(model.snapshot?.images.count == 2)
        #expect(model.canTrain)
    }

    @Test @MainActor
    func manualReviewSavesNormalConventionOnTheActualImage() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        try await load(model, fixture.source)
        let image = try #require(model.snapshot?.images.first { $0.role == .depth })
        model.review(image: image, category: "Bricks", role: .normal, normalConvention: "directx", verified: true, note: "Reviewed normal")
        try await finishLoading(model)
        let refreshed = try #require(model.snapshot?.images.first { $0.id == image.id })
        #expect(refreshed.role == .normal)
        #expect(refreshed.normalConvention == "directx")
        #expect(refreshed.note == "Reviewed normal")
        let tag = try #require(try ImageMetadataStore.read(image.url))
        #expect(tag.normalConvention == "directx")
        #expect(tag.verified)
    }

    @MainActor private func load(_ model: FactoryModel, _ url: URL) async throws {
        model.openFolder(url)
        try await finishLoading(model)
    }

    @MainActor private func finishLoading(_ model: FactoryModel) async throws {
        for _ in 0..<100 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.errorMessage == nil)
        _ = try #require(model.snapshot)
        #expect(!model.busy)
    }

    @MainActor private func render(model: FactoryModel, size: NSSize, name: String) async throws {
        _ = NSApplication.shared
        let view = NSHostingView(rootView: LoomRoomView(model: model))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        view.frame = NSRect(origin: .zero, size: size)
        window.orderFrontRegardless()
        try await Task.sleep(for: .milliseconds(200))
        window.layoutIfNeeded()
        view.layoutSubtreeIfNeeded()
        #expect(view.frame.width == size.width)
        #expect(view.frame.height == size.height)
        guard let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
            Issue.record("The native Loom Room view could not render")
            window.close()
            return
        }
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1000)
        if let destination = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let directory = URL(fileURLWithPath: destination, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try png.write(to: directory.appendingPathComponent(name))
        }
        window.close()
    }
}

private struct Fixture {
    let root: URL
    let source: URL
    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-factory-ui-\(UUID().uuidString)", isDirectory: true)
        source = root.appendingPathComponent("Source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        for (name, role) in [("brick_diff.png", MapRole.diffuse), ("brick_disp.png", MapRole.depth)] {
            let url = source.appendingPathComponent(name)
            var pixels = [UInt8](repeating: 255, count: 256 * 256 * 4)
            for y in 0..<256 {
                for x in 0..<256 {
                    let joint = y % 32 < 3 || (x + (y / 32 % 2) * 32) % 64 < 3
                    let offset = (y * 256 + x) * 4
                    if role == .diffuse {
                        pixels[offset] = joint ? 70 : UInt8(160 + x % 18)
                        pixels[offset + 1] = joint ? 75 : UInt8(85 + y % 10)
                        pixels[offset + 2] = joint ? 80 : 64
                    } else {
                        let level: UInt8 = joint ? 60 : 185
                        pixels[offset] = level; pixels[offset + 1] = level; pixels[offset + 2] = level
                    }
                }
            }
            let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
            let image = try #require(CGImage(width: 256, height: 256, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: 256 * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
            let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            #expect(CGImageDestinationFinalize(destination))
            try ImageMetadataStore.write(ImageMetadata(pairingKey: "brick", role: role, category: "Bricks", verified: true), to: url)
        }
    }
    func remove() { try? FileManager.default.removeItem(at: root) }
}
