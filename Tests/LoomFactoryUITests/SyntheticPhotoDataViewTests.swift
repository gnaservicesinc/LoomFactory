import AppKit
import CoreGraphics
import ImageIO
import LoomCore
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

@Suite(.serialized)
struct SyntheticPhotoDataViewTests {
    @Test @MainActor func controlsRenderAndModelActionUsesSavedRecipe() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let material = root.appendingPathComponent("Paper")
        try FileManager.default.createDirectory(at: material, withIntermediateDirectories: true)
        let imageURL = material.appendingPathComponent("color.png")
        let context = try #require(CGContext(data: nil, width: 1024, height: 1024, bitsPerComponent: 8,
            bytesPerRow: 4096, space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.setFillColor(CGColor(gray: 0.4, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: 1024, height: 1024))
        let destination = try #require(CGImageDestinationCreateWithURL(imageURL as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try #require(context.makeImage()), nil)
        #expect(CGImageDestinationFinalize(destination))
        try ImageMetadataStore.write(.init(pairingKey: "Paper", role: .diffuse, category: "Paper"), to: imageURL)
        let before = try Data(contentsOf: imageURL)
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(root)
        for _ in 0..<200 { if !model.busy { break }; try await Task.sleep(for: .milliseconds(20)) }
        _ = try #require(model.snapshot)
        _ = NSApplication.shared
        let hosting = NSHostingView(rootView: SyntheticPhotoDataSheet(model: model).environment(\.colorScheme, .light))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 800, height: 780),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false; window.contentView = hosting
        window.appearance = NSAppearance(named: .aqua)
        hosting.frame = NSRect(x: 0, y: 0, width: 800, height: 780)
        window.orderFrontRegardless()
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(250))
        hosting.layoutSubtreeIfNeeded()
        let bitmap = try #require(hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds))
        hosting.cacheDisplay(in: hosting.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1000)
        if let folder = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let url = URL(fileURLWithPath: folder)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            try png.write(to: url.appendingPathComponent("synthetic-photo-controls.png"))
        }
        var recipe = SyntheticPhotoRecipe(); recipe.materialPercentage = 100; recipe.variantsPerCrop = 1
        recipe.blur = false; recipe.noise = false; recipe.lighting = false; recipe.perspective = false
        model.generateSyntheticPhotoData(recipe)
        for _ in 0..<500 { if !model.busy { break }; try await Task.sleep(for: .milliseconds(20)) }
        #expect(!model.busy)
        #expect(model.errorMessage == nil)
        #expect(model.savedSyntheticPhotoRecipe() == recipe)
        let output = try #require(model.lastSyntheticPhotoDirectory)
        #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("manifest.json").path))
        #expect(try Data(contentsOf: imageURL) == before)
        let otherSource = root.appendingPathComponent("Other Source", isDirectory: true)
        try FileManager.default.createDirectory(at: otherSource, withIntermediateDirectories: true)
        model.openFolder(otherSource)
        #expect(model.lastSyntheticPhotoDirectory == nil)
        #expect(model.lastPhotoEnhancerDirectory == nil)
        for _ in 0..<100 { if !model.busy { break }; try await Task.sleep(for: .milliseconds(20)) }
        #expect(!model.busy)
        #expect(model.errorMessage == nil)
    }
}
