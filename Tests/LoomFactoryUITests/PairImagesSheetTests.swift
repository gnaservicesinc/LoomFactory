import AppKit
import Foundation
import ImageIO
import LoomCore
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

@Suite(.serialized)
struct PairImagesSheetTests {
    @Test
    func checkboxesUseCurrentMembershipAndForgetDeletedSelections() {
        let diffuse = image("diffuse", role: .diffuse)
        let depth = image("depth", role: .depth)
        let added = image("added", role: .roughness)
        var selection = PairImagesSelection()
        selection.select(diffuse.id, isSelected: true, in: [diffuse, depth])
        selection.select(depth.id, isSelected: true, in: [diffuse, depth])
        #expect(selection.currentIDs(in: [diffuse, depth]) == [depth.id, diffuse.id])
        // Submission checks membership immediately, before a UI update prunes it.
        #expect(selection.currentIDs(in: [diffuse]) == [diffuse.id])
        selection.retainExisting(in: [diffuse, added])
        #expect(selection.selectedImageIDs == [diffuse.id])
        selection.select(depth.id, isSelected: true, in: [diffuse, added])
        #expect(selection.selectedImageIDs == [diffuse.id])
        selection.retainExisting(in: [diffuse, depth, added])
        #expect(selection.selectedImageIDs == [diffuse.id])
        selection.select(diffuse.id, isSelected: false, in: [diffuse, depth, added])
        #expect(selection.currentIDs(in: [diffuse, depth, added]).isEmpty)
    }

    @Test @MainActor
    func nativePairingSheetRendersCheckboxesAndFolderPollingWithoutATable() async throws {
        let fixture = try PairingFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        for _ in 0..<100 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.errorMessage == nil)
        let snapshot = try #require(model.snapshot)
        #expect(snapshot.images.count == 2)
        let depth = try #require(snapshot.images.first { $0.role == .depth })
        _ = NSApplication.shared
        let view = NSHostingView(rootView: PairImagesSheet(model: model, selectedImageIDs: Set(snapshot.images.map(\.id))))
        let window = NSWindow(contentRect: NSRect(x: -20000, y: -20000, width: 680, height: 540),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFrontRegardless()
        defer { window.close() }
        try await render(view, name: "pair-images-checkboxes-before-poll.png")
        #expect(!containsTable(view))
        try FileManager.default.removeItem(at: depth.url)
        await model.pollFolder()
        #expect(model.snapshot?.images.count == 1)
        try await render(view, name: "pair-images-checkboxes-after-poll.png")
        #expect(!containsTable(view))
        let diffuse = try #require(model.snapshot?.images.first)
        try FileManager.default.removeItem(at: diffuse.url)
        await model.pollFolder()
        #expect(model.snapshot?.images.isEmpty == true)
        try await render(view, name: "pair-images-checkboxes-empty.png")
        #expect(!containsTable(view))
    }

    private func image(_ id: String, role: MapRole) -> SourceImage {
        SourceImage(id: id, url: URL(fileURLWithPath: "/Example Source/\(id).png"),
                    displayName: id, role: role, category: "Bark", verified: true, note: "",
                    pairingKey: "material", width: 32, height: 32)
    }

    @MainActor private func containsTable(_ view: NSView) -> Bool {
        view is NSTableView || view.subviews.contains(where: containsTable)
    }

    @MainActor private func render(_ view: NSView, name: String) async throws {
        try await Task.sleep(for: .milliseconds(100))
        view.layoutSubtreeIfNeeded()
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1000)
        if let destination = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let directory = URL(fileURLWithPath: destination, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try png.write(to: directory.appendingPathComponent(name))
        }
    }
}

private struct PairingFixture {
    let source: URL
    init() throws {
        source = FileManager.default.temporaryDirectory.appendingPathComponent("loom-pairing-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        for (name, role) in [("diffuse", MapRole.diffuse), ("depth", MapRole.depth)] {
            let url = source.appendingPathComponent("\(name).png")
            let size = 32
            let provider = try #require(CGDataProvider(data: Data(repeating: 160, count: size * size * 4) as CFData))
            let image = try #require(CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
            let writer = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(writer, image, nil)
            #expect(CGImageDestinationFinalize(writer))
            try ImageMetadataStore.write(.init(pairingKey: "material", role: role, category: "Bark", verified: true), to: url)
        }
    }
    func remove() { try? FileManager.default.removeItem(at: source) }
}
