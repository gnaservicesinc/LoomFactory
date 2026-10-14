import AppKit
import Foundation
import ImageIO
import LoomCore
import SwiftUI
import Testing
import UniformTypeIdentifiers
@testable import LoomFactory

@Suite(.serialized)
struct MaterialBrowserTests {
    @Test
    func selectionExpandsOnlyItsMaterialAndResolutionAndKeepsUserExpansion() throws {
        let source = URL(fileURLWithPath: "/Example Source", isDirectory: true)
        let images = [
            image(id: "first-low", source: source, material: "bark_brown_01", size: 1024),
            image(id: "first-high", source: source, material: "bark_brown_01", size: 2048),
            image(id: "second-low", source: source, material: "bark_willow", size: 1024)
        ]
        let folders = SourceMaterialFolder.group(images: images, sourceFolder: source)
        #expect(folders.count == 2)
        let first = try #require(folders.first { $0.displayName == "Bark Brown 01" })
        let second = try #require(folders.first { $0.displayName == "Bark Willow" })
        let high = try #require(first.resolutions.first { $0.width == 2048 })
        var expansion = MaterialBrowserExpansion()
        expansion.reveal(selectedImageID: "first-high", in: folders)
        #expect(expansion.materials == [first.id])
        #expect(expansion.resolutions == [high.id])
        expansion.reveal(selectedImageID: "second-low", in: folders)
        #expect(expansion.materials == [first.id, second.id])
        expansion.materials.remove(first.id)
        expansion.retainExisting(in: SourceMaterialFolder.group(images: images, sourceFolder: source))
        #expect(expansion.materials == [second.id])
        #expect(expansion.resolutions.contains(high.id))
    }

    @Test
    func pollingRemovesDeletedExpansionWithoutOpeningOtherMaterials() throws {
        let source = URL(fileURLWithPath: "/Example Source", isDirectory: true)
        let first = image(id: "first", source: source, material: "bark_brown_01", size: 1024)
        let second = image(id: "second", source: source, material: "bark_willow", size: 1024)
        let folders = SourceMaterialFolder.group(images: [first, second], sourceFolder: source)
        var expansion = MaterialBrowserExpansion()
        expansion.reveal(selectedImageID: first.id, in: folders)
        expansion.retainExisting(in: SourceMaterialFolder.group(images: [second], sourceFolder: source))
        #expect(expansion.materials.isEmpty)
        #expect(expansion.resolutions.isEmpty)
        expansion.reveal(selectedImageID: "deleted", in: folders)
        #expect(expansion.materials.isEmpty)
    }

    @Test
    func collapsingRowsKeepsSelectionAndRemembersChildExpansion() throws {
        let source = URL(fileURLWithPath: "/Example Source", isDirectory: true)
        let selected = image(id: "first-low", source: source, material: "bark_brown_01", size: 1024)
        let higher = image(id: "first-high", source: source, material: "bark_brown_01", size: 2048)
        let folders = SourceMaterialFolder.group(images: [selected, higher], sourceFolder: source)
        let folder = try #require(folders.first)
        var expansion = MaterialBrowserExpansion()
        #expect(expansion.rows(in: folders).map(\.id) == ["material:" + folder.id])
        expansion.reveal(selectedImageID: selected.id, in: folders)
        let openRows = expansion.rows(in: folders)
        #expect(openRows.count == 4)
        #expect(openRows.map(\.id).contains("image:" + selected.id))
        #expect(!openRows.map(\.id).contains("image:" + higher.id))
        let resolution = try #require(openRows.first { $0.level == 1 })
        #expect(resolution.parentID == "material:" + folder.id)
        expansion.setExpanded(false, for: .material(folder))
        #expect(expansion.rows(in: folders).count == 1)
        expansion.setExpanded(true, for: .material(folder))
        #expect(expansion.rows(in: folders).map(\.id) == openRows.map(\.id))
        expansion.setExpanded(false, for: resolution)
        #expect(expansion.rows(in: folders).count == 3)
        #expect(expansion.rows(in: folders).allSatisfy { $0.level != 2 })
    }

    @Test
    func keyboardNavigationUsesCurrentRowsAndStaleButtonActionsAreIgnored() throws {
        let source = URL(fileURLWithPath: "/Example Source", isDirectory: true)
        let selected = image(id: "first", source: source, material: "bark_brown", size: 1024)
        let folders = SourceMaterialFolder.group(images: [selected], sourceFolder: source)
        var expansion = MaterialBrowserExpansion()
        let material = try #require(expansion.rows(in: folders).first)
        #expect(expansion.move(.right, from: material.id, in: folders)?.id == material.id)
        #expect(expansion.isExpanded(material))
        let movedResolution = expansion.move(.right, from: material.id, in: folders)
        let resolution = try #require(movedResolution)
        #expect(resolution.level == 1)
        #expect(expansion.move(.right, from: resolution.id, in: folders)?.id == resolution.id)
        let movedMap = expansion.move(.down, from: resolution.id, in: folders)
        let map = try #require(movedMap)
        #expect(map.id == "image:" + selected.id)
        #expect(expansion.move(.down, from: map.id, in: folders)?.id == map.id)
        #expect(expansion.move(.left, from: map.id, in: folders)?.id == resolution.id)
        #expect(expansion.move(.left, from: resolution.id, in: folders)?.id == resolution.id)
        #expect(!expansion.isExpanded(resolution))
        #expect(expansion.move(.up, from: resolution.id, in: folders)?.id == material.id)
        let activatedResolution = expansion.activate(resolution, in: folders)
        let activatedMap = expansion.activate(map, in: folders)
        let activatedDeletedMap = expansion.activate(map, in: [])
        #expect(activatedResolution)
        #expect(activatedMap)
        #expect(!activatedDeletedMap)
        #expect(expansion.move(.down, from: map.id, in: []) == nil)
        #expect(expansion.move(.up, from: "deleted", in: folders)?.id == material.id)
    }

    @Test
    func collapsedMaterialSelectionResolvesEveryMapAndResolutionWithoutExpansion() throws {
        let source = URL(fileURLWithPath: "/Example Source", isDirectory: true)
        let images = [image(id: "first-low", source: source, material: "bark_brown", size: 1024),
                      image(id: "first-high", source: source, material: "bark_brown", size: 2048)]
        let folders = SourceMaterialFolder.group(images: images, sourceFolder: source)
        let folder = try #require(folders.first)
        let expansion = MaterialBrowserExpansion()
        let selection = try #require(MaterialBrowserSelection.row(id: "material:" + folder.id, in: folders))
        #expect(selection.images.count == 2)
        #expect(selection.isReviewed)
        #expect(MaterialBrowserSelection.representative(in: selection.images)?.role == .diffuse)
        #expect(expansion.rows(in: folders).count == 1)
        let resolution = try #require(folder.resolutions.last)
        #expect(MaterialBrowserSelection.row(id: "resolution:" + resolution.id, in: folders)?.images.count == 1)
        #expect(MaterialBrowserSelection.row(id: "removed", in: folders) == nil)
    }

    @Test @MainActor
    func collapsedFolderPreviewAndOneActionReviewWorkAcrossHiddenMaps() async throws {
        let fixture = try BrowserFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        for _ in 0..<100 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        let folder = try #require(model.materialFolders.first)
        model.selectedMaterialBrowserRowID = "material:" + folder.id
        #expect(model.selectedMaterialBrowserRow?.images.count == 4)
        model.setMaterialReviewed(imageIDs: [try #require(folder.resolutions.first?.images.first).id], verified: false)
        for _ in 0..<100 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.materialFolders.first { $0.id == folder.id }?.resolutions.flatMap(\.images).allSatisfy { !$0.verified } == true)
        model.fileFilter = .unreviewed
        #expect(model.visibleMaterialFolders.count == 1)
        _ = NSApplication.shared
        let view = NSHostingView(rootView: MaterialSelectionViewer(model: model)
            .padding(16).background(Color(nsColor: .windowBackgroundColor)).preferredColorScheme(.dark))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 800, height: 600),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: .darkAqua)
        window.contentView = view
        window.orderFrontRegardless()
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(250))
        window.layoutIfNeeded()
        view.layoutSubtreeIfNeeded()
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1000)
        if let destination = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let directory = URL(fileURLWithPath: destination, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try png.write(to: directory.appendingPathComponent("material-folder-preview-without-expansion.png"))
        }
        model.setMaterialReviewed(imageIDs: model.selectedMaterialBrowserRow?.images.map(\.id) ?? [], verified: true)
        for _ in 0..<100 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.visibleMaterialFolders.isEmpty)
        #expect(model.materialFolders.allSatisfy { $0.resolutions.flatMap(\.images).allSatisfy(\.verified) })
    }

    @Test @MainActor
    func folderPreviewKeepsItsMaterialWhenTheRepresentativeIsRemovedAndFallsBackWhenEmpty() async throws {
        let fixture = try BrowserFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        for _ in 0..<100 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.errorMessage == nil)
        let firstFolder = try #require(model.materialFolders.first)
        let selectedFolder = try #require(model.materialFolders.last)
        #expect(firstFolder.id != selectedFolder.id)
        let representative = try #require(MaterialBrowserSelection.representative(in: selectedFolder.resolutions.flatMap(\.images)))
        model.selectedMaterialBrowserRowID = "material:" + selectedFolder.id
        model.selectedImageID = representative.id
        try FileManager.default.removeItem(at: representative.url)
        await model.pollFolder()
        #expect(model.selectedMaterialBrowserRowID == "material:" + selectedFolder.id)
        #expect(model.selectedImageID != representative.id)
        #expect(model.selectedImage?.url.deletingLastPathComponent().standardizedFileURL == selectedFolder.url)
        #expect(model.selectedMaterialBrowserRow?.images.count == 3)

        for image in try #require(model.selectedMaterialBrowserRow).images {
            try FileManager.default.removeItem(at: image.url)
        }
        await model.pollFolder()
        #expect(model.selectedMaterialBrowserRowID == "material:" + firstFolder.id)
        #expect(model.selectedImage != nil)
        #expect(model.materialFolders.allSatisfy { $0.id != selectedFolder.id })
    }

    @Test @MainActor
    func thousandMapSearchAndPollingKeepTheUIExecutorResponsive() async throws {
        let fixture = try BrowserFixture(materialCount: 250)
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        for _ in 0..<500 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.errorMessage == nil)
        #expect(model.snapshot?.images.count == 1000)
        _ = NSApplication.shared
        let view = NSHostingView(rootView: MaterialBrowser(model: model))
        let window = NSWindow(contentRect: NSRect(x: -20000, y: -20000, width: 380, height: 700),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFrontRegardless()
        defer { window.close() }
        try await settle(view)
        let ticker = Task { @MainActor () -> Double in
            var last = DispatchTime.now().uptimeNanoseconds
            var longest = 0.0
            for _ in 0..<50 {
                try await Task.sleep(for: .milliseconds(10))
                let now = DispatchTime.now().uptimeNanoseconds
                longest = max(longest, Double(now - last) / 1_000_000_000)
                last = now
            }
            return longest
        }
        await model.pollFolder()
        for index in 0..<12 {
            model.imageSearch = index.isMultiple(of: 2) ? "material 1" : ""
            view.layoutSubtreeIfNeeded()
            try await Task.sleep(for: .milliseconds(5))
        }
        model.imageSearch = ""
        await model.pollFolder()
        let longest = try await ticker.value
        #expect(longest < 0.25, "UI executor stalled for \(longest) seconds during 1,000-map searches/polls")
        #expect(model.visibleImages.count == 1000)
        #expect(!containsTable(view))
        if let destination = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let directory = URL(fileURLWithPath: destination, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let evidence: [String: Any] = ["source_materials": 250, "map_files": 1000,
                "search_changes": 12, "folder_polls": 2, "main_actor_heartbeat_seconds": 0.01,
                "maximum_main_actor_heartbeat_gap_seconds": longest]
            try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
                .write(to: directory.appendingPathComponent("materials-ui-responsiveness.json"))
        }
    }

    @Test @MainActor
    func nativeBrowserSurvivesSelectionFilteringAndFolderPollingWithoutATable() async throws {
        let fixture = try BrowserFixture()
        defer { fixture.remove() }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(fixture.source)
        for _ in 0..<100 {
            if !model.busy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(model.errorMessage == nil)
        _ = try #require(model.snapshot)
        let folder = try #require(model.materialFolders.first)
        let selected = try #require(model.selectedImage)
        let resolution = try #require(folder.resolutions.first { $0.images.contains { $0.id == selected.id } })
        let target = try #require(resolution.images.first { $0.role == .depth })
        _ = NSApplication.shared
        let view = NSHostingView(rootView: MaterialBrowser(model: model))
        let window = NSWindow(contentRect: NSRect(x: -20000, y: -20000, width: 380, height: 700),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFrontRegardless()
        defer { window.close() }
        try await settle(view)
        #expect(!containsTable(view))
        model.selectedImageID = target.id
        try await settle(view)
        #expect(model.selectedImageID == target.id)
        #expect(!containsTable(view))

        model.fileFilter = .needsReview
        try await settle(view)
        #expect(model.selectedImageID == nil)
        #expect(model.visibleMaterialFolders.isEmpty)
        #expect(!containsTable(view))
        model.fileFilter = .all
        try await settle(view)
        #expect(model.selectedImage?.role == .diffuse)
        model.selectedImageID = target.id
        try await settle(view)
        #expect(model.selectedImageID == target.id)

        // Remove the selected row while preserving its material/resolution IDs.
        try FileManager.default.removeItem(at: target.url)
        await model.pollFolder()
        try await settle(view)
        #expect(model.selectedImageID != target.id)
        #expect(model.selectedImage?.role == .diffuse)
        #expect(model.visibleImages.allSatisfy { $0.id != target.id })
        #expect(!containsTable(view))
        try FileManager.default.removeItem(at: folder.url)
        await model.pollFolder()
        try await settle(view)
        #expect(model.visibleMaterialFolders.allSatisfy { $0.id != folder.id })
        #expect(model.selectedImage != nil)
        #expect(!containsTable(view))

        model.imageSearch = "no matching material"
        try await settle(view)
        #expect(model.selectedImageID == nil)
        #expect(!containsTable(view))

        model.imageSearch = ""
        try await settle(view)
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        #expect(png.count > 1000)
        if let destination = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let directory = URL(fileURLWithPath: destination, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try png.write(to: directory.appendingPathComponent("material-browser-after-selection-filter-poll.png"))
        }
    }

    @MainActor private func settle(_ view: NSView) async throws {
        try await Task.sleep(for: .milliseconds(100))
        view.layoutSubtreeIfNeeded()
    }

    @MainActor private func containsTable(_ view: NSView) -> Bool {
        view is NSTableView || view.subviews.contains(where: containsTable)
    }

    private func image(id: String, source: URL, material: String, size: Int) -> SourceImage {
        SourceImage(id: id, url: source.appendingPathComponent(material, isDirectory: true).appendingPathComponent("\(id).png"),
                    displayName: id, role: .diffuse, category: "Bark", verified: true, note: "",
                    pairingKey: "\(material)/\(size)", width: size, height: size)
    }
}

private struct BrowserFixture {
    let source: URL
    init(materialCount: Int = 2) throws {
        source = FileManager.default.temporaryDirectory.appendingPathComponent("loom-browser-\(UUID().uuidString)", isDirectory: true)
        let materials = materialCount == 2 ? ["bark_brown", "bark_willow"] :
            (0..<materialCount).map { "bark_material_\($0)" }
        for material in materials {
            for size in [32, 64] {
                for (suffix, role) in [("diff", MapRole.diffuse), ("disp", MapRole.depth)] {
                    let url = source.appendingPathComponent(material).appendingPathComponent("\(material)_\(suffix)_\(size).png")
                    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                    let provider = try #require(CGDataProvider(data: Data(repeating: 160, count: size * size * 4) as CFData))
                    let image = try #require(CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
                    let writer = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
                    CGImageDestinationAddImage(writer, image, nil)
                    #expect(CGImageDestinationFinalize(writer))
                    try ImageMetadataStore.write(.init(pairingKey: "\(material)/\(size)", role: role, category: "Bark", verified: true), to: url)
                }
            }
        }
    }
    func remove() { try? FileManager.default.removeItem(at: source) }
}
