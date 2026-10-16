import AppKit
import LoomCore
import SwiftUI
import Testing
@testable import LoomFactory

@Suite(.serialized)
struct PhotoEnhancerViewTests {
    @Test @MainActor func trainingControlsRenderWithoutStartingGPUWork() async throws {
        let model = FactoryModel(rememberSourceFolder: false)
        _ = NSApplication.shared
        let hosting = NSHostingView(rootView: PhotoEnhancerSheet(model: model).environment(\.colorScheme,.light))
        let window = NSWindow(contentRect:NSRect(x:0,y:0,width:820,height:820),styleMask:[.borderless],backing:.buffered,defer:false)
        window.isReleasedWhenClosed = false; window.contentView = hosting
        window.appearance = NSAppearance(named:.aqua); window.orderFrontRegardless()
        try await Task.sleep(for:.milliseconds(250))
        defer { window.close() }
        hosting.frame = NSRect(x:0,y:0,width:820,height:820); hosting.layoutSubtreeIfNeeded()
        let bitmap = try #require(hosting.bitmapImageRepForCachingDisplay(in:hosting.bounds))
        hosting.cacheDisplay(in:hosting.bounds,to:bitmap)
        let png = try #require(bitmap.representation(using:.png,properties:[:]))
        #expect(png.count > 1000); #expect(!model.busy)
        if let folder = ProcessInfo.processInfo.environment["LOOM_FACTORY_UI_RENDER_OUTPUT"] {
            let root = URL(fileURLWithPath:folder); try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
            try png.write(to:root.appendingPathComponent("photo-enhancer-training.png"))
        }
    }
}
