import AppKit
import ImageIO
import SwiftUI

private final class FactoryBitmapBox: @unchecked Sendable {
    let image: NSImage
    init(_ image: NSImage) { self.image = image }
}

struct FactoryImage: View {
    let url: URL
    let revision: String
    var maximumPixels = 1400
    @State private var image: NSImage?
    @State private var couldNotRead = false

    var body: some View {
        ZStack {
            Color.primary.opacity(0.035)
            if let image {
                Image(nsImage: image).resizable().interpolation(.high)
                    .aspectRatio(contentMode: .fit)
            } else if couldNotRead {
                Label("Image could not be read", systemImage: "photo.badge.exclamationmark")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                ProgressView().controlSize(.small)
            }
        }
        .clipped()
        .task(id: "\(url.path)|\(revision)|\(maximumPixels)") {
            image = nil
            couldNotRead = false
            let sourceURL = url
            let pixels = maximumPixels
            let result = await Task.detached(priority: .userInitiated) { () -> FactoryBitmapBox? in
                guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
                      let bitmap = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceCreateThumbnailWithTransform: true,
                        kCGImageSourceThumbnailMaxPixelSize: pixels,
                        kCGImageSourceShouldCacheImmediately: true
                      ] as CFDictionary) else { return nil }
                return FactoryBitmapBox(NSImage(cgImage: bitmap, size: .zero))
            }.value
            guard !Task.isCancelled else { return }
            image = result?.image
            couldNotRead = result == nil
        }
    }
}
