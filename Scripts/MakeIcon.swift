import AppKit
import Foundation

let destination = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let iconset = destination.appendingPathComponent("LoomFactory.iconset", isDirectory: true)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)

for (size, filename) in [
    (16, "icon_16x16.png"), (32, "icon_16x16@2x.png"),
    (32, "icon_32x32.png"), (64, "icon_32x32@2x.png"),
    (128, "icon_128x128.png"), (256, "icon_128x128@2x.png"),
    (256, "icon_256x256.png"), (512, "icon_256x256@2x.png"),
    (512, "icon_512x512.png"), (1024, "icon_512x512@2x.png")
] {
    let image = NSImage(size: NSSize(width: size, height: size))
    image.lockFocus()
    let scale = CGFloat(size) / 1024
    let inset = CGFloat(40) * scale
    let shape = NSBezierPath(roundedRect: NSRect(x: inset, y: inset, width: CGFloat(size) - 2 * inset, height: CGFloat(size) - 2 * inset), xRadius: 210 * scale, yRadius: 210 * scale)
    let gradient = NSGradient(starting: NSColor(srgbRed: 0.21, green: 0.33, blue: 0.69, alpha: 1), ending: NSColor(srgbRed: 0.46, green: 0.69, blue: 0.97, alpha: 1))!
    gradient.draw(in: shape, angle: 70)
    for row in 0..<3 {
        for column in 0..<3 {
            NSColor.white.withAlphaComponent(row == column ? 1 : 0.58).setFill()
            NSBezierPath(roundedRect: NSRect(x: CGFloat(278 + column * 172) * scale, y: CGFloat(278 + row * 172) * scale, width: 124 * scale, height: 124 * scale), xRadius: 26 * scale, yRadius: 26 * scale).fill()
        }
    }
    image.unlockFocus()
    guard let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:]) else {
        fatalError("Could not render the Loom Factory icon")
    }
    try png.write(to: iconset.appendingPathComponent(filename))
}
let task = Process()
task.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
task.arguments = ["-c", "icns", iconset.path, "-o", destination.appendingPathComponent("LoomFactory.icns").path]
try task.run()
task.waitUntilExit()
guard task.terminationStatus == 0 else { fatalError("iconutil could not create the application icon") }
try FileManager.default.removeItem(at: iconset)
