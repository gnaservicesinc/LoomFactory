import Foundation
import LoomCore

/// Launches only a new background Blender child. Never touches a running GUI.
final class BlenderExportProcess: @unchecked Sendable {
    private let lock = NSLock()
    private let process = Process()
    private var cancelled = false

    static var installedBlender: URL? {
        ["/Applications/Blender.app", NSHomeDirectory() + "/Applications/Blender.app"]
            .map { URL(fileURLWithPath: $0).appendingPathComponent("Contents/MacOS/Blender") }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    func cancel() {
        lock.lock(); defer { lock.unlock() }
        cancelled = true
        if process.isRunning { process.terminate() }
    }

    func run(blender: URL, manifest: URL) throws -> URL {
        let root = manifest.deletingLastPathComponent()
        let log = root.appendingPathComponent("blender-export.log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log)
        defer { try? handle.close() }
        try lock.withLock {
            if cancelled { throw CancellationError() }
            process.executableURL = blender
            process.arguments = ["--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
                "--python", root.appendingPathComponent("BlenderExport.py").path, "--", manifest.path]
            process.standardOutput = handle; process.standardError = handle
            try process.run()
        }
        process.waitUntilExit()
        if lock.withLock({ cancelled }) { throw CancellationError() }
        let output = root.appendingPathComponent("Material Review.blend")
        guard process.terminationStatus == 0, FileManager.default.fileExists(atPath: output.path),
              FileManager.default.fileExists(atPath: root.appendingPathComponent("export-receipt.json").path) else {
            let detail = (try? String(contentsOf: log, encoding: .utf8)).map { String($0.suffix(1600)) } ?? ""
            throw LoomError("Blender export did not finish. Exact maps and the retry script remain in \(root.path).\n\(detail)")
        }
        return output
    }
}
