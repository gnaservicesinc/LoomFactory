import CryptoKit
import Foundation
import XCTest
@testable import LoomCore

final class CategorizerPerformanceTests: XCTestCase {
    func testRealSourceFolderUsesOneNativePredictionPerMaterialAndNoneOnResume() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let sourcePath = environment["LOOM_FACTORY_CATEGORIZER_SOURCE"],
              let modelPath = environment["LOOM_FACTORY_CATEGORIZER_MODEL"] else {
            throw XCTSkip("Set a real source folder and native classifier to benchmark categorization on disposable APFS clones.")
        }
        let source = URL(fileURLWithPath: sourcePath, isDirectory: true)
        let originalTags = try metadataFingerprint(source)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-categorizer-replay-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let copy = Process(); copy.executableURL = URL(fileURLWithPath: "/bin/cp")
        copy.arguments = ["-cR", source.path, root.path]
        try copy.run(); copy.waitUntilExit()
        XCTAssertEqual(copy.terminationStatus, 0)
        // Project outputs are irrelevant to this corpus replay. Source images
        // are independent APFS clones, never symlinks to the user's originals.
        let storage = root.appendingPathComponent(".loom-factory")
        if FileManager.default.fileExists(atPath: storage.path) { try FileManager.default.removeItem(at: storage) }
        let project = try LoomProject(sourceFolder: root)
        let initial = try await project.snapshot()
        for image in initial.images {
            guard try image.url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true else {
                throw LoomError("This performance fixture requires independent source files, not imported links.")
            }
            var metadata = try XCTUnwrap(ImageMetadataStore.read(image.url))
            metadata.category = nil; metadata.verified = false
            try ImageMetadataStore.write(metadata, to: image.url)
        }
        let expectedFolders = Set(initial.images.filter { $0.role == .diffuse }.map { $0.url.deletingLastPathComponent() }).count
        let started = Date()
        let summary = try await project.categorize(using: URL(fileURLWithPath: modelPath), progress: { _, _ in })
        let seconds = Date().timeIntervalSince(started)
        XCTAssertEqual(summary.predictionCount, expectedFolders)
        XCTAssertEqual(summary.folderCount, expectedFolders)
        XCTAssertEqual(summary.reusedFolderCount, 0)
        XCTAssertTrue(summary.conflictingFolders.isEmpty)
        let categorized = try await project.snapshot()
        for (_, members) in Dictionary(grouping: categorized.images, by: { $0.url.deletingLastPathComponent() }) {
            XCTAssertEqual(Set(members.compactMap(\.category)).count, 1)
            XCTAssertTrue(members.allSatisfy { $0.category != nil && !$0.verified })
        }
        let resumedAt = Date()
        let resumed = try await project.categorize(using: URL(fileURLWithPath: modelPath), progress: { _, _ in })
        let resumeSeconds = Date().timeIntervalSince(resumedAt)
        XCTAssertEqual(resumed.predictionCount, 0)
        XCTAssertEqual(resumed.folderCount, 0)
        XCTAssertEqual(originalTags, try metadataFingerprint(source))
        let record: [String: Any] = ["source": sourcePath, "source_image_count": initial.images.count,
            "map_roles": Dictionary(grouping: initial.images, by: { $0.role.rawValue }).mapValues(\.count),
            "browser_material_folders": SourceMaterialFolder.group(images: initial.images, sourceFolder: root).count,
            "diffuse_image_count": initial.images.filter { $0.role == .diffuse }.count,
            "material_folder_count": expectedFolders, "prediction_count": summary.predictionCount,
            "seconds": seconds, "resume_prediction_count": resumed.predictionCount, "resume_seconds": resumeSeconds,
            "source_metadata_unchanged": true, "originals_used_via_independent_apfs_clones": true]
        let data = try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: data, as: UTF8.self))
        if let output = environment["LOOM_FACTORY_CATEGORIZER_PROOF"] { try data.write(to: URL(fileURLWithPath: output)) }
    }

    private func metadataFingerprint(_ folder: URL) throws -> String {
        let files = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])!
        var records: [(String, ImageMetadata)] = []
        while let file = files.nextObject() as? URL {
            guard try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true,
                  let tag = try ImageMetadataStore.read(file) else { continue }
            records.append((file.path, tag))
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var hash = SHA256()
        for (path, tag) in records.sorted(by: { $0.0 < $1.0 }) {
            hash.update(data: Data(path.utf8)); hash.update(data: try encoder.encode(tag))
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
