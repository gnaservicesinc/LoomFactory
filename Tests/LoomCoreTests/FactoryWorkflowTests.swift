import Foundation
import XCTest
@testable import LoomCore

final class FactoryWorkflowTests: XCTestCase {
    func testFreshActorWorkflowTrainsMapImportsClassifierAndExportsOneNativePackage() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let realInput = environment["LOOM_FACTORY_TEST_DIFFUSE"],
              let realTarget = environment["LOOM_FACTORY_TEST_DEPTH"], let nativeClassifier = environment["LOOM_FACTORY_TEST_CLASSIFIER"] else {
            throw XCTSkip("Set diffuse/depth fixtures and an externally trained Bricks classifier to execute package integration.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("factory-workflow-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let input = root.appendingPathComponent("brick_diff.png"), depth = root.appendingPathComponent("brick_disp.png")
        try FileManager.default.copyItem(at: URL(fileURLWithPath: realInput), to: input)
        try FileManager.default.copyItem(at: URL(fileURLWithPath: realTarget), to: depth)
        let project = try LoomProject(sourceFolder: root)
        let current = try await project.snapshot()
        let diffuse = try XCTUnwrap(current.images.first { $0.role == .diffuse })
        try await project.updateImage(id: diffuse.id, category: "Bricks", verified: false, note: "Integration fixture")
        let settings = TrainingSettings(imageSize: 64, epochs: 1, batchSize: 1, shuffleEachEpoch: false,
                                        exaggerateFeatures: true, modelWidth: 8)
        let map = try await project.trainMap(datasetID: ImageMetadataStore.categoryID("Bricks"), map: .depth,
                                            settings: settings, progress: { _ in })
        let classifier = try await project.importClassifier(from: URL(fileURLWithPath: nativeClassifier))
        XCTAssertEqual(map.datasetName, "Bricks")
        XCTAssertNotNil(map.coreMLPath)
        XCTAssertEqual(classifier.kind, "classifier")
        XCTAssertNil(classifier.weightsPath)
        let newFolder = root.appendingPathComponent("New Brick", isDirectory: true)
        try FileManager.default.createDirectory(at: newFolder, withIntermediateDirectories: true)
        let untagged = newFolder.appendingPathComponent("new_brick_diff.png")
        try FileManager.default.copyItem(at: URL(fileURLWithPath: realInput), to: untagged)
        _ = try await project.snapshot()
        try await project.categorize(using: classifier.directory.appendingPathComponent(classifier.coreMLPath!), progress: { _, _ in })
        let categorized = try await project.snapshot()
        let predicted = try XCTUnwrap(categorized.images.first { $0.url.lastPathComponent == untagged.lastPathComponent })
        XCTAssertEqual(predicted.category, "Bricks")
        XCTAssertFalse(predicted.verified)
        let archive = try await project.exportPackage(name: "Integration Texture Loom", version: "1.0.0")
        XCTAssertTrue(archive.path.hasPrefix(current.sourceURL.path + "/.loom-factory/exports/"))
        let extracted = root.appendingPathComponent(".loom-factory/unpacked")
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = ["-x", "-k", archive.path, extracted.path]
        try process.run(); process.waitUntilExit(); XCTAssertEqual(process.terminationStatus, 0)
        let data = try Data(contentsOf: extracted.appendingPathComponent("manifest.json"))
        let manifest = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(manifest["schema"] as? String, "texture-loom.package.v1")
        XCTAssertEqual(manifest["id"] as? String, current.projectID)
        let categories = try XCTUnwrap(manifest["categories"] as? [[String: Any]])
        XCTAssertEqual(categories.count, 1)
        XCTAssertEqual(categories[0]["id"] as? String, ImageMetadataStore.categoryID("Bricks"))
        let files = try XCTUnwrap(manifest["files"] as? [[String: String]])
        XCTAssertTrue(files.contains { $0["path"]?.hasSuffix(".safetensors") == true })
        XCTAssertTrue(files.contains { $0["path"]?.hasSuffix(".mlmodel") == true })
        XCTAssertEqual(try Data(contentsOf: input), try Data(contentsOf: URL(fileURLWithPath: realInput)))
        XCTAssertEqual(try Data(contentsOf: depth), try Data(contentsOf: URL(fileURLWithPath: realTarget)))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent(".loom-factory/runs").path))
        let final = try await project.snapshot()
        XCTAssertEqual(final.artifacts.count, 2)
        if let validationOutput = environment["LOOM_FACTORY_WORKFLOW_OUTPUT"] {
            let output = URL(fileURLWithPath: validationOutput, isDirectory: true)
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            let saved = output.appendingPathComponent("IntegrationTextureLoom.textureloom")
            if FileManager.default.fileExists(atPath: saved.path) { try FileManager.default.removeItem(at: saved) }
            try FileManager.default.copyItem(at: archive, to: saved)
            try data.write(to: output.appendingPathComponent("workflow-manifest.json"))
        }
    }
}
