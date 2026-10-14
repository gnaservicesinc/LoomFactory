import Foundation
import ImageIO
import LoomNative
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

@MainActor
final class ModelManagementExportTests: XCTestCase {
    // These are package assembly fixtures. They validate exact assignments and
    // persisted review decisions without training or claiming model quality.

    func testExportKeepsChosenHistoricalModelsAndClassifierDespiteApprovedNewerCandidates() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let classifier = try await fixture.classifier(marker: "chosen classifier")
        try await fixture.project.selectClassifierVersion(runID: classifier.id)
        _ = try await fixture.classifier(marker: "newer unchosen classifier")
        var chosen: [String: ModelArtifact] = [:]
        var newest: [String: ModelArtifact] = [:]
        for category in fixture.categories {
            let historical = try fixture.artifact("old-" + category.id, category: category,
                                                  date: Date(timeIntervalSince1970: 100))
            try await fixture.project.selectModelVersion(runID: historical.id, datasetID: category.id, map: .depth)
            // A newly completed candidate must not replace an explicit choice.
            newest[category.id] = try fixture.artifact("new-" + category.id, category: category,
                                                      date: Date(timeIntervalSince1970: 200))
            chosen[category.id] = historical
        }
        let category = try XCTUnwrap(fixture.categories.first)
        let approvedCandidate = try XCTUnwrap(newest[category.id])
        try await fixture.project.setModelApproval(runID: approvedCandidate.id, approved: true)
        try await fixture.project.setModelRequirement(slot: ModelSlot(group: .dataset(category.id), product: .depth),
                                                      required: false)
        let reopened = try LoomProject(sourceFolder: fixture.root)
        let coverage = try await reopened.modelCoverage()
        let depth = try XCTUnwrap(coverage.first { $0.slot.datasetID == category.id && $0.slot.product == .depth })
        XCTAssertEqual(depth.selectedRunID, chosen[category.id]?.id)
        XCTAssertFalse(depth.isRequired, "An optional requirement must preserve its chosen model")
        XCTAssertEqual(depth.status, .selected, "Approval of an unchosen candidate must not qualify the chosen model")
        XCTAssertEqual(depth.approvedCount, 1)
        XCTAssertEqual(depth.candidates.first(where: { $0.id == approvedCandidate.id })?.isSelected, false)
        let restoredClassifier = try await reopened.selectedClassifierArtifact()
        XCTAssertEqual(restoredClassifier?.id, classifier.id)

        let archive = try await reopened.exportPackage(name: "Managed Choices", version: "1.0.0")
        let unpacked = try fixture.extract(archive)
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(contentsOf: unpacked.appendingPathComponent("manifest.json"))) as? [String: Any])
        XCTAssertEqual(manifest["schema"] as? String, "texture-loom.package.v1")
        let records = try XCTUnwrap(manifest["categories"] as? [[String: Any]])
        XCTAssertEqual(Set(records.compactMap { $0["id"] as? String }), Set(fixture.categories.map(\.id)))
        for record in records {
            let id = try XCTUnwrap(record["id"] as? String)
            let models = try XCTUnwrap(record["models"] as? [[String: Any]])
            XCTAssertEqual(models.compactMap { $0["map"] as? String }, ["depth"],
                           "Management readiness must preserve v1 partial package export")
            let artifact = try XCTUnwrap(chosen[id])
            XCTAssertEqual(try Data(contentsOf: unpacked.appendingPathComponent("models/\(id)/depth/model.safetensors")),
                           try Data(contentsOf: artifact.directory.appendingPathComponent("model.safetensors")))
        }
        let classifierDescriptor = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(contentsOf: unpacked.appendingPathComponent("classifier/model.json"))) as? [String: Any])
        XCTAssertEqual(classifierDescriptor["assembly_fixture_marker"] as? String, "chosen classifier")
    }

    func testMissingChosenMapDoesNotPromoteNewerCandidateOrExportIt() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        _ = try await fixture.classifier(marker: "classifier")
        let category = try XCTUnwrap(fixture.categories.first)
        let selected = try fixture.artifact("selected", category: category, date: Date(timeIntervalSince1970: 100))
        try await fixture.project.selectModelVersion(runID: selected.id, datasetID: category.id, map: .depth)
        let other = try fixture.artifact("other", category: category, date: Date(timeIntervalSince1970: 200))
        for remaining in fixture.categories.dropFirst() {
            _ = try fixture.artifact("support-" + remaining.id, category: remaining,
                                     date: Date(timeIntervalSince1970: 200))
        }
        try await fixture.project.setModelApproval(runID: other.id, approved: true)
        try FileManager.default.removeItem(at: selected.directory.appendingPathComponent("artifact.json"))
        let reopened = try LoomProject(sourceFolder: fixture.root)
        let coverage = try await reopened.modelCoverage()
        let depth = try XCTUnwrap(coverage.first { $0.slot.datasetID == category.id && $0.slot.product == .depth })
        XCTAssertEqual(depth.status, .unavailable)
        XCTAssertEqual(depth.selectedRunID, selected.id)
        XCTAssertEqual(depth.effectiveRunID, selected.id)
        XCTAssertNil(depth.selectedArtifact)
        XCTAssertEqual(depth.approvedCount, 1)
        await assertExportFails(reopened, containing: "selected depth version is missing")
    }

    func testMissingChosenClassifierDoesNotFallBackToNewerImportedClassifier() async throws {
        let fixture = try await Fixture(); defer { fixture.remove() }
        let selected = try await fixture.classifier(marker: "chosen classifier")
        try await fixture.project.selectClassifierVersion(runID: selected.id)
        let newer = try await fixture.classifier(marker: "newer classifier")
        try await fixture.project.setModelApproval(runID: newer.id, approved: true)
        try FileManager.default.removeItem(at: selected.directory.appendingPathComponent("artifact.json"))
        let reopened = try LoomProject(sourceFolder: fixture.root)
        let coverage = try await reopened.modelCoverage()
        let classifier = try XCTUnwrap(coverage.first { $0.slot.product == .classifier })
        XCTAssertEqual(classifier.status, .unavailable)
        XCTAssertEqual(classifier.selectedRunID, selected.id)
        XCTAssertEqual(classifier.approvedCount, 1)
        XCTAssertNil(classifier.selectedArtifact)
        await assertExportFails(reopened, containing: "selected classifier version is missing")
    }

    private func assertExportFails(_ project: LoomProject, containing message: String,
                                   file: StaticString = #filePath, line: UInt = #line) async {
        do {
            _ = try await project.exportPackage(name: "Managed Choices", version: "1.0.0")
            XCTFail("An unavailable explicit assignment must block export", file: file, line: line)
        } catch {
            XCTAssertTrue(error.localizedDescription.contains(message), error.localizedDescription, file: file, line: line)
        }
    }

    private struct Fixture {
        let root: URL
        let project: LoomProject
        let projectID: String
        let categories: [(id: String, name: String)]
        let classifierURL: URL

        init() async throws {
            let temporary = FileManager.default.temporaryDirectory
                .appendingPathComponent("loom-management-export-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
            root = temporary.resolvingSymlinksInPath().standardizedFileURL
            project = try LoomProject(sourceFolder: root)
            projectID = try await project.snapshot().projectID
            classifierURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Resources/DTTC3.mlmodel")
            let interface = try await nativeClassifierInterface(at: classifierURL)
            categories = interface.labels.map { label in
                let name = ImageMetadataStore.friendlyName(label)
                return (ImageMetadataStore.categoryID(name), name)
            }
            for (index, category) in categories.enumerated() {
                try Self.image(root.appendingPathComponent("Material-\(index)/sample_diff.png"),
                               category: category.name, pairingKey: "material-\(index)")
            }
        }

        func remove() { try? FileManager.default.removeItem(at: root) }

        func artifact(_ id: String, category: (id: String, name: String),
                      date: Date, map: MapRole = .depth) throws -> ModelArtifact {
            let directory = project.projectFolder.appendingPathComponent("runs/" + id, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let settings = NativeFactoryTraining.settingsJSON(TrainingSettings(imageSize: 32,
                epochs: 1, modelWidth: 8, modelBlocks: 1, modelArchitecture: NativeArchitecture.identifier))
            try Data((id + "-weights").utf8).write(to: directory.appendingPathComponent("model.safetensors"))
            try Data((id + "-native-assembly-fixture").utf8).write(to: directory.appendingPathComponent("model.mlmodel"))
            let descriptor: [String: Any] = ["schema": "loom-factory-model-v1", "status": "completed",
                "project_id": projectID, "kind": "map", "dataset_id": category.id,
                "dataset_name": category.name, "map": map.rawValue,
                "architecture": NativeArchitecture.identifier,
                "architecture_config": ["id": NativeArchitecture.identifier, "width": 8, "blocks": 1],
                "image_size": 32, "weights_path": "model.safetensors", "coreml_path": "model.mlmodel",
                "weights_sha256": try TrainingRunFiles.sha256(directory.appendingPathComponent("model.safetensors")),
                "coreml_sha256": try TrainingRunFiles.sha256(directory.appendingPathComponent("model.mlmodel")),
                "input_name": "rgb", "output_name": "material_map", "training": settings]
            try NativeFactoryTraining.writeJSON(descriptor, to: directory.appendingPathComponent("model.json"))
            let artifact = ModelArtifact(id: id, kind: "map", datasetID: category.id, datasetName: category.name,
                map: map, directory: directory, createdAt: date, epochs: 1, loss: nil,
                coreMLPath: "model.mlmodel", weightsPath: "model.safetensors", architecture: NativeArchitecture.identifier)
            try JSONEncoder().encode(artifact).write(to: directory.appendingPathComponent("artifact.json"))
            return artifact
        }

        func classifier(marker: String) async throws -> ModelArtifact {
            let artifact = try await project.importClassifier(from: classifierURL)
            let descriptor = artifact.directory.appendingPathComponent("model.json")
            var value = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: descriptor)) as? [String: Any])
            value["assembly_fixture_marker"] = marker
            try NativeFactoryTraining.writeJSON(value, to: descriptor)
            return artifact
        }

        func extract(_ archive: URL) throws -> URL {
            let destination = project.projectFolder.appendingPathComponent("unpacked-" + UUID().uuidString)
            let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
            process.arguments = ["-x", "-k", archive.path, destination.path]
            try process.run(); process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus, 0)
            return destination
        }

        private static func image(_ file: URL, category: String, pairingKey: String) throws {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            let provider = try XCTUnwrap(CGDataProvider(data: Data(repeating: 120, count: 32 * 32 * 3) as CFData))
            let image = try XCTUnwrap(CGImage(width: 32, height: 32, bitsPerComponent: 8, bitsPerPixel: 24,
                bytesPerRow: 96, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [], provider: provider,
                decode: nil, shouldInterpolate: false, intent: .defaultIntent))
            let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
            try ImageMetadataStore.write(ImageMetadata(pairingKey: pairingKey, role: .diffuse,
                                                      category: category), to: file)
        }
    }
}
