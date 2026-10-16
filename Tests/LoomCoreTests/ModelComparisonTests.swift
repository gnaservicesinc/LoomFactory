import CoreGraphics
import Foundation
import ImageIO
import LoomNative
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class ModelComparisonTests: XCTestCase {
    func testComparisonRefusesAnotherProjectActorsOperationBeforeReadingModels() async throws {
        let (project, _, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: project.sourceFolder) }
        let lease = try XCTUnwrap(TrainingQueueLease.requireAvailable(projectFolder: project.projectFolder))
        do {
            _ = try await project.compareModels(firstArtifactID: "first", secondArtifactID: "second",
                pairID: "pair", progress: { _ in XCTFail("A busy project must not start inference") })
            XCTFail("Comparison must share the training operation lease")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Another model operation"), error.localizedDescription) }
        withExtendedLifetime(lease) {}
        XCTAssertFalse(FileManager.default.fileExists(atPath: project.projectFolder.appendingPathComponent("comparisons").path))
    }

    func testRealSavedModelsUseTheSameNativeCropAndFixedSixteenBitScaleWithoutChangingSources() async throws {
        let (project, input, target) = try fixture()
        let root = project.sourceFolder
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = try await project.snapshot(), datasetID = ImageMetadataStore.categoryID("Bricks")
        let first = try model(project: project, projectID: snapshot.projectID, id: "quarter", size: 32, value: 0.25,
            input: input, split: "validation", export: true)
        let second = try model(project: project, projectID: snapshot.projectID, id: "three-quarter", size: 64, value: 0.75,
            input: input, split: "training", exaggerate: true, export: true)
        let originals = try [input, target].map { try Data(contentsOf: $0) }
        let options = try await project.comparePairOptions(datasetID: datasetID, map: .depth,
            artifactIDs: [first.id, second.id])
        let pair = try XCTUnwrap(options.first)
        XCTAssertEqual(pair.seenByFirst, .heldOut)
        XCTAssertEqual(pair.seenBySecond, .training)
        let result = try await project.compareModels(firstArtifactID: first.id, secondArtifactID: second.id,
            pairID: pair.id, progress: { _ in })
        XCTAssertEqual(result.cropSize, 32)
        XCTAssertEqual(result.firstArtifactID, first.id)
        XCTAssertEqual(result.secondArtifactID, second.id)
        XCTAssertEqual(result.firstMAE, 0.05, accuracy: 2e-4)
        XCTAssertEqual(result.secondMAE, 0.55, accuracy: 2e-4)
        XCTAssertEqual(result.outputDifferenceMAE, 0.5, accuracy: 2e-4)
        XCTAssertTrue(result.targetNote.contains("exaggerated"))
        for (url, original) in zip([input, target], originals) { XCTAssertEqual(try Data(contentsOf: url), original) }

        let firstPixels = try NativeImages.readTarget(from: result.firstOutputURL, mapKind: "depth")
        let secondPixels = try NativeImages.readTarget(from: result.secondOutputURL, mapKind: "depth")
        let targetPixels = try NativeImages.readTarget(from: result.targetURL, mapKind: "depth")
        XCTAssertEqual(firstPixels.width, 32); XCTAssertEqual(firstPixels.height, 32)
        XCTAssertTrue(firstPixels.values.allSatisfy { abs($0 - 0.25) < 2e-4 })
        XCTAssertTrue(secondPixels.values.allSatisfy { abs($0 - 0.75) < 2e-4 })
        XCTAssertEqual(targetPixels.values, [Float](repeating: 0.2, count: 32 * 32),
            "The target preview must contain the center crop, without the contrasting outer border")
        for url in [result.inputURL, result.targetURL, result.firstOutputURL, result.secondOutputURL] {
            let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
            let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
            XCTAssertEqual(image.bitsPerComponent, 16)
        }
        let originalPixels = try NativeImages.readRGB(from: input)
        let cropPixels = try NativeImages.readRGB(from: result.inputURL)
        for y in 0..<32 {
            for x in 0..<32 {
                for channel in 0..<3 {
                    XCTAssertEqual(cropPixels.values[(y * 32 + x) * 3 + channel],
                        originalPixels.values[((y + 16) * 64 + x + 16) * 3 + channel], accuracy: 1 / 65535)
                }
            }
        }
        let record = try NativeFactoryTraining.json(result.directory.appendingPathComponent("comparison.json"))
        let crop = try XCTUnwrap(record["crop"] as? [String: Int])
        XCTAssertEqual(crop, ["x": 16, "y": 16, "size": 32])
        XCTAssertEqual(record["membership_a"] as? String, "heldOut")
        XCTAssertEqual(record["membership_b"] as? String, "training")

        let reversed = try await project.compareModels(firstArtifactID: second.id, secondArtifactID: first.id,
            pairID: pair.id, progress: { _ in })
        XCTAssertEqual(reversed.firstMAE, result.secondMAE, accuracy: 2e-4)
        XCTAssertEqual(reversed.secondMAE, result.firstMAE, accuracy: 2e-4)
        XCTAssertNotEqual(reversed.directory, result.directory)
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.firstOutputURL.path), "Previous comparisons remain available")
    }

    func testMembershipFollowsWholeMaterialAncestorsAndTreatsMissingOrCyclicAncestryAsUnknown() async throws {
        let (project, input, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: project.sourceFolder) }
        let snapshot = try await project.snapshot(), datasetID = ImageMetadataStore.categoryID("Bricks")
        let first = try model(project: project, projectID: snapshot.projectID, id: "independent", size: 32,
            value: 0.25, input: input, split: "validation", export: false)
        let parentID = "parent"
        let lineage = TrainingLineage(parentRunID: parentID, source: .model, weightsSHA256: "fixture")
        let second = try model(project: project, projectID: snapshot.projectID, id: "continued", size: 32,
            value: 0.75, input: input, split: "validation", lineage: lineage, export: false)
        let parent = project.projectFolder.appendingPathComponent("runs/" + parentID, isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        try metadata(projectID: snapshot.projectID, id: parentID, lineage: nil, at: parent)
        // Different file/resolution, same material folder: it was still seen.
        let otherResolution = input.deletingLastPathComponent().appendingPathComponent("diffuse-128.png")
        try request(input: otherResolution, split: "training", at: parent)
        var options = try await project.comparePairOptions(datasetID: datasetID, map: .depth,
            artifactIDs: [first.id, second.id])
        XCTAssertEqual(options.first?.seenByFirst, .heldOut)
        XCTAssertEqual(options.first?.seenBySecond, .training)

        try request(input: otherResolution, split: "validation", at: parent)
        options = try await project.comparePairOptions(datasetID: datasetID, map: .depth,
            artifactIDs: [first.id, second.id])
        XCTAssertEqual(options.first?.seenBySecond, .heldOut)

        try FileManager.default.removeItem(at: parent.appendingPathComponent("run-version.json"))
        options = try await project.comparePairOptions(datasetID: datasetID, map: .depth,
            artifactIDs: [first.id, second.id])
        XCTAssertEqual(options.first?.seenBySecond, .unknown)

        try metadata(projectID: snapshot.projectID, id: parentID,
            lineage: TrainingLineage(parentRunID: "continued", source: .model, weightsSHA256: "fixture"), at: parent)
        options = try await project.comparePairOptions(datasetID: datasetID, map: .depth,
            artifactIDs: [first.id, second.id])
        XCTAssertEqual(options.first?.seenBySecond, .unknown)
    }

    func testMembershipIndexKeepsUnrelatedUnknownSplitsLocalAndKnownTrainingWinsOverIncompleteAncestry() async throws {
        let (project, input, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: project.sourceFolder) }
        let projectID = try await project.snapshot().projectID
        let first = try model(project: project, projectID: projectID, id: "holdout", size: 32,
            value: 0.25, input: input, split: "validation", export: false)
        let second = try model(project: project, projectID: projectID, id: "trained", size: 32,
            value: 0.75, input: input, split: "training",
            lineage: TrainingLineage(parentRunID: "missing-parent", source: .model, weightsSHA256: "fixture"), export: false)
        let unrelated = project.sourceFolder.appendingPathComponent("unrelated/diffuse.png")
        try NativeFactoryTraining.writeJSON(["samples": [
            ["input": input.path, "split": "validation"],
            ["input": unrelated.path, "split": "unknown"]
        ]], to: first.directory.appendingPathComponent("request.json"))
        let options = try await project.comparePairOptions(datasetID: ImageMetadataStore.categoryID("Bricks"), map: .depth,
            artifactIDs: [first.id, second.id])
        XCTAssertEqual(options.count, 1)
        XCTAssertEqual(options.first?.seenByFirst, .heldOut)
        XCTAssertEqual(options.first?.seenBySecond, .training)
    }

    private func fixture() throws -> (LoomProject, URL, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("model-comparison-" + UUID().uuidString)
        let folder = root.appendingPathComponent("brick", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let input = folder.appendingPathComponent("diffuse.png"), target = folder.appendingPathComponent("depth.png")
        var rgb = [UInt16](repeating: 0, count: 64 * 64 * 3)
        for index in rgb.indices {
            let code: Int = (index * 127 + 971) % 65536
            rgb[index] = UInt16(code)
        }
        let scalar = (0..<(64 * 64)).map { pixel -> UInt16 in
            let x = pixel % 64, y = pixel / 64
            return (16..<48).contains(x) && (16..<48).contains(y) ? 13107 : 58982
        }
        try writePNG(rgb, channels: 3, to: input)
        try writePNG(scalar, channels: 1, to: target)
        for (url, role) in [(input, MapRole.diffuse), (target, .depth)] {
            try ImageMetadataStore.write(ImageMetadata(pairingKey: "brick", role: role, category: "Bricks",
                roleWasManuallySelected: true, pairingWasManuallySelected: true), to: url)
        }
        let project = try LoomProject(sourceFolder: root)
        let canonicalFolder = project.sourceFolder.appendingPathComponent("brick", isDirectory: true)
        return (project, canonicalFolder.appendingPathComponent("diffuse.png"), canonicalFolder.appendingPathComponent("depth.png"))
    }

    private func model(project: LoomProject, projectID: String, id: String, size: Int, value: Float,
                       input: URL, split: String, exaggerate: Bool = false, lineage: TrainingLineage? = nil,
                       export: Bool) throws -> ModelArtifact {
        let directory = project.projectFolder.appendingPathComponent("runs/" + id, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 1)
        var weights = Dictionary(uniqueKeysWithValues: architecture.parameterShapes.map { name, shape in
            (name, NativeTensor(shape: shape, values: [Float](repeating: 0, count: shape.reduce(1, *))))
        })
        weights["output.bias"]!.values[0] = log(value / (1 - value))
        let modelURL = directory.appendingPathComponent("model.mlmodel")
        if export {
            let compiled = try NativeCoreML.export(weights: weights, architecture: architecture, imageSize: size, to: modelURL)
            try FileManager.default.removeItem(at: compiled)
        } else { try Data("comparison-options-fixture".utf8).write(to: modelURL) }
        try NativeWeights.write(weights, to: directory.appendingPathComponent("model.safetensors"))
        let artifact = ModelArtifact(id: id, kind: "map", datasetID: ImageMetadataStore.categoryID("Bricks"),
            datasetName: "Bricks", map: .depth, directory: directory, epochs: 1, loss: nil,
            coreMLPath: "model.mlmodel", weightsPath: "model.safetensors", architecture: NativeArchitecture.identifier)
        try JSONEncoder().encode(artifact).write(to: directory.appendingPathComponent("artifact.json"))
        try NativeFactoryTraining.writeJSON(["schema": "loom-factory-model-v1", "status": "completed",
            "project_id": projectID, "dataset_id": ImageMetadataStore.categoryID("Bricks"), "map": "depth",
            "architecture": NativeArchitecture.identifier, "image_size": size,
            "coreml_path": "model.mlmodel", "weights_path": "model.safetensors",
            "coreml_sha256": TrainingRunFiles.sha256(modelURL),
            "training": ["exaggerate_features": exaggerate]], to: directory.appendingPathComponent("model.json"))
        try metadata(projectID: projectID, id: id, lineage: lineage, at: directory)
        try request(input: input, split: split, at: directory)
        return artifact
    }

    private func metadata(projectID: String, id: String, lineage: TrainingLineage?, at directory: URL) throws {
        try TrainingRunFiles.writeMetadata(TrainingRunMetadata(projectID: projectID, runID: id,
            datasetID: ImageMetadataStore.categoryID("Bricks"), map: .depth, label: "Test fixture",
            createdAt: .now, settings: TrainingSettings(imageSize: 32, modelWidth: 8, modelBlocks: 1),
            lineage: lineage, status: "completed"), at: directory)
    }
    private func request(input: URL, split: String, at directory: URL) throws {
        try NativeFactoryTraining.writeJSON(["samples": [["input": input.path, "split": split]]],
            to: directory.appendingPathComponent("request.json"))
    }
    private func writePNG(_ codes: [UInt16], channels: Int, to url: URL) throws {
        let data = codes.withUnsafeBytes { Data($0) }
        let provider = try XCTUnwrap(CGDataProvider(data: data as CFData))
        let image = try XCTUnwrap(CGImage(width: 64, height: 64, bitsPerComponent: 16,
            bitsPerPixel: 16 * channels, bytesPerRow: 64 * channels * 2,
            space: channels == 1 ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: [.byteOrder16Little], provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }
}
