import CoreGraphics
import Foundation
import ImageIO
import LoomNative
import XCTest
@testable import LoomCore

final class ModelTestingTests: XCTestCase {
    func testOverlapTilesKeepEveryNativePixelAndRectangularDimensions() throws {
        let width = 73, height = 49, count = width * height
        var values = [Float](repeating: 0, count: count * 3)
        for pixel in 0..<count {
            values[pixel * 3] = Float(pixel % 997) / 997
            values[pixel * 3 + 1] = 0.2; values[pixel * 3 + 2] = 0.8
        }
        let image = NativeImage(width: width, height: height, channels: 3, values: values)
        let plan = try ModelTestTilePlan(width: width, height: height, tileSize: 32)
        XCTAssertGreaterThan(plan.tiles.count, 1)
        XCTAssertTrue(plan.tiles.contains { $0.x + $0.width == width && $0.y + $0.height == height })
        let output = try ModelTestImage.predict(image: image, plan: plan, map: .roughness,
            cancellation: NativeCancellation(), progress: { _ in }) { input, h, w in Array(input.prefix(h * w)) }
        XCTAssertEqual(output.count, count)
        for pixel in 0..<count { XCTAssertEqual(output[pixel], values[pixel * 3], accuracy: 2e-7) }

        let small = try ModelTestTilePlan(width: 7, height: 13, tileSize: 32)
        XCTAssertEqual(small.tiles.count, 1)
        XCTAssertEqual(small.tiles[0].width, 7); XCTAssertEqual(small.tiles[0].height, 13)
    }

    func testNormalOverlapBlendsDirectionsThenRestoresUnitVectors() throws {
        let width = 49, height = 33, count = width * height
        let image = NativeImage(width: width, height: height, channels: 3,
            values: [Float](repeating: 0.5, count: count * 3))
        let plan = try ModelTestTilePlan(width: width, height: height, tileSize: 32)
        var visit = 0
        let output = try ModelTestImage.predict(image: image, plan: plan, map: .normal,
            cancellation: NativeCancellation(), progress: { _ in }) { _, h, w in
                defer { visit += 1 }
                let pixels = h * w
                var prediction = [Float](repeating: 0.5, count: pixels * 3)
                for pixel in 0..<pixels { prediction[(visit.isMultiple(of: 2) ? 0 : pixels) + pixel] = 1 }
                return prediction
            }
        var blended = false
        for pixel in 0..<count {
            let x = output[pixel] * 2 - 1, y = output[count + pixel] * 2 - 1, z = output[2 * count + pixel] * 2 - 1
            XCTAssertEqual(sqrt(x * x + y * y + z * z), 1, accuracy: 2e-6)
            if x > 0.1 && y > 0.1 { blended = true }
        }
        XCTAssertTrue(blended, "The overlap must blend both predicted directions rather than choose a tile")
    }

    func testNormalPNGKeepsOpenGLChannelsOnFixedSixteenBitScale() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("normal-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory); NativeImages.clearDecodedPixelCache() }
        let output = directory.appendingPathComponent("normal.png"), width = 7, height = 5, count = width * height
        let values = [Float](repeating: 0.5, count: count)
            + [Float](repeating: 0.8, count: count) + [Float](repeating: 0.9, count: count)
        try ModelTestImage.writePNG(values, channels: 3, width: width, height: height, to: output)
        let decoded = try NativeImages.readTarget(from: output, mapKind: "normal", normalConvention: "OpenGL")
        XCTAssertEqual(decoded.width, width); XCTAssertEqual(decoded.height, height)
        for pixel in 0..<count {
            XCTAssertEqual(decoded.values[pixel * 3], 0.5, accuracy: 1 / 65535)
            XCTAssertEqual(decoded.values[pixel * 3 + 1], 0.8, accuracy: 1 / 65535)
            XCTAssertEqual(decoded.values[pixel * 3 + 2], 0.9, accuracy: 1 / 65535)
        }
    }

    func testSavedModelAndCheckpointSelectionPreserveExactSourceAndPublishFullSizePNG() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let original = try Data(contentsOf: fixture.input)
        let checkpoint = try fixture.selection(.checkpoint)
        let deployment = try fixture.selection(.model)
        XCTAssertNotEqual(checkpoint.weightsURL, deployment.weightsURL)
        for selection in [checkpoint, deployment] {
            let savedWeights = try Data(contentsOf: selection.weightsURL)
            let output = fixture.root.appendingPathComponent("outputs/" + selection.source.rawValue)
            let result = try ModelTestWorker.run(model: selection, inputURL: fixture.input, output: output,
                cancellation: NativeCancellation(), progress: { _ in }) { _, h, w in
                    [Float](repeating: 0.25, count: h * w)
                }
            XCTAssertEqual(result.width, fixture.width); XCTAssertEqual(result.height, fixture.height)
            XCTAssertEqual(result.inputURL, fixture.input)
            XCTAssertEqual(result.outputURL, output.appendingPathComponent("depth.png"))
            XCTAssertEqual(result.source, selection.source)
            XCTAssertEqual(try Data(contentsOf: fixture.input), original)
            XCTAssertEqual(try Data(contentsOf: selection.weightsURL), savedWeights)
            let decoded = try NativeImages.readTarget(from: result.outputURL, mapKind: "depth")
            XCTAssertEqual(decoded.width, fixture.width); XCTAssertEqual(decoded.height, fixture.height)
            XCTAssertTrue(decoded.values.allSatisfy { abs($0 - 0.25) < 1 / 65535 })
            let source = try XCTUnwrap(CGImageSourceCreateWithURL(result.outputURL as CFURL, nil))
            let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
            XCTAssertEqual(image.bitsPerComponent, 16)
            let record = try NativeFactoryTraining.json(output.appendingPathComponent("test.json"))
            XCTAssertEqual(record["weights_sha256"] as? String, selection.weightsSHA256)
            XCTAssertEqual(record["input_sha256"] as? String, try TrainingRunFiles.sha256(fixture.input))
            XCTAssertEqual(record["weight_source"] as? String, selection.source.rawValue)
            XCTAssertTrue(result.note.contains("16 bits"))
        }
    }

    func testSelectionRejectsOtherDatasetMapChangedArchitectureAndWeights() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        XCTAssertThrowsError(try ModelTestSelection.load(runID: fixture.runID, source: .checkpoint,
            datasetID: "other", map: .depth, projectFolder: fixture.project.projectFolder, projectID: fixture.projectID))
        XCTAssertThrowsError(try ModelTestSelection.load(runID: fixture.runID, source: .checkpoint,
            datasetID: fixture.datasetID, map: .normal, projectFolder: fixture.project.projectFolder, projectID: fixture.projectID))
        let descriptorURL = fixture.run.appendingPathComponent("checkpoint.json")
        let original = try Data(contentsOf: descriptorURL)
        var descriptor = try NativeFactoryTraining.json(descriptorURL)
        descriptor["architecture"] = "unregistered-model-v1"
        try NativeFactoryTraining.writeJSON(descriptor, to: descriptorURL)
        XCTAssertThrowsError(try fixture.selection(.checkpoint))
        try original.write(to: descriptorURL)
        let weights = fixture.run.appendingPathComponent("checkpoint.safetensors")
        try Data("changed checkpoint".utf8).write(to: weights)
        XCTAssertThrowsError(try fixture.selection(.checkpoint))
    }

    func testScalarEndpointPredictionsReportSaturationAndPreserveFloatPixels() throws {
        let fixture = try Fixture(map: .roughness); defer { fixture.remove() }
        let selection = try fixture.selection(.model)
        for (value, endpoint) in [(Float(0), "black"), (Float(1), "white")] {
            let output = fixture.root.appendingPathComponent("outputs/" + endpoint)
            let result = try ModelTestWorker.run(model: selection, inputURL: fixture.input, output: output,
                cancellation: NativeCancellation(), progress: { _ in }) { _, h, w in
                    [Float](repeating: value, count: h * w)
                }
            XCTAssertTrue(result.note.hasPrefix("The model produced an almost entirely \(endpoint) map"))
            let decoded = try NativeImages.readTarget(from: result.outputURL, mapKind: "roughness")
            XCTAssertTrue(decoded.values.allSatisfy { $0 == value }, "Diagnostics must not rescale endpoint pixels")
            let record = try NativeFactoryTraining.json(output.appendingPathComponent("test.json"))
            let statistics = try XCTUnwrap(record["output_statistics"] as? [String: Any])
            XCTAssertEqual(statistics["minimum"] as? Double, Double(value))
            XCTAssertEqual(statistics["maximum"] as? Double, Double(value))
            XCTAssertEqual(statistics["mean"] as? Double, Double(value))
            XCTAssertEqual(statistics["near_\(endpoint)_fraction"] as? Double, 1)
            XCTAssertEqual(statistics["value_count"] as? Int, fixture.width * fixture.height)
            XCTAssertNotNil(record["output_diagnostic"] as? String)
        }
    }

    func testBrightTexturedPredictionReportsNumericRangeWithoutSaturationAlert() throws {
        let fixture = try Fixture(map: .roughness); defer { fixture.remove() }
        let selection = try fixture.selection(.model)
        let output = fixture.root.appendingPathComponent("outputs/bright-texture")
        let result = try ModelTestWorker.run(model: selection, inputURL: fixture.input, output: output,
            cancellation: NativeCancellation(), progress: { _ in }) { _, h, w in
                (0..<(h * w)).map { $0.isMultiple(of: 2) ? Float(0.98) : Float(1) }
            }
        XCTAssertFalse(result.note.contains("Training may have saturated"))
        let record = try NativeFactoryTraining.json(output.appendingPathComponent("test.json"))
        let statistics = try XCTUnwrap(record["output_statistics"] as? [String: Any])
        XCTAssertEqual(try XCTUnwrap(statistics["minimum"] as? Double), 0.98, accuracy: 1e-6)
        XCTAssertEqual(statistics["maximum"] as? Double, 1)
        XCTAssertGreaterThan(try XCTUnwrap(statistics["mean"] as? Double), 0.98)
        XCTAssertLessThan(try XCTUnwrap(statistics["near_white_fraction"] as? Double), 0.999)
        XCTAssertTrue(record["output_diagnostic"] is NSNull)
        let decoded = try NativeImages.readTarget(from: result.outputURL, mapKind: "roughness")
        XCTAssertEqual(decoded.values.min()!, 0.98, accuracy: 2e-5)
        XCTAssertEqual(decoded.values.max(), 1)
    }

    func testEndpointStatisticsRecognizeNearWhiteFloatsAndExcludeNormalMaps() throws {
        let statistics = try ModelTestOutputStatistics(values: [0.999994, 1, 0.999999], cancellation: NativeCancellation())
        XCTAssertEqual(statistics.nearWhiteFraction, 1)
        XCTAssertNotNil(statistics.saturationNote(map: .roughness))
        XCTAssertNil(statistics.saturationNote(map: .normal))
    }

    func testCancellationAndChangedOriginalLeaveNoPublishedOrPartialOutput() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let selection = try fixture.selection(.checkpoint)
        let cancelledOutput = fixture.root.appendingPathComponent("outputs/cancelled")
        let cancellation = NativeCancellation()
        XCTAssertThrowsError(try ModelTestWorker.run(model: selection, inputURL: fixture.input, output: cancelledOutput,
            cancellation: cancellation, progress: { event in
                if event.message.hasPrefix("Saving ") { cancellation.cancel() }
            }) { _, h, w in [Float](repeating: 0.75, count: h * w) }) { error in
                XCTAssertTrue(error is CancellationError)
            }
        XCTAssertFalse(FileManager.default.fileExists(atPath: cancelledOutput.path))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: cancelledOutput.deletingLastPathComponent().path), [])

        let changedOutput = fixture.root.appendingPathComponent("outputs/changed")
        var changed = false
        XCTAssertThrowsError(try ModelTestWorker.run(model: selection, inputURL: fixture.input, output: changedOutput,
            cancellation: NativeCancellation(), progress: { _ in }) { _, h, w in
                if !changed { try Data("replaced source".utf8).write(to: fixture.input); changed = true }
                return [Float](repeating: 0.75, count: h * w)
            }) { error in XCTAssertTrue(error.localizedDescription.contains("diffuse image changed")) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: changedOutput.path))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: changedOutput.deletingLastPathComponent().path), [])
    }

    /// Explicit opt-in: runs real inference against a user-selected source
    /// project's saved checkpoint and leaves the generated image for inspection.
    func testOptInExistingSavedCheckpointProducesNativeOutput() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["LOOM_FACTORY_MODEL_TEST_SOURCE_ROOT"] else {
            throw XCTSkip("Set LOOM_FACTORY_MODEL_TEST_SOURCE_ROOT to test an existing saved checkpoint.")
        }
        let project = try LoomProject(sourceFolder: URL(fileURLWithPath: path))
        let versions = try await project.modelVersions()
        let version = try XCTUnwrap(versions.first { candidate in
            candidate.checkpoint != nil && (environment["LOOM_FACTORY_MODEL_TEST_RUN_ID"].map { $0 == candidate.id } ?? true)
        })
        let input: URL
        if let path = environment["LOOM_FACTORY_MODEL_TEST_INPUT"] { input = URL(fileURLWithPath: path) }
        else {
            let request = try NativeFactoryTraining.json(version.directory.appendingPathComponent("request.json"))
            let samples = try XCTUnwrap(request["samples"] as? [[String: Any]])
            let path = samples.compactMap { $0["input"] as? String }.first { path in
                guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
                      let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] else { return false }
                return props[kCGImagePropertyPixelWidth] as? Int == 1024 && props[kCGImagePropertyPixelHeight] as? Int == 1024
            }
            input = URL(fileURLWithPath: try XCTUnwrap(path))
        }
        let sourceHash = try TrainingRunFiles.sha256(input)
        let result = try await project.testModel(datasetID: version.datasetID, map: version.map, runID: version.id,
            source: .checkpoint, inputURL: input, progress: { _ in })
        XCTAssertEqual(try TrainingRunFiles.sha256(input), sourceHash)
        let output = try NativeImages.readTarget(from: result.outputURL, mapKind: version.map.rawValue)
        XCTAssertEqual(output.width, result.width); XCTAssertEqual(output.height, result.height)
        XCTAssertTrue(output.values.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
        print("LOOM_FACTORY_MODEL_TEST_OUTPUT=\(result.outputURL.path)")
    }

    private struct Fixture {
        let project: LoomProject
        let root: URL, input: URL, run: URL
        let runID = "saved-model-test", datasetID = "bricks", projectID: String
        let width = 73, height = 49
        let map: MapRole

        init(map: MapRole = .depth) throws {
            self.map = map
            let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("model-test-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
            project = try LoomProject(sourceFolder: temporary)
            root = project.sourceFolder
            let document = try JSONDecoder().decode(ProjectDocument.self,
                from: Data(contentsOf: project.projectFolder.appendingPathComponent("project.json")))
            projectID = document.id
            input = root.appendingPathComponent("external-diffuse.png")
            try ModelTestImage.writePNG([Float](repeating: 0.2, count: width * height * 3), channels: 3,
                width: width, height: height, to: input)
            run = project.projectFolder.appendingPathComponent("runs/" + runID)
            try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
            let architecture = try NativeArchitecture(mapKind: map.rawValue, width: 8, blocks: 1)
            let weights = Dictionary(uniqueKeysWithValues: architecture.parameterShapes.map { name, shape in
                (name, NativeTensor(shape: shape, values: [Float](repeating: 0, count: shape.reduce(1, *))))
            })
            let settings = NativeFactoryTraining.settingsJSON(TrainingSettings(imageSize: 32, modelWidth: 8,
                modelBlocks: 1, modelArchitecture: architecture.identifier))
            for source in [TrainingWeightSource.checkpoint, .model] {
                let filename = source == .checkpoint ? "checkpoint" : "model"
                let weightFile = run.appendingPathComponent(filename + ".safetensors")
                try NativeWeights.write(weights, to: weightFile)
                try NativeFactoryTraining.writeJSON(["schema": "loom-factory-\(filename)-v1", "status": "completed",
                    "project_id": projectID, "dataset_id": datasetID, "kind": "map", "map": map.rawValue,
                    "architecture": architecture.identifier, "architecture_config": architecture.descriptor,
                    "weights_path": filename + ".safetensors", "weights_sha256": TrainingRunFiles.sha256(weightFile),
                    "image_size": 32, (source == .checkpoint ? "settings" : "training"): settings],
                    to: run.appendingPathComponent(filename + ".json"))
            }
        }

        func selection(_ source: TrainingWeightSource) throws -> ModelTestSelection {
            try ModelTestSelection.load(runID: runID, source: source, datasetID: datasetID, map: map,
                projectFolder: project.projectFolder, projectID: projectID)
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
    }
}
