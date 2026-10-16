import Foundation
import LoomNative
import Testing
@testable import LoomCore

struct ModelTestContextTilesTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["LOOM_PAPER_CONTEXT_PARITY"] == "1"))
    func savedPaperTwoKContextTilesMatchOneCompleteReflectedField() throws {
        let source = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        let project = source.appendingPathComponent(".loom-factory")
        let lease = try TrainingQueueLease.requireAvailable(projectFolder: project)
        defer { withExtendedLifetime(lease) {}; NativeImages.clearDecodedPixelCache() }
        let descriptor = try NativeFactoryTraining.json(project.appendingPathComponent("runs/ec471005-e4a6-49b7-8bd2-13920027ad6f/model.json"))
        let model = try ModelTestSelection.load(runID: "ec471005-e4a6-49b7-8bd2-13920027ad6f", source: .model,
            datasetID: "dataset-382635c9325bf327", map: .depth, projectFolder: project,
            projectID: try #require(descriptor["project_id"] as? String))
        let cancellation = NativeCancellation()
        let inputURL = source.appendingPathComponent("Paper003/Paper003_2K-PNG_Color.png")
        let hash = try TrainingRunFiles.sha256(inputURL)
        let image = try NativeImages.readRGB(from: inputURL, cancellation: cancellation)
        let (halo, phase) = try ModelTestContextPlan.halo(for: model.architecture)
        let session = try NativeTrainingSession(architecture: model.architecture,
            options: NativeTrainingOptions(scalarLoss: .absoluteV1), seed: 0,
            weights: NativeWeights.read(from: model.weightsURL), cancellation: cancellation)
        let predictor: ModelTestWorker.Predictor = { values, h, w in
            try session.predict(input: values, batchSize: 1, height: h, width: w)
        }
        let tiled = try ModelTestImage.predictContext(image: image,
            plan: ModelTestContextPlan(width: image.width, height: image.height, tileSize: 1024, halo: halo, phase: phase),
            map: .depth, cancellation: cancellation, progress: { _ in }, predictor: predictor)
        let completePlan = try ModelTestContextPlan(width: image.width, height: image.height,
            tileSize: max(image.width, image.height) + 2*halo, halo: halo, phase: phase)
        #expect(completePlan.tiles.count == 1)
        let complete = try ModelTestImage.predictContext(image: image, plan: completePlan,
            map: .depth, cancellation: cancellation, progress: { _ in }, predictor: predictor)
        let maximum = zip(tiled, complete).reduce(Float(0)) { max($0, abs($1.0-$1.1)) }
        #expect(maximum < 0.000002)
        #expect(try TrainingRunFiles.sha256(inputURL) == hash)
        #expect(try TrainingRunFiles.sha256(model.weightsURL) == model.weightsSHA256)
        let output = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["LOOM_PAPER_CONTEXT_OUTPUT"]))
        try NativeFactoryTraining.writeJSON(["width": image.width, "height": image.height,
            "maximum_absolute_difference": maximum, "tolerance": 0.000002,
            "reference_input_size": completePlan.inputWidth, "halo": halo, "phase": phase,
            "weights_sha256": model.weightsSHA256, "source_sha256": hash,
            "training_updates": 0], to: output.appendingPathComponent("production-parity.json"))
    }

    @Test func ownsEveryPixelOnceAndKeepsPyramidPhaseForOddRectangles() throws {
        let p = try ModelTestContextPlan(width: 73, height: 49, tileSize: 32, halo: 8, phase: 4)
        var visits = [Int](repeating: 0, count: 73*49)
        for t in p.tiles {
            #expect((t.x-p.halo)%4 == 0 && (t.y-p.halo)%4 == 0)
            for y in t.y..<(t.y+t.height) { for x in t.x..<(t.x+t.width) { visits[y*73+x] += 1 } }
        }
        #expect(visits.allSatisfy { $0 == 1 })
        let values = (0..<(73*49*3)).map { Float($0%997)/997 }
        let image = NativeImage(width: 73, height: 49, channels: 3, values: values)
        let output = try ModelTestImage.predictContext(image: image, plan: p, map: .depth,
            cancellation: NativeCancellation(), progress: { _ in }) { input, h, w in Array(input.prefix(h*w)) }
        #expect(output == stride(from: 0, to: values.count, by: 3).map { values[$0] })
        #expect(ModelTestContextPlan.reflected(-1, length: 5) == 1)
        #expect(ModelTestContextPlan.reflected(5, length: 5) == 3)
        #expect(ModelTestContextPlan.reflected(-100, length: 1) == 0)
    }

    @Test func discardsBoundaryBiasThatLegacyFeatherLeaksIntoTheOutput() throws {
        let image = NativeImage(width: 73, height: 49, channels: 3,
            values: [Float](repeating: 0.5, count: 73*49*3))
        let predictor: ModelTestWorker.Predictor = { _, h, w in
            (0..<(h*w)).map { i in
                let x=i%w,y=i/w
                return min(x,y,w-1-x,h-1-y)<8 ? Float(0.1) : Float(0.5)
            }
        }
        let legacy = try ModelTestImage.predict(image: image,
            plan: ModelTestTilePlan(width: 73, height: 49, tileSize: 32), map: .depth,
            cancellation: NativeCancellation(), progress: { _ in }, predictor: predictor)
        #expect(legacy[24*73+24] < 0.49)
        let context = try ModelTestImage.predictContext(image: image,
            plan: ModelTestContextPlan(width: 73, height: 49, tileSize: 32, halo: 8, phase: 4), map: .depth,
            cancellation: NativeCancellation(), progress: { _ in }, predictor: predictor)
        #expect(context.allSatisfy { $0 == 0.5 })
    }

    @Test func modelHaloCoversTheSavedFiniteConvolutionFamiliesAndRejectsCompact() throws {
        let pyramid = try NativeArchitecture(mapKind: "depth", width: 55, blocks: 4, identifier: NativeArchitecture.pyramidIdentifier)
        let (halo, phase) = try ModelTestContextPlan.halo(for: pyramid)
        #expect(halo == 44 && phase == 4)
        let p = try ModelTestContextPlan(width: 2048, height: 2048, tileSize: 1024, halo: halo, phase: phase)
        #expect(p.tiles.count == 9 && p.coreWidth == 936)
        let compact = try NativeArchitecture(mapKind: "depth", width: 32, blocks: 1, identifier: NativeArchitecture.compactIdentifier)
        #expect(throws: (any Error).self) { try ModelTestContextPlan.halo(for: compact) }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["LOOM_PAPER_CONTEXT_INFERENCE"] == "1"))
    func savedPaperWeightsCompareLegacyAndContextWithoutTraining() async throws {
        let source = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        let output = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["LOOM_PAPER_CONTEXT_OUTPUT"]))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let project = try LoomProject(sourceFolder: source)
        let run = "ec471005-e4a6-49b7-8bd2-13920027ad6f", dataset = "dataset-382635c9325bf327"
        let selection = source.appendingPathComponent(".loom-factory/model-selection.json")
        let before = try Data(contentsOf: selection)
        var records: [[String: Any]] = []
        for (size, mode) in [(1024, ModelTestBoundaryHandling.legacy), (1024, .context), (2048, .context)] {
            let image = source.appendingPathComponent("Paper003/Paper003_\(size == 1024 ? "1K" : "2K")-PNG_Color.png")
            let result = try await project.testModel(datasetID: dataset, map: .depth, runID: run, source: .model,
                inputURL: image, boundaryHandling: mode, progress: { _ in })
            records.append(["size": size, "mode": mode.rawValue, "directory": result.directory.path,
                "output": result.outputURL.path, "tile_count": result.tileCount, "input_sha256": result.inputSHA256 ?? ""])
            #expect(try Data(contentsOf: selection) == before)
            try JSONSerialization.data(withJSONObject: records, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("saved-weight-inference.json"), options: .atomic)
        }
    }
}
