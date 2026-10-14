import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class TrainingTelemetryWorkflowTests: XCTestCase {
    private final class Events: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [TrainingEvent] = []
        func append(_ event: TrainingEvent) {
            lock.lock(); defer { lock.unlock() }
            values.append(event)
        }
        func snapshot() -> [TrainingEvent] {
            lock.lock(); defer { lock.unlock() }
            return values
        }
    }

    func testActualNativeRunReportsImagesEpochsOptimizerStepsTimingsAndSavedCompletion() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal GPU") }
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let project = try LoomProject(sourceFolder: root)
        let snapshot = try await project.snapshot()
        let originals = try Dictionary(uniqueKeysWithValues: snapshot.images.map { ($0.url, try Data(contentsOf: $0.url)) })
        let settings = TrainingSettings(imageSize: 32, epochs: 2, batchSize: 1, shuffleEachEpoch: false,
            flipCrops: true, rotateCrops: true, modelWidth: 8, modelBlocks: 1,
            gradientAccumulationSteps: 2, validationFraction: 0.25)
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        let plan = try XCTUnwrap(snapshot.trainingPlan(datasetID: datasetID, map: .depth, settings: settings))
        let stepsPerEpoch = settings.optimizerStepsPerEpoch(pairCount: plan.trainingPairCount)
        let totalOptimizerUpdates = stepsPerEpoch * settings.epochs
        let trainingSampleVisits = plan.trainingPairCount * settings.epochs
        let validationSampleVisits = plan.validationPairCount * settings.epochs
        XCTAssertEqual(plan.trainingPairCount, 3)
        XCTAssertEqual(plan.validationPairCount, 1)
        XCTAssertEqual(stepsPerEpoch, 2)
        XCTAssertEqual(totalOptimizerUpdates, 4)

        let events = Events()
        let artifact = try await project.trainMap(datasetID: datasetID, map: .depth, settings: settings,
            progress: { events.append($0) })
        let captured = events.snapshot()
        XCTAssertFalse(captured.isEmpty)
        XCTAssertFalse(captured.contains { $0.message.localizedCaseInsensitiveContains("native GPU batch") })
        let snapshots = try captured.map { try XCTUnwrap($0.trainingProgress, "Missing structured progress for \($0.type)") }
        let expectedStages: Set<TrainingStage> = [.checkingSources, .preparingModel, .preparingImage, .training,
            .optimizing, .validating, .checkpointing, .verifyingSources, .exporting, .completed]
        XCTAssertTrue(expectedStages.isSubset(of: Set(snapshots.map(\.stage))))
        for state in snapshots {
            XCTAssertEqual(state.totalEpochs, settings.epochs)
            XCTAssertEqual(state.stepsPerEpoch, stepsPerEpoch)
            XCTAssertEqual(state.totalSteps, totalOptimizerUpdates)
            XCTAssertTrue(state.elapsedSeconds.isFinite && state.elapsedSeconds >= 0)
            XCTAssertTrue(state.epochElapsedSeconds.isFinite && state.epochElapsedSeconds >= 0)
            XCTAssertGreaterThanOrEqual(state.completedSteps, 0)
            XCTAssertLessThanOrEqual(state.completedSteps, totalOptimizerUpdates)
            if state.stage == .validating {
                XCTAssertEqual(state.imagesPerEpoch, plan.validationPairCount)
                XCTAssertGreaterThanOrEqual(state.imageIndex, 0)
                XCTAssertLessThanOrEqual(state.imageIndex, state.imagesPerEpoch,
                    "Validation must not retain the last training image index")
            }
        }

        let trainingNames = Set(plan.training.map { $0.input.trainingDisplayName })
        for type in ["image_preparing", "image_training"] {
            let imageEvents = captured.filter { $0.type == type }
            XCTAssertEqual(imageEvents.count, trainingSampleVisits)
            for event in imageEvents {
                let state = try XCTUnwrap(event.trainingProgress)
                let image = try XCTUnwrap(state.currentImage)
                XCTAssertTrue(trainingNames.contains(image), "Progress must name the actual selected input")
                XCTAssertEqual(state.imagesPerEpoch, plan.trainingPairCount)
                XCTAssertTrue((1...state.imagesPerEpoch).contains(state.imageIndex))
                XCTAssertTrue((1...settings.epochs).contains(state.epoch))
                XCTAssertEqual(state.stage, type == "image_preparing" ? .preparingImage : .training)
            }
        }
        let validationEvents = captured.filter { $0.type == "validation_image" }
        XCTAssertEqual(validationEvents.count, validationSampleVisits)
        for event in validationEvents {
            XCTAssertEqual(event.trainingProgress?.currentImage, plan.validation[0].input.trainingDisplayName)
            XCTAssertEqual(event.trainingProgress?.imageIndex, 1)
        }

        let steps = captured.filter { $0.type == "progress" }
        XCTAssertEqual(steps.count, totalOptimizerUpdates,
            "An optimizer step includes two image visits, except the partial final window")
        XCTAssertEqual(steps.compactMap(\.update), [1, 2, 3, 4])
        XCTAssertEqual(steps.compactMap { $0.trainingProgress?.epoch }, [1, 1, 2, 2])
        XCTAssertEqual(steps.compactMap { $0.trainingProgress?.stepInEpoch }, [1, 2, 1, 2])
        XCTAssertEqual(steps.compactMap { $0.trainingProgress?.completedImages }, [2, 3, 5, 6])
        XCTAssertEqual(steps.compactMap { $0.trainingProgress?.imageIndex }, [2, 3, 2, 3])
        for event in steps {
            let state = try XCTUnwrap(event.trainingProgress)
            let timing = try XCTUnwrap(event.cycleTiming, "Each completed optimizer cycle must report its measured phases")
            for seconds in [timing.preparationSeconds, timing.preparationWaitSeconds, timing.trainingSeconds,
                            timing.optimizerSeconds, timing.totalSeconds] {
                XCTAssertTrue(seconds.isFinite && seconds >= 0)
            }
            XCTAssertGreaterThanOrEqual(timing.totalSeconds + 0.000_001,
                timing.preparationWaitSeconds + timing.trainingSeconds + timing.optimizerSeconds)
            XCTAssertEqual(state.lastCycleSeconds, timing.totalSeconds)
            XCTAssertEqual(state.completedSteps, event.update)
            XCTAssertEqual(event.totalUpdates, totalOptimizerUpdates)
        }
        XCTAssertNil(steps.first?.trainingProgress?.averageCycleSeconds, "Compilation must not create a steady-speed estimate")
        XCTAssertNil(steps.first?.trainingProgress?.totalRemainingSeconds)
        let steady = try XCTUnwrap(steps.dropFirst().first?.trainingProgress)
        XCTAssertTrue(try XCTUnwrap(steady.averageCycleSeconds).isFinite)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(steady.totalRemainingSeconds), 0)

        let completed = try XCTUnwrap(captured.last { $0.type == "completed" }?.trainingProgress)
        XCTAssertEqual(completed.stage, .completed)
        XCTAssertEqual(completed.epoch, settings.epochs)
        XCTAssertEqual(completed.completedSteps, totalOptimizerUpdates)
        XCTAssertEqual(completed.completedImages, trainingSampleVisits)
        XCTAssertEqual(completed.epochRemainingSeconds, 0)
        XCTAssertEqual(completed.totalRemainingSeconds, 0)
        XCTAssertEqual(artifact.epochs, settings.epochs)
        for (url, bytes) in originals { XCTAssertEqual(try Data(contentsOf: url), bytes) }

        let lines = try String(contentsOf: artifact.directory.appendingPathComponent("training.log"), encoding: .utf8)
            .split(separator: "\n")
        let log = try lines.map { line in
            try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
        }
        XCTAssertEqual(log.count, captured.count)
        XCTAssertTrue(log.allSatisfy { $0["training_progress"] is [String: Any] })
        XCTAssertEqual(log.filter { $0["cycle_timing"] is [String: Any] }.count, totalOptimizerUpdates)
        let savedCompletion = try XCTUnwrap(log.last?["training_progress"] as? [String: Any])
        XCTAssertEqual(savedCompletion["stage"] as? String, TrainingStage.completed.rawValue)
        XCTAssertEqual(savedCompletion["completedSteps"] as? Int, totalOptimizerUpdates)
        XCTAssertEqual(savedCompletion["totalRemainingSeconds"] as? Double, 0)
    }

    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("training-telemetry-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for material in 0..<4 {
            let folder = root.appendingPathComponent("material-\(material)", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for role in [MapRole.diffuse, .depth] {
                let size = 64
                let stem = role == .diffuse ? "diff" : "disp"
                let file = folder.appendingPathComponent("material_\(material)_\(stem).png")
                let bytes = (0..<(size * size * 3)).map { index -> UInt8 in
                    let pixel = role == .depth ? index / 3 : index
                    return UInt8((pixel * 7 + material * 17 + 31) % 255)
                }
                let provider = try XCTUnwrap(CGDataProvider(data: Data(bytes) as CFData))
                let image = try XCTUnwrap(CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 24,
                    bytesPerRow: size * 3, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider,
                    decode: nil, shouldInterpolate: false, intent: .defaultIntent))
                let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL,
                    UTType.png.identifier as CFString, 1, nil))
                CGImageDestinationAddImage(destination, image, nil)
                XCTAssertTrue(CGImageDestinationFinalize(destination))
                var metadata = ImageMetadata(pairingKey: "material-\(material)", role: role)
                metadata.category = "Bricks"
                metadata.roleWasManuallySelected = true; metadata.pairingWasManuallySelected = true
                try ImageMetadataStore.write(metadata, to: file)
            }
        }
        return root
    }
}
