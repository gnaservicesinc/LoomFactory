import CoreGraphics
import Foundation
import ImageIO
import LoomNative
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

@MainActor
final class TrainingSeedPolicyTests: XCTestCase {
    func testRandomPreferencePersistsWhileEveryStartRecordsItsActualSeed() async throws {
        let fixture = try Fixture(materialCount: 1)
        defer { fixture.remove() }
        let settings = fixture.settings(seed: -1, epochs: 1)
        try await fixture.project.updateSettings(ProjectSettings(training: settings))
        for _ in 0..<2 {
            let artifact = try await fixture.project.trainMap(datasetID: fixture.datasetID, map: .depth,
                settings: settings, start: .fresh, progress: { _ in })
            let request = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("request.json"))
            let metadata = try XCTUnwrap(TrainingRunFiles.metadata(at: artifact.directory))
            let seed = try XCTUnwrap(request["seed"] as? Int)
            XCTAssertTrue((0...Int(Int32.max)).contains(seed))
            XCTAssertEqual(request["requested_seed"] as? Int, -1)
            XCTAssertEqual(metadata.settings.seed, seed)
            XCTAssertEqual(metadata.settings.requestedSeed, -1)
            XCTAssertEqual(request["resume_state_policy"] as? String, "fresh-optimizer-and-sampler")
            let snapshot = try await fixture.project.snapshot()
            XCTAssertEqual(snapshot.settings.training.seed, -1, "A receipt must not replace the user's random-seed preference")
        }
        let reopened = try LoomProject(sourceFolder: fixture.root)
        let reopenedSettings = try await reopened.snapshot().settings.training
        XCTAssertEqual(reopenedSettings.seed, -1)
        XCTAssertNil(reopenedSettings.requestedSeed)
    }

    func testRandomCheckpointContinuationStartsFreshStateWhileLegacyFixedSeedResumesExactly() async throws {
        let fixture = try Fixture(materialCount: 1)
        defer { fixture.remove() }
        let settings = fixture.settings(seed: 37, epochs: 2)
        let baseline = try await fixture.project.trainMap(datasetID: fixture.datasetID, map: .depth,
            settings: settings, start: .fresh, progress: { _ in })
        let expected = try NativeWeights.read(from: baseline.directory.appendingPathComponent("model.safetensors"))
        let snapshot = try await fixture.project.snapshot()
        let dataset = try XCTUnwrap(snapshot.datasets.first { $0.id == fixture.datasetID })
        let plan = try XCTUnwrap(snapshot.trainingPlan(datasetID: dataset.id, map: .depth, settings: settings))
        XCTAssertEqual(plan.trainingPairCount, 1)
        let runID = "seed-policy-stopped-" + UUID().uuidString.lowercased()
        let directory = fixture.project.projectFolder.appendingPathComponent("runs/" + runID)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let control = TrainingRunControl(runID: runID)
        defer { control.finish() }
        let pending = Task { try await control.request(stop: true) }
        try await waitForPending(control)
        let root = fixture.project.sourceFolder, projectID = snapshot.projectID
        do {
            _ = try await Task.detached {
                try NativeFactoryTraining.run(id: runID, root: root, output: directory, projectID: projectID,
                    dataset: dataset, map: .depth, plan: plan, settings: settings, resume: nil,
                    cancellation: NativeCancellation(), progress: { _ in }, control: control)
            }.value
            XCTFail("The first checkpoint request must save and stop")
        } catch is TrainingSavedStop {}
        let receipt = try await pending.value
        XCTAssertEqual(receipt.completedUpdates, 1)
        XCTAssertTrue(receipt.stopped)
        control.finish()

        // Older checkpoints omit execution-only cache controls and the new
        // preference field. Their fixed seed and exact optimizer cursor remain
        // compatible with today's execution settings.
        let checkpointURL = directory.appendingPathComponent("checkpoint.json")
        var saved = try NativeFactoryTraining.json(checkpointURL)
        var savedSettings = try XCTUnwrap(saved["settings"] as? [String: Any])
        for key in ["requested_seed", "cache_images_in_memory", "image_cache_limit_mb"] { savedSettings.removeValue(forKey: key) }
        saved["settings"] = savedSettings
        try NativeFactoryTraining.writeJSON(saved, to: checkpointURL)
        let originalDescriptor = try Data(contentsOf: checkpointURL)
        let weightsPath = try XCTUnwrap(saved["weights_path"] as? String)
        let originalWeights = try Data(contentsOf: directory.appendingPathComponent(weightsPath))

        let fixedEvents = Messages()
        let fixed = try await fixture.project.trainMap(datasetID: dataset.id, map: .depth, settings: settings,
            start: .checkpoint(runID: runID), progress: { fixedEvents.add($0.message) })
        XCTAssertTrue(fixedEvents.snapshot().contains { $0.contains("Resuming checkpoint with optimizer state") })
        XCTAssertEqual(try NativeWeights.read(from: fixed.directory.appendingPathComponent("model.safetensors")), expected)
        let fixedCheckpoint = try NativeFactoryTraining.json(fixed.directory.appendingPathComponent("checkpoint.json"))
        XCTAssertEqual(fixedCheckpoint["optimizer_updates"] as? Int, 2)
        XCTAssertEqual(fixedCheckpoint["updates"] as? Int, 2)

        var random = settings
        random.seed = -1
        let randomEvents = Messages()
        let continued = try await fixture.project.trainMap(datasetID: dataset.id, map: .depth, settings: random,
            start: .checkpoint(runID: runID), progress: { randomEvents.add($0.message) })
        XCTAssertTrue(randomEvents.snapshot().contains { $0.contains("fresh optimizer, shuffle and augmentation state") })
        let request = try NativeFactoryTraining.json(continued.directory.appendingPathComponent("request.json"))
        XCTAssertEqual(request["requested_seed"] as? Int, -1)
        XCTAssertTrue((0...Int(Int32.max)).contains(try XCTUnwrap(request["seed"] as? Int)))
        let randomCheckpoint = try NativeFactoryTraining.json(continued.directory.appendingPathComponent("checkpoint.json"))
        XCTAssertEqual(randomCheckpoint["optimizer_updates"] as? Int, 2,
            "Fresh optimizer state starts at zero rather than adding the source checkpoint's update")
        XCTAssertEqual(randomCheckpoint["updates"] as? Int, 2,
            "A new sampler visits its image in both epochs rather than resuming after the first image")
        XCTAssertEqual(randomCheckpoint["prior_updates"] as? Int, 1)
        XCTAssertEqual(randomCheckpoint["prior_images"] as? Int, 1)
        XCTAssertEqual((randomCheckpoint["history"] as? [[String: Any]])?.count, 2)
        XCTAssertEqual(try Data(contentsOf: checkpointURL), originalDescriptor)
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent(weightsPath)), originalWeights)
    }

    func testCachePreferenceAndBudgetPersistWithoutChangingTheRandomPreference() async throws {
        let fixture = try Fixture(materialCount: 1)
        defer { fixture.remove() }
        var settings = fixture.settings(seed: -1, epochs: 1)
        settings.cacheImagesInMemory = false
        settings.imageCacheLimitMB = 64
        try await fixture.project.updateSettings(ProjectSettings(training: settings))
        let reopened = try LoomProject(sourceFolder: fixture.root)
        var restored = try await reopened.snapshot().settings.training
        XCTAssertFalse(restored.cacheImagesInMemory)
        XCTAssertEqual(restored.imageCacheLimitMB, 64)
        XCTAssertEqual(restored.seed, -1)
        restored.cacheImagesInMemory = true
        try await reopened.updateSettings(ProjectSettings(training: restored))
        let encoded = try JSONEncoder().encode(restored)
        XCTAssertEqual(try JSONDecoder().decode(TrainingSettings.self, from: encoded), restored)
    }

    private func waitForPending(_ control: TrainingRunControl) async throws {
        for _ in 0..<1000 {
            if control.hasPendingRequest { return }
            try await Task.sleep(for: .milliseconds(1))
        }
        throw LoomError("Seed policy checkpoint request did not queue")
    }

    private struct Fixture {
        let root: URL
        let project: LoomProject
        let datasetID = ImageMetadataStore.categoryID("Bricks")
        init(materialCount: Int) throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-seed-policy-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            for material in 0..<materialCount {
                let folder = root.appendingPathComponent("material-\(material)")
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                for role in [MapRole.diffuse, .depth] {
                    let url = folder.appendingPathComponent(role.rawValue + ".png")
                    let channels = role == .diffuse ? 3 : 1
                    let bytes = (0..<(64 * 64 * channels)).map { UInt8(($0 * 7 + material * 31 + 13) % 251) }
                    let provider = try XCTUnwrap(CGDataProvider(data: Data(bytes) as CFData))
                    let image = try XCTUnwrap(CGImage(width: 64, height: 64, bitsPerComponent: 8, bitsPerPixel: 8 * channels,
                        bytesPerRow: 64 * channels, space: channels == 3 ? CGColorSpaceCreateDeviceRGB() : CGColorSpaceCreateDeviceGray(),
                        bitmapInfo: [], provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
                    let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
                    CGImageDestinationAddImage(destination, image, nil)
                    XCTAssertTrue(CGImageDestinationFinalize(destination))
                    try ImageMetadataStore.write(ImageMetadata(pairingKey: "material-\(material)", role: role,
                        category: "Bricks", roleWasManuallySelected: true, pairingWasManuallySelected: true), to: url)
                }
            }
            project = try LoomProject(sourceFolder: root)
        }
        func settings(seed: Int, epochs: Int) -> TrainingSettings {
            TrainingSettings(imageSize: 32, epochs: epochs, shuffleEachEpoch: true, flipCrops: true, rotateCrops: true,
                seed: seed, modelWidth: 8, modelBlocks: 1, cropRefresh: .perSample, augmentationRefresh: .perSample,
                validationFraction: 0, checkpointEveryEpochs: 100, cacheImagesInMemory: false, imageCacheLimitMB: 64)
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
    }
    private final class Messages: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [String] = []
        func add(_ value: String) { lock.withLock { values.append(value) } }
        func snapshot() -> [String] { lock.withLock { values } }
    }
}
