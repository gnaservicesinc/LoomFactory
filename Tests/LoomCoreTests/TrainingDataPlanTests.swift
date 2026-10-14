import Foundation
import XCTest
@testable import LoomCore

final class TrainingDataPlanTests: XCTestCase {
    private let root = URL(fileURLWithPath: "/source")

    func testSourceResolutionLevelsChooseExistingOriginalsAndUseEachMaterialsFirstUsableGrid() throws {
        let images = pairs("all-grids", sizes: [512, 1024, 2048, 4096, 8192])
            + pairs("missing-grid", sizes: [1024, 4096, 8192])
            + pairs("large-only", sizes: [4096])
        let snapshot = snapshot(images)
        var settings = TrainingSettings(imageSize: 1024, sourceResolutionLevels: 1, validationFraction: 0)
        let base = try plan(snapshot, settings)
        XCTAssertEqual(base.sourceImageCount, 18)
        XCTAssertEqual(base.sourceMaterialCount, 3)
        XCTAssertEqual(base.eligiblePairCount, 8)
        XCTAssertEqual(base.totalPairCount, 3)
        XCTAssertEqual(base.excludedPairCount, 5)
        XCTAssertEqual(Set(base.training.map(\.id)), ["all-grids-1024", "missing-grid-1024", "large-only-4096"])
        XCTAssertEqual(base.resolutions.first?.width, 512)
        XCTAssertEqual(base.resolutions.first?.eligiblePairCount, 0)

        settings.sourceResolutionLevels = 2
        let next = try plan(snapshot, settings)
        XCTAssertEqual(next.totalPairCount, 4)
        XCTAssertEqual(Set(next.training.map(\.id)), ["all-grids-1024", "all-grids-2048", "missing-grid-1024", "large-only-4096"])
        settings.sourceResolutionLevels = 3
        let higher = try plan(snapshot, settings)
        XCTAssertEqual(higher.totalPairCount, 6)
        XCTAssertEqual(higher.excludedPairCount, 2)
        XCTAssertEqual(higher.resolutions.first(where: { $0.width == 8192 })?.eligiblePairCount, 2)
        XCTAssertEqual(higher.resolutions.first(where: { $0.width == 8192 })?.totalPairCount, 0)

        settings.sourceResolutionLevels = 0
        let all = try plan(snapshot, settings)
        XCTAssertEqual(all.totalPairCount, 8)
        XCTAssertEqual(all.excludedPairCount, 0)
        XCTAssertEqual(Set(all.training.map(\.id)), Set(try TrainingPairSelector.select(images: images, dataset: "Bark", map: .depth, size: 1024).map(\.id)))
        for (levels, selected) in [(1, base), (2, next), (3, higher), (0, all)] {
            XCTAssertEqual(selected.resolutions.reduce(0) { $0 + $1.sourceImageCount }, selected.sourceImageCount)
            XCTAssertEqual(selected.resolutions.reduce(0) { $0 + $1.eligiblePairCount }, selected.eligiblePairCount)
            XCTAssertEqual(selected.resolutions.reduce(0) { $0 + $1.trainingPairCount }, selected.trainingPairCount)
            XCTAssertEqual(selected.resolutions.reduce(0) { $0 + $1.validationPairCount }, selected.validationPairCount)
            settings.sourceResolutionLevels = levels
            XCTAssertEqual(selected.training.map(\.input), try TrainingPairSelector.select(images: images, dataset: "Bark", map: .depth, settings: settings).map(\.input))
        }
        XCTAssertEqual(snapshot.images, images, "Planning must not change source records or their dimensions")
    }

    func testCountsIncludeIgnoredAndUnpairedSourceFilesButOnlyCurrentEligibleMapsEnterTraining() throws {
        var images = pairs("usable", sizes: [1024])
            + pairs("ignored-target", sizes: [1024])
            + pairs("too-small", sizes: [512])
        images[3].ignored = true
        images.append(image("unpaired", role: .diffuse, size: 2048))
        images.append(image("ignored-only", role: .unknown, size: 1024, ignored: true))
        var otherCategory = image("other-category", role: .diffuse, size: 1024)
        otherCategory.category = "Wood"
        images.append(otherCategory)
        let selected = try plan(snapshot(images), .init(imageSize: 1024, validationFraction: 0))
        XCTAssertEqual(selected.sourceImageCount, 8)
        XCTAssertEqual(selected.sourceMaterialCount, 5)
        XCTAssertEqual(selected.ignoredSourceImageCount, 2)
        XCTAssertEqual(selected.eligiblePairCount, 1)
        XCTAssertEqual(selected.eligibleMaterialCount, 1)
        XCTAssertEqual(selected.trainingPairCount, 1)
        XCTAssertEqual(selected.unavailableMaterialCount, 4)
        XCTAssertFalse(selected.training[0].input.verified)
        let counts = selected.resolutions.first(where: { $0.width == 1024 })
        XCTAssertEqual(counts?.sourceImageCount, 5)
        XCTAssertEqual(counts?.eligiblePairCount, 1)
        XCTAssertEqual(counts?.totalPairCount, 1)
    }

    func testHoldoutKeepsEverySelectedResolutionOfTheSamePhysicalFolderTogether() throws {
        let images = (0..<5).flatMap { pairs("material-\($0)", sizes: [1024, 2048, 4096]) }
        let settings = TrainingSettings(imageSize: 1024, sourceResolutionLevels: 2, epochs: 3,
            batchSize: 2, gradientAccumulationSteps: 2, validationFraction: 0.4, validationSeed: 7)
        let selected = try plan(snapshot(images), settings)
        let repeated = try plan(snapshot(images.reversed()), settings)
        XCTAssertEqual(selected.sourceImageCount, 30)
        XCTAssertEqual(selected.sourceMaterialCount, 5)
        XCTAssertEqual(selected.eligiblePairCount, 15)
        XCTAssertEqual(selected.totalPairCount, 10)
        XCTAssertEqual(selected.trainingPairCount, 6)
        XCTAssertEqual(selected.validationPairCount, 4)
        XCTAssertEqual(selected.trainingMaterialCount, 3)
        XCTAssertEqual(selected.validationMaterialCount, 2)
        XCTAssertEqual(Set(selected.validation.map(\.id)), Set(repeated.validation.map(\.id)))
        let trainingFolders = Set(selected.training.map { $0.input.url.deletingLastPathComponent() })
        let validationFolders = Set(selected.validation.map { $0.input.url.deletingLastPathComponent() })
        XCTAssertTrue(trainingFolders.isDisjoint(with: validationFolders))
        for resolution in selected.resolutions.filter({ $0.width != 4096 }) {
            XCTAssertEqual(resolution.trainingPairCount, 3)
            XCTAssertEqual(resolution.validationPairCount, 2)
        }
        XCTAssertEqual(selected.batchesPerEpoch, 3)
        XCTAssertEqual(selected.optimizerStepsPerEpoch, 2)
        XCTAssertEqual(selected.totalBatches, 9)
        XCTAssertEqual(selected.totalOptimizerUpdates, 6)
        XCTAssertEqual(selected.trainingSampleVisits, 18)
        XCTAssertEqual(selected.validationSampleVisits, 12)
    }

    func testPartialBatchAndAccumulationWindowCountsMatchNativeTrainingLoop() throws {
        let images = (0..<5).flatMap { pairs("material-\($0)", sizes: [1024]) }
        let settings = TrainingSettings(imageSize: 1024, epochs: 3, batchSize: 2,
            gradientAccumulationSteps: 2, validationFraction: 0)
        let selected = try plan(snapshot(images), settings)
        var batches = 0, updates = 0, visits = 0
        for window in stride(from: 0, to: selected.trainingPairCount, by: settings.effectiveBatchSize) {
            let end = min(window + settings.effectiveBatchSize, selected.trainingPairCount)
            for offset in stride(from: window, to: end, by: settings.batchSize) {
                visits += min(offset + settings.batchSize, end) - offset
                batches += 1
            }
            updates += 1
        }
        XCTAssertEqual(selected.batchesPerEpoch, batches)
        XCTAssertEqual(selected.optimizerStepsPerEpoch, updates)
        XCTAssertEqual(selected.trainingSampleVisits, visits * settings.epochs)
        XCTAssertEqual(selected.totalBatches, 9)
        XCTAssertEqual(selected.totalOptimizerUpdates, 6)
        XCTAssertEqual(selected.validationSampleVisits, 0)
    }

    func testNormalConventionAndDimensionChecksStillApplyBeforeLevelSelection() throws {
        var normal = image("normal-material", role: .normal, size: 1024)
        let diffuse = image("normal-material", role: .diffuse, size: 1024)
        normal = SourceImage(id: normal.id, url: normal.url, displayName: normal.displayName, role: .normal,
            category: "Bark", verified: false, note: "", pairingKey: normal.pairingKey, width: 1024, height: 1024, normalConvention: "DirectX")
        let directX = try snapshot([diffuse, normal]).trainingPlan(datasetID: "bark", map: .normal, settings: .init(imageSize: 1024))
        XCTAssertEqual(directX?.sourceImageCount, 2)
        XCTAssertEqual(directX?.totalPairCount, 0)
        let mismatched = SourceImage(id: "mismatch", url: normal.url, displayName: normal.displayName, role: .depth,
            category: "Bark", verified: false, note: "", pairingKey: diffuse.pairingKey, width: 2048, height: 1024)
        XCTAssertThrowsError(try snapshot([diffuse, mismatched]).trainingPlan(datasetID: "bark", map: .depth, settings: .init(imageSize: 1024)))
    }

    private func plan(_ snapshot: ProjectSnapshot, _ settings: TrainingSettings) throws -> TrainingDataPlan {
        try XCTUnwrap(snapshot.trainingPlan(datasetID: "bark", map: .depth, settings: settings))
    }

    private func snapshot(_ images: [SourceImage]) -> ProjectSnapshot {
        ProjectSnapshot(sourceURL: root, projectID: "project", settings: .init(), images: images,
            datasets: [.init(id: "bark", name: "Bark", imageCount: images.count, verifiedCount: 0, pairCounts: [:])], artifacts: [])
    }

    private func pairs(_ material: String, sizes: [Int]) -> [SourceImage] {
        sizes.flatMap { size in [MapRole.diffuse, .depth].map { image(material, role: $0, size: size) } }
    }

    private func image(_ material: String, role: MapRole, size: Int, ignored: Bool = false) -> SourceImage {
        let id = "\(material)-\(role.rawValue)-\(size)"
        return SourceImage(id: id, url: root.appendingPathComponent(material).appendingPathComponent("\(role.rawValue)-\(size).png"),
            displayName: material, role: role, category: "Bark", verified: false, note: "", pairingKey: "\(material)-\(size)",
            width: size, height: size, ignored: ignored)
    }
}
