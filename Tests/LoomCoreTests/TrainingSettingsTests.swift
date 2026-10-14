import Foundation
import LoomNative
import XCTest
@testable import LoomCore

final class TrainingSettingsTests: XCTestCase {
    func testOlderProjectSettingsPreserveTheirValuesAndAcquireNewControls() throws {
        let json = Data("""
        {"imageSize":2048,"epochs":37,"batchSize":2,"learningRate":0.0007,
         "shuffleEachEpoch":false,"flipCrops":true,"rotateCrops":false,
         "exaggerateFeatures":true,"seed":7,"modelWidth":24,"continueLatestWeights":false}
        """.utf8)
        let settings = try JSONDecoder().decode(TrainingSettings.self, from: json)
        XCTAssertEqual(settings.imageSize, 2048); XCTAssertEqual(settings.modelWidth, 24)
        XCTAssertEqual(settings.modelArchitecture, NativeArchitecture.identifier)
        XCTAssertEqual(settings.scalarLoss, .absoluteV1)
        XCTAssertEqual(settings.epochs, 37); XCTAssertEqual(settings.learningRate, 0.0007)
        XCTAssertFalse(settings.shuffleEachEpoch); XCTAssertTrue(settings.flipCrops)
        XCTAssertEqual(settings.optimizer, .adamW); XCTAssertEqual(settings.gradientAccumulationSteps, 1)
        XCTAssertEqual(settings.cropRefresh, .perEpoch); XCTAssertEqual(settings.augmentationRefresh, .perEpoch)
        XCTAssertEqual(settings.sourceResolutionLevels, 0, "Saved projects keep all original source resolutions")
        XCTAssertNil(settings.validationIssue)
    }

    func testAllNewTrainingControlsRoundTripWithTheirSourceProject() throws {
        let settings = TrainingSettings(imageSize: 512, sourceResolutionLevels: 3, epochs: 23, batchSize: 2, learningRate: 0.0001,
            seed: 123, modelWidth: 116, modelBlocks: 6, optimizer: .lion, weightDecay: 0.1,
            beta1: 0.95, beta2: 0.98, gradientAccumulationSteps: 4, gradientClipNorm: 0.5,
            learningRateSchedule: .cosine, warmupEpochs: 2, minimumLearningRateRatio: 0.05,
            cropRefresh: .perRun, augmentationRefresh: .perSample, device: .metal,
            validationFraction: 0.2, validationSeed: 19, earlyStoppingPatience: 5,
            exportBestWeights: false, checkpointEveryEpochs: 3)
        XCTAssertEqual(try JSONDecoder().decode(TrainingSettings.self, from: JSONEncoder().encode(settings)), settings)
        XCTAssertEqual(settings.effectiveBatchSize, 8)
        XCTAssertEqual(settings.optimizerStepsPerEpoch(pairCount: 17), 3)
        XCTAssertEqual(settings.batchesPerEpoch(pairCount: 17), 9)
        XCTAssertEqual(TrainingSettings(optimizer: .lion).beta2, 0.99)
    }

    func testSourceResolutionLevelsDefaultToBaseAndRejectUnsupportedValues() {
        var settings = TrainingSettings()
        XCTAssertEqual(settings.sourceResolutionLevels, 1)
        for levels in 0...3 {
            settings.sourceResolutionLevels = levels
            XCTAssertNil(settings.validationIssue)
        }
        for levels in [-1, 4, Int.max] {
            settings.sourceResolutionLevels = levels
            XCTAssertEqual(settings.validationIssue, "Source resolution levels must be one to three, or all available.")
        }
    }

    func testParameterCountsAreDerivedFromFreshConvolutionsAndSafeWhileEditing() {
        XCTAssertEqual(TrainingSettings(modelWidth: 24, modelArchitecture: NativeArchitecture.identifier).parameterCount(map: .depth), 42_553)
        XCTAssertEqual(TrainingSettings(modelWidth: 84, modelArchitecture: NativeArchitecture.identifier).parameterCount(map: .depth), 511_813)
        XCTAssertEqual(TrainingSettings(modelWidth: 84, modelArchitecture: NativeArchitecture.identifier).parameterCount(map: .normal), 513_327)
        var settings = TrainingSettings()
        settings.modelWidth = Int.max; settings.modelBlocks = Int.max
        settings.batchSize = Int.max; settings.gradientAccumulationSteps = Int.max
        XCTAssertEqual(settings.parameterCount(map: .depth), 0)
        XCTAssertEqual(settings.effectiveBatchSize, 0)
        XCTAssertEqual(settings.optimizerStepsPerEpoch(pairCount: Int.max), 0)
        XCTAssertNotNil(settings.validationIssue)
    }

    func testNewProjectsUsePyramidAndParameterCountMatchesItsNativeConstructor() throws {
        let settings = TrainingSettings()
        XCTAssertEqual(settings.modelArchitecture, NativeArchitecture.pyramidIdentifier)
        XCTAssertEqual(settings.scalarLoss, .logitCrossEntropyV2)
        XCTAssertEqual(settings.modelWidth, 38)
        for map in MapRole.targets {
            let architecture = try NativeArchitecture(mapKind: map.rawValue, width: settings.modelWidth,
                blocks: settings.modelBlocks, identifier: settings.modelArchitecture)
            XCTAssertEqual(settings.parameterCount(map: map), architecture.parameterCount)
            XCTAssertTrue((1_900_000...2_100_000).contains(architecture.parameterCount))
        }
        var unknown = settings; unknown.modelArchitecture = "unregistered-network"
        XCTAssertNotNil(unknown.validationIssue)
        XCTAssertEqual(unknown.parameterCount(map: .depth), 0)
    }

    func testFreshRunMigrationKeepsApproximateCapacityAndOtherTrainingChoices() {
        var settings = TrainingSettings(imageSize: 1024, epochs: 500, learningRate: 0.0002,
            seed: 9587, modelWidth: 164, gradientAccumulationSteps: 2,
            modelArchitecture: NativeArchitecture.identifier)
        let oldCount = settings.parameterCount(map: .depth)
        settings.usePyramidForFreshRun(map: .depth)
        XCTAssertEqual(settings.modelArchitecture, NativeArchitecture.pyramidIdentifier)
        XCTAssertEqual(settings.modelWidth, 38)
        XCTAssertLessThan(abs(settings.parameterCount(map: .depth) - oldCount), oldCount / 20)
        XCTAssertEqual(settings.imageSize, 1024); XCTAssertEqual(settings.epochs, 500)
        XCTAssertEqual(settings.learningRate, 0.0002); XCTAssertEqual(settings.seed, 9587)
        XCTAssertEqual(settings.gradientAccumulationSteps, 2)
        let repeatSettings = settings
        settings.usePyramidForFreshRun(map: .depth)
        XCTAssertEqual(settings, repeatSettings)
    }

    func testOnlyActiveOptimizerCoefficientsCanDisableTraining() {
        var settings = TrainingSettings(optimizer: .sgd)
        settings.beta1 = 1; settings.beta2 = 1
        XCTAssertNil(settings.validationIssue)
        settings.optimizer = .adamW; settings.beta1 = 0.9; settings.beta2 = 0.999; settings.momentum = 2
        XCTAssertNil(settings.validationIssue)
        settings.optimizer = .sgd
        XCTAssertNotNil(settings.validationIssue)
    }

    func testHeldOutSplitKeepsAllResolutionsOfEachMaterialTogetherAndNeverLosesPairs() {
        let root = URL(fileURLWithPath: "/source")
        let pairs = (0..<5).flatMap { material in
            [64, 128].map { size in
                let folder = root.appendingPathComponent("material-\(material)")
                return TrainingPair(id: "\(material)-\(size)", input: image(folder, .diffuse, size), target: image(folder, .depth, size))
            }
        }
        let settings = TrainingSettings(imageSize: 32, validationFraction: 0.4, validationSeed: 7)
        let first = TrainingDataPlan(pairs: pairs, settings: settings, sourceFolder: root)
        let repeatPlan = TrainingDataPlan(pairs: pairs.reversed(), settings: settings, sourceFolder: root)
        XCTAssertEqual(first.trainingPairCount, 6); XCTAssertEqual(first.validationPairCount, 4)
        XCTAssertEqual(first.trainingMaterialCount, 3); XCTAssertEqual(first.validationMaterialCount, 2)
        let trainingFolders = Set(first.training.map { $0.input.url.deletingLastPathComponent() })
        let validationFolders = Set(first.validation.map { $0.input.url.deletingLastPathComponent() })
        XCTAssertTrue(trainingFolders.isDisjoint(with: validationFolders))
        XCTAssertEqual(Set(first.validation.map(\.id)), Set(repeatPlan.validation.map(\.id)))
        XCTAssertEqual(Set((first.training + first.validation).map(\.id)), Set(pairs.map(\.id)))
        let singleton = TrainingDataPlan(pairs: Array(pairs.prefix(2)), settings: settings, sourceFolder: root)
        XCTAssertEqual(singleton.trainingPairCount, 2); XCTAssertEqual(singleton.validationPairCount, 0)
    }

    private func image(_ folder: URL, _ role: MapRole, _ size: Int) -> SourceImage {
        let id = "\(role.rawValue)-\(size)"
        return SourceImage(id: folder.path + id, url: folder.appendingPathComponent(id + ".png"), displayName: id,
            role: role, category: "Bark", verified: true, note: "", pairingKey: folder.path + "-\(size)", width: size, height: size)
    }
}
