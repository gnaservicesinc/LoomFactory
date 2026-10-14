import Foundation
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
        XCTAssertEqual(settings.epochs, 37); XCTAssertEqual(settings.learningRate, 0.0007)
        XCTAssertFalse(settings.shuffleEachEpoch); XCTAssertTrue(settings.flipCrops)
        XCTAssertEqual(settings.optimizer, .adamW); XCTAssertEqual(settings.gradientAccumulationSteps, 1)
        XCTAssertEqual(settings.cropRefresh, .perEpoch); XCTAssertEqual(settings.augmentationRefresh, .perEpoch)
        XCTAssertNil(settings.validationIssue)
    }

    func testAllNewTrainingControlsRoundTripWithTheirSourceProject() throws {
        let settings = TrainingSettings(imageSize: 512, epochs: 23, batchSize: 2, learningRate: 0.0001,
            seed: 123, modelWidth: 116, modelBlocks: 6, optimizer: .lion, weightDecay: 0.1,
            beta1: 0.95, beta2: 0.98, gradientAccumulationSteps: 4, gradientClipNorm: 0.5,
            learningRateSchedule: .cosine, warmupEpochs: 2, minimumLearningRateRatio: 0.05,
            cropRefresh: .perRun, augmentationRefresh: .perSample, device: .metal,
            validationFraction: 0.2, validationSeed: 19, earlyStoppingPatience: 5,
            exportBestWeights: false, checkpointEveryEpochs: 3)
        XCTAssertEqual(try JSONDecoder().decode(TrainingSettings.self, from: JSONEncoder().encode(settings)), settings)
        XCTAssertEqual(settings.effectiveBatchSize, 8)
        XCTAssertEqual(settings.optimizerStepsPerEpoch(pairCount: 17), 3)
        XCTAssertEqual(TrainingSettings(optimizer: .lion).beta2, 0.99)
    }

    func testParameterCountsAreDerivedFromFreshConvolutionsAndSafeWhileEditing() {
        XCTAssertEqual(TrainingSettings(modelWidth: 24).parameterCount(map: .depth), 42_553)
        XCTAssertEqual(TrainingSettings(modelWidth: 84).parameterCount(map: .depth), 511_813)
        XCTAssertEqual(TrainingSettings(modelWidth: 84).parameterCount(map: .normal), 513_327)
        var settings = TrainingSettings()
        settings.modelWidth = Int.max; settings.modelBlocks = Int.max
        settings.batchSize = Int.max; settings.gradientAccumulationSteps = Int.max
        XCTAssertEqual(settings.parameterCount(map: .depth), 0)
        XCTAssertEqual(settings.effectiveBatchSize, 0)
        XCTAssertEqual(settings.optimizerStepsPerEpoch(pairCount: Int.max), 0)
        XCTAssertNotNil(settings.validationIssue)
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
