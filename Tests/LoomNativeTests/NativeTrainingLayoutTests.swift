import Metal
import XCTest
@testable import LoomNative

final class NativeTrainingLayoutTests: XCTestCase {
    func testCalibrationPreservesWeightsMomentsAndRealAccumulation() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal GPU") }
        let architecture = try NativeArchitecture(mapKind: "depth", width: 9, blocks: 1)
        let options = try NativeTrainingOptions(scalarLoss: .logitCrossEntropyV2)
        let previous = try NativeTrainingSession(architecture: architecture, options: options, seed: 9587,
            channelsLast: true)
        _ = try previous.train(input: Array(repeating: 0.4, count: 3 * 32 * 32),
            target: Array(repeating: 0.8, count: 32 * 32), batchSize: 1, height: 32, width: 32)
        _ = try previous.applyGradients(learningRate: 0.0001)
        let before = try previous.weights(), moments = try previous.optimizerState()
        XCTAssertTrue(moments.values.contains { $0.values.contains { $0 != 0 } })
        let model = try NativeTrainingSession(architecture: architecture, options: options, seed: 0,
            weights: before, optimizerState: moments, initialUpdateCount: previous.updateCount)
        let size = 512
        let input = (0..<(3 * size * size)).map { Float(($0 * 13) % 53) / 52 }
        let target = (0..<(size * size)).map { Float(($0 * 7) % 31) / 30 }
        let loss = try model.train(input: input, target: target, batchSize: 1, height: size, width: size, weight: 0.3)
        let selection = try XCTUnwrap(model.trainingLayoutSelection)
        XCTAssertTrue(selection.gradientsAgree)
        XCTAssertGreaterThan(selection.nchwSeconds, 0)
        XCTAssertGreaterThan(selection.nhwcSeconds, 0)
        XCTAssertEqual(model.channelsLast, selection.channelsLast)
        XCTAssertEqual(model.updateCount, previous.updateCount, "Calibration cannot count as an optimizer update")
        XCTAssertEqual(try model.weights(), before)
        XCTAssertEqual(try model.optimizerState(), moments)
        let reference = try NativeTrainingSession(architecture: architecture, options: options, seed: 0,
            weights: before, optimizerState: moments, initialUpdateCount: previous.updateCount, channelsLast: model.channelsLast)
        XCTAssertEqual(loss, try reference.train(input: input, target: target, batchSize: 1, height: size, width: size, weight: 0.3))
        XCTAssertEqual(try model.accumulatedGradients(), try reference.accumulatedGradients(),
            "Only the real weighted microbatch belongs in accumulation")
        XCTAssertNil(reference.trainingLayoutSelection, "Explicit layouts bypass calibration")
        for session in [model, reference] { _ = try session.applyGradients(learningRate: 0.0001) }
        XCTAssertEqual(try model.weights(), try reference.weights())
        XCTAssertEqual(try model.optimizerState(), try reference.optimizerState())
    }

    func testFullGradientComparisonRejectsEqualNormCorruption() {
        let reference = ["w": NativeTensor(shape: [2], values: [1, 2])]
        XCTAssertTrue(NativeTrainingLayout.gradientsAgree(reference: reference,
            candidate: ["w": NativeTensor(shape: [2], values: [1.0001, 1.9999])]))
        for bad in [[Float(2), 1], [-1, -2], [.nan, 2], [.infinity, 2]] {
            XCTAssertFalse(NativeTrainingLayout.gradientsAgree(reference: reference,
                candidate: ["w": NativeTensor(shape: [2], values: bad)]))
        }
        XCTAssertFalse(NativeTrainingLayout.gradientsAgree(reference: reference, candidate: [:]))
        XCTAssertFalse(NativeTrainingLayout.gradientsAgree(reference: reference,
            candidate: ["w": NativeTensor(shape: [1, 2], values: [1, 2])]))
        XCTAssertFalse(NativeTrainingLayout.gradientsAgree(
            reference: ["w": NativeTensor(shape: [1], values: [0])],
            candidate: ["w": NativeTensor(shape: [1], values: [1e-8])]))
    }

    func testOptionalCalibrationFailureFallsBackAndCancellationPropagates() throws {
        let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 1)
        let cancellation = NativeCancellation()
        func select() throws -> NativeTrainingLayoutSelection {
            try NativeTrainingLayout.select(architecture: architecture, options: NativeTrainingOptions(),
                adapter: nil, input: [], target: [], batch: 1, height: 512, width: 512,
                weights: { throw NativeTrainingError.unavailable("Diagnostic allocation unavailable") },
                adapters: { nil }, cancellation: cancellation)
        }
        let fallback = try select()
        XCTAssertTrue(fallback.channelsLast)
        XCTAssertFalse(fallback.gradientsAgree)
        XCTAssertTrue(try XCTUnwrap(fallback.fallbackReason).contains("Diagnostic allocation unavailable"))
        cancellation.cancel()
        XCTAssertThrowsError(try select()) { XCTAssertTrue($0 is CancellationError) }
    }

    func testSmallShapesPyramidAndCancelledRunsDoNotCalibrateOrUpdate() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal GPU") }
        for identifier in [NativeArchitecture.identifier, NativeArchitecture.pyramidIdentifier] {
            let architecture = try NativeArchitecture(mapKind: "depth", width: 8, blocks: 1, identifier: identifier)
            let token = NativeCancellation()
            let model = try NativeTrainingSession(architecture: architecture, options: NativeTrainingOptions(), seed: 42,
                cancellation: token)
            let input = Array(repeating: Float(0.3), count: 3 * 32 * 32)
            let target = Array(repeating: Float(0.8), count: 32 * 32)
            _ = try model.train(input: input, target: target, batchSize: 1, height: 32, width: 32)
            XCTAssertNil(model.trainingLayoutSelection)
            XCTAssertThrowsError(try model.train(input: [], target: [], batchSize: -1, height: 1024, width: 1024))
            XCTAssertThrowsError(try model.train(input: [], target: [], batchSize: Int.max, height: Int.max, width: Int.max))
            let before = try model.weights()
            token.cancel()
            XCTAssertThrowsError(try model.train(input: input, target: target, batchSize: 1, height: 32, width: 32))
            XCTAssertEqual(try model.weights(), before)
            XCTAssertEqual(model.updateCount, 0)
        }
    }
}
