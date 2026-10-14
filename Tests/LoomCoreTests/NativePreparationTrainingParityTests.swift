import Foundation
import LoomNative
import Metal
import XCTest
@testable import LoomCore

final class NativePreparationTrainingParityTests: XCTestCase {
    private struct Batch: Sendable {
        let inputs: [Float]
        let targets: [Float]
        let count: Int
        let windowCount: Int
        let beginsWindow: Bool
        let endsWindow: Bool
    }

    private struct Measurement {
        let losses: [Double]
        let norms: [Double]
        let updates: Int
        let weights: [String: NativeTensor]
        let optimizer: [String: NativeTensor]
    }

    func testPreparingAheadPreservesCropOrderAccumulationLossesWeightsAndOptimizer() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal GPU") }
        let serial = try train(preparingAhead: false)
        let ahead = try train(preparingAhead: true)
        XCTAssertEqual(serial.updates, 4, "Two uneven accumulation windows in each of two epochs")
        XCTAssertEqual(ahead.updates, serial.updates)
        XCTAssertEqual(serial.losses.count, 6, "Two batches in the three-image window, one in the two-image window")
        XCTAssertEqual(ahead.losses.count, serial.losses.count)
        for (left, right) in zip(serial.losses, ahead.losses) { XCTAssertEqual(left, right, accuracy: 1e-8) }
        for (left, right) in zip(serial.norms, ahead.norms) { XCTAssertEqual(left, right, accuracy: 1e-8) }
        XCTAssertEqual(ahead.weights, serial.weights, "Lookahead must preserve every native optimizer update")
        XCTAssertEqual(ahead.optimizer, serial.optimizer, "Lookahead must preserve optimizer moments")
    }

    private func train(preparingAhead: Bool) throws -> Measurement {
        let size = 48
        let input = NativeImage(width: size, height: size, channels: 3,
            values: (0..<(size * size * 3)).map { Float(($0 * 7 + 13) % 251) / 250 })
        let target = NativeImage(width: size, height: size, channels: 1,
            values: (0..<(size * size)).map { Float(($0 * 11 + 31) % 251) / 250 })
        let cancellation = NativeCancellation()
        let session = try NativeTrainingSession(architecture: NativeArchitecture(mapKind: "depth", width: 2, blocks: 1),
            options: NativeTrainingOptions(optimizer: "adamw", weightDecay: 0.01, gradientClipNorm: 0.1),
            seed: 42, cancellation: cancellation)
        // A smaller final microbatch and a smaller final accumulation window
        // exercise sample weighting without changing optimizer boundaries.
        let descriptors: [(epoch: Int, offset: Int, count: Int, windowCount: Int, begins: Bool, ends: Bool)] =
            (0..<2).flatMap { epoch in
                [(epoch, 0, 2, 3, true, false), (epoch, 2, 1, 3, false, true), (epoch, 3, 2, 2, true, true)]
            }
        let prepare: @Sendable (Int) throws -> Batch = { index in
            let item = descriptors[index]
            var inputs: [Float] = [], targets: [Float] = []
            for sample in item.offset..<(item.offset + item.count) {
                let transform = NativeCropTransform(x: sample * 2, y: item.epoch * 3, size: 32,
                    horizontalFlip: sample % 2 == 0, verticalFlip: item.epoch == 1,
                    quarterTurns: (sample + item.epoch) % 4)
                let pair = try NativeImages.pair(input: input, target: target, transform: transform,
                    mapKind: "depth", cancellation: cancellation)
                inputs.append(contentsOf: pair.input.values); targets.append(contentsOf: pair.target.values)
            }
            return Batch(inputs: inputs, targets: targets, count: item.count, windowCount: item.windowCount,
                beginsWindow: item.begins, endsWindow: item.ends)
        }
        let pipeline = NativePreparationPipeline<Batch>(cancellation: cancellation)
        defer { pipeline.close() }
        if preparingAhead { try pipeline.submit { try prepare(0) } }
        var losses: [Double] = [], norms: [Double] = []
        for index in descriptors.indices {
            let batch = preparingAhead ? try pipeline.take() : try prepare(index)
            if preparingAhead, index + 1 < descriptors.count {
                let next = index + 1
                try pipeline.submit { try prepare(next) }
            }
            if batch.beginsWindow { session.resetAccumulatedGradients() }
            losses.append(try session.train(input: batch.inputs, target: batch.targets, batchSize: batch.count,
                height: 32, width: 32, weight: Float(batch.count) / Float(batch.windowCount)))
            if batch.endsWindow { norms.append(try session.applyGradients(learningRate: 0.001)) }
        }
        pipeline.close()
        return Measurement(losses: losses, norms: norms, updates: session.updateCount,
            weights: try session.weights(), optimizer: try session.optimizerState())
    }
}
