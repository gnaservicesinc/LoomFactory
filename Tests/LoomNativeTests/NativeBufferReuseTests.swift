import Foundation
import Metal
import XCTest
@testable import LoomNative

final class NativeBufferReuseTests: XCTestCase {
    func testFullResolutionNarrowGraphsReuseResultsWithoutCPUGradientReadback() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal GPU") }
        for identifier in [NativeArchitecture.identifier, NativeArchitecture.pyramidIdentifier] {
            let architecture = try NativeArchitecture(mapKind: "depth", width: 2, blocks: 1, identifier: identifier)
            let weights = architecture.parameterShapes.mapValues {
                NativeTensor(shape: $0, values: Array(repeating: 0, count: $0.reduce(1, *)))
            }
            let session = try NativeTrainingSession(architecture: architecture,
                options: NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0, gradientClipNorm: 0),
                seed: 42, weights: weights)
            let size = 1024
            let input = Array(repeating: Float(0), count: 3 * size * size)
            let target = Array(repeating: Float(1), count: size * size)
            var expectedBias: Float = 0
            for update in 0..<4 {
                let sigmoid = 1 / (1 + exp(-expectedBias))
                let expectedNorm = sigmoid * (1 - sigmoid)
                _ = try session.train(input: input, target: target, batchSize: 1, height: size, width: size)
                let norm = try session.applyGradients(learningRate: 0.001)
                XCTAssertEqual(norm, Double(expectedNorm), accuracy: 2e-6, "\(identifier) update \(update)")
                expectedBias += expectedNorm * 0.001
                if update > 1 {
                    XCTAssertEqual(session.lastExecutionMeasurements?.resultBufferAllocationCount, 0)
                }
            }
            let updated = try session.weights()
            XCTAssertEqual(updated["output.bias"]!.values[0], expectedBias, accuracy: 2e-6, identifier)
            for (name, tensor) in updated where name != "output.bias" {
                XCTAssertTrue(tensor.values.allSatisfy { $0 == 0 }, "Masked gradient changed \(identifier) \(name)")
            }
        }
    }

    func testRepeatedAccumulationAlternatesResultsWithoutOverwritingFeeds() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal GPU") }
        let architecture = try NativeArchitecture(mapKind: "depth", width: 4, blocks: 1,
                                                  identifier: NativeArchitecture.pyramidIdentifier)
        let weights = architecture.parameterShapes.mapValues {
            NativeTensor(shape: $0, values: Array(repeating: 0, count: $0.reduce(1, *)))
        }
        let session = try NativeTrainingSession(architecture: architecture,
            options: NativeTrainingOptions(optimizer: "sgd", weightDecay: 0, momentum: 0, gradientClipNorm: 0),
            seed: 42, weights: weights)
        let size = 16, target = Array(repeating: Float(1), count: 16 * 16)
        var expectedBias: Float = 0
        for update in 0..<8 {
            let sigmoid = 1 / (1 + exp(-expectedBias))
            let expectedNorm = sigmoid * (1 - sigmoid)
            for batch in 0..<3 {
                let input = Array(repeating: Float(batch) / 2, count: 3 * size * size)
                let loss = try session.train(input: input, target: target, batchSize: 1,
                                             height: size, width: size, weight: 1 / 3)
                XCTAssertEqual(loss, Double(1 - sigmoid), accuracy: 1e-6)
                if update > 0 || batch > 1 {
                    let measurement = try XCTUnwrap(session.lastExecutionMeasurements)
                    XCTAssertEqual(measurement.feedBufferAllocationCount, 0)
                    XCTAssertEqual(measurement.resultBufferAllocationCount, 0)
                }
            }
            // Read neither gradients nor weights between training and optimizer;
            // CPU materialization must not be necessary for correct GPU state.
            let norm = try session.applyGradients(learningRate: 0.01)
            XCTAssertEqual(norm, Double(expectedNorm), accuracy: 2e-6)
            expectedBias += expectedNorm * 0.01
            if update > 1 {
                let measurement = try XCTUnwrap(session.lastExecutionMeasurements)
                XCTAssertEqual(measurement.feedBufferAllocationCount, 0)
                XCTAssertEqual(measurement.resultBufferAllocationCount, 0)
            }
        }
        let updated = try session.weights()
        XCTAssertEqual(updated["output.bias"]!.values[0], expectedBias, accuracy: 2e-6)
        for (name, tensor) in updated where name != "output.bias" {
            XCTAssertTrue(tensor.values.allSatisfy { $0 == 0 }, "Masked gradient changed \(name)")
        }
    }

    func testChangedFeedsAndEvictedShapesMatchFreshGraphResults() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("No Metal GPU") }
        for map in ["depth", "normal"] {
            let architecture = try NativeArchitecture(mapKind: map, width: 4, blocks: 1,
                                                      identifier: NativeArchitecture.pyramidIdentifier)
            let options = try NativeTrainingOptions()
            let cached = try NativeTrainingSession(architecture: architecture, options: options, seed: 42)
            // Exercise changing content, batch size, odd rectangles, and eviction
            // after the fifth shape, then revisit an evicted graph.
            let shapes = [(1, 7, 5), (2, 5, 7), (1, 9, 9), (1, 3, 11), (1, 11, 3), (1, 7, 5)]
            for (index, dimensions) in shapes.enumerated() {
                let (batch, height, width) = dimensions
                for content in 0..<2 {
                    let pixels = batch * height * width
                    let input = (0..<(3 * pixels)).map { Float(($0 * 7 + index * 13 + content * 19) % 101) / 100 }
                    let target = (0..<(architecture.outputChannels * pixels)).map { Float(($0 + content * 23) % 101) / 100 }
                    let fresh = try NativeTrainingSession(architecture: architecture, options: options, seed: 42)
                    XCTAssertEqual(try cached.predict(input: input, batchSize: batch, height: height, width: width),
                                   try fresh.predict(input: input, batchSize: batch, height: height, width: width))
                    if content > 0 {
                        XCTAssertEqual(cached.lastExecutionMeasurements?.feedBufferAllocationCount, 0)
                        XCTAssertEqual(cached.lastExecutionMeasurements?.resultBufferAllocationCount, 0)
                    }
                    XCTAssertEqual(try cached.evaluate(input: input, target: target, batchSize: batch, height: height, width: width),
                                   try fresh.evaluate(input: input, target: target, batchSize: batch, height: height, width: width), accuracy: 1e-7)
                    if content > 0 {
                        XCTAssertEqual(cached.lastExecutionMeasurements?.feedBufferAllocationCount, 0)
                        XCTAssertEqual(cached.lastExecutionMeasurements?.resultBufferAllocationCount, 0)
                    }
                }
            }
        }
    }
}
