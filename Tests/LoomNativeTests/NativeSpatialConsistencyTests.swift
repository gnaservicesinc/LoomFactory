import Metal
import XCTest
@testable import LoomNative

final class NativeSpatialConsistencyTests: XCTestCase {
    func testFullResolutionPredictionsDoNotDependOnCropGridPhase() throws {
        if MTLCreateSystemDefaultDevice() == nil { throw XCTSkip("No Metal GPU") }
        let image = NativeImage(width: 128, height: 128, channels: 3,
            values: (0..<(128 * 128 * 3)).map { Float(($0 * 71 + 17) % 257) / 256 })
        let target = NativeImage(width: 128, height: 128, channels: 1,
            values: Array(repeating: 0.5, count: 128 * 128))
        for channelsLast in [false, true] {
            let model = try NativeTrainingSession(architecture: NativeArchitecture(mapKind: "depth", width: 3, blocks: 1),
                options: NativeTrainingOptions(scalarLoss: .nativeDetailV3), seed: 42, channelsLast: channelsLast)
            func predict(x: Int, y: Int) throws -> [Float] {
                let pair = try NativeImages.pair(input: image, target: target,
                    transform: NativeCropTransform(x: x, y: y, size: 64), mapKind: "depth")
                return try model.predict(input: pair.input.values, batchSize: 1, height: 64, width: 64)
            }
            let reference = try predict(x: 32, y: 32)
            for (dx, dy) in [(1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1)] {
                let shifted = try predict(x: 32 + dx, y: 32 + dy)
                var maximumError: Float = 0
                // The margin exceeds the receptive-field radius. Both outputs
                // see identical original context at these corresponding pixels.
                for y in 16..<44 { for x in 16..<44 {
                    maximumError = max(maximumError, abs(reference[(y + dy) * 64 + x + dx] - shifted[y * 64 + x]))
                } }
                XCTAssertLessThan(maximumError, 2e-6, "Crop shift \(dx),\(dy); channelsLast=\(channelsLast)")
            }
        }
    }
}
