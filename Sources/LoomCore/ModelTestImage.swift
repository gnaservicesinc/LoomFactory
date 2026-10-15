import CoreGraphics
import Foundation
import ImageIO
import LoomNative
import UniformTypeIdentifiers

struct ModelTestTilePlan {
    struct Tile {
        let x: Int, y: Int, width: Int, height: Int
    }
    let width: Int, height: Int, overlap: Int
    let tiles: [Tile]

    init(width: Int, height: Int, tileSize: Int) throws {
        guard width > 0, height > 0, (1...8192).contains(tileSize),
              width <= Int.max / height / 3 else { throw LoomError("The test image has invalid native dimensions.") }
        self.width = width; self.height = height
        let tileWidth = min(width, tileSize), tileHeight = min(height, tileSize)
        let overlap = max(0, min(64, min(tileWidth, tileHeight) / 4))
        self.overlap = overlap
        func starts(length: Int, size: Int) -> [Int] {
            guard length > size else { return [0] }
            let step = max(1, size - 2 * overlap)
            var values = [0]
            while values.last! + size < length { values.append(min(length - size, values.last! + step)) }
            return values
        }
        tiles = starts(length: height, size: tileHeight).flatMap { y in
            starts(length: width, size: tileWidth).map { x in Tile(x: x, y: y, width: tileWidth, height: tileHeight) }
        }
    }
}

enum ModelTestImage {
    /// Returns NCHW values. Tile inputs are exact copies of the original native
    /// pixels; the feather affects predictions only, never the diffuse image.
    static func predict(image: NativeImage, plan: ModelTestTilePlan, map: MapRole,
                        cancellation: NativeCancellation, progress: @escaping @Sendable (TrainingEvent) -> Void,
                        predictor: ModelTestWorker.Predictor) throws -> [Float] {
        let channels = map == .normal ? 3 : 1, count = image.width * image.height
        guard image.width == plan.width, image.height == plan.height, image.channels == 3,
              image.values.count == count * 3 else { throw LoomError("The native test image dimensions changed.") }
        var sums = [Float](repeating: 0, count: count * channels)
        var weights = [Float](repeating: 0, count: count)
        for (index, tile) in plan.tiles.enumerated() {
            try cancellation.check()
            progress(.init(type: "status", message: "Testing tile \(index + 1) of \(plan.tiles.count)…",
                progress: Double(index) / Double(plan.tiles.count)))
            try autoreleasepool {
                let tileCount = tile.width * tile.height
                var input = [Float](repeating: 0, count: tileCount * 3)
                for y in 0..<tile.height {
                    if y % 16 == 0 { try cancellation.check() }
                    for x in 0..<tile.width {
                        let source = ((tile.y + y) * image.width + tile.x + x) * 3
                        for channel in 0..<3 { input[channel * tileCount + y * tile.width + x] = image.values[source + channel] }
                    }
                }
                let prediction = try predictor(input, tile.height, tile.width)
                guard prediction.count == tileCount * channels,
                      prediction.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
                    throw LoomError("The saved model produced invalid native output pixels.")
                }
                try cancellation.check()
                func feather(_ coordinate: Int, size: Int, origin: Int, limit: Int) -> Float {
                    guard plan.overlap > 0 else { return 1 }
                    var value: Float = 1
                    if origin > 0 { value = min(value, Float(coordinate + 1) / Float(plan.overlap + 1)) }
                    if origin + size < limit { value = min(value, Float(size - coordinate) / Float(plan.overlap + 1)) }
                    return value
                }
                for y in 0..<tile.height {
                    if y % 16 == 0 { try cancellation.check() }
                    let wy = feather(y, size: tile.height, origin: tile.y, limit: image.height)
                    for x in 0..<tile.width {
                        let weight = wy * feather(x, size: tile.width, origin: tile.x, limit: image.width)
                        let destination = (tile.y + y) * image.width + tile.x + x
                        weights[destination] += weight
                        for channel in 0..<channels {
                            sums[channel * count + destination] += prediction[channel * tileCount + y * tile.width + x] * weight
                        }
                    }
                }
            }
        }
        for pixel in 0..<count {
            if pixel % 65536 == 0 { try cancellation.check() }
            guard weights[pixel] > 0 else { throw LoomError("The test output has an uncovered native pixel.") }
            for channel in 0..<channels { sums[channel * count + pixel] /= weights[pixel] }
            if map == .normal {
                let x = sums[pixel] * 2 - 1, y = sums[count + pixel] * 2 - 1, z = sums[2 * count + pixel] * 2 - 1
                let length = max(sqrt(x * x + y * y + z * z), 1e-6)
                sums[pixel] = x / length * 0.5 + 0.5
                sums[count + pixel] = y / length * 0.5 + 0.5
                sums[2 * count + pixel] = z / length * 0.5 + 0.5
            }
        }
        return sums
    }

    static func writePNG(_ values: [Float], channels: Int, width: Int, height: Int, to url: URL,
                         clampPreview: Bool = false,
                         cancellation: NativeCancellation = NativeCancellation()) throws {
        let count = width * height
        guard [1, 3].contains(channels), width > 0, height > 0, values.count == count * channels,
              values.allSatisfy({ $0.isFinite && (clampPreview || ($0 >= 0 && $0 <= 1)) }) else {
            throw LoomError("Cannot save invalid model test pixels.")
        }
        var data = Data(count: count * channels * MemoryLayout<UInt16>.stride)
        try data.withUnsafeMutableBytes { storage in
            let pixels = storage.bindMemory(to: UInt16.self)
            for pixel in 0..<count {
                if pixel % 65536 == 0 { try cancellation.check() }
                for channel in 0..<channels {
                    let value = values[channel * count + pixel]
                    let preview = clampPreview ? min(1, max(0, value)) : value
                    pixels[pixel * channels + channel] = UInt16((preview * 65535).rounded()).bigEndian
                }
            }
        }
        let space = channels == 1 ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: data as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 16, bitsPerPixel: 16 * channels,
                bytesPerRow: width * channels * 2, space: space,
                bitmapInfo: [.byteOrder16Big, CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)],
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw LoomError("Could not create the full-resolution model test output.")
        }
        try cancellation.check()
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw LoomError("Could not save the full-resolution model test output.") }
    }
}

/// Statistics describe the unchanged Float32 prediction, independently of its
/// PNG preview. Endpoint checks apply to scalar maps; normal components can
/// legitimately contain endpoints without representing a white or black map.
struct ModelTestOutputStatistics {
    static let endpointTolerance = 1e-5
    let minimum: Float
    let maximum: Float
    let mean: Double
    let nearWhiteFraction: Double
    let nearBlackFraction: Double
    let valueCount: Int

    init(values: [Float], cancellation: NativeCancellation) throws {
        guard !values.isEmpty else { throw LoomError("Cannot inspect an empty model output.") }
        var minimum = Float.greatestFiniteMagnitude, maximum = -Float.greatestFiniteMagnitude
        var sum: Double = 0, nearWhite = 0, nearBlack = 0
        for (index, value) in values.enumerated() {
            if index % 65536 == 0 { try cancellation.check() }
            guard value.isFinite else { throw LoomError("Cannot inspect non-finite model output pixels.") }
            minimum = min(minimum, value); maximum = max(maximum, value)
            let numeric = Double(value)
            sum += numeric
            if abs(1 - numeric) <= Self.endpointTolerance { nearWhite += 1 }
            if abs(numeric) <= Self.endpointTolerance { nearBlack += 1 }
        }
        self.minimum = minimum; self.maximum = maximum
        valueCount = values.count
        mean = sum / Double(valueCount)
        nearWhiteFraction = Double(nearWhite) / Double(valueCount)
        nearBlackFraction = Double(nearBlack) / Double(valueCount)
    }

    var record: [String: Any] {
        ["minimum": minimum, "maximum": maximum, "mean": mean,
         "near_white_fraction": nearWhiteFraction, "near_black_fraction": nearBlackFraction,
         "endpoint_tolerance": Self.endpointTolerance, "value_count": valueCount]
    }

    func saturationNote(map: MapRole) -> String? {
        guard map == .depth || map == .roughness else { return nil }
        let endpoint: String
        if nearWhiteFraction > 0.999 { endpoint = "white" }
        else if nearBlackFraction > 0.999 { endpoint = "black" }
        else { return nil }
        return "The model produced an almost entirely \(endpoint) map: over 99.9% of values are within 0.00001 of the endpoint. Training may have saturated."
    }
}
