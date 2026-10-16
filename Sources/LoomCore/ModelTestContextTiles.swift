import Foundation
import LoomNative

public enum ModelTestBoundaryHandling: String, Sendable, CaseIterable {
    case context = "context-reflect-valid-v1"
    case legacy = "linear-feather-overlap-v1"
}

/// For finite-support convolutions only. Compact's global attention must never
/// use this path. The learned graph and weights stay unchanged.
struct ModelTestContextPlan {
    struct Tile { let x: Int, y: Int, width: Int, height: Int }
    let width: Int, height: Int, halo: Int, phase: Int
    let inputWidth: Int, inputHeight: Int, coreWidth: Int, coreHeight: Int
    let tiles: [Tile]

    static func halo(for architecture: NativeArchitecture) throws -> (Int, Int) {
        guard !architecture.isCompact else { throw LoomError("Compact attention requires the complete native field.") }
        // Pyramid's longest path: 3x3 input, two stride-2 encoders, 2*blocks
        // quarter-scale convolutions, two nearest/3x3 decoders, 3x3 output.
        // 8*blocks+12 conservatively covers every nearest-neighbor phase.
        return architecture.isPyramid ? (8 * architecture.blocks + 12, 4) : (2 * architecture.blocks + 2, 1)
    }

    init(width: Int, height: Int, tileSize: Int, halo: Int, phase: Int) throws {
        guard width > 0, height > 0, width <= Int.max / height / 3,
              tileSize > 0, tileSize <= 8192, halo > 0, phase > 0,
              halo % phase == 0, tileSize >= 2 * halo + phase else {
            throw LoomError("The inference tile is too small for this model's reliable context. Choose legacy edge handling to compare it.")
        }
        self.width = width; self.height = height; self.halo = halo; self.phase = phase
        func size(_ length: Int) -> Int {
            min(tileSize / phase * phase, (length + 2 * halo + phase - 1) / phase * phase)
        }
        let iw = size(width), ih = size(height), cw = iw - 2 * halo, ch = ih - 2 * halo
        inputWidth = iw; inputHeight = ih; coreWidth = cw; coreHeight = ch
        tiles = stride(from: 0, to: height, by: ch).flatMap { y in
            stride(from: 0, to: width, by: cw).map { x in
                Tile(x: x, y: y, width: min(cw, width-x), height: min(ch, height-y))
            }
        }
    }

    static func reflected(_ index: Int, length: Int) -> Int {
        guard length > 1 else { return 0 }
        let period = 2 * (length - 1)
        let n = (index % period + period) % period
        return n < length ? n : period - n
    }
}

extension ModelTestImage {
    /// Every native output pixel is owned by exactly one reliable tile core.
    /// Internal context is copied from adjacent native pixels. Only coordinates
    /// beyond the original image are reflected in the temporary model input;
    /// no file is padded, resized, blurred or normalized.
    static func predictContext(image: NativeImage, plan: ModelTestContextPlan, map: MapRole,
                               cancellation: NativeCancellation, progress: @escaping @Sendable (TrainingEvent) -> Void,
                               predictor: ModelTestWorker.Predictor) throws -> [Float] {
        let count = image.width * image.height, channels = map == .normal ? 3 : 1
        guard image.width == plan.width, image.height == plan.height, image.channels == 3,
              image.values.count == count * 3 else { throw LoomError("The native test image dimensions changed.") }
        var output = [Float](repeating: 0, count: count * channels)
        let w = plan.inputWidth, h = plan.inputHeight, n = w * h
        for (index, tile) in plan.tiles.enumerated() {
            try cancellation.check()
            progress(.init(type: "status", message: "Testing context tile \(index + 1) of \(plan.tiles.count)…",
                progress: Double(index) / Double(plan.tiles.count)))
            try autoreleasepool {
                var input = [Float](repeating: 0, count: n * 3)
                for y in 0..<h {
                    if y % 16 == 0 { try cancellation.check() }
                    let sy = ModelTestContextPlan.reflected(tile.y + y - plan.halo, length: image.height)
                    for x in 0..<w {
                        let sx = ModelTestContextPlan.reflected(tile.x + x - plan.halo, length: image.width)
                        for c in 0..<3 { input[c*n+y*w+x] = image.values[(sy*image.width+sx)*3+c] }
                    }
                }
                let prediction = try predictor(input, h, w)
                guard prediction.count == n * channels,
                      prediction.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
                    throw LoomError("The saved model produced invalid native output pixels.")
                }
                for y in 0..<tile.height {
                    if y % 16 == 0 { try cancellation.check() }
                    for x in 0..<tile.width {
                        let source = (y+plan.halo)*w+x+plan.halo
                        let target = (tile.y+y)*image.width+tile.x+x
                        for c in 0..<channels { output[c*count+target] = prediction[c*n+source] }
                    }
                }
            }
        }
        if map == .normal {
            for i in 0..<count {
                if i % 65536 == 0 { try cancellation.check() }
                let x = output[i]*2-1, y = output[count+i]*2-1, z = output[2*count+i]*2-1
                let length = max(sqrt(x*x+y*y+z*z), 1e-6)
                output[i] = x/length*0.5+0.5; output[count+i] = y/length*0.5+0.5; output[2*count+i] = z/length*0.5+0.5
            }
        }
        return output
    }
}
