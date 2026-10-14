import Foundation
import MetalPerformanceShadersGraph

public enum NativeTrainingError: Error, LocalizedError {
    case invalid(String)
    case unavailable(String)
    case nonFinite(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message), .unavailable(let message), .nonFinite(let message): message }
    }
}

/// Immutable native model constructors. Boundary tensors remain NCHW and all
/// learned convolution kernels use OIHW, including pyramid-v1's lower scales.
public struct NativeArchitecture: Sendable, Equatable {
    public static let identifier = "loom-factory-material-residual-v1"
    public static let pyramidIdentifier = "loom-factory-material-pyramid-v1"
    public let identifier: String
    public let mapKind: String
    public let width: Int
    public let blocks: Int
    public var outputChannels: Int { mapKind == "normal" ? 3 : 1 }
    public init(mapKind: String, width: Int, blocks: Int, identifier: String = Self.identifier) throws {
        guard [Self.identifier, Self.pyramidIdentifier].contains(identifier),
              ["depth", "roughness", "normal"].contains(mapKind), (1...512).contains(width), (1...64).contains(blocks) else {
            throw NativeTrainingError.invalid("Unsupported native architecture identity, map kind, or dimensions.")
        }
        self.identifier = identifier; self.mapKind = mapKind; self.width = width; self.blocks = blocks
    }
    public var isPyramid: Bool { identifier == Self.pyramidIdentifier }
    public var parameterShapes: [String: [Int]] {
        var shapes: [String: [Int]] = ["input.weight": [width, 3, 3, 3], "input.bias": [width],
                                      "output.weight": [outputChannels, width, 3, 3], "output.bias": [outputChannels]]
        let bodyWidth = isPyramid ? width * 4 : width
        if isPyramid {
            for (name, input, output) in [("encoder.half", width, width * 2),
                                          ("encoder.quarter", width * 2, width * 4),
                                          ("decoder.half", width * 4, width * 2),
                                          ("decoder.full", width * 2, width)] {
                shapes[name + ".weight"] = [output, input, 3, 3]
                shapes[name + ".bias"] = [output]
            }
        }
        for block in 0..<blocks {
            for conv in 1...2 {
                shapes["body.\(block).conv\(conv).weight"] = [bodyWidth, bodyWidth, 3, 3]
                shapes["body.\(block).conv\(conv).bias"] = [bodyWidth]
            }
        }
        return shapes
    }
    public var convolutionNames: [String] {
        ["input"] + (isPyramid ? ["encoder.half", "encoder.quarter"] : [])
            + (0..<blocks).flatMap { ["body.\($0).conv1", "body.\($0).conv2"] }
            + (isPyramid ? ["decoder.half", "decoder.full"] : []) + ["output"]
    }
    public func convolutionStride(_ name: String) -> Int {
        isPyramid && ["encoder.half", "encoder.quarter"].contains(name) ? 2 : 1
    }
    public var parameterCount: Int { parameterShapes.values.reduce(0) { $0 + $1.reduce(1, *) } }
    public var descriptor: [String: Any] {
        var result: [String: Any] = ["id": identifier, "width": width, "blocks": blocks, "input_channels": 3,
         "output_channels": outputChannels, "input": "RGB sRGB float32 NCHW [0,1]",
         "output": "float32 NCHW [0,1]", "padding": "zero",
         "normal_convention": mapKind == "normal" ? "OpenGL" as Any : NSNull()]
        if isPyramid {
            result["scale_channels"] = [width, width * 2, width * 4]
            result["residual_scale"] = 4
            result["upsampling"] = "nearest-2x-trim-to-skip"
        }
        return result
    }
}

public struct NativeAdapterConfiguration: Sendable, Equatable {
    public let rank: Int
    public let alpha: Double
    public init(rank: Int, alpha: Double) throws {
        guard (1...256).contains(rank), alpha.isFinite, alpha > 0, Float(alpha).isFinite else {
            throw NativeTrainingError.invalid("LoRA rank and alpha must be positive finite values.")
        }
        self.rank = rank; self.alpha = alpha
    }
}

/// Deterministic initialization, independent of global RNG state. The uniform
/// bound is 1/sqrt(fan-in), matching Conv2d's Kaiming-uniform initialization.
struct NativeRandom {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        return value ^ (value >> 31)
    }
    mutating func values(count: Int, bound: Float) -> [Float] {
        (0..<count).map { _ in (Float(Double(next() >> 11) / 9_007_199_254_740_992) * 2 - 1) * bound }
    }
}

extension MPSGraph {
    func scalar(_ value: Double) -> MPSGraphTensor { constant(value, dataType: .float32) }
    func scaled(_ value: MPSGraphTensor, _ scale: Double) -> MPSGraphTensor {
        multiplication(value, scalar(scale), name: nil)
    }
    func normalizedVector(_ value: MPSGraphTensor) -> MPSGraphTensor {
        let squares = square(with: value, name: nil)
        let sum = reductionSum(with: squares, axis: 1, name: nil)
        // Clamp squared length before sqrt: this preserves normalize(eps=1e-6)
        // while avoiding an undefined sqrt derivative for a zero vector.
        let length = squareRoot(with: maximum(sum, scalar(1e-12), name: nil), name: nil)
        return division(value, length, name: nil)
    }
    func convolution(_ source: MPSGraphTensor, weight: MPSGraphTensor, kernel: Int, stride: Int = 1,
                     channelsLast: Bool = false) -> MPSGraphTensor {
        let padding = kernel / 2
        let descriptor = MPSGraphConvolution2DOpDescriptor(strideInX: stride, strideInY: stride,
            dilationRateInX: 1, dilationRateInY: 1, groups: 1,
            paddingLeft: padding, paddingRight: padding, paddingTop: padding, paddingBottom: padding,
            paddingStyle: .explicit, dataLayout: channelsLast ? .NHWC : .NCHW,
            weightsLayout: channelsLast ? .HWIO : .OIHW)!
        let kernelWeights = channelsLast ? transpose(weight, permutation: [2, 3, 1, 0], name: nil) : weight
        return convolution2D(source, weights: kernelWeights, descriptor: descriptor, name: nil)
    }
    func materialNetwork(input: MPSGraphTensor, architecture: NativeArchitecture,
                         parameters: [String: MPSGraphTensor], adapter: NativeAdapterConfiguration?,
                         channelsLast: Bool = false) -> MPSGraphTensor {
        func conv(_ source: MPSGraphTensor, _ name: String) -> MPSGraphTensor {
            let stride = architecture.convolutionStride(name)
            var value = convolution(source, weight: parameters[name + ".weight"]!, kernel: 3,
                                    stride: stride, channelsLast: channelsLast)
            let bias = reshape(parameters[name + ".bias"]!, shape: channelsLast ? [1, 1, 1, -1] : [1, -1, 1, 1], name: nil)
            value = addition(value, bias, name: nil)
            if let adapter {
                let down = convolution(source, weight: parameters[name + ".down.weight"]!, kernel: 3,
                                       stride: stride, channelsLast: channelsLast)
                let up = convolution(down, weight: parameters[name + ".up.weight"]!, kernel: 1, channelsLast: channelsLast)
                value = addition(value, scaled(up, adapter.alpha / Double(adapter.rank)), name: nil)
            }
            return value
        }
        // Internal layout can change execution without changing checkpoint or
        // boundary tensor identities. Normal/loss still operate on NCHW below.
        let activation = channelsLast ? transpose(input, permutation: [0, 2, 3, 1], name: nil) : input
        var value = reLU(with: conv(activation, "input"), name: nil)
        let full = value
        let half: MPSGraphTensor?
        if architecture.isPyramid {
            let encodedHalf = reLU(with: conv(value, "encoder.half"), name: nil)
            half = encodedHalf
            value = reLU(with: conv(encodedHalf, "encoder.quarter"), name: nil)
        } else { half = nil }
        for block in 0..<architecture.blocks {
            let first = reLU(with: conv(value, "body.\(block).conv1"), name: nil)
            value = reLU(with: addition(value, conv(first, "body.\(block).conv2"), name: nil), name: nil)
        }
        if let half {
            value = nearestDouble(value, matching: half, channelsLast: channelsLast)
            value = reLU(with: addition(conv(value, "decoder.half"), half, name: nil), name: nil)
            value = nearestDouble(value, matching: full, channelsLast: channelsLast)
            value = reLU(with: addition(conv(value, "decoder.full"), full, name: nil), name: nil)
        }
        value = sigmoid(with: conv(value, "output"), name: nil)
        if channelsLast { value = transpose(value, permutation: [0, 3, 1, 2], name: nil) }
        if architecture.mapKind == "normal" {
            value = addition(scaled(normalizedVector(subtraction(scaled(value, 2), scalar(1), name: nil)), 0.5), scalar(0.5), name: nil)
        }
        return value
    }
    /// Exact nearest 2x replication followed by bottom/right trimming. The
    /// resize primitive supplies its native GPU gradient for training.
    private func nearestDouble(_ source: MPSGraphTensor, matching skip: MPSGraphTensor,
                               channelsLast: Bool) -> MPSGraphTensor {
        // The current Metal resize gradient rejects some NHWC channel counts.
        // Keep this operation in NCHW while convolutions retain their measured
        // faster layout; the public tensor contract and precision stay intact.
        if channelsLast {
            let value = nearestDouble(transpose(source, permutation: [0, 3, 1, 2], name: nil),
                matching: transpose(skip, permutation: [0, 3, 1, 2], name: nil), channelsLast: false)
            return transpose(value, permutation: [0, 2, 3, 1], name: nil)
        }
        let shape = source.shape!.map(\.intValue), target = skip.shape!.map(\.intValue)
        let heightAxis = channelsLast ? 1 : 2, widthAxis = channelsLast ? 2 : 3
        var result = resize(source, size: [NSNumber(value: shape[heightAxis] * 2), NSNumber(value: shape[widthAxis] * 2)],
            mode: .nearest, centerResult: true, alignCorners: false, layout: .NCHW, name: nil)
        if shape[heightAxis] * 2 != target[heightAxis] || shape[widthAxis] * 2 != target[widthAxis] {
            var ends = shape; ends[heightAxis] = target[heightAxis]; ends[widthAxis] = target[widthAxis]
            result = sliceTensor(result, starts: [0, 0, 0, 0], ends: ends.map(NSNumber.init(value:)),
                strides: [1, 1, 1, 1], name: nil)
        }
        return result
    }
    func materialLoss(prediction: MPSGraphTensor, target: MPSGraphTensor, mapKind: String) -> MPSGraphTensor {
        let error: MPSGraphTensor
        if mapKind == "normal" {
            let predicted = normalizedVector(subtraction(scaled(prediction, 2), scalar(1), name: nil))
            let expected = normalizedVector(subtraction(scaled(target, 2), scalar(1), name: nil))
            let dot = reductionSum(with: multiplication(predicted, expected, name: nil), axis: 1, name: nil)
            error = subtraction(scalar(1), dot, name: nil)
        } else {
            error = absolute(with: subtraction(prediction, target, name: nil), name: nil)
        }
        return mean(of: error, axes: [0, 1, 2, 3], name: nil)
    }
}
