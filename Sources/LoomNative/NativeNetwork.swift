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

/// The deployed residual-v1 contract. Parameter tensors use PyTorch's OIHW layout
/// and the original state-dictionary names so existing projects remain readable.
public struct NativeArchitecture: Sendable, Equatable {
    public static let identifier = "loom-factory-material-residual-v1"
    public let mapKind: String
    public let width: Int
    public let blocks: Int
    public var outputChannels: Int { mapKind == "normal" ? 3 : 1 }
    public init(mapKind: String, width: Int, blocks: Int) throws {
        guard ["depth", "roughness", "normal"].contains(mapKind), (1...512).contains(width), (1...64).contains(blocks) else {
            throw NativeTrainingError.invalid("Unsupported residual-v1 map kind or architecture dimensions.")
        }
        self.mapKind = mapKind; self.width = width; self.blocks = blocks
    }
    public var parameterShapes: [String: [Int]] {
        var shapes: [String: [Int]] = ["input.weight": [width, 3, 3, 3], "input.bias": [width],
                                      "output.weight": [outputChannels, width, 3, 3], "output.bias": [outputChannels]]
        for block in 0..<blocks {
            for conv in 1...2 {
                shapes["body.\(block).conv\(conv).weight"] = [width, width, 3, 3]
                shapes["body.\(block).conv\(conv).bias"] = [width]
            }
        }
        return shapes
    }
    public var convolutionNames: [String] {
        ["input"] + (0..<blocks).flatMap { ["body.\($0).conv1", "body.\($0).conv2"] } + ["output"]
    }
    public var parameterCount: Int { parameterShapes.values.reduce(0) { $0 + $1.reduce(1, *) } }
    public var descriptor: [String: Any] {
        ["id": Self.identifier, "width": width, "blocks": blocks, "input_channels": 3,
         "output_channels": outputChannels, "input": "RGB sRGB float32 NCHW [0,1]",
         "output": "float32 NCHW [0,1]", "padding": "zero",
         "normal_convention": mapKind == "normal" ? "OpenGL" as Any : NSNull()]
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
    func convolution(_ source: MPSGraphTensor, weight: MPSGraphTensor, kernel: Int) -> MPSGraphTensor {
        let padding = kernel / 2
        let descriptor = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1,
            dilationRateInX: 1, dilationRateInY: 1, groups: 1,
            paddingLeft: padding, paddingRight: padding, paddingTop: padding, paddingBottom: padding,
            paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
        return convolution2D(source, weights: weight, descriptor: descriptor, name: nil)
    }
    func materialNetwork(input: MPSGraphTensor, architecture: NativeArchitecture,
                         parameters: [String: MPSGraphTensor], adapter: NativeAdapterConfiguration?) -> MPSGraphTensor {
        func conv(_ source: MPSGraphTensor, _ name: String) -> MPSGraphTensor {
            var value = convolution(source, weight: parameters[name + ".weight"]!, kernel: 3)
            let bias = reshape(parameters[name + ".bias"]!, shape: [1, -1, 1, 1], name: nil)
            value = addition(value, bias, name: nil)
            if let adapter {
                let down = convolution(source, weight: parameters[name + ".down.weight"]!, kernel: 3)
                let up = convolution(down, weight: parameters[name + ".up.weight"]!, kernel: 1)
                value = addition(value, scaled(up, adapter.alpha / Double(adapter.rank)), name: nil)
            }
            return value
        }
        var value = reLU(with: conv(input, "input"), name: nil)
        for block in 0..<architecture.blocks {
            let first = reLU(with: conv(value, "body.\(block).conv1"), name: nil)
            value = reLU(with: addition(value, conv(first, "body.\(block).conv2"), name: nil), name: nil)
        }
        value = sigmoid(with: conv(value, "output"), name: nil)
        if architecture.mapKind == "normal" {
            value = addition(scaled(normalizedVector(subtraction(scaled(value, 2), scalar(1), name: nil)), 0.5), scalar(0.5), name: nil)
        }
        return value
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
