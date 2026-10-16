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
    /// A distinct inference identity: negative hidden features retain a 0.1
    /// slope instead of permanently losing their teaching signal at ReLU.
    /// Existing residual-v1 and Pyramid checkpoints keep their original graph.
    public static let residualV2Identifier = "loom-factory-material-residual-v2"
    public static let pyramidIdentifier = "loom-factory-material-pyramid-v1"
    public static let pyramidV2Identifier = "loom-factory-material-pyramid-v2"
    public static let compactIdentifier = "loom-factory-material-compact-naf-v1"
    public static let photoCompactIdentifier = "loom-factory-photo-compact-naf-v1"
    public static var supportedIdentifiers: [String] { NativeArchitectureConstructor.all.map(\.id) }
    public let identifier: String
    public let mapKind: String
    public let width: Int
    public let blocks: Int
    public var product: NativeModelProduct { NativeModelProduct(rawValue: mapKind)! }
    public var isPhoto: Bool { product.isPhoto }
    public var outputChannels: Int { product.channels }
    public var outputName: String { product.outputName }
    public init(mapKind: String, width: Int, blocks: Int, identifier: String = Self.identifier) throws {
        guard let product = NativeModelProduct(rawValue: mapKind), product.constructors.contains(identifier), (1...512).contains(width), (1...64).contains(blocks),
              ![Self.compactIdentifier, Self.photoCompactIdentifier].contains(identifier) || blocks == 1 else {
            throw NativeTrainingError.invalid("Unsupported native architecture identity, map kind, or dimensions.")
        }
        self.identifier = identifier; self.mapKind = mapKind; self.width = width; self.blocks = blocks
    }
    public var isPyramid: Bool { identifier == Self.pyramidIdentifier || isPyramidV2 }
    public var isPyramidV2: Bool { identifier == Self.pyramidV2Identifier }
    public var isCompact: Bool { identifier == Self.compactIdentifier || identifier == Self.photoCompactIdentifier }
    public var isResidualV2: Bool { identifier == Self.residualV2Identifier }
    public var usesLeakyActivation: Bool { isResidualV2 || isPyramidV2 }
    public var hiddenNegativeSlope: Double { usesLeakyActivation ? 0.1 : 0 }
    public var isFullResolutionResidual: Bool { identifier == Self.identifier || isResidualV2 }
    public var residualNegativeSlope: Double { isResidualV2 ? 0.1 : 0 }
    public var parameterShapes: [String: [Int]] {
        if isCompact { return NativeCompactNetwork.parameterShapes(width: width, outputChannels: outputChannels) }
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
        if isCompact { return NativeCompactNetwork.convolutionNames }
        return ["input"] + (isPyramid ? ["encoder.half", "encoder.quarter"] : [])
            + (0..<blocks).flatMap { ["body.\($0).conv1", "body.\($0).conv2"] }
            + (isPyramid ? ["decoder.half", "decoder.full"] : []) + ["output"]
    }
    public func convolutionStride(_ name: String) -> Int {
        if isCompact { return name.hasPrefix("down.") ? 2 : 1 }
        return isPyramid && ["encoder.half", "encoder.quarter"].contains(name) ? 2 : 1
    }
    public func convolutionKernelSize(_ name: String) -> Int {
        if isCompact {
            if name.hasPrefix("down.") { return 2 }
            return name == "input" || name == "output" || name.hasSuffix(".depthwise") ? 3 : 1
        }
        return 3
    }
    public func convolutionGroups(_ name: String) -> Int {
        guard isCompact, name.hasSuffix(".depthwise") else { return 1 }
        return parameterShapes[name + ".weight"]![0]
    }
    /// Depthwise kernels, affine normalization and residual gains remain part
    /// of the frozen base. Compact LoRA adapts only its dense convolutions.
    public var adaptableConvolutionNames: [String] {
        isCompact ? convolutionNames.filter { !$0.hasSuffix(".depthwise") } : convolutionNames
    }
    public var usesLinearOutput: Bool { isCompact }
    /// Compact's three unpadded stride-two levels require exact multiples of
    /// eight. Unsupported grids fail rather than resizing or padding images.
    public func supportsGrid(height: Int, width: Int) -> Bool {
        !isCompact || (height >= 32 && width >= 32 && height % 8 == 0 && width % 8 == 0)
    }
    public var parameterCount: Int { parameterShapes.values.reduce(0) { $0 + $1.reduce(1, *) } }
    public var descriptor: [String: Any] {
        var result: [String: Any] = ["id": identifier, "width": width, "blocks": blocks, "input_channels": 3,
         "output_channels": outputChannels, "input": "RGB sRGB float32 NCHW [0,1]",
         "output": "float32 NCHW [0,1]", "padding": "zero",
         "normal_convention": mapKind == "normal" ? "OpenGL" as Any : NSNull()]
        if usesLeakyActivation {
            result["hidden_activation"] = "leaky-relu"
            result["negative_slope"] = hiddenNegativeSlope
        }
        if isPyramidV2 {
            result["head_activation"] = mapKind == "normal" ? "unit-linear-vector-v1" : "sigmoid"
        }
        if isPyramid {
            result["scale_channels"] = [width, width * 2, width * 4]
            result["residual_scale"] = 4
            result["upsampling"] = "nearest-2x-trim-to-skip"
        } else if isCompact {
            result["output"] = "float32 NCHW raw linear values"
            result["head_activation"] = "linear"
            result["padding"] = "reflect-v1"
            result["scale_channels"] = [width, width * 2, width * 4, width * 8]
            result["blocks_per_stage"] = 1
            result["stage_count"] = 7
            result["upsampling"] = "nearest-2x-exact-repeat"
            result["grid_multiple"] = 8
            result["minimum_grid_size"] = 32
            result["normal_output_transform"] = "none"
            result["lora_scope"] = "dense-convolutions-only"
        }
        if isPhoto {
            result["product"] = mapKind
            result["head_activation"] = "input-plus-linear-residual"
            result["output"] = "sRGB numeric RGB float32 NCHW raw; no normal-vector transform"
            result["initial_output"] = "identity-zero-residual-head"
            result["lora_scope"] = "unsupported-v1"
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
        materialPrediction(logits: materialLogits(input: input, architecture: architecture,
            parameters: parameters, adapter: adapter, channelsLast: channelsLast), architecture: architecture)
    }
    /// Pre-activation NCHW values are available to stable scalar training losses;
    /// exported predictions keep the exact same sigmoid/vector interpretation.
    func materialLogits(input: MPSGraphTensor, architecture: NativeArchitecture,
                        parameters: [String: MPSGraphTensor], adapter: NativeAdapterConfiguration?,
                        channelsLast: Bool = false) -> MPSGraphTensor {
        if architecture.isCompact {
            let residual = compactMaterialNetwork(input: input, architecture: architecture, parameters: parameters,
                adapter: adapter, channelsLast: channelsLast)
            return architecture.isPhoto ? addition(input, residual, name: "photo_rgb_residual") : residual
        }
        func activation(_ value: MPSGraphTensor) -> MPSGraphTensor {
            architecture.usesLeakyActivation
                ? leakyReLU(with: value, alpha: architecture.hiddenNegativeSlope, name: nil)
                : reLU(with: value, name: nil)
        }
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
        let source = channelsLast ? transpose(input, permutation: [0, 2, 3, 1], name: nil) : input
        var value = activation(conv(source, "input"))
        let full = value
        let half: MPSGraphTensor?
        if architecture.isPyramid {
            let encodedHalf = activation(conv(value, "encoder.half"))
            half = encodedHalf
            value = activation(conv(encodedHalf, "encoder.quarter"))
        } else { half = nil }
        for block in 0..<architecture.blocks {
            let first = activation(conv(value, "body.\(block).conv1"))
            value = activation(addition(value, conv(first, "body.\(block).conv2"), name: nil))
        }
        if let half {
            value = nearestDouble(value, matching: half, channelsLast: channelsLast)
            value = activation(addition(conv(value, "decoder.half"), half, name: nil))
            value = nearestDouble(value, matching: full, channelsLast: channelsLast)
            value = activation(addition(conv(value, "decoder.full"), full, name: nil))
        }
        value = conv(value, "output")
        if channelsLast { value = transpose(value, permutation: [0, 3, 1, 2], name: nil) }
        return value
    }
    func materialPrediction(logits: MPSGraphTensor, architecture: NativeArchitecture) -> MPSGraphTensor {
        if architecture.usesLinearOutput { return logits }
        if architecture.isPyramidV2 && architecture.mapKind == "normal" {
            return addition(scaled(normalizedVector(logits), 0.5), scalar(0.5), name: nil)
        }
        var value = sigmoid(with: logits, name: nil)
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
    /// Soft-target BCE evaluated from logits. Its derivative is sigmoid(z)-y,
    /// so an incorrectly saturated white/black scalar map can still recover.
    /// The positive/negative parts cancel |z|'s derivative at z=0 and avoid
    /// subtracting large products near target endpoints. The exponential cannot
    /// overflow. Targets are used without normalization.
    func scalarLogitCrossEntropy(logits: MPSGraphTensor, target: MPSGraphTensor) -> MPSGraphTensor {
        let absolute = absolute(with: logits, name: nil)
        let positive = addition(scaled(absolute, 0.5), scaled(logits, 0.5), name: nil)
        let negative = subtraction(scaled(absolute, 0.5), scaled(logits, 0.5), name: nil)
        let linear = addition(multiplication(subtraction(scalar(1), target, name: nil), positive, name: nil),
            multiplication(target, negative, name: nil), name: nil)
        let softplusTail = logarithm(with: addition(scalar(1),
            exponent(with: scaled(absolute, -1), name: nil), name: nil), name: nil)
        return mean(of: addition(linear, softplusTail, name: nil), axes: [0, 1, 2, 3], name: nil)
    }

    /// Match signed native pixel differences at 1, 2 and 4 pixels in both axes.
    /// No resized targets, edge enhancement, absolute-gradient ambiguity, or
    /// image smoothing: every term compares the same original pixel positions.
    /// Averaging directions/offsets keeps the contract independent of image size.
    /// Combined with logit BCE, saturated wrong predictions still have a
    /// corrective level gradient even when sigmoid's detail gradient vanishes.
    func nativeDetailLoss(prediction: MPSGraphTensor, target: MPSGraphTensor) -> MPSGraphTensor {
        let shape = prediction.shape!.map(\.intValue)
        let residual = subtraction(prediction, target, name: nil)
        var terms: [MPSGraphTensor] = []
        for offset in [1, 2, 4] {
            for axis in [2, 3] where shape[axis] > offset {
                var starts = [0, 0, 0, 0], ends = shape
                starts[axis] = offset
                let right = sliceTensor(residual, starts: starts.map(NSNumber.init(value:)),
                    ends: shape.map(NSNumber.init(value:)), strides: [1, 1, 1, 1], name: nil)
                ends[axis] -= offset
                let left = sliceTensor(residual, starts: [0, 0, 0, 0],
                    ends: ends.map(NSNumber.init(value:)), strides: [1, 1, 1, 1], name: nil)
                terms.append(mean(of: absolute(with: subtraction(right, left, name: nil), name: nil),
                    axes: [0, 1, 2, 3], name: nil))
            }
        }
        guard let first = terms.first else { return scalar(0) }
        return scaled(terms.dropFirst().reduce(first) { addition($0, $1, name: nil) }, 1 / Double(terms.count))
    }
}
