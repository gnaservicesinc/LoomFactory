import Foundation
import MetalPerformanceShadersGraph

// NAF-style block design adapted from megvii-research/NAFNet (MIT).
// Copyright (c) 2022 megvii-model. See the retained NAFNet license notice.
// The native task-specific topology uses no upstream pretrained weights.

enum NativeCompactNetwork {
    static let blockNames = (0..<3).map { "encoder.\($0)" } + ["middle"]
        + (0..<3).reversed().map { "decoder.\($0)" }
    static let blockConvolutions = ["expand", "depthwise", "attention", "project", "ffn_expand", "ffn_project"]
    static var convolutionNames: [String] {
        ["input"] + (0..<3).flatMap { level in
            blockConvolutions.map { "encoder.\(level)." + $0 } + ["down.\(level)"]
        } + blockConvolutions.map { "middle." + $0 }
            + (0..<3).reversed().flatMap { level in
                ["up.\(level)"] + blockConvolutions.map { "decoder.\(level)." + $0 }
            } + ["output"]
    }
    static func parameterShapes(width: Int, outputChannels: Int) -> [String: [Int]] {
        var shapes: [String: [Int]] = [:]
        func convolution(_ name: String, _ incoming: Int, _ outgoing: Int, kernel: Int = 1, groups: Int = 1) {
            shapes[name + ".weight"] = [outgoing, incoming / groups, kernel, kernel]
            shapes[name + ".bias"] = [1, outgoing, 1, 1]
        }
        func block(_ name: String, _ channels: Int) {
            for index in 1...2 {
                shapes[name + ".norm\(index).weight"] = [1, channels, 1, 1]
                shapes[name + ".norm\(index).bias"] = [1, channels, 1, 1]
            }
            convolution(name + ".expand", channels, channels * 2)
            convolution(name + ".depthwise", channels * 2, channels * 2, kernel: 3, groups: channels * 2)
            convolution(name + ".attention", channels, channels)
            convolution(name + ".project", channels, channels)
            convolution(name + ".ffn_expand", channels, channels * 2)
            convolution(name + ".ffn_project", channels, channels)
            shapes[name + ".beta"] = [1, channels, 1, 1]
            shapes[name + ".gamma"] = [1, channels, 1, 1]
        }
        convolution("input", 3, width, kernel: 3)
        for level in 0..<3 {
            let channels = width << level
            block("encoder.\(level)", channels)
            convolution("down.\(level)", channels, channels * 2, kernel: 2)
            convolution("up.\(level)", channels * 2, channels)
            block("decoder.\(level)", channels)
        }
        block("middle", width * 8)
        convolution("output", width, outputChannels, kernel: 3)
        return shapes
    }
    /// Exact historical SplitMix64/24-bit uniform initialization. This is
    /// separate from residual-v1's established Kaiming initialization stream.
    static func initialWeights(architecture: NativeArchitecture, seed: UInt64) -> [String: NativeTensor] {
        var random = NativeRandom(state: seed), result: [String: NativeTensor] = [:]
        let shapes = architecture.parameterShapes
        for name in shapes.keys.sorted() {
            let shape = shapes[name]!
            result[name] = NativeTensor(shape: shape, values: initialValues(name: name, shape: shape, random: &random, photo: architecture.isPhoto))
        }
        return result
    }
    static func initialValues(name: String, shape: [Int], random: inout NativeRandom, photo: Bool = false) -> [Float] {
        let count = shape.reduce(1, *)
        if photo && name.hasPrefix("output.") { return Array(repeating: 0, count: count) }
        if name.contains(".norm"), name.hasSuffix(".weight") { return Array(repeating: 1, count: count) }
        if name.hasSuffix(".beta") || name.hasSuffix(".gamma") { return Array(repeating: 0.1, count: count) }
        if name == "output.bias" { return count == 3 ? [0.5, 0.5, 1] : [0.5] }
        if name.hasSuffix(".bias") { return Array(repeating: 0, count: count) }
        let incoming = shape[1] * shape[2] * shape[3]
        let outgoing = (name.contains("depthwise") ? 1 : shape[0]) * shape[2] * shape[3]
        let limit = sqrt(Float(6) / Float(incoming + outgoing))
        return (0..<count).map { _ in
            let unit = Float(random.next() >> 40) / Float(1 << 24)
            return (unit * 2 - 1) * limit
        }
    }
}

extension MPSGraph {
    /// Seven NAF-style stages with original-resolution encoder skips. Every
    /// learned weight, intermediate and prediction remains Float32. The head
    /// is linear for scalar and normal components; no output transform follows.
    func compactMaterialNetwork(input: MPSGraphTensor, architecture: NativeArchitecture,
                                parameters: [String: MPSGraphTensor], adapter: NativeAdapterConfiguration?,
                                channelsLast: Bool) -> MPSGraphTensor {
        let channelAxis = channelsLast ? 3 : 1
        func affine(_ name: String) -> MPSGraphTensor {
            let tensor = parameters[name]!
            return channelsLast ? reshape(tensor, shape: [1, 1, 1, -1], name: nil) : tensor
        }
        func rawConvolution(_ input: MPSGraphTensor, _ weight: MPSGraphTensor,
                            kernel: Int, stride: Int, groups: Int) -> MPSGraphTensor {
            var source = input
            if stride == 1 && kernel > 1 {
                let padding = kernel / 2
                let edges = channelsLast ? [0, padding, padding, 0] : [0, 0, padding, padding]
                source = padTensor(source, with: .reflect, leftPadding: edges.map(NSNumber.init(value:)),
                    rightPadding: edges.map(NSNumber.init(value:)), constantValue: 0, name: nil)
            }
            let descriptor = MPSGraphConvolution2DOpDescriptor(strideInX: stride, strideInY: stride,
                dilationRateInX: 1, dilationRateInY: 1, groups: groups,
                paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0,
                paddingStyle: .explicit, dataLayout: channelsLast ? .NHWC : .NCHW,
                weightsLayout: channelsLast ? .HWIO : .OIHW)!
            let kernelWeights = channelsLast ? transpose(weight, permutation: [2, 3, 1, 0], name: nil) : weight
            return convolution2D(source, weights: kernelWeights, descriptor: descriptor, name: nil)
        }
        func conv(_ input: MPSGraphTensor, _ name: String) -> MPSGraphTensor {
            let kernel = architecture.convolutionKernelSize(name), stride = architecture.convolutionStride(name)
            let groups = architecture.convolutionGroups(name)
            var value = addition(rawConvolution(input, parameters[name + ".weight"]!, kernel: kernel,
                stride: stride, groups: groups), affine(name + ".bias"), name: nil)
            if let adapter, groups == 1, let downWeight = parameters[name + ".down.weight"],
               let upWeight = parameters[name + ".up.weight"] {
                let down = rawConvolution(input, downWeight, kernel: kernel, stride: stride, groups: 1)
                let up = rawConvolution(down, upWeight, kernel: 1, stride: 1, groups: 1)
                value = addition(value, scaled(up, adapter.alpha / Double(adapter.rank)), name: nil)
            }
            return value
        }
        func normalize(_ input: MPSGraphTensor, _ name: String) -> MPSGraphTensor {
            let mean = self.mean(of: input, axes: [NSNumber(value: channelAxis)], name: nil)
            let centered = subtraction(input, mean, name: nil)
            let variance = self.mean(of: multiplication(centered, centered, name: nil),
                axes: [NSNumber(value: channelAxis)], name: nil)
            let normalized = division(centered, squareRoot(with: addition(variance, scalar(1e-6), name: nil), name: nil), name: nil)
            return addition(multiplication(normalized, affine(name + ".weight"), name: nil), affine(name + ".bias"), name: nil)
        }
        func gate(_ input: MPSGraphTensor, channels: Int) -> MPSGraphTensor {
            let first = sliceTensor(input, dimension: channelAxis, start: 0, length: channels, name: nil)
            let second = sliceTensor(input, dimension: channelAxis, start: channels, length: channels, name: nil)
            return multiplication(first, second, name: nil)
        }
        func block(_ input: MPSGraphTensor, _ name: String, channels: Int) -> MPSGraphTensor {
            var value = conv(normalize(input, name + ".norm1"), name + ".expand")
            value = gate(conv(value, name + ".depthwise"), channels: channels)
            let attention = conv(mean(of: value, axes: channelsLast ? [1, 2] : [2, 3], name: nil), name + ".attention")
            value = conv(multiplication(value, attention, name: nil), name + ".project")
            let residual = addition(input, multiplication(value, affine(name + ".beta"), name: nil), name: nil)
            value = conv(normalize(residual, name + ".norm2"), name + ".ffn_expand")
            value = conv(gate(value, channels: channels), name + ".ffn_project")
            return addition(residual, multiplication(value, affine(name + ".gamma"), name: nil), name: nil)
        }
        func nearestRepeat(_ source: MPSGraphTensor) -> MPSGraphTensor {
            // Preserve the historical exact nearest duplication and its sum
            // gradient. MPSGraphTileOp does not implement its derivative.
            let nchw = channelsLast ? transpose(source, permutation: [0, 3, 1, 2], name: nil) : source
            let shape = nchw.shape!.map(\.intValue)
            let expanded = reshape(nchw, shape: [shape[0], shape[1], shape[2], 1, shape[3], 1].map(NSNumber.init(value:)), name: nil)
            let rows = concatTensors([expanded, expanded], dimension: 3, name: nil)
            let samples = concatTensors([rows, rows], dimension: 5, name: nil)
            let repeated = reshape(samples, shape: [shape[0], shape[1], shape[2] * 2, shape[3] * 2].map(NSNumber.init(value:)), name: nil)
            return channelsLast ? transpose(repeated, permutation: [0, 2, 3, 1], name: nil) : repeated
        }
        let source = channelsLast ? transpose(input, permutation: [0, 2, 3, 1], name: nil) : input
        var value = conv(source, "input"), skips: [MPSGraphTensor] = []
        for level in 0..<3 {
            value = block(value, "encoder.\(level)", channels: architecture.width << level)
            skips.append(value)
            value = conv(value, "down.\(level)")
        }
        value = block(value, "middle", channels: architecture.width * 8)
        for level in (0..<3).reversed() {
            value = addition(nearestRepeat(conv(value, "up.\(level)")), skips[level], name: nil)
            value = block(value, "decoder.\(level)", channels: architecture.width << level)
        }
        value = conv(value, "output")
        return channelsLast ? transpose(value, permutation: [0, 3, 1, 2], name: nil) : value
    }
}
