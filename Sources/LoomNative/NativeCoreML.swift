import CoreML
import Foundation

/// Exports the existing residual-v1 product using Core ML's public protobuf
/// specification. The generated model contains actual learned Float32 weights.
/// Specification fields are documented in Apple's mlmodel/format/{Model,
/// FeatureTypes,NeuralNetwork}.proto; no converter subprocess is involved.
public enum NativeCoreML {
    @discardableResult
    public static func export(weights: [String: NativeTensor], architecture: NativeArchitecture,
                              imageSize: Int, to url: URL) throws -> URL {
        let bytes = try modelData(weights: weights, mapKind: architecture.mapKind,
                                  width: architecture.width, blocks: architecture.blocks,
                                  imageSize: imageSize)
        try bytes.write(to: url, options: .atomic)
        let compiled = try MLModel.compileModel(at: url)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        guard model.modelDescription.inputDescriptionsByName["rgb"]?.multiArrayConstraint?.shape.count == 4,
              model.modelDescription.outputDescriptionsByName["material_map"] != nil else {
            throw NativeIOError.invalid("The compiled Core ML model does not retain the residual-v1 tensor interface.")
        }
        return compiled
    }

    static func modelData(weights: [String: NativeTensor], mapKind: String,
                          width: Int, blocks: Int, imageSize: Int) throws -> Data {
        guard ["depth", "roughness", "normal"].contains(mapKind), width > 0,
              blocks >= 0, blocks <= 256, (32...8192).contains(imageSize) else {
            throw NativeIOError.invalid("Invalid residual-v1 architecture or Core ML default image size.")
        }
        let channels = mapKind == "normal" ? 3 : 1
        var expected: [String: [Int]] = ["input.weight": [width, 3, 3, 3], "input.bias": [width],
                                         "output.weight": [channels, width, 3, 3], "output.bias": [channels]]
        for index in 0..<blocks {
            for convolution in ["conv1", "conv2"] {
                expected["body.\(index).\(convolution).weight"] = [width, width, 3, 3]
                expected["body.\(index).\(convolution).bias"] = [width]
            }
        }
        guard Set(expected.keys) == Set(weights.keys) else {
            throw NativeIOError.invalid("Core ML export requires exactly the residual-v1 parameter names.")
        }
        for (name, shape) in expected {
            guard let tensor = weights[name], tensor.shape == shape else {
                throw NativeIOError.invalid("Core ML export weight \(name) has an incompatible shape.")
            }
            try NativeWeights.validate(tensor, name: name)
        }

        var network = CoreMLWire()
        network.integer(5, 1) // EXACT_ARRAY_MAPPING retains NCHW rank 4.
        network.integer(6, 1) // RANK4_IMAGE_MAPPING disables legacy rank-5 mapping.
        func layer(_ name: String, _ inputs: [String], _ output: String, _ field: Int, _ params: CoreMLWire) {
            var node = CoreMLWire()
            node.string(1, name)
            for input in inputs { node.string(2, input) }
            node.string(3, output)
            var tensor = CoreMLWire(); tensor.integer(1, 4); tensor.integers(2, [1, -1, -1, -1])
            for _ in inputs { node.message(4, tensor) }
            node.message(5, tensor)
            node.message(field, params)
            network.message(1, node)
        }
        func convolution(_ name: String, _ input: String, _ output: String) {
            let weight = weights[name + ".weight"]!
            let bias = weights[name + ".bias"]!
            var params = CoreMLWire()
            params.integer(1, weight.shape[0])
            params.integer(2, weight.shape[1])
            params.integer(10, 1)
            params.integers(20, [3, 3])
            params.integers(30, [1, 1])
            params.integers(40, [1, 1])
            params.message(51, CoreMLWire()) // SAME, odd 3x3 kernel, stride 1: zero padding of 1.
            params.integer(70, 1)
            var kernel = CoreMLWire(); kernel.floats(1, weight.values)
            var offset = CoreMLWire(); offset.floats(1, bias.values)
            params.message(90, kernel); params.message(91, offset)
            layer(name, [input], output, 100, params)
        }
        func activation(_ name: String, _ input: String, _ output: String, _ kind: Int) {
            var params = CoreMLWire(); params.message(kind, CoreMLWire())
            layer(name, [input], output, 130, params)
        }
        func scalar(_ name: String, _ input: String, _ output: String, field: Int, value: Float) {
            var params = CoreMLWire(); params.float(1, value)
            layer(name, [input], output, field, params)
        }

        convolution("input", "rgb", "input_conv")
        activation("input_relu", "input_conv", "input_relu", 10)
        var value = "input_relu"
        for index in 0..<blocks {
            let prefix = "body.\(index)"
            convolution(prefix + ".conv1", value, prefix + ".conv1")
            activation(prefix + ".relu1", prefix + ".conv1", prefix + ".relu1", 10)
            convolution(prefix + ".conv2", prefix + ".relu1", prefix + ".conv2")
            layer(prefix + ".residual", [value, prefix + ".conv2"], prefix + ".residual", 230, CoreMLWire())
            activation(prefix + ".relu2", prefix + ".residual", prefix + ".relu2", 10)
            value = prefix + ".relu2"
        }
        convolution("output", value, "logits")
        activation("output_sigmoid", "logits", mapKind == "normal" ? "sigmoid" : "material_value", 40)
        if mapKind == "normal" {
            // Match torch.normalize(v, dim=1, eps=1e-6): denominator is
            // max(sqrt(sum(v*v)), eps), not sqrt(sum(v*v) + eps).
            scalar("normal_scale", "sigmoid", "normal_scaled", field: 231, value: 2)
            scalar("normal_center", "normal_scaled", "normal_vector", field: 230, value: -1)
            var reduce = CoreMLWire(); reduce.integers(1, [1]); reduce.integer(2, 1)
            layer("normal_length", ["normal_vector"], "normal_length", 1255, reduce)
            var clamp = CoreMLWire(); clamp.float(1, 1e-6); clamp.float(2, Float.greatestFiniteMagnitude)
            layer("normal_denominator", ["normal_length"], "normal_denominator", 660, clamp)
            layer("normal_unit", ["normal_vector", "normal_denominator"], "normal_unit", 890, CoreMLWire())
            scalar("normal_half", "normal_unit", "normal_half", field: 231, value: 0.5)
            scalar("normal_output", "normal_half", "material_value", field: 230, value: 0.5)
        }
        var outputShape = CoreMLWire(); outputShape.integers(1, [1, channels, 0, 0])
        layer("output_nchw", ["material_value"], "material_map", 1150, outputShape)

        var description = CoreMLWire()
        description.message(1, feature(name: "rgb", channels: 3, size: imageSize, flexible: true))
        description.message(10, feature(name: "material_map", channels: channels, size: imageSize, flexible: true))
        var metadata = CoreMLWire()
        metadata.string(1, "Texture Loom model trained natively in Loom Factory")
        for (key, value) in ["loom.schema": "loom-factory-model-v1", "loom.architecture": "loom-factory-material-residual-v1"] {
            var pair = CoreMLWire(); pair.string(1, key); pair.string(2, value)
            metadata.message(100, pair)
        }
        description.message(100, metadata)
        var model = CoreMLWire()
        // macOS 13 matches the original deployed minimum. Older specification
        // versions return legacy rank-5 arrays when spatial output sizes vary.
        model.integer(1, 7)
        model.message(2, description)
        model.message(500, network)
        return model.data
    }

    private static func feature(name: String, channels: Int, size: Int, flexible: Bool) -> CoreMLWire {
        var array = CoreMLWire()
        array.integers(1, [1, channels, size, size])
        array.integer(2, 65568) // FLOAT32
        if flexible {
            var ranges = CoreMLWire()
            for limits in [(1, 1), (channels, channels), (32, 8192), (32, 8192)] {
                var range = CoreMLWire(); range.integer(1, limits.0); range.integer(2, limits.1)
                ranges.message(1, range)
            }
            array.message(31, ranges)
        }
        var type = CoreMLWire(); type.message(5, array)
        var feature = CoreMLWire(); feature.string(1, name); feature.message(3, type)
        return feature
    }
}

/// Minimal protobuf writer for the exported public messages. Numeric arrays
/// use packed encoding and learned float bit patterns are preserved exactly.
private struct CoreMLWire {
    var data = Data()
    mutating func varint(_ value: UInt64) {
        var remainder = value
        while remainder >= 0x80 {
            data.append(UInt8(remainder & 0x7f) | 0x80)
            remainder >>= 7
        }
        data.append(UInt8(remainder))
    }
    mutating func integer(_ field: Int, _ value: Int) {
        varint(UInt64(field << 3)); varint(UInt64(bitPattern: Int64(value)))
    }
    mutating func bytes(_ field: Int, _ value: Data) {
        varint(UInt64(field << 3 | 2)); varint(UInt64(value.count)); data.append(value)
    }
    mutating func message(_ field: Int, _ value: CoreMLWire) { bytes(field, value.data) }
    mutating func string(_ field: Int, _ value: String) { bytes(field, Data(value.utf8)) }
    mutating func float(_ field: Int, _ value: Float) {
        varint(UInt64(field << 3 | 5))
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    mutating func floats(_ field: Int, _ values: [Float]) {
        let packed = values.withUnsafeBytes { Data($0) }
        bytes(field, packed)
    }
    mutating func integers(_ field: Int, _ values: [Int]) {
        var packed = CoreMLWire()
        for value in values { packed.varint(UInt64(bitPattern: Int64(value))) }
        bytes(field, packed.data)
    }
}
