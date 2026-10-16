import CoreML
import Foundation

/// Exports the versioned native products using Core ML's public protobuf
/// specification. The generated model contains actual learned Float32 weights.
/// Specification fields are documented in Apple's mlmodel/format/{Model,
/// FeatureTypes,NeuralNetwork}.proto; no converter subprocess is involved.
public enum NativeCoreML {
    @discardableResult
    public static func export(weights: [String: NativeTensor], architecture: NativeArchitecture,
                              imageSize: Int, to url: URL) throws -> URL {
        let bytes = try modelData(weights: weights, mapKind: architecture.mapKind,
                                  width: architecture.width, blocks: architecture.blocks,
                                  imageSize: imageSize, identifier: architecture.identifier)
        try bytes.write(to: url, options: .atomic)
        let compiled = try MLModel.compileModel(at: url)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        guard model.modelDescription.inputDescriptionsByName["rgb"]?.multiArrayConstraint?.shape.count == 4,
              model.modelDescription.outputDescriptionsByName[architecture.outputName] != nil else {
            throw NativeIOError.invalid("The compiled Core ML model does not retain the native NCHW tensor interface.")
        }
        return compiled
    }

    static func modelData(weights: [String: NativeTensor], mapKind: String,
                          width: Int, blocks: Int, imageSize: Int,
                          identifier: String = NativeArchitecture.identifier) throws -> Data {
        guard (32...8192).contains(imageSize) else {
            throw NativeIOError.invalid("Invalid Core ML default image size.")
        }
        let architecture = try NativeArchitecture(mapKind: mapKind, width: width, blocks: blocks, identifier: identifier)
        let channels = mapKind == "normal" ? 3 : 1
        let expected = architecture.parameterShapes
        guard Set(expected.keys) == Set(weights.keys) else {
            throw NativeIOError.invalid("Core ML export requires exactly the \(identifier) parameter names.")
        }
        for (name, shape) in expected {
            guard let tensor = weights[name], tensor.shape == shape else {
                throw NativeIOError.invalid("Core ML export weight \(name) has an incompatible shape.")
            }
            try NativeWeights.validate(tensor, name: name)
        }
        if architecture.isCompact {
            guard imageSize.isMultiple(of: 8) else {
                throw NativeIOError.invalid("Compact Core ML export requires dimensions divisible by eight without resizing or padding.")
            }
            return compactModelData(weights: weights, architecture: architecture, imageSize: imageSize)
        }

        var network = CoreMLWire()
        network.integer(5, 1) // EXACT_ARRAY_MAPPING retains NCHW rank 4.
        network.integer(6, 1) // RANK4_IMAGE_MAPPING disables legacy rank-5 mapping.
        func layer(_ name: String, _ inputs: [String], _ output: String, _ field: Int, _ params: CoreMLWire,
                   inputRanks: [Int]? = nil, outputRank: Int = 4) {
            var node = CoreMLWire()
            node.string(1, name)
            for input in inputs { node.string(2, input) }
            node.string(3, output)
            for (index, _) in inputs.enumerated() {
                var tensor = CoreMLWire(); tensor.integer(1, inputRanks?[index] ?? 4)
                node.message(4, tensor)
            }
            var outputTensor = CoreMLWire(); outputTensor.integer(1, outputRank)
            node.message(5, outputTensor)
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
            let stride = architecture.convolutionStride(name)
            params.integers(30, [stride, stride])
            params.integers(40, [1, 1])
            if stride == 2 {
                // Explicit symmetric pad 1 preserves the native sampling origin
                // for even as well as odd dimensions. SAME can shift it.
                var edge = CoreMLWire(); edge.integer(1, 1); edge.integer(2, 1)
                var borders = CoreMLWire(); borders.message(10, edge); borders.message(10, edge)
                var padding = CoreMLWire(); padding.message(1, borders)
                params.message(50, padding)
            } else {
                params.message(51, CoreMLWire()) // SAME 3x3 stride 1: symmetric zero pad 1.
            }
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
        func hiddenActivation(_ name: String, _ input: String, _ output: String) {
            guard architecture.usesLeakyActivation else { activation(name, input, output, 10); return }
            // ActivationParams.leakyReLU = 15, ActivationLeakyReLU.alpha = 1.
            // The separate model identity preserves existing exported graphs.
            var slope = CoreMLWire(); slope.float(1, Float(architecture.hiddenNegativeSlope))
            var params = CoreMLWire(); params.message(15, slope)
            layer(name, [input], output, 130, params)
        }
        func scalar(_ name: String, _ input: String, _ output: String, field: Int, value: Float) {
            var params = CoreMLWire(); params.float(1, value)
            layer(name, [input], output, field, params)
        }
        func nearestDouble(_ name: String, _ input: String, matching reference: String) -> String {
            var upsample = CoreMLWire(); upsample.integers(1, [2, 2]); upsample.integer(5, 0)
            layer(name + ".nearest", [input], name + ".nearest", 210, upsample)
            // Derive spatial ends at inference. The legacy Crop layer requires
            // static reference dimensions, while ND slice supports flexible sizes.
            layer(name + ".shape", [reference], name + ".shape", 1065, CoreMLWire(), outputRank: 1)
            var zeros = CoreMLWire(); zeros.integers(1, [4])
            var values = CoreMLWire(); values.floats(1, [0, 0, 0, 0]); zeros.message(2, values)
            layer(name + ".begin", [], name + ".begin", 1070, zeros, outputRank: 1)
            var crop = CoreMLWire(); crop.integers(2, [1, 1, 1, 1]); crop.integers(4, [1, 1, 0, 0])
            crop.integers(5, [1, 1, 1, 1]); crop.integers(6, [0, 0, 0, 0])
            layer(name + ".trim", [name + ".nearest", name + ".begin", name + ".shape"],
                name + ".trim", 1000, crop, inputRanks: [4, 1, 1])
            return name + ".trim"
        }

        convolution("input", "rgb", "input_conv")
        hiddenActivation("input_relu", "input_conv", "input_relu")
        var value = "input_relu"
        if architecture.isPyramid {
            convolution("encoder.half", value, "encoder.half.conv")
            hiddenActivation("encoder.half.relu", "encoder.half.conv", "encoder.half.relu")
            convolution("encoder.quarter", "encoder.half.relu", "encoder.quarter.conv")
            hiddenActivation("encoder.quarter.relu", "encoder.quarter.conv", "encoder.quarter.relu")
            value = "encoder.quarter.relu"
        }
        for index in 0..<blocks {
            let prefix = "body.\(index)"
            convolution(prefix + ".conv1", value, prefix + ".conv1")
            hiddenActivation(prefix + ".relu1", prefix + ".conv1", prefix + ".relu1")
            convolution(prefix + ".conv2", prefix + ".relu1", prefix + ".conv2")
            layer(prefix + ".residual", [value, prefix + ".conv2"], prefix + ".residual", 230, CoreMLWire())
            hiddenActivation(prefix + ".relu2", prefix + ".residual", prefix + ".relu2")
            value = prefix + ".relu2"
        }
        if architecture.isPyramid {
            value = nearestDouble("decoder.half", value, matching: "encoder.half.relu")
            convolution("decoder.half", value, "decoder.half.conv")
            layer("decoder.half.skip", ["decoder.half.conv", "encoder.half.relu"], "decoder.half.skip", 230, CoreMLWire())
            hiddenActivation("decoder.half.relu", "decoder.half.skip", "decoder.half.relu")
            value = nearestDouble("decoder.full", "decoder.half.relu", matching: "input_relu")
            convolution("decoder.full", value, "decoder.full.conv")
            layer("decoder.full.skip", ["decoder.full.conv", "input_relu"], "decoder.full.skip", 230, CoreMLWire())
            hiddenActivation("decoder.full.relu", "decoder.full.skip", "decoder.full.relu")
            value = "decoder.full.relu"
        }
        convolution("output", value, "logits")
        if mapKind != "normal" || !architecture.isPyramidV2 {
            activation("output_sigmoid", "logits", mapKind == "normal" ? "sigmoid" : "material_value", 40)
        }
        if mapKind == "normal" {
            // Match torch.normalize(v, dim=1, eps=1e-6): denominator is
            // max(sqrt(sum(v*v)), eps), not sqrt(sum(v*v) + eps).
            let vector: String
            if architecture.isPyramidV2 { vector = "logits" }
            else {
                scalar("normal_scale", "sigmoid", "normal_scaled", field: 231, value: 2)
                scalar("normal_center", "normal_scaled", "normal_vector", field: 230, value: -1)
                vector = "normal_vector"
            }
            var reduce = CoreMLWire(); reduce.integers(1, [1]); reduce.integer(2, 1)
            layer("normal_length", [vector], "normal_length", 1255, reduce)
            var clamp = CoreMLWire(); clamp.float(1, 1e-6); clamp.float(2, Float.greatestFiniteMagnitude)
            layer("normal_denominator", ["normal_length"], "normal_denominator", 660, clamp)
            layer("normal_unit", [vector, "normal_denominator"], "normal_unit", 890, CoreMLWire())
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
        for (key, value) in ["loom.schema": "loom-factory-model-v1", "loom.architecture": identifier] {
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

    /// The compact model's linear output, reflection borders, channel-wise
    /// normalization and exact nearest replication are inference contracts.
    /// Keep this constructor separate so older exported graphs remain unchanged.
    private static func compactModelData(weights: [String: NativeTensor], architecture: NativeArchitecture,
                                         imageSize: Int) -> Data {
        var network = CoreMLWire()
        network.integer(5, 1) // EXACT_ARRAY_MAPPING
        network.integer(6, 1) // RANK4_IMAGE_MAPPING
        func layer(_ name: String, _ inputs: [String], _ outputs: [String], _ field: Int, _ params: CoreMLWire) {
            var node = CoreMLWire()
            node.string(1, name)
            for input in inputs { node.string(2, input) }
            for output in outputs { node.string(3, output) }
            for _ in inputs {
                var tensor = CoreMLWire(); tensor.integer(1, 4); node.message(4, tensor)
            }
            for _ in outputs {
                var tensor = CoreMLWire(); tensor.integer(1, 4); node.message(5, tensor)
            }
            node.message(field, params); network.message(1, node)
        }
        func operation(_ name: String, _ inputs: [String], _ field: Int, _ params: CoreMLWire = CoreMLWire()) -> String {
            layer(name, inputs, [name], field, params); return name
        }
        func constant(_ name: String, _ values: [Float], shape: [Int]) -> String {
            var params = CoreMLWire(); params.integers(1, shape)
            var data = CoreMLWire(); data.floats(1, values); params.message(2, data)
            return operation(name, [], 1070, params)
        }
        func parameter(_ name: String) -> String {
            let tensor = weights[name]!
            return constant(name, tensor.values, shape: tensor.shape)
        }
        let epsilon = constant("normalization_epsilon", [1e-6], shape: [1, 1, 1, 1])
        func mean(_ name: String, _ input: String, axes: [Int]) -> String {
            var params = CoreMLWire(); params.integers(1, axes); params.integer(2, 1)
            return operation(name, [input], 1280, params)
        }
        func convolution(_ name: String, _ input: String, stride: Int = 1, groups: Int = 1) -> String {
            let weight = weights[name + ".weight"]!, bias = weights[name + ".bias"]!
            let kernel = weight.shape[2]
            var source = input
            if stride == 1, kernel > 1 {
                var edge = CoreMLWire(); edge.integer(1, kernel / 2); edge.integer(2, kernel / 2)
                var borders = CoreMLWire(); borders.message(10, edge); borders.message(10, edge)
                var padding = CoreMLWire(); padding.message(2, CoreMLWire()); padding.message(10, borders)
                source = operation(name + ".reflect", [input], 200, padding)
            }
            var params = CoreMLWire()
            params.integer(1, weight.shape[0]); params.integer(2, weight.shape[1]); params.integer(10, groups)
            params.integers(20, [kernel, kernel]); params.integers(30, [stride, stride]); params.integers(40, [1, 1])
            params.message(50, CoreMLWire()) // VALID: reflection, if needed, is explicit above.
            params.integer(70, 1)
            var values = CoreMLWire(); values.floats(1, weight.values); params.message(90, values)
            var offsets = CoreMLWire(); offsets.floats(1, bias.values); params.message(91, offsets)
            return operation(name, [source], 100, params)
        }
        func normalize(_ name: String, _ input: String) -> String {
            let average = mean(name + ".mean", input, axes: [1])
            let centered = operation(name + ".centered", [input, average], 905)
            let squares = operation(name + ".squares", [centered, centered], 900)
            let variance = mean(name + ".variance", squares, axes: [1])
            let adjusted = operation(name + ".adjusted_variance", [variance, epsilon], 880)
            var squareRoot = CoreMLWire(); squareRoot.integer(1, 0)
            let denominator = operation(name + ".denominator", [adjusted], 220, squareRoot)
            let normalized = operation(name + ".normalized", [centered, denominator], 890)
            let gain = parameter(name + ".weight"), bias = parameter(name + ".bias")
            let scaled = operation(name + ".scaled", [normalized, gain], 900)
            return operation(name + ".output", [scaled, bias], 880)
        }
        func gate(_ name: String, _ input: String) -> String {
            var split = CoreMLWire(); split.integer(1, 1); split.integer(2, 2)
            let first = name + ".first", second = name + ".second"
            layer(name + ".split", [input], [first, second], 975, split)
            return operation(name, [first, second], 900)
        }
        func block(_ name: String, _ input: String, channels: Int) -> String {
            var value = convolution(name + ".expand", normalize(name + ".norm1", input))
            value = convolution(name + ".depthwise", value, groups: channels * 2)
            value = gate(name + ".gate1", value)
            let pooled = mean(name + ".pooled", value, axes: [2, 3])
            let attention = convolution(name + ".attention", pooled)
            value = operation(name + ".attended", [value, attention], 900)
            value = convolution(name + ".project", value)
            let beta = parameter(name + ".beta")
            let scaled = operation(name + ".scaled_residual", [value, beta], 900)
            let residual = operation(name + ".residual", [input, scaled], 880)
            value = convolution(name + ".ffn_expand", normalize(name + ".norm2", residual))
            value = convolution(name + ".ffn_project", gate(name + ".gate2", value))
            let gamma = parameter(name + ".gamma")
            value = operation(name + ".scaled_ffn", [value, gamma], 900)
            return operation(name + ".output", [residual, value], 880)
        }
        var value = convolution("input", "rgb"), skips: [String] = []
        for level in 0..<3 {
            value = block("encoder.\(level)", value, channels: architecture.width << level)
            skips.append(value)
            value = convolution("down.\(level)", value, stride: 2)
        }
        value = block("middle", value, channels: architecture.width * 8)
        for level in (0..<3).reversed() {
            value = convolution("up.\(level)", value)
            var upsample = CoreMLWire(); upsample.integers(1, [2, 2]); upsample.integer(5, 0)
            value = operation("up.\(level).nearest", [value], 210, upsample)
            value = operation("up.\(level).skip", [value, skips[level]], 880)
            value = block("decoder.\(level)", value, channels: architecture.width << level)
        }
        value = convolution("output", value)
        if architecture.isPhoto { value = operation("photo_rgb_residual", ["rgb", value], 880) }
        var outputShape = CoreMLWire(); outputShape.integers(1, [1, architecture.outputChannels, 0, 0])
        layer("output_nchw", [value], [architecture.outputName], 1150, outputShape)

        var description = CoreMLWire()
        description.message(1, feature(name: "rgb", channels: 3, size: imageSize, flexible: true))
        description.message(10, feature(name: architecture.outputName, channels: architecture.outputChannels, size: imageSize, flexible: true))
        var metadata = CoreMLWire(); metadata.string(1, "Texture Loom model trained natively in Loom Factory")
        var attributes = ["loom.schema": "loom-factory-model-v1", "loom.architecture": architecture.identifier]
        if architecture.isPhoto { attributes["loom.schema"] = "loom-factory-photo-model-v1"; attributes["loom.product"] = architecture.mapKind; attributes["loom.width"] = String(architecture.width) }
        for (key, value) in attributes {
            var pair = CoreMLWire(); pair.string(1, key); pair.string(2, value); metadata.message(100, pair)
        }
        description.message(100, metadata)
        var model = CoreMLWire(); model.integer(1, 7); model.message(2, description); model.message(500, network)
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
