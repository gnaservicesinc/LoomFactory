import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Stop is independent of the serial GPU worker. The UI never has to wait for
/// that worker's actor or dispatch queue to set the cancellation flag.
public final class NativeCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var requested = false
    public init() {}
    public func cancel() { lock.withLock { requested = true } }
    public var isCancelled: Bool { lock.withLock { requested } }
    public func check() throws { if isCancelled { throw CancellationError() } }
}

private struct NativeBatchShape: Hashable { let batch: Int; let height: Int; let width: Int }

/// Every feed and persistent result owns contiguous Metal storage through the
/// complete graph call. No checkpoint tensor depends on graph scratch storage.
private struct NativeTensorStorage {
    let data: MPSGraphTensorData
    let backingBuffer: MTLBuffer
    init(data: MPSGraphTensorData, backingBuffer: MTLBuffer) {
        self.data = data; self.backingBuffer = backingBuffer
    }
}

private final class NativeBatchGraph {
    let graph = MPSGraph()
    let input: MPSGraphTensor
    let target: MPSGraphTensor
    let weight: MPSGraphTensor
    let prediction: MPSGraphTensor
    let loss: MPSGraphTensor
    var parameters: [String: MPSGraphTensor] = [:]
    var accumulators: [String: MPSGraphTensor] = [:]
    var accumulated: [String: MPSGraphTensor] = [:]
    init(shape: NativeBatchShape, architecture: NativeArchitecture, shapes: [String: [Int]],
         trainable: Set<String>, adapter: NativeAdapterConfiguration?) {
        input = graph.placeholder(shape: [shape.batch, 3, shape.height, shape.width].map(NSNumber.init(value:)), dataType: .float32, name: "rgb")
        target = graph.placeholder(shape: [shape.batch, architecture.outputChannels, shape.height, shape.width].map(NSNumber.init(value:)), dataType: .float32, name: "target")
        weight = graph.placeholder(shape: [], dataType: .float32, name: "accumulation_weight")
        for name in shapes.keys.sorted() {
            parameters[name] = graph.placeholder(shape: shapes[name]!.map(NSNumber.init(value:)), dataType: .float32, name: name)
        }
        prediction = graph.materialNetwork(input: input, architecture: architecture, parameters: parameters, adapter: adapter)
        loss = graph.materialLoss(prediction: prediction, target: target, mapKind: architecture.mapKind)
        let weightedLoss = graph.multiplication(loss, weight, name: nil)
        let names = trainable.sorted()
        let gradients = graph.gradients(of: weightedLoss, with: names.map { parameters[$0]! }, name: nil)
        for name in names {
            let placeholder = graph.placeholder(shape: shapes[name]!.map(NSNumber.init(value:)), dataType: .float32, name: "accumulated." + name)
            accumulators[name] = placeholder
            accumulated[name] = graph.addition(placeholder, gradients[parameters[name]!]!, name: "sum." + name)
        }
    }
    var trainingTargets: [MPSGraphTensor] { [loss] + accumulated.keys.sorted().compactMap { accumulated[$0] } }
}

/// A session has one serial owner. GPU tensors, gradients and optimizer moments
/// remain resident across steps; only loss/norm cross to the CPU during training.
/// Whole checkpoints are transferred by the explicit weights/state methods.
public final class NativeTrainingSession {
    public let architecture: NativeArchitecture
    public let options: NativeTrainingOptions
    public let adapter: NativeAdapterConfiguration?
    public let cancellation: NativeCancellation
    public let deviceName: String
    public private(set) var updateCount: Int
    public var trainableParameterCount: Int { trainableShapes.values.reduce(0) { $0 + $1.reduce(1, *) } }
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let shapes: [String: [Int]]
    private let trainableShapes: [String: [Int]]
    private var parameters: [String: NativeTensorStorage] = [:]
    private var accumulation: [String: NativeTensorStorage] = [:]
    private var zeroAccumulation: [String: NativeTensorStorage] = [:]
    private var first: [String: NativeTensorStorage] = [:]
    private var second: [String: NativeTensorStorage] = [:]
    private var batchGraphs: [NativeBatchShape: NativeBatchGraph] = [:]
    private let optimizerGraph: NativeOptimizerGraph
    private var accumulatedBatches = 0

    public init(architecture: NativeArchitecture, options: NativeTrainingOptions,
                seed: UInt64, weights: [String: NativeTensor]? = nil,
                adapter: NativeAdapterConfiguration? = nil, adapterWeights: [String: NativeTensor]? = nil,
                optimizerState: [String: NativeTensor]? = nil, initialUpdateCount: Int = 0,
                device: String = "auto", cancellation: NativeCancellation = NativeCancellation()) throws {
        try cancellation.check()
        guard ["auto", "mps"].contains(device) else {
            throw NativeTrainingError.unavailable("Native training requires the Apple GPU. Select Automatic or Apple GPU (Metal).")
        }
        guard let metal = MTLCreateSystemDefaultDevice(), let queue = metal.makeCommandQueue(), MPSSupportsMTLDevice(metal) else {
            throw NativeTrainingError.unavailable("No Apple GPU supporting native MPSGraph training is available.")
        }
        guard initialUpdateCount >= 0 else { throw NativeTrainingError.invalid("Optimizer step count cannot be negative.") }
        if adapter != nil && weights == nil { throw NativeTrainingError.invalid("LoRA training requires the exact installed base weights.") }
        if adapter == nil && adapterWeights != nil { throw NativeTrainingError.invalid("Adapter weights require a LoRA configuration.") }
        self.architecture = architecture; self.options = options; self.adapter = adapter
        self.cancellation = cancellation; self.device = metal; self.queue = queue
        deviceName = metal.name; updateCount = initialUpdateCount
        var allShapes = architecture.parameterShapes
        var trainedShapes = allShapes
        if let adapter {
            trainedShapes = [:]
            for name in architecture.convolutionNames {
                let baseShape = allShapes[name + ".weight"]!
                trainedShapes[name + ".down.weight"] = [adapter.rank, baseShape[1], 3, 3]
                trainedShapes[name + ".up.weight"] = [baseShape[0], adapter.rank, 1, 1]
            }
            allShapes.merge(trainedShapes) { _, new in new }
        }
        shapes = allShapes; trainableShapes = trainedShapes
        optimizerGraph = NativeOptimizerGraph(shapes: trainedShapes, options: options)
        if let weights { try Self.validate(weights, expected: architecture.parameterShapes, purpose: "Base weights") }
        if let adapterWeights { try Self.validate(adapterWeights, expected: trainedShapes, purpose: "Adapter weights") }
        var random = NativeRandom(state: seed)
        for name in allShapes.keys.sorted() {
            try cancellation.check()
            let shape = allShapes[name]!
            let supplied = weights?[name] ?? adapterWeights?[name]
            let values: [Float]
            if let supplied { values = supplied.values }
            else if name.hasSuffix(".up.weight") { values = Array(repeating: 0, count: shape.reduce(1, *)) }
            else {
                let fanIn: Int
                if shape.count == 4 { fanIn = shape[1] * shape[2] * shape[3] }
                else { fanIn = allShapes[String(name.dropLast(4)) + "weight"]!.dropFirst().reduce(1, *) }
                values = random.values(count: shape.reduce(1, *), bound: 1 / sqrt(Float(fanIn)))
            }
            parameters[name] = try tensorData(values, shape: shape)
        }
        var stateShapes: [String: [Int]] = [:]
        for (name, shape) in trainedShapes { stateShapes["first." + name] = shape; stateShapes["second." + name] = shape }
        if let optimizerState { try Self.validate(optimizerState, expected: stateShapes, purpose: "Optimizer state") }
        for (name, shape) in trainedShapes {
            let zeros = Array(repeating: Float(0), count: shape.reduce(1, *))
            accumulation[name] = try tensorData(zeros, shape: shape)
            zeroAccumulation[name] = accumulation[name]
            first[name] = try tensorData(optimizerState?["first." + name]?.values ?? zeros, shape: shape)
            second[name] = try tensorData(optimizerState?["second." + name]?.values ?? zeros, shape: shape)
        }
    }

    /// Adds the weighted mean batch gradient. Use batchCount/windowCount for an
    /// accumulation window with a smaller final batch, matching sample means.
    @discardableResult
    public func train(input: [Float], target: [Float], batchSize: Int, height: Int, width: Int, weight: Float = 1) throws -> Double {
        try cancellation.check()
        guard weight.isFinite, weight > 0 else { throw NativeTrainingError.invalid("Gradient accumulation weight must be positive and finite.") }
        let compiled = try batchGraph(input: input, target: target, batchSize: batchSize, height: height, width: width)
        var feeds = parameterFeeds(compiled)
        feeds[compiled.input] = try tensorData(input, shape: [batchSize, 3, height, width])
        feeds[compiled.target] = try tensorData(target, shape: [batchSize, architecture.outputChannels, height, width])
        feeds[compiled.weight] = try tensorData([weight], shape: [])
        for (name, placeholder) in compiled.accumulators { feeds[placeholder] = accumulation[name]! }
        try cancellation.check()
        let results = try runGraph(compiled.graph, feeds: feeds, targets: compiled.trainingTargets)
        let loss = Double(read(results[compiled.loss]!, count: 1)[0])
        guard loss.isFinite else { throw NativeTrainingError.nonFinite("Training produced a non-finite loss. The previous checkpoint remains intact.") }
        try cancellation.check()
        for (name, result) in compiled.accumulated { accumulation[name] = results[result]! }
        accumulatedBatches += 1
        return loss
    }

    @discardableResult
    public func applyGradients(learningRate: Double) throws -> Double {
        try cancellation.check()
        guard learningRate.isFinite, learningRate >= 0, learningRate <= 1, accumulatedBatches > 0 else {
            throw NativeTrainingError.invalid("A finite learning rate and accumulated gradients are required for an optimizer step.")
        }
        let compiled = optimizerGraph
        var feeds: [MPSGraphTensor: NativeTensorStorage] = [:]
        for name in trainableShapes.keys {
            feeds[compiled.parameters[name]!] = parameters[name]!
            feeds[compiled.gradients[name]!] = accumulation[name]!
            feeds[compiled.first[name]!] = first[name]!
            feeds[compiled.second[name]!] = second[name]!
        }
        feeds[compiled.rate] = try tensorData([Float(learningRate)], shape: [])
        feeds[compiled.firstCorrection] = try tensorData([Float(1 - pow(options.beta1, Double(updateCount + 1)))], shape: [])
        feeds[compiled.secondCorrection] = try tensorData([Float(1 - pow(options.beta2, Double(updateCount + 1)))], shape: [])
        let results = try runGraph(compiled.graph, feeds: feeds, targets: compiled.targets)
        let norm = Double(read(results[compiled.norm]!, count: 1)[0])
        let invalidWeights = read(results[compiled.valid]!, count: 1)[0]
        guard norm.isFinite, invalidWeights == 0 else {
            throw NativeTrainingError.nonFinite("Training produced non-finite gradients or weights. The last finite checkpoint was preserved.")
        }
        try cancellation.check()
        for (name, result) in compiled.updated { parameters[name] = results[result]! }
        for (name, result) in compiled.updatedFirst { first[name] = results[result]! }
        for (name, result) in compiled.updatedSecond { second[name] = results[result]! }
        updateCount += 1
        resetAccumulatedGradients()
        return norm
    }

    public func resetAccumulatedGradients() {
        accumulation = zeroAccumulation
        accumulatedBatches = 0
    }

    public func evaluate(input: [Float], target: [Float], batchSize: Int, height: Int, width: Int) throws -> Double {
        try cancellation.check()
        let compiled = try batchGraph(input: input, target: target, batchSize: batchSize, height: height, width: width)
        var feeds = parameterFeeds(compiled)
        feeds[compiled.input] = try tensorData(input, shape: [batchSize, 3, height, width])
        feeds[compiled.target] = try tensorData(target, shape: [batchSize, architecture.outputChannels, height, width])
        let result = try runGraph(compiled.graph, feeds: feeds, targets: [compiled.loss])
        let loss = Double(read(result[compiled.loss]!, count: 1)[0])
        try cancellation.check()
        guard loss.isFinite else { throw NativeTrainingError.nonFinite("Held-out evaluation produced a non-finite loss.") }
        return loss
    }

    public func predict(input: [Float], batchSize: Int, height: Int, width: Int) throws -> [Float] {
        try cancellation.check()
        let count = try validateBatch(input: input, target: nil, batchSize: batchSize, height: height, width: width)
        let compiled = try batchGraph(input: input, target: nil, batchSize: batchSize, height: height, width: width)
        var feeds = parameterFeeds(compiled)
        feeds[compiled.input] = try tensorData(input, shape: [batchSize, 3, height, width])
        let result = try runGraph(compiled.graph, feeds: feeds, targets: [compiled.prediction])
        let values = read(result[compiled.prediction]!, count: count * architecture.outputChannels)
        try cancellation.check()
        guard values.allSatisfy(\.isFinite) else { throw NativeTrainingError.nonFinite("Prediction produced non-finite pixels.") }
        return values
    }

    /// Available after cancellation as well, so Stop can save the latest finite
    /// update. No pending partial accumulation is presented as an optimizer step.
    public func weights() throws -> [String: NativeTensor] {
        try export(names: Array(architecture.parameterShapes.keys))
    }
    public func adapterWeights() throws -> [String: NativeTensor] {
        guard adapter != nil else { throw NativeTrainingError.invalid("This session does not train a LoRA adapter.") }
        return try export(names: Array(trainableShapes.keys))
    }
    public func optimizerState() throws -> [String: NativeTensor] {
        var state: [String: NativeTensor] = [:]
        for (name, shape) in trainableShapes {
            state["first." + name] = NativeTensor(shape: shape, values: read(first[name]!, count: shape.reduce(1, *)))
            state["second." + name] = NativeTensor(shape: shape, values: read(second[name]!, count: shape.reduce(1, *)))
        }
        return state
    }
    public func accumulatedGradients() throws -> [String: NativeTensor] {
        Dictionary(uniqueKeysWithValues: trainableShapes.map { name, shape in
            (name, NativeTensor(shape: shape, values: read(accumulation[name]!, count: shape.reduce(1, *))))
        })
    }

    /// LoRA fusion uses the exact frozen kernels and rank update OIHW, without
    /// replacing or writing the installed base. This path only runs at export.
    public func fusedWeights() throws -> [String: NativeTensor] {
        var result = try weights()
        guard let adapter else { return result }
        let deltas = try adapterWeights()
        let scale = Float(adapter.alpha / Double(adapter.rank))
        for name in architecture.convolutionNames {
            try cancellation.check()
            let down = deltas[name + ".down.weight"]!
            let up = deltas[name + ".up.weight"]!
            var kernel = result[name + ".weight"]!
            let fanIn = down.shape.dropFirst().reduce(1, *)
            for output in 0..<kernel.shape[0] {
                for rank in 0..<adapter.rank {
                    let factor = up.values[output * adapter.rank + rank] * scale
                    if factor == 0 { continue }
                    for index in 0..<fanIn { kernel.values[output * fanIn + index] += factor * down.values[rank * fanIn + index] }
                }
            }
            guard kernel.values.allSatisfy(\.isFinite) else { throw NativeTrainingError.nonFinite("LoRA fusion produced non-finite weights.") }
            result[name + ".weight"] = kernel
        }
        return result
    }

    private func tensorData(_ values: [Float], shape: [Int]) throws -> NativeTensorStorage {
        // Own the feed's Metal storage explicitly. NSData-backed feeds may be
        // uploaded lazily; short-lived Swift Data must never own live GPU state.
        let buffer = values.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }
        guard let buffer else { throw NativeTrainingError.unavailable("The Apple GPU could not allocate native training tensor storage.") }
        return NativeTensorStorage(data: MPSGraphTensorData(buffer, shape: shape.map(NSNumber.init(value:)), dataType: .float32), backingBuffer: buffer)
    }
    private func runGraph(_ graph: MPSGraph, feeds: [MPSGraphTensor: NativeTensorStorage], targets: [MPSGraphTensor]) throws -> [MPSGraphTensor: NativeTensorStorage] {
        // Automatically allocated target data, fed into another graph without
        // CPU materialization, produced allocation-dependent gradients at full
        // resolution despite an identical loss. Explicit output buffers keep graph results
        // independent of the allocator and require no CPU gradient readback.
        var results: [MPSGraphTensor: NativeTensorStorage] = [:]
        guard let initialization = queue.makeCommandBuffer(), let blit = initialization.makeBlitCommandEncoder() else {
            throw NativeTrainingError.unavailable("The Apple GPU could not initialize native training result storage.")
        }
        for tensor in targets {
            guard let shape = tensor.shape, tensor.dataType == .float32 else {
                blit.endEncoding()
                throw NativeTrainingError.invalid("Native training requires fixed float32 result shapes.")
            }
            let byteCount = max(1, shape.reduce(1) { $0 * $1.intValue }) * MemoryLayout<Float>.size
            guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
                blit.endEncoding()
                throw NativeTrainingError.unavailable("The Apple GPU could not allocate native training result storage.")
            }
            blit.fill(buffer: buffer, range: 0..<byteCount, value: 0)
            results[tensor] = NativeTensorStorage(data: MPSGraphTensorData(buffer, shape: shape, dataType: .float32), backingBuffer: buffer)
        }
        blit.endEncoding()
        initialization.commit()
        initialization.waitUntilCompleted()
        guard initialization.status == .completed else {
            throw NativeTrainingError.unavailable("The Apple GPU could not initialize native training results: \(initialization.error?.localizedDescription ?? "command did not complete").")
        }
        // This MPSGraph API is synchronous: all output buffers are complete
        // before callers publish the next accumulation or optimizer state.
        withExtendedLifetime((feeds, results)) {
            graph.run(with: queue, feeds: feeds.mapValues(\.data), targetOperations: nil, resultsDictionary: results.mapValues(\.data))
        }
        return results
    }
    private func read(_ tensor: NativeTensorStorage, count: Int) -> [Float] {
        withExtendedLifetime(tensor) { read(tensor.data, count: count) }
    }
    private func read(_ tensor: MPSGraphTensorData, count: Int) -> [Float] {
        var values = Array(repeating: Float(0), count: count)
        values.withUnsafeMutableBytes { tensor.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        return values
    }
    private func export(names: [String]) throws -> [String: NativeTensor] {
        var result: [String: NativeTensor] = [:]
        for name in names {
            let shape = shapes[name]!
            let values = read(parameters[name]!, count: shape.reduce(1, *))
            guard values.allSatisfy(\.isFinite) else { throw NativeTrainingError.nonFinite("Cannot checkpoint non-finite weights.") }
            result[name] = NativeTensor(shape: shape, values: values)
        }
        return result
    }
    private func parameterFeeds(_ compiled: NativeBatchGraph) -> [MPSGraphTensor: NativeTensorStorage] {
        Dictionary(uniqueKeysWithValues: compiled.parameters.map { name, placeholder in (placeholder, parameters[name]!) })
    }
    private func batchGraph(input: [Float], target: [Float]?, batchSize: Int, height: Int, width: Int) throws -> NativeBatchGraph {
        _ = try validateBatch(input: input, target: target, batchSize: batchSize, height: height, width: width)
        let shape = NativeBatchShape(batch: batchSize, height: height, width: width)
        if let compiled = batchGraphs[shape] { return compiled }
        try cancellation.check()
        let compiled = NativeBatchGraph(shape: shape, architecture: architecture, shapes: shapes,
            trainable: Set(trainableShapes.keys), adapter: adapter)
        // Native-resolution batches normally have one or two sizes. Bound graph
        // cache growth when a caller evaluates many differing dimensions.
        if batchGraphs.count >= 4 { batchGraphs.removeAll() }
        batchGraphs[shape] = compiled
        return compiled
    }
    private func validateBatch(input: [Float], target: [Float]?, batchSize: Int, height: Int, width: Int) throws -> Int {
        guard (1...256).contains(batchSize), (1...8192).contains(height), (1...8192).contains(width) else {
            throw NativeTrainingError.invalid("Invalid native batch dimensions.")
        }
        let pixels = batchSize * height * width
        guard input.count == pixels * 3, target == nil || target!.count == pixels * architecture.outputChannels else {
            throw NativeTrainingError.invalid("Native NCHW input or target shape does not match its pixels.")
        }
        guard input.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
              target == nil || target!.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
            throw NativeTrainingError.invalid("Native input and target pixels must be finite values in [0,1].")
        }
        return pixels
    }
    private static func validate(_ values: [String: NativeTensor], expected: [String: [Int]], purpose: String) throws {
        guard Set(values.keys) == Set(expected.keys) else { throw NativeTrainingError.invalid("\(purpose) parameter names do not match residual-v1.") }
        for (name, shape) in expected {
            guard values[name]!.shape == shape, values[name]!.values.count == shape.reduce(1, *), values[name]!.values.allSatisfy(\.isFinite) else {
                throw NativeTrainingError.invalid("\(purpose) tensor \(name) has an incompatible shape or non-finite values.")
            }
        }
    }
}
