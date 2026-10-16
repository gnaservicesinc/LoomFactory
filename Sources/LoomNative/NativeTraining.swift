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

/// Measurements of the latest training, optimizer, evaluation, or prediction
/// call. The synchronous graph API does not expose its internal GPU timestamps;
/// gpuSpanSeconds remains zero rather than inferring utilization from CPU waits.
public struct NativeGraphExecutionMeasurements: Sendable {
    public fileprivate(set) var feedSeconds: Double = 0
    public fileprivate(set) var resultSetupSeconds: Double = 0
    public fileprivate(set) var encodingSeconds: Double = 0
    public fileprivate(set) var waitSeconds: Double = 0
    public fileprivate(set) var gpuSpanSeconds: Double = 0
    public fileprivate(set) var scalarReadSeconds: Double = 0
    public fileprivate(set) var totalSeconds: Double = 0
    public fileprivate(set) var feedBufferAllocationCount: Int = 0
    public fileprivate(set) var resultBufferAllocationCount: Int = 0
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

/// Graph calls are synchronous and have one serial owner. Retain fixed feed
/// storage and at most two result buffers per target, alternating when the last
/// result is an input to this call. Never clear or write a currently fed buffer.
/// This keeps autorelease drains without allocating every tensor on every step.
private final class NativeGraphStorage {
    // Create a fresh TensorData wrapper when reusing a buffer. Reusing the
    // wrapper itself failed the full-resolution gradient regression, even with
    // disjoint feed and result buffers. Preserve the uncached binding lifetime.
    var feeds: [MPSGraphTensor: MTLBuffer] = [:]
    var results: [MPSGraphTensor: [MTLBuffer]] = [:]
    var feedBufferAllocationCount = 0
    func beginOperation() { feedBufferAllocationCount = 0 }
}

private final class NativeBatchGraph {
    let graph = MPSGraph()
    let storage = NativeGraphStorage()
    let input: MPSGraphTensor
    let target: MPSGraphTensor
    let weight: MPSGraphTensor
    let prediction: MPSGraphTensor
    let loss: MPSGraphTensor
    var parameters: [String: MPSGraphTensor] = [:]
    var accumulators: [String: MPSGraphTensor] = [:]
    var accumulated: [String: MPSGraphTensor] = [:]
    init(shape: NativeBatchShape, architecture: NativeArchitecture, shapes: [String: [Int]],
         trainable: Set<String>, adapter: NativeAdapterConfiguration?, channelsLast: Bool,
         scalarLoss: NativeScalarLoss) {
        input = graph.placeholder(shape: [shape.batch, 3, shape.height, shape.width].map(NSNumber.init(value:)), dataType: .float32, name: "rgb")
        target = graph.placeholder(shape: [shape.batch, architecture.outputChannels, shape.height, shape.width].map(NSNumber.init(value:)), dataType: .float32, name: "target")
        weight = graph.placeholder(shape: [], dataType: .float32, name: "accumulation_weight")
        for name in shapes.keys.sorted() {
            parameters[name] = graph.placeholder(shape: shapes[name]!.map(NSNumber.init(value:)), dataType: .float32, name: name)
        }
        let logits = graph.materialLogits(input: input, architecture: architecture, parameters: parameters,
            adapter: adapter, channelsLast: channelsLast)
        prediction = graph.materialPrediction(logits: logits, architecture: architecture)
        if architecture.isCompact {
            loss = scalarLoss == .squaredStructureV1
                ? graph.squaredStructureLoss(prediction: prediction, target: target)
                : graph.linearDetailLoss(prediction: prediction, target: target)
        } else if architecture.mapKind == "normal" || scalarLoss == .absoluteV1 {
            loss = graph.materialLoss(prediction: prediction, target: target, mapKind: architecture.mapKind)
        } else {
            let level = graph.scalarLogitCrossEntropy(logits: logits, target: target)
            loss = scalarLoss == .nativeDetailV3
                ? graph.addition(level, graph.scaled(graph.nativeDetailLoss(prediction: prediction, target: target), 2), name: nil)
                : level
        }
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
    public private(set) var channelsLast: Bool
    public private(set) var trainingLayoutSelection: NativeTrainingLayoutSelection?
    public private(set) var updateCount: Int
    public private(set) var lastExecutionMeasurements: NativeGraphExecutionMeasurements?
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
    private let optimizerStorage = NativeGraphStorage()
    private var accumulatedBatches = 0
    private let automaticallySelectLayout: Bool

    public init(architecture: NativeArchitecture, options: NativeTrainingOptions,
                seed: UInt64, weights: [String: NativeTensor]? = nil,
                adapter: NativeAdapterConfiguration? = nil, adapterWeights: [String: NativeTensor]? = nil,
                optimizerState: [String: NativeTensor]? = nil, initialUpdateCount: Int = 0,
                device: String = "auto", cancellation: NativeCancellation = NativeCancellation(),
                channelsLast: Bool? = nil) throws {
        try cancellation.check()
        guard ["auto", "mps"].contains(device) else {
            throw NativeTrainingError.unavailable("Native training requires the Apple GPU. Select Automatic or Apple GPU (Metal).")
        }
        guard let metal = MTLCreateSystemDefaultDevice(), let queue = metal.makeCommandQueue(), MPSSupportsMTLDevice(metal) else {
            throw NativeTrainingError.unavailable("No Apple GPU supporting native MPSGraph training is available.")
        }
        guard architecture.isCompact == options.scalarLoss.isCompactObjective else {
            throw NativeTrainingError.invalid("Compact NAF requires its recorded linear detail objective; residual models retain their own objective.")
        }
        if architecture.isCompact, channelsLast == true {
            throw NativeTrainingError.invalid("Compact NAF currently uses its verified NCHW layout.")
        }
        guard initialUpdateCount >= 0 else { throw NativeTrainingError.invalid("Optimizer step count cannot be negative.") }
        if architecture.isPhoto && adapter != nil { throw NativeTrainingError.invalid("Photo enhancer adapters are not supported in v1.") }
        if adapter != nil && weights == nil { throw NativeTrainingError.invalid("LoRA training requires the exact installed base weights.") }
        if adapter == nil && adapterWeights != nil { throw NativeTrainingError.invalid("Adapter weights require a LoRA configuration.") }
        self.architecture = architecture; self.options = options; self.adapter = adapter
        self.cancellation = cancellation; self.device = metal; self.queue = queue
        // Layout performance varies with width and GPU. Large residual batches
        // calibrate both unchanged Float32 graphs before the first real update.
        // Explicit diagnostic choices and Pyramid keep their requested layout.
        self.channelsLast = channelsLast ?? (!architecture.isPyramid && !architecture.isCompact)
        automaticallySelectLayout = channelsLast == nil && !architecture.isPyramid && !architecture.isCompact
        deviceName = metal.name; updateCount = initialUpdateCount
        var allShapes = architecture.parameterShapes
        var trainedShapes = allShapes
        if let adapter {
            trainedShapes = [:]
            for name in architecture.adaptableConvolutionNames {
                let baseShape = allShapes[name + ".weight"]!
                trainedShapes[name + ".down.weight"] = [adapter.rank, baseShape[1], baseShape[2], baseShape[3]]
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
            else if architecture.isCompact, architecture.parameterShapes[name] != nil {
                values = NativeCompactNetwork.initialValues(name: name, shape: shape, random: &random, photo: architecture.isPhoto)
            }
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
        // A detached training worker may run for hours without returning to
        // a run loop. Drain transient Objective-C/Metal graph objects each
        // operation; persistent tensors retain their owned buffers.
        return try autoreleasepool {
            let executionStart = ProcessInfo.processInfo.systemUptime
            try cancellation.check()
            guard weight.isFinite, weight > 0 else { throw NativeTrainingError.invalid("Gradient accumulation weight must be positive and finite.") }
            if automaticallySelectLayout, trainingLayoutSelection == nil, min(height, width) >= 512,
               (1...256).contains(batchSize), (1...8192).contains(height), (1...8192).contains(width),
               UInt64(device.currentAllocatedSize) < device.recommendedMaxWorkingSetSize,
               UInt64(batchSize) * UInt64(height) * UInt64(width) * UInt64(architecture.width) * 4
                    * UInt64(2 * architecture.blocks + 12)
                    <= (device.recommendedMaxWorkingSetSize - UInt64(device.currentAllocatedSize)) / 2 {
                _ = try validateBatch(input: input, target: target, batchSize: batchSize, height: height, width: width)
                let selection = try NativeTrainingLayout.select(architecture: architecture, options: options,
                    adapter: adapter, input: input, target: target,
                    batch: batchSize, height: height, width: width, weights: { try self.weights() },
                    adapters: { try self.adapter.map { _ in try self.adapterWeights() } }, cancellation: cancellation)
                if channelsLast != selection.channelsLast {
                    batchGraphs.removeAll()
                    channelsLast = selection.channelsLast
                }
                trainingLayoutSelection = selection
            }
            let compiled = try batchGraph(input: input, target: target, batchSize: batchSize, height: height, width: width)
            compiled.storage.beginOperation()
            var feeds = parameterFeeds(compiled)
            feeds[compiled.input] = try cachedFeed(input, tensor: compiled.input, storage: compiled.storage)
            feeds[compiled.target] = try cachedFeed(target, tensor: compiled.target, storage: compiled.storage)
            feeds[compiled.weight] = try cachedFeed([weight], tensor: compiled.weight, storage: compiled.storage)
            for (name, placeholder) in compiled.accumulators { feeds[placeholder] = accumulation[name]! }
            try cancellation.check()
            let results = try runGraph(compiled.graph, storage: compiled.storage, feeds: feeds, targets: compiled.trainingTargets,
                                       feedSeconds: ProcessInfo.processInfo.systemUptime - executionStart)
            let loss = Double(readScalar(results[compiled.loss]!))
            guard loss.isFinite else { throw NativeTrainingError.nonFinite("Training produced a non-finite loss. The previous checkpoint remains intact.") }
            try cancellation.check()
            for (name, result) in compiled.accumulated { accumulation[name] = results[result]! }
            accumulatedBatches += 1
            lastExecutionMeasurements?.totalSeconds = ProcessInfo.processInfo.systemUptime - executionStart
            return loss
        }
    }

    @discardableResult
    public func applyGradients(learningRate: Double) throws -> Double {
        return try autoreleasepool {
            let executionStart = ProcessInfo.processInfo.systemUptime
            try cancellation.check()
            guard learningRate.isFinite, learningRate >= 0, learningRate <= 1, accumulatedBatches > 0 else {
                throw NativeTrainingError.invalid("A finite learning rate and accumulated gradients are required for an optimizer step.")
            }
            let compiled = optimizerGraph
            optimizerStorage.beginOperation()
            var feeds: [MPSGraphTensor: NativeTensorStorage] = [:]
            for name in trainableShapes.keys {
                feeds[compiled.parameters[name]!] = parameters[name]!
                feeds[compiled.gradients[name]!] = accumulation[name]!
                feeds[compiled.first[name]!] = first[name]!
                feeds[compiled.second[name]!] = second[name]!
            }
            feeds[compiled.rate] = try cachedFeed([Float(learningRate)], tensor: compiled.rate, storage: optimizerStorage)
            feeds[compiled.clipThreshold] = try cachedFeed([Float(options.gradientClipThreshold(updateCount: updateCount))], tensor: compiled.clipThreshold, storage: optimizerStorage)
            feeds[compiled.firstCorrection] = try cachedFeed([Float(1 - pow(options.beta1, Double(updateCount + 1)))], tensor: compiled.firstCorrection, storage: optimizerStorage)
            feeds[compiled.secondCorrection] = try cachedFeed([Float(1 - pow(options.beta2, Double(updateCount + 1)))], tensor: compiled.secondCorrection, storage: optimizerStorage)
            let results = try runGraph(compiled.graph, storage: optimizerStorage, feeds: feeds, targets: compiled.targets,
                                       feedSeconds: ProcessInfo.processInfo.systemUptime - executionStart)
            let norm = Double(readScalar(results[compiled.norm]!))
            let invalidWeights = readScalar(results[compiled.valid]!)
            guard norm.isFinite, invalidWeights == 0 else {
                throw NativeTrainingError.nonFinite("Training produced non-finite gradients or weights. The last finite checkpoint was preserved.")
            }
            try cancellation.check()
            for (name, result) in compiled.updated { parameters[name] = results[result]! }
            for (name, result) in compiled.updatedFirst { first[name] = results[result]! }
            for (name, result) in compiled.updatedSecond { second[name] = results[result]! }
            updateCount += 1
            resetAccumulatedGradients()
            lastExecutionMeasurements?.totalSeconds = ProcessInfo.processInfo.systemUptime - executionStart
            return norm
        }
    }

    public func resetAccumulatedGradients() {
        accumulation = zeroAccumulation
        accumulatedBatches = 0
    }

    public func evaluate(input: [Float], target: [Float], batchSize: Int, height: Int, width: Int) throws -> Double {
        return try autoreleasepool {
            let executionStart = ProcessInfo.processInfo.systemUptime
            try cancellation.check()
            let compiled = try batchGraph(input: input, target: target, batchSize: batchSize, height: height, width: width)
            compiled.storage.beginOperation()
            var feeds = parameterFeeds(compiled)
            feeds[compiled.input] = try cachedFeed(input, tensor: compiled.input, storage: compiled.storage)
            feeds[compiled.target] = try cachedFeed(target, tensor: compiled.target, storage: compiled.storage)
            let result = try runGraph(compiled.graph, storage: compiled.storage, feeds: feeds, targets: [compiled.loss],
                                      feedSeconds: ProcessInfo.processInfo.systemUptime - executionStart)
            let loss = Double(readScalar(result[compiled.loss]!))
            try cancellation.check()
            guard loss.isFinite else { throw NativeTrainingError.nonFinite("Held-out evaluation produced a non-finite loss.") }
            lastExecutionMeasurements?.totalSeconds = ProcessInfo.processInfo.systemUptime - executionStart
            return loss
        }
    }

    public func predict(input: [Float], batchSize: Int, height: Int, width: Int) throws -> [Float] {
        return try autoreleasepool {
            let executionStart = ProcessInfo.processInfo.systemUptime
            try cancellation.check()
            let count = try validateBatch(input: input, target: nil, batchSize: batchSize, height: height, width: width)
            let compiled = try batchGraph(input: input, target: nil, batchSize: batchSize, height: height, width: width)
            compiled.storage.beginOperation()
            var feeds = parameterFeeds(compiled)
            feeds[compiled.input] = try cachedFeed(input, tensor: compiled.input, storage: compiled.storage)
            let result = try runGraph(compiled.graph, storage: compiled.storage, feeds: feeds, targets: [compiled.prediction],
                                      feedSeconds: ProcessInfo.processInfo.systemUptime - executionStart)
            let values = read(result[compiled.prediction]!, count: count * architecture.outputChannels)
            try cancellation.check()
            guard values.allSatisfy(\.isFinite) else { throw NativeTrainingError.nonFinite("Prediction produced non-finite pixels.") }
            lastExecutionMeasurements?.totalSeconds = ProcessInfo.processInfo.systemUptime - executionStart
            return values
        }
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
        for name in architecture.adaptableConvolutionNames {
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
        return try autoreleasepool {
            // Own the feed's Metal storage explicitly. NSData-backed feeds may be
            // uploaded lazily; short-lived Swift Data must never own live GPU state.
            let buffer = values.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            guard let buffer else { throw NativeTrainingError.unavailable("The Apple GPU could not allocate native training tensor storage.") }
            return NativeTensorStorage(data: MPSGraphTensorData(buffer, shape: shape.map(NSNumber.init(value:)), dataType: .float32), backingBuffer: buffer)
        }
    }
    private func cachedFeed(_ values: [Float], tensor: MPSGraphTensor,
                            storage: NativeGraphStorage) throws -> NativeTensorStorage {
        if let existing = storage.feeds[tensor] {
            values.withUnsafeBytes { bytes in
                existing.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
            return NativeTensorStorage(data: MPSGraphTensorData(existing, shape: tensor.shape!, dataType: .float32), backingBuffer: existing)
        }
        let created = try tensorData(values, shape: tensor.shape!.map(\.intValue))
        storage.feeds[tensor] = created.backingBuffer
        storage.feedBufferAllocationCount += 1
        return created
    }
    private func runGraph(_ graph: MPSGraph, storage: NativeGraphStorage,
                          feeds: [MPSGraphTensor: NativeTensorStorage],
                          targets: [MPSGraphTensor], feedSeconds: Double) throws -> [MPSGraphTensor: NativeTensorStorage] {
        // Automatically allocated target data, fed into another graph without
        // CPU materialization, produced allocation-dependent gradients at full
        // resolution despite an identical loss. Preserve explicit, zeroed output
        // buffers; no CPU gradient readback is required.
        let setupStart = ProcessInfo.processInfo.systemUptime
        var measurements = NativeGraphExecutionMeasurements()
        measurements.feedSeconds = feedSeconds
        measurements.feedBufferAllocationCount = storage.feedBufferAllocationCount
        var results: [MPSGraphTensor: NativeTensorStorage] = [:]
        let activeBuffers = Set(feeds.values.map { ObjectIdentifier($0.backingBuffer) })
        guard let initialization = queue.makeCommandBuffer(), let blit = initialization.makeBlitCommandEncoder() else {
            throw NativeTrainingError.unavailable("The Apple GPU could not initialize native training result storage.")
        }
        for tensor in targets {
            guard let shape = tensor.shape, shape.allSatisfy({ $0.intValue > 0 }), tensor.dataType == .float32 else {
                blit.endEncoding()
                throw NativeTrainingError.invalid("Native training requires fixed float32 result shapes.")
            }
            let byteCount = max(1, shape.reduce(1) { $0 * $1.intValue }) * MemoryLayout<Float>.size
            let buffer: MTLBuffer
            if let reusable = storage.results[tensor]?.first(where: { !activeBuffers.contains(ObjectIdentifier($0)) }) {
                buffer = reusable
            } else {
                // Only the latest result for a target can be a feed. A third
                // slot would indicate an ownership bug rather than justify
                // unbounded cache growth.
                guard (storage.results[tensor]?.count ?? 0) < 2 else {
                    blit.endEncoding()
                    throw NativeTrainingError.invalid("Native training result storage overlaps active inputs.")
                }
                guard let created = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
                    blit.endEncoding()
                    throw NativeTrainingError.unavailable("The Apple GPU could not allocate native training result storage.")
                }
                buffer = created
                storage.results[tensor, default: []].append(buffer)
                measurements.resultBufferAllocationCount += 1
            }
            let result = NativeTensorStorage(data: MPSGraphTensorData(buffer, shape: shape, dataType: .float32), backingBuffer: buffer)
            blit.fill(buffer: result.backingBuffer, range: 0..<byteCount, value: 0)
            results[tensor] = result
        }
        blit.endEncoding()
        let graphFeeds = feeds.mapValues(\.data)
        let graphResults = results.mapValues(\.data)
        measurements.resultSetupSeconds = ProcessInfo.processInfo.systemUptime - setupStart
        let waitStart = ProcessInfo.processInfo.systemUptime
        initialization.commit()
        initialization.waitUntilCompleted()
        measurements.waitSeconds = ProcessInfo.processInfo.systemUptime - waitStart
        guard initialization.status == .completed else {
            throw NativeTrainingError.unavailable("The Apple GPU could not initialize native training results: \(initialization.error?.localizedDescription ?? "command did not complete").")
        }
        let runStart = ProcessInfo.processInfo.systemUptime
        withExtendedLifetime((feeds, results)) {
            graph.run(with: queue, feeds: graphFeeds, targetOperations: nil, resultsDictionary: graphResults)
        }
        // The reference API includes execution in this blocking call and
        // does not expose the graph's underlying GPU command buffers.
        measurements.encodingSeconds = ProcessInfo.processInfo.systemUptime - runStart
        measurements.totalSeconds = feedSeconds + ProcessInfo.processInfo.systemUptime - setupStart
        lastExecutionMeasurements = measurements
        return results
    }
    private func readScalar(_ tensor: NativeTensorStorage) -> Float {
        let start = ProcessInfo.processInfo.systemUptime
        let value = read(tensor, count: 1)[0]
        lastExecutionMeasurements?.scalarReadSeconds += ProcessInfo.processInfo.systemUptime - start
        return value
    }
    private func read(_ tensor: NativeTensorStorage, count: Int) -> [Float] {
        withExtendedLifetime(tensor) { read(tensor.data, count: count) }
    }
    private func read(_ tensor: MPSGraphTensorData, count: Int) -> [Float] {
        return autoreleasepool {
            var values = Array(repeating: Float(0), count: count)
            values.withUnsafeMutableBytes { tensor.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
            return values
        }
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
            trainable: Set(trainableShapes.keys), adapter: adapter, channelsLast: channelsLast,
            scalarLoss: options.scalarLoss)
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
        if !architecture.supportsGrid(height: height, width: width) {
            throw NativeTrainingError.invalid("Compact NAF needs native dimensions of at least 32 pixels and multiples of 8. Choose a matching native crop; originals are unchanged.")
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
        guard Set(values.keys) == Set(expected.keys) else { throw NativeTrainingError.invalid("\(purpose) parameter names do not match the saved architecture.") }
        for (name, shape) in expected {
            guard values[name]!.shape == shape, values[name]!.values.count == shape.reduce(1, *), values[name]!.values.allSatisfy(\.isFinite) else {
                throw NativeTrainingError.invalid("\(purpose) tensor \(name) has an incompatible shape or non-finite values.")
            }
        }
    }
}
