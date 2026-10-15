import Foundation
import MetalPerformanceShadersGraph

/// Loss contracts are versioned independently of the unchanged inference graph.
/// Existing checkpoints retain their recorded objective unless explicitly changed.
public enum NativeScalarLoss: String, Codable, CaseIterable, Sendable {
    case absoluteV1 = "absolute-v1"
    case logitCrossEntropyV2 = "logit-cross-entropy-v2"
    case nativeDetailV3 = "native-detail-v3"

    public var displayName: String {
        switch self {
        case .absoluteV1: "Original absolute error (v1)"
        case .logitCrossEntropyV2: "Stable pixel loss (v2)"
        case .nativeDetailV3: "Native pixel detail (v3)"
        }
    }
}

public struct NativeTrainingOptions: Sendable, Equatable {
    public let optimizer: String
    public let weightDecay: Double
    public let beta1: Double
    public let beta2: Double
    public let momentum: Double
    public let gradientClipNorm: Double
    public let gradientClipPolicy: String
    public let gradientClipWarmupStartNorm: Double
    public let gradientClipWarmupUpdates: Int
    public let scalarLoss: NativeScalarLoss
    public init(optimizer: String = "adamw", weightDecay: Double = 0.01, beta1: Double = 0.9,
                beta2: Double = 0.999, momentum: Double = 0.9, gradientClipNorm: Double = 1,
                gradientClipPolicy: String = "fixed", gradientClipWarmupStartNorm: Double = 0.25,
                gradientClipWarmupUpdates: Int = 100,
                scalarLoss: NativeScalarLoss = .absoluteV1) throws {
        let activeBeta1 = optimizer == "sgd" ? 0.9 : beta1
        let activeBeta2 = optimizer == "sgd" ? 0.999 : beta2
        let activeMomentum = optimizer == "sgd" ? momentum : 0.9
        let activeClipNorm = gradientClipPolicy == "off" ? 0 : gradientClipNorm
        let usesWarmup = gradientClipPolicy == "warmup" && activeClipNorm > 0
        let activeWarmupStart = usesWarmup ? gradientClipWarmupStartNorm : 0.25
        let activeWarmupUpdates = usesWarmup ? gradientClipWarmupUpdates : 100
        guard ["adamw", "adam", "sgd", "lion"].contains(optimizer), weightDecay.isFinite,
              (0...10).contains(weightDecay), ["fixed", "warmup", "off"].contains(gradientClipPolicy),
              activeClipNorm.isFinite, (0...1_000_000).contains(activeClipNorm),
              activeWarmupStart.isFinite, activeWarmupStart > 0,
              !usesWarmup || activeWarmupStart <= activeClipNorm,
              (1...1_000_000).contains(activeWarmupUpdates),
              activeBeta1.isFinite, activeBeta2.isFinite, (0..<1).contains(activeBeta1), (0..<1).contains(activeBeta2),
              activeMomentum.isFinite, (0...1).contains(activeMomentum) else {
            throw NativeTrainingError.invalid("Invalid native optimizer settings.")
        }
        self.optimizer = optimizer; self.weightDecay = weightDecay; self.beta1 = activeBeta1
        self.beta2 = activeBeta2; self.momentum = activeMomentum; self.gradientClipNorm = activeClipNorm
        self.gradientClipPolicy = gradientClipPolicy; self.gradientClipWarmupStartNorm = activeWarmupStart
        self.gradientClipWarmupUpdates = activeWarmupUpdates
        self.scalarLoss = scalarLoss
    }
    /// The threshold for the next update. Warmup starts tight and reaches the
    /// final cap after this many successful updates. Restored step counts keep
    /// their progress; accumulated batches and skipped updates do not advance it.
    /// This is a tunable global-norm schedule, not delta-GClip or a lower bound.
    public func gradientClipThreshold(updateCount: Int) -> Double {
        guard gradientClipPolicy != "off", gradientClipNorm > 0 else { return 0 }
        guard gradientClipPolicy == "warmup" else { return gradientClipNorm }
        let completed = min(max(0, updateCount), gradientClipWarmupUpdates)
        let fraction = Double(completed) / Double(gradientClipWarmupUpdates)
        return gradientClipWarmupStartNorm + (gradientClipNorm - gradientClipWarmupStartNorm) * fraction
    }
    public static func learningRate(base: Double, step: Int, totalSteps: Int, stepsPerEpoch: Int,
                                    warmupEpochs: Int, schedule: String, minimumRatio: Double) -> Double {
        let warmupSteps = warmupEpochs * stepsPerEpoch
        if step < warmupSteps { return base * Double(step + 1) / Double(warmupSteps) }
        guard schedule == "cosine" else { return base }
        let decaySteps = totalSteps - warmupSteps
        let fraction = decaySteps > 1 ? Double(step - warmupSteps) / Double(decaySteps - 1) : 0
        return base * (minimumRatio + (1 - minimumRatio) * (1 + cos(.pi * fraction)) / 2)
    }
}

/// All optimizer arithmetic stays on Metal. First/second moments are fed back as
/// GPU tensor data, rather than round-tripping millions of weights through Swift.
final class NativeOptimizerGraph {
    let graph = MPSGraph()
    var parameters: [String: MPSGraphTensor] = [:]
    var gradients: [String: MPSGraphTensor] = [:]
    var first: [String: MPSGraphTensor] = [:]
    var second: [String: MPSGraphTensor] = [:]
    var updated: [String: MPSGraphTensor] = [:]
    var updatedFirst: [String: MPSGraphTensor] = [:]
    var updatedSecond: [String: MPSGraphTensor] = [:]
    let rate: MPSGraphTensor
    let firstCorrection: MPSGraphTensor
    let secondCorrection: MPSGraphTensor
    let clipThreshold: MPSGraphTensor
    let norm: MPSGraphTensor
    let valid: MPSGraphTensor

    init(shapes: [String: [Int]], options: NativeTrainingOptions) {
        rate = graph.placeholder(shape: [], dataType: .float32, name: "learning_rate")
        firstCorrection = graph.placeholder(shape: [], dataType: .float32, name: "bias_correction_1")
        secondCorrection = graph.placeholder(shape: [], dataType: .float32, name: "bias_correction_2")
        clipThreshold = graph.placeholder(shape: [], dataType: .float32, name: "gradient_clip_threshold")
        var sum = graph.scalar(0)
        for name in shapes.keys.sorted() {
            let shape = shapes[name]!.map(NSNumber.init(value:))
            parameters[name] = graph.placeholder(shape: shape, dataType: .float32, name: name)
            gradients[name] = graph.placeholder(shape: shape, dataType: .float32, name: "gradient." + name)
            first[name] = graph.placeholder(shape: shape, dataType: .float32, name: "first." + name)
            second[name] = graph.placeholder(shape: shape, dataType: .float32, name: "second." + name)
            let squared = graph.square(with: gradients[name]!, name: nil)
            let reduce = graph.reductionSum(with: squared, axes: nil, name: nil)
            sum = graph.addition(sum, graph.reshape(reduce, shape: [], name: nil), name: nil)
        }
        norm = graph.squareRoot(with: sum, name: "gradient_norm")
        // One scale for the entire accumulated gradient vector preserves its
        // direction, and is applied exactly once before optimizer arithmetic.
        let clippedScale = graph.minimum(graph.scalar(1), graph.division(clipThreshold,
            graph.addition(norm, graph.scalar(1e-6), name: nil), name: nil), name: nil)
        let scale = graph.select(predicate: graph.greaterThan(clipThreshold, graph.scalar(0), name: nil),
            trueTensor: clippedScale, falseTensor: graph.scalar(1), name: nil)
        var finite = graph.scalar(0)
        for name in shapes.keys.sorted() {
            let parameter = parameters[name]!
            var gradient = graph.multiplication(gradients[name]!, scale, name: nil)
            if options.optimizer == "adam" || options.optimizer == "sgd" {
                gradient = graph.addition(gradient, graph.scaled(parameter, options.weightDecay), name: nil)
            }
            let direction: MPSGraphTensor
            let oldFirst = first[name]!
            if options.optimizer == "sgd" {
                // PyTorch's first momentum buffer equals the first gradient;
                // zero initialization followed by m*g+g has that same result.
                let buffer = graph.addition(graph.scaled(oldFirst, options.momentum), gradient, name: nil)
                updatedFirst[name] = buffer
                direction = buffer
            } else if options.optimizer == "lion" {
                let blend = graph.addition(graph.scaled(oldFirst, options.beta1), graph.scaled(gradient, 1 - options.beta1), name: nil)
                direction = graph.sign(with: blend, name: nil)
                updatedFirst[name] = graph.addition(graph.scaled(oldFirst, options.beta2), graph.scaled(gradient, 1 - options.beta2), name: nil)
            } else {
                let moment = graph.addition(graph.scaled(oldFirst, options.beta1), graph.scaled(gradient, 1 - options.beta1), name: nil)
                let variance = graph.addition(graph.scaled(second[name]!, options.beta2),
                    graph.scaled(graph.square(with: gradient, name: nil), 1 - options.beta2), name: nil)
                updatedFirst[name] = moment; updatedSecond[name] = variance
                let correctedMoment = graph.division(moment, firstCorrection, name: nil)
                let denominator = graph.addition(graph.squareRoot(with: graph.division(variance, secondCorrection, name: nil), name: nil), graph.scalar(1e-8), name: nil)
                direction = graph.division(correctedMoment, denominator, name: nil)
            }
            var value = parameter
            if options.optimizer == "adamw" || options.optimizer == "lion" {
                let decay = graph.subtraction(graph.scalar(1), graph.scaled(rate, options.weightDecay), name: nil)
                value = graph.multiplication(value, decay, name: nil)
            }
            updated[name] = graph.subtraction(value, graph.multiplication(rate, direction, name: nil), name: nil)
            let invalid = graph.select(predicate: graph.isFinite(with: updated[name]!, name: nil),
                                       trueTensor: graph.scalar(0), falseTensor: graph.scalar(1), name: nil)
            let proof = graph.reductionMaximum(with: invalid, axes: nil, name: nil)
            finite = graph.addition(finite, graph.reshape(proof, shape: [], name: nil), name: nil)
        }
        valid = finite
    }
    var targets: [MPSGraphTensor] {
        [norm, valid] + updated.keys.sorted().compactMap { updated[$0] }
            + updatedFirst.keys.sorted().compactMap { updatedFirst[$0] }
            + updatedSecond.keys.sorted().compactMap { updatedSecond[$0] }
    }
}
