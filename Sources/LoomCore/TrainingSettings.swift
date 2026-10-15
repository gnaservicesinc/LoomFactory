import Foundation
import LoomNative

public enum TrainingNetwork: String, Codable, CaseIterable, Sendable, Identifiable {
    case pyramid = "loom-factory-material-pyramid-v1"
    case residual = "loom-factory-material-residual-v1"
    public var id: String { rawValue }
    public var displayName: String { self == .pyramid ? "Pyramid (experimental)" : "Original residual" }
    public var detail: String {
        self == .pyramid
            ? "Downsamples learned features. Saved brick models show sampling-grid artifacts; use only for controlled comparisons."
            : "Runs every layer at native resolution without downsampling or upsampling feature grids. Default for new models."
    }
}

public enum TrainingOptimizer: String, Codable, CaseIterable, Sendable, Identifiable {
    case adamW = "adamw", adam, sgd, lion
    public var id: String { rawValue }
    public var displayName: String {
        switch self { case .adamW: "AdamW"; case .adam: "Adam"; case .sgd: "SGD"; case .lion: "Lion" }
    }
}
public enum TrainingRefresh: String, Codable, CaseIterable, Sendable, Identifiable {
    case perSample = "per_sample", perEpoch = "per_epoch", perRun = "per_run"
    public var id: String { rawValue }
    public var displayName: String {
        switch self { case .perSample: "Every image visit"; case .perEpoch: "Once per epoch"; case .perRun: "Once per run" }
    }
}
public enum TrainingLearningRateSchedule: String, Codable, CaseIterable, Sendable, Identifiable {
    case constant, cosine
    public var id: String { rawValue }
    public var displayName: String { self == .constant ? "Constant" : "Cosine decay" }
}
public enum TrainingGradientClipping: String, Codable, CaseIterable, Sendable, Identifiable {
    case warmup, fixed, off
    public var id: String { rawValue }
    public var displayName: String {
        switch self {
        case .warmup: "Warmup global norm"
        case .fixed: "Fixed global norm"
        case .off: "Off"
        }
    }
}
public enum TrainingDevice: String, Codable, CaseIterable, Sendable, Identifiable {
    case automatic = "auto", metal = "mps", cpu
    public var id: String { rawValue }
    public var displayName: String {
        switch self { case .automatic: "Automatic"; case .metal: "Apple GPU (Metal)"; case .cpu: "CPU" }
    }
}

public struct TrainingSettings: Codable, Sendable, Equatable {
    public var imageSize: Int
    /// One selects each material's smallest usable source grid; two and three
    /// also allow grids up to 2x and 4x that size. Zero keeps all source grids.
    public var sourceResolutionLevels: Int
    public var epochs: Int
    public var batchSize: Int
    public var learningRate: Double
    public var shuffleEachEpoch: Bool
    public var flipCrops: Bool
    public var rotateCrops: Bool
    public var exaggerateFeatures: Bool
    /// -1 chooses a system random seed once when a fresh run starts. Saved run
    /// settings contain the resolved nonnegative seed for reproducible replay.
    public var seed: Int
    public var modelArchitecture: String
    /// Immutable training objective; inference architecture and pixels stay unchanged.
    public var scalarLoss: NativeScalarLoss
    public var modelWidth: Int
    public var continueLatestWeights: Bool?
    public var modelBlocks: Int
    public var optimizer: TrainingOptimizer
    public var weightDecay: Double
    public var beta1: Double
    public var beta2: Double
    public var momentum: Double
    public var gradientAccumulationSteps: Int
    public var gradientClipNorm: Double
    public var gradientClipPolicy: TrainingGradientClipping
    public var gradientClipWarmupStartNorm: Double
    /// Completed optimizer updates, rather than visits or microbatches.
    public var gradientClipWarmupUpdates: Int
    public var learningRateSchedule: TrainingLearningRateSchedule
    public var warmupEpochs: Int
    public var minimumLearningRateRatio: Double
    public var cropRefresh: TrainingRefresh
    public var augmentationRefresh: TrainingRefresh
    public var device: TrainingDevice
    public var validationFraction: Double
    public var validationSeed: Int
    public var earlyStoppingPatience: Int
    public var exportBestWeights: Bool
    public var checkpointEveryEpochs: Int

    public init(imageSize: Int = 1024, sourceResolutionLevels: Int = 1, epochs: Int = 10, batchSize: Int = 1,
                learningRate: Double = 0.001, shuffleEachEpoch: Bool = true,
                flipCrops: Bool = false, rotateCrops: Bool = false,
                exaggerateFeatures: Bool = false, seed: Int = -1, modelWidth: Int = 164,
                continueLatestWeights: Bool = false, modelBlocks: Int = 4,
                optimizer: TrainingOptimizer = .adamW, weightDecay: Double = 0.01,
                beta1: Double = 0.9, beta2: Double? = nil, momentum: Double = 0.9,
                gradientAccumulationSteps: Int = 1, gradientClipNorm: Double = 1,
                gradientClipPolicy: TrainingGradientClipping = .warmup,
                gradientClipWarmupStartNorm: Double = 0.25, gradientClipWarmupUpdates: Int = 100,
                learningRateSchedule: TrainingLearningRateSchedule = .constant, warmupEpochs: Int = 0,
                minimumLearningRateRatio: Double = 0.1, cropRefresh: TrainingRefresh = .perEpoch,
                augmentationRefresh: TrainingRefresh = .perEpoch, device: TrainingDevice = .automatic,
                validationFraction: Double = 0.1, validationSeed: Int = 42, earlyStoppingPatience: Int = 0,
                exportBestWeights: Bool = true, checkpointEveryEpochs: Int = 1,
                modelArchitecture: String = NativeArchitecture.identifier,
                scalarLoss: NativeScalarLoss = .logitCrossEntropyV2) {
        self.imageSize = imageSize; self.sourceResolutionLevels = sourceResolutionLevels
        self.epochs = epochs; self.batchSize = batchSize
        self.learningRate = learningRate; self.shuffleEachEpoch = shuffleEachEpoch
        self.flipCrops = flipCrops; self.rotateCrops = rotateCrops
        self.exaggerateFeatures = exaggerateFeatures; self.seed = seed; self.modelWidth = modelWidth
        self.modelArchitecture = modelArchitecture
        self.scalarLoss = scalarLoss
        self.continueLatestWeights = continueLatestWeights; self.modelBlocks = modelBlocks
        self.optimizer = optimizer; self.weightDecay = weightDecay; self.beta1 = beta1
        self.beta2 = beta2 ?? (optimizer == .lion ? 0.99 : 0.999); self.momentum = momentum
        self.gradientAccumulationSteps = gradientAccumulationSteps; self.gradientClipNorm = gradientClipNorm
        self.gradientClipPolicy = gradientClipPolicy; self.gradientClipWarmupStartNorm = gradientClipWarmupStartNorm
        self.gradientClipWarmupUpdates = gradientClipWarmupUpdates
        self.learningRateSchedule = learningRateSchedule; self.warmupEpochs = warmupEpochs
        self.minimumLearningRateRatio = minimumLearningRateRatio
        self.cropRefresh = cropRefresh; self.augmentationRefresh = augmentationRefresh; self.device = device
        self.validationFraction = validationFraction; self.validationSeed = validationSeed
        self.earlyStoppingPatience = earlyStoppingPatience; self.exportBestWeights = exportBestWeights
        self.checkpointEveryEpochs = checkpointEveryEpochs
    }

    public var effectiveBatchSize: Int {
        guard (1...64).contains(batchSize), (1...1024).contains(gradientAccumulationSteps) else { return 0 }
        return batchSize * gradientAccumulationSteps
    }
    /// Resolve the sentinel only once; callers persist this copy in the run's
    /// metadata, request, checkpoints and exported model.
    mutating func resolveRandomSeedForRun() {
        var generator = SystemRandomNumberGenerator()
        resolveRandomSeedForRun(using: &generator)
    }

    mutating func resolveRandomSeedForRun<R: RandomNumberGenerator>(using generator: inout R) {
        guard seed == -1 else { return }
        seed = Int.random(in: 0...Int(Int32.max), using: &generator)
    }
    public func optimizerStepsPerEpoch(pairCount: Int) -> Int {
        guard pairCount > 0, effectiveBatchSize > 0 else { return 0 }
        return (pairCount - 1) / effectiveBatchSize + 1
    }
    public func batchesPerEpoch(pairCount: Int) -> Int {
        guard pairCount > 0, (1...64).contains(batchSize) else { return 0 }
        return (pairCount - 1) / batchSize + 1
    }
    public func parameterCount(map: MapRole) -> Int {
        guard (8...256).contains(modelWidth), (1...12).contains(modelBlocks), MapRole.targets.contains(map) else { return 0 }
        return (try? NativeArchitecture(mapKind: map.rawValue, width: modelWidth, blocks: modelBlocks,
            identifier: modelArchitecture).parameterCount) ?? 0
    }
    /// Switching a saved project's fresh run keeps its approximate capacity,
    /// while its saved checkpoints keep their exact recorded architecture.
    public mutating func useFullResolutionForFreshRun(map: MapRole) {
        if map != .normal { scalarLoss = .logitCrossEntropyV2 }
        guard modelArchitecture != NativeArchitecture.identifier else { return }
        let budget = parameterCount(map: map)
        modelArchitecture = NativeArchitecture.identifier
        guard budget > 0 else { modelWidth = 164; return }
        modelWidth = (8...256).min { left, right in
            let a = try? NativeArchitecture(mapKind: map.rawValue, width: left, blocks: modelBlocks,
                identifier: modelArchitecture).parameterCount
            let b = try? NativeArchitecture(mapKind: map.rawValue, width: right, blocks: modelBlocks,
                identifier: modelArchitecture).parameterCount
            return abs((a ?? 0) - budget) < abs((b ?? 0) - budget)
        } ?? 164
    }
    public var validationIssue: String? {
        if !(32...8192).contains(imageSize) { return "Training dimensions must be between 32 and 8,192 pixels." }
        if !(0...3).contains(sourceResolutionLevels) { return "Source resolution levels must be one to three, or all available." }
        if !(1...100_000).contains(epochs) { return "Epochs must be between 1 and 100,000." }
        if !(1...64).contains(batchSize) { return "Batch size must be between 1 and 64." }
        if !learningRate.isFinite || !(0 < learningRate && learningRate <= 1) { return "Learning rate must be greater than zero and at most one." }
        if !(8...256).contains(modelWidth) || !(1...12).contains(modelBlocks) { return "Model width must be 8–256 and residual blocks must be 1–12." }
        if TrainingNetwork(rawValue: modelArchitecture) == nil { return "This model uses an unsupported network architecture. Choose Pyramid or Original residual." }
        if !(1...1024).contains(gradientAccumulationSteps) { return "Gradient accumulation must be between 1 and 1,024 batches." }
        if gradientClipPolicy != .off {
            if !gradientClipNorm.isFinite || !(0...1000).contains(gradientClipNorm) { return "Gradient clipping must be between zero and 1,000. Zero disables clipping." }
            if gradientClipPolicy == .warmup && gradientClipNorm > 0 {
                if !gradientClipWarmupStartNorm.isFinite || gradientClipWarmupStartNorm <= 0 || gradientClipWarmupStartNorm > gradientClipNorm { return "Initial gradient norm must be positive and at most the final maximum norm." }
                if !(1...1_000_000).contains(gradientClipWarmupUpdates) { return "Clipping warmup must use one to 1,000,000 optimizer updates." }
            }
        }
        if !weightDecay.isFinite || !(0...10).contains(weightDecay) { return "Weight decay must be between zero and 10." }
        if !beta1.isFinite || !beta2.isFinite || !momentum.isFinite { return "Optimizer coefficients must be finite numbers." }
        if optimizer != .sgd && (!beta1.isFinite || !beta2.isFinite || !(0..<1).contains(beta1) || !(0..<1).contains(beta2)) { return "Optimizer betas must be between zero and one, excluding one." }
        if optimizer == .sgd && (!momentum.isFinite || !(0...1).contains(momentum)) { return "Momentum must be between zero and one." }
        if warmupEpochs < 0 || warmupEpochs >= epochs { return "Warmup must be shorter than the training run. Zero disables warmup." }
        if !minimumLearningRateRatio.isFinite || !(0...1).contains(minimumLearningRateRatio) { return "Minimum learning-rate ratio must be between zero and one." }
        if !validationFraction.isFinite || !(0...0.5).contains(validationFraction) { return "Validation must use between zero and 50% of the material folders." }
        if !(-1...Int(Int32.max)).contains(seed) { return "Random seed must be -1 for a new system seed, or zero to 2,147,483,647." }
        if !(0...Int(Int32.max)).contains(validationSeed) { return "Validation seed must be between zero and 2,147,483,647." }
        if !(0...100_000).contains(earlyStoppingPatience) { return "Early stopping patience must be zero to 100,000 epochs." }
        if earlyStoppingPatience > 0 && validationFraction == 0 { return "Enable held-out validation to use early stopping." }
        if !(1...100_000).contains(checkpointEveryEpochs) { return "Checkpoint frequency must be between one and 100,000 epochs." }
        return nil
    }

    enum CodingKeys: String, CodingKey {
        case imageSize, sourceResolutionLevels, epochs, batchSize, learningRate, shuffleEachEpoch, flipCrops, rotateCrops, exaggerateFeatures, seed, modelWidth, continueLatestWeights
        case modelArchitecture, scalarLoss, modelBlocks, optimizer, weightDecay, beta1, beta2, momentum, gradientAccumulationSteps, gradientClipNorm
        case gradientClipPolicy, gradientClipWarmupStartNorm, gradientClipWarmupUpdates
        case learningRateSchedule, warmupEpochs, minimumLearningRateRatio, cropRefresh, augmentationRefresh, device
        case validationFraction, validationSeed, earlyStoppingPatience, exportBestWeights, checkpointEveryEpochs
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = TrainingSettings()
        imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? d.imageSize
        // Earlier projects trained every eligible physical source resolution.
        // Preserve that choice while new projects start with the base grid.
        sourceResolutionLevels = try c.decodeIfPresent(Int.self, forKey: .sourceResolutionLevels) ?? 0
        epochs = try c.decodeIfPresent(Int.self, forKey: .epochs) ?? d.epochs
        batchSize = try c.decodeIfPresent(Int.self, forKey: .batchSize) ?? d.batchSize
        learningRate = try c.decodeIfPresent(Double.self, forKey: .learningRate) ?? d.learningRate
        shuffleEachEpoch = try c.decodeIfPresent(Bool.self, forKey: .shuffleEachEpoch) ?? d.shuffleEachEpoch
        flipCrops = try c.decodeIfPresent(Bool.self, forKey: .flipCrops) ?? d.flipCrops
        rotateCrops = try c.decodeIfPresent(Bool.self, forKey: .rotateCrops) ?? d.rotateCrops
        exaggerateFeatures = try c.decodeIfPresent(Bool.self, forKey: .exaggerateFeatures) ?? d.exaggerateFeatures
        // Missing seeds belonged to the original deterministic seed-zero
        // contract; only newly constructed settings default to system random.
        seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? 0
        // Absence identifies the original full-resolution architecture. Never
        // reinterpret an existing run's width as a pyramid channel count.
        modelArchitecture = try c.decodeIfPresent(String.self, forKey: .modelArchitecture) ?? NativeArchitecture.identifier
        // Missing IDs identify the original sigmoid/L1 training contract.
        scalarLoss = try c.decodeIfPresent(NativeScalarLoss.self, forKey: .scalarLoss) ?? .absoluteV1
        modelWidth = try c.decodeIfPresent(Int.self, forKey: .modelWidth)
            ?? (modelArchitecture == NativeArchitecture.identifier ? 84 : 38)
        continueLatestWeights = try c.decodeIfPresent(Bool.self, forKey: .continueLatestWeights) ?? false
        modelBlocks = try c.decodeIfPresent(Int.self, forKey: .modelBlocks) ?? d.modelBlocks
        optimizer = try c.decodeIfPresent(TrainingOptimizer.self, forKey: .optimizer) ?? d.optimizer
        weightDecay = try c.decodeIfPresent(Double.self, forKey: .weightDecay) ?? d.weightDecay
        beta1 = try c.decodeIfPresent(Double.self, forKey: .beta1) ?? d.beta1
        beta2 = try c.decodeIfPresent(Double.self, forKey: .beta2) ?? (optimizer == .lion ? 0.99 : d.beta2)
        momentum = try c.decodeIfPresent(Double.self, forKey: .momentum) ?? d.momentum
        gradientAccumulationSteps = try c.decodeIfPresent(Int.self, forKey: .gradientAccumulationSteps) ?? d.gradientAccumulationSteps
        gradientClipNorm = try c.decodeIfPresent(Double.self, forKey: .gradientClipNorm) ?? d.gradientClipNorm
        // Existing saved runs keep their fixed threshold, including zero/off.
        gradientClipPolicy = try c.decodeIfPresent(TrainingGradientClipping.self, forKey: .gradientClipPolicy) ?? .fixed
        gradientClipWarmupStartNorm = try c.decodeIfPresent(Double.self, forKey: .gradientClipWarmupStartNorm) ?? d.gradientClipWarmupStartNorm
        gradientClipWarmupUpdates = try c.decodeIfPresent(Int.self, forKey: .gradientClipWarmupUpdates) ?? d.gradientClipWarmupUpdates
        learningRateSchedule = try c.decodeIfPresent(TrainingLearningRateSchedule.self, forKey: .learningRateSchedule) ?? d.learningRateSchedule
        warmupEpochs = try c.decodeIfPresent(Int.self, forKey: .warmupEpochs) ?? d.warmupEpochs
        minimumLearningRateRatio = try c.decodeIfPresent(Double.self, forKey: .minimumLearningRateRatio) ?? d.minimumLearningRateRatio
        cropRefresh = try c.decodeIfPresent(TrainingRefresh.self, forKey: .cropRefresh) ?? d.cropRefresh
        augmentationRefresh = try c.decodeIfPresent(TrainingRefresh.self, forKey: .augmentationRefresh) ?? d.augmentationRefresh
        device = try c.decodeIfPresent(TrainingDevice.self, forKey: .device) ?? d.device
        validationFraction = try c.decodeIfPresent(Double.self, forKey: .validationFraction) ?? d.validationFraction
        validationSeed = try c.decodeIfPresent(Int.self, forKey: .validationSeed) ?? d.validationSeed
        earlyStoppingPatience = try c.decodeIfPresent(Int.self, forKey: .earlyStoppingPatience) ?? d.earlyStoppingPatience
        exportBestWeights = try c.decodeIfPresent(Bool.self, forKey: .exportBestWeights) ?? d.exportBestWeights
        checkpointEveryEpochs = try c.decodeIfPresent(Int.self, forKey: .checkpointEveryEpochs) ?? d.checkpointEveryEpochs
    }
}
