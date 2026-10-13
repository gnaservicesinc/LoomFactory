import Foundation

public enum MapRole: String, Codable, CaseIterable, Sendable, Identifiable {
    case diffuse, depth, roughness, normal, unknown
    public var id: String { rawValue }
    public var displayName: String {
        switch self {
        case .diffuse: "Diffuse"
        case .depth: "Depth"
        case .roughness: "Roughness"
        case .normal: "Normal"
        case .unknown: "Choose map type"
        }
    }
    public static var targets: [MapRole] { [.depth, .roughness, .normal] }
}

public struct TrainingSettings: Codable, Sendable, Equatable {
    public var imageSize: Int
    public var epochs: Int
    public var batchSize: Int
    public var learningRate: Double
    public var shuffleEachEpoch: Bool
    public var flipCrops: Bool
    public var rotateCrops: Bool
    public var exaggerateFeatures: Bool
    public var seed: Int
    public var modelWidth: Int
    public var continueLatestWeights: Bool?
    public init(imageSize: Int = 2048, epochs: Int = 10, batchSize: Int = 1,
                learningRate: Double = 0.001, shuffleEachEpoch: Bool = true,
                flipCrops: Bool = false, rotateCrops: Bool = false,
                exaggerateFeatures: Bool = false, seed: Int = 0, modelWidth: Int = 24, continueLatestWeights: Bool = false) {
        self.imageSize = imageSize; self.epochs = epochs; self.batchSize = batchSize
        self.learningRate = learningRate; self.shuffleEachEpoch = shuffleEachEpoch
        self.flipCrops = flipCrops; self.rotateCrops = rotateCrops
        self.exaggerateFeatures = exaggerateFeatures; self.seed = seed; self.modelWidth = modelWidth
        self.continueLatestWeights = continueLatestWeights
    }
}

public struct ProjectSettings: Codable, Sendable, Equatable {
    public var modelName: String
    public var modelVersion: String
    public var huggingFaceRepository: String
    public var training: TrainingSettings
    public init(modelName: String = "Texture Loom", modelVersion: String = "1.0.0",
                huggingFaceRepository: String = "", training: TrainingSettings = .init()) {
        self.modelName = modelName; self.modelVersion = modelVersion
        self.huggingFaceRepository = huggingFaceRepository; self.training = training
    }
}

public struct SourceImage: Codable, Identifiable, Sendable, Equatable {
    public let id: String
    public let url: URL
    public let displayName: String
    public var role: MapRole
    public var category: String?
    public var verified: Bool
    public var note: String
    public let pairingKey: String
    public let width: Int
    public let height: Int
    public let normalConvention: String?
    public var dimensions: String { "\(width) × \(height)" }
    public init(id: String, url: URL, displayName: String, role: MapRole, category: String?,
                verified: Bool, note: String, pairingKey: String, width: Int, height: Int, normalConvention: String? = nil) {
        self.id = id; self.url = url; self.displayName = displayName; self.role = role
        self.category = category; self.verified = verified; self.note = note
        self.pairingKey = pairingKey; self.width = width; self.height = height
        self.normalConvention = normalConvention
    }
}

public struct DatasetCategory: Codable, Identifiable, Sendable, Equatable {
    public let id: String
    public let name: String
    public let imageCount: Int
    public let verifiedCount: Int
    public let pairCounts: [String: Int]
    public init(id: String, name: String, imageCount: Int, verifiedCount: Int, pairCounts: [String: Int]) {
        self.id = id; self.name = name; self.imageCount = imageCount
        self.verifiedCount = verifiedCount; self.pairCounts = pairCounts
    }
}

public struct ModelArtifact: Codable, Identifiable, Sendable, Equatable {
    public let id: String
    public let kind: String
    public let datasetID: String?
    public let datasetName: String?
    public let map: MapRole?
    public let directory: URL
    public let createdAt: Date
    public let epochs: Int
    public let loss: Double?
    public let coreMLPath: String?
    public let weightsPath: String?
    public let architecture: String
    public init(id: String, kind: String, datasetID: String?, datasetName: String?, map: MapRole?,
                directory: URL, createdAt: Date = .now, epochs: Int, loss: Double?,
                coreMLPath: String?, weightsPath: String?, architecture: String) {
        self.id = id; self.kind = kind; self.datasetID = datasetID; self.datasetName = datasetName
        self.map = map; self.directory = directory; self.createdAt = createdAt
        self.epochs = epochs; self.loss = loss; self.coreMLPath = coreMLPath
        self.weightsPath = weightsPath; self.architecture = architecture
    }
}

public struct ProjectSnapshot: Sendable {
    public let sourceURL: URL
    public let projectID: String
    public let settings: ProjectSettings
    public let images: [SourceImage]
    public let datasets: [DatasetCategory]
    public let artifacts: [ModelArtifact]
    public init(sourceURL: URL, projectID: String, settings: ProjectSettings,
                images: [SourceImage], datasets: [DatasetCategory], artifacts: [ModelArtifact]) {
        self.sourceURL = sourceURL; self.projectID = projectID; self.settings = settings
        self.images = images; self.datasets = datasets; self.artifacts = artifacts
    }
    public func eligiblePairCount(datasetID: String, map: MapRole, imageSize: Int) -> Int {
        guard let dataset = datasets.first(where: { $0.id == datasetID }) else { return 0 }
        return (try? TrainingPairSelector.select(images: images, dataset: dataset.name, map: map, size: imageSize).count) ?? 0
    }
}

public struct TrainingEvent: Sendable {
    public let type: String
    public let message: String
    public let progress: Double?
    public let epoch: Int?
    public let loss: Double?
    public init(type: String, message: String, progress: Double? = nil, epoch: Int? = nil, loss: Double? = nil) {
        self.type = type; self.message = message; self.progress = progress; self.epoch = epoch; self.loss = loss
    }
}

public struct LoomError: Error, LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}
