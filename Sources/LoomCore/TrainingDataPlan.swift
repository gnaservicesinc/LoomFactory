import CryptoKit
import Foundation

public struct TrainingDataPlan: Sendable {
    public let trainingPairCount: Int
    public let validationPairCount: Int
    public let trainingMaterialCount: Int
    public let validationMaterialCount: Int
    public var totalPairCount: Int { trainingPairCount + validationPairCount }
    let training: [TrainingPair]
    let validation: [TrainingPair]

    init(pairs: [TrainingPair], settings: TrainingSettings, sourceFolder: URL) {
        let groups = Dictionary(grouping: pairs) { $0.input.url.deletingLastPathComponent().standardizedFileURL }
        let root = sourceFolder.standardizedFileURL.path
        let ordered = groups.keys.sorted { left, right in
            func key(_ folder: URL) -> String {
                let relative = folder.path.hasPrefix(root + "/") ? String(folder.path.dropFirst(root.count + 1)) : folder.lastPathComponent
                return SHA256.hash(data: Data("\(settings.validationSeed)|\(relative)".utf8)).map { String(format: "%02x", $0) }.joined()
            }
            let a = key(left), b = key(right)
            return a == b ? left.path < right.path : a < b
        }
        let heldOutCount = settings.validationFraction > 0 && ordered.count > 1
            ? min(ordered.count - 1, max(1, Int((Double(ordered.count) * settings.validationFraction).rounded()))) : 0
        let heldOut = Set(ordered.prefix(heldOutCount))
        validation = pairs.filter { heldOut.contains($0.input.url.deletingLastPathComponent().standardizedFileURL) }
        training = pairs.filter { !heldOut.contains($0.input.url.deletingLastPathComponent().standardizedFileURL) }
        trainingPairCount = training.count; validationPairCount = validation.count
        validationMaterialCount = heldOut.count; trainingMaterialCount = groups.count - heldOut.count
    }
}

extension ProjectSnapshot {
    public func trainingPlan(datasetID: String, map: MapRole, settings: TrainingSettings) throws -> TrainingDataPlan? {
        guard let dataset = datasets.first(where: { $0.id == datasetID }) else { return nil }
        if let issue = settings.validationIssue { throw LoomError(issue) }
        return TrainingDataPlan(pairs: try TrainingPairSelector.select(images: images, dataset: dataset.name, map: map, size: settings.imageSize),
                                settings: settings, sourceFolder: sourceURL)
    }
}
