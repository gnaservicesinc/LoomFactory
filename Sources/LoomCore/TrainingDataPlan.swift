import CryptoKit
import Foundation

public struct TrainingResolutionPlan: Identifiable, Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let sourceImageCount: Int
    public let sourceMaterialCount: Int
    public let eligiblePairCount: Int
    public let trainingPairCount: Int
    public let validationPairCount: Int
    public var id: String { "\(width)x\(height)" }
    public var displayName: String { "\(width.formatted()) × \(height.formatted())" }
    public var totalPairCount: Int { trainingPairCount + validationPairCount }
    public var excludedPairCount: Int { eligiblePairCount - totalPairCount }
}

public struct TrainingDataPlan: Sendable {
    /// All current files tagged into this category, including ignored and
    /// unpaired files. These counts come from the current source scan.
    public let sourceImageCount: Int
    public let sourceMaterialCount: Int
    public let ignoredSourceImageCount: Int
    public let eligiblePairCount: Int
    public let eligibleMaterialCount: Int
    public let trainingPairCount: Int
    public let validationPairCount: Int
    public let trainingMaterialCount: Int
    public let validationMaterialCount: Int
    public var totalPairCount: Int { trainingPairCount + validationPairCount }
    public var excludedPairCount: Int { eligiblePairCount - totalPairCount }
    public var unavailableMaterialCount: Int { sourceMaterialCount - trainingMaterialCount - validationMaterialCount }
    public let batchesPerEpoch: Int
    public let totalBatches: Int
    public let trainingSampleVisits: Int
    public let validationSampleVisits: Int
    public let optimizerStepsPerEpoch: Int
    public let totalOptimizerUpdates: Int
    public let resolutions: [TrainingResolutionPlan]
    let training: [TrainingPair]
    let validation: [TrainingPair]

    init(pairs: [TrainingPair], settings: TrainingSettings, sourceFolder: URL,
         sourceImages: [SourceImage]? = nil, eligiblePairs: [TrainingPair]? = nil) {
        let eligible = eligiblePairs ?? pairs
        let sources = sourceImages ?? eligible.flatMap { [$0.input, $0.target] }
        let uniqueSources = Dictionary(grouping: sources) { $0.url.standardizedFileURL }.values.compactMap(\.first)
        sourceImageCount = uniqueSources.count
        sourceMaterialCount = Set(uniqueSources.map { $0.url.deletingLastPathComponent().standardizedFileURL }).count
        ignoredSourceImageCount = uniqueSources.filter(\.ignored).count
        eligiblePairCount = eligible.count
        eligibleMaterialCount = Set(eligible.map { $0.input.url.deletingLastPathComponent().standardizedFileURL }).count
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
        batchesPerEpoch = settings.batchesPerEpoch(pairCount: training.count)
        optimizerStepsPerEpoch = settings.optimizerStepsPerEpoch(pairCount: training.count)
        totalBatches = plannedVisits(batchesPerEpoch, epochs: settings.epochs)
        totalOptimizerUpdates = plannedVisits(optimizerStepsPerEpoch, epochs: settings.epochs)
        trainingSampleVisits = plannedVisits(training.count, epochs: settings.epochs)
        validationSampleVisits = plannedVisits(validation.count, epochs: settings.epochs)

        func resolutionKey(_ image: SourceImage) -> String { "\(image.width)x\(image.height)" }
        let sourceGrids = Dictionary(grouping: uniqueSources, by: resolutionKey)
        let eligibleGrids = Dictionary(grouping: eligible) { resolutionKey($0.input) }
        let trainingGrids = Dictionary(grouping: training) { resolutionKey($0.input) }
        let validationGrids = Dictionary(grouping: validation) { resolutionKey($0.input) }
        resolutions = sourceGrids.map { key, images in
            let first = images[0]
            return TrainingResolutionPlan(width: first.width, height: first.height,
                sourceImageCount: images.count,
                sourceMaterialCount: Set(images.map { $0.url.deletingLastPathComponent().standardizedFileURL }).count,
                eligiblePairCount: eligibleGrids[key]?.count ?? 0,
                trainingPairCount: trainingGrids[key]?.count ?? 0,
                validationPairCount: validationGrids[key]?.count ?? 0)
        }.sorted {
            let left = min($0.width, $0.height), right = min($1.width, $1.height)
            if left != right { return left < right }
            return $0.width == $1.width ? $0.height < $1.height : $0.width < $1.width
        }
    }
}

private func plannedVisits(_ count: Int, epochs: Int) -> Int {
    guard count > 0, epochs > 0 else { return 0 }
    let (total, overflow) = count.multipliedReportingOverflow(by: epochs)
    return overflow ? Int.max : total
}

extension ProjectSnapshot {
    public func trainingPlan(datasetID: String, map: MapRole, settings: TrainingSettings) throws -> TrainingDataPlan? {
        guard let dataset = datasets.first(where: { $0.id == datasetID }) else { return nil }
        if let issue = settings.validationIssue { throw LoomError(issue) }
        let eligible = try TrainingPairSelector.select(images: images, dataset: dataset.name, map: map, size: settings.imageSize)
        let selected = TrainingPairSelector.selectSourceResolutions(pairs: eligible, levels: settings.sourceResolutionLevels)
        return TrainingDataPlan(pairs: selected, settings: settings, sourceFolder: sourceURL,
                                sourceImages: images.filter { $0.category == dataset.name }, eligiblePairs: eligible)
    }
}
