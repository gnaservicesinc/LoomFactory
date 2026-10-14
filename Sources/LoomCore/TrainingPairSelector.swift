import Foundation

struct TrainingPair: Sendable {
    let id: String
    let input: SourceImage
    let target: SourceImage
}

enum TrainingPairSelector {
    static func select(images: [SourceImage], dataset: String, map: MapRole, settings: TrainingSettings) throws -> [TrainingPair] {
        if let issue = settings.validationIssue { throw LoomError(issue) }
        let pairs = try select(images: images, dataset: dataset, map: map, size: settings.imageSize)
        return selectSourceResolutions(pairs: pairs, levels: settings.sourceResolutionLevels)
    }

    static func selectSourceResolutions(pairs: [TrainingPair], levels: Int) -> [TrainingPair] {
        guard levels > 0 else { return pairs }
        let groups = Dictionary(grouping: pairs) { $0.input.url.deletingLastPathComponent().standardizedFileURL }
        let factor = 1 << (min(levels, 3) - 1)
        let maximumDimensions = groups.mapValues { group in
            let base = group.map { min($0.input.width, $0.input.height) }.min() ?? 0
            let (maximum, overflow) = base.multipliedReportingOverflow(by: factor)
            return overflow ? Int.max : maximum
        }
        // This only chooses originals already present in the current scan.
        // The trainer takes exact crops from those originals at imageSize.
        return pairs.filter { pair in
            let folder = pair.input.url.deletingLastPathComponent().standardizedFileURL
            return min(pair.input.width, pair.input.height) <= (maximumDimensions[folder] ?? 0)
        }
    }

    static func select(images: [SourceImage], dataset: String, map: MapRole, size: Int) throws -> [TrainingPair] {
        let members = images.filter { $0.category == dataset && !$0.ignored }
        let families = Dictionary(grouping: members, by: \.pairingKey)
        var pairs: [TrainingPair] = []
        for key in families.keys.sorted() {
            let group = families[key]!
            let inputs = group.filter { $0.role == .diffuse }.sorted { $0.id < $1.id }
            let targets = group.filter { $0.role == map && (map != .normal || $0.normalConvention?.lowercased() != "directx") }
            guard !inputs.isEmpty, !targets.isEmpty else { continue }
            if inputs.count > 1 {
                let folders = Set((inputs + targets).map { $0.url.deletingLastPathComponent().standardizedFileURL })
                guard folders.count == 1 else {
                    throw LoomError("Keep diffuse variants and their shared maps in one material folder so training and validation stay separate.")
                }
            }
            for input in inputs {
                let matching = targets.filter { $0.width == input.width && $0.height == input.height }
                guard !matching.isEmpty else {
                    throw LoomError("The diffuse and \(map.displayName.lowercased()) maps for \(input.displayName) have different dimensions. Originals were left unchanged.")
                }
                guard matching.count == 1 else {
                    throw LoomError("\(input.displayName) contains several \(map.displayName.lowercased()) maps at the same resolution. Pair the intended maps before training.")
                }
                guard input.width >= size, input.height >= size else { continue }
                // Color variants are independent inputs to the shared target.
                // Preserve existing singleton IDs and distinguish variant crops.
                let id = inputs.count == 1 ? key : key + "|" + input.id
                pairs.append(TrainingPair(id: id, input: input, target: matching[0]))
            }
        }
        return pairs
    }
}
