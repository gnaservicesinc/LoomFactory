import Foundation

struct TrainingPair: Sendable {
    let id: String
    let input: SourceImage
    let target: SourceImage
}

enum TrainingPairSelector {
    static func select(images: [SourceImage], dataset: String, map: MapRole, size: Int) throws -> [TrainingPair] {
        let members = images.filter { $0.category == dataset && $0.verified && !$0.ignored }
        let families = Dictionary(grouping: members, by: \.pairingKey)
        var pairs: [TrainingPair] = []
        for key in families.keys.sorted() {
            let group = families[key]!
            let inputs = group.filter { $0.role == .diffuse }
            let targets = group.filter { $0.role == map && (map != .normal || $0.normalConvention?.lowercased() != "directx") }
            guard !inputs.isEmpty, !targets.isEmpty else { continue }
            guard inputs.count == 1, targets.count == 1 else {
                throw LoomError("\(group[0].displayName) contains several maps of the same type. Pair the intended maps before training.")
            }
            let input = inputs[0], target = targets[0]
            guard input.width == target.width, input.height == target.height else {
                throw LoomError("The diffuse and \(map.displayName.lowercased()) maps for \(input.displayName) have different dimensions. Originals were left unchanged.")
            }
            guard input.width >= size, input.height >= size else { continue }
            pairs.append(TrainingPair(id: key, input: input, target: target))
        }
        return pairs
    }
}
