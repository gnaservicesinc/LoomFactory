import Foundation
import LoomNative

struct FactoryImageBatch: Sendable {
    let pairs: [TrainingPair]
    let crops: [NativeCropTransform]
    let offset: Int
    var imageNames: String { pairs.map { $0.input.trainingDisplayName }.joined(separator: ", ") }
}

extension SourceImage {
    var trainingDisplayName: String { url.deletingLastPathComponent().lastPathComponent + " / " + url.lastPathComponent }
}

struct FactoryPreparedBatch: Sendable {
    let input: [Float]
    let target: [Float]
    let preparationSeconds: Double

    static func prepare(_ batch: FactoryImageBatch, map: MapRole, settings: TrainingSettings,
                        cancellation: NativeCancellation, imageCache: NativeImageCache? = nil) throws -> FactoryPreparedBatch {
        let start = ProcessInfo.processInfo.systemUptime
        var input: [Float] = [], target: [Float] = []
        input.reserveCapacity(batch.pairs.count * 3 * settings.imageSize * settings.imageSize)
        target.reserveCapacity(batch.pairs.count * (map == .normal ? 3 : 1) * settings.imageSize * settings.imageSize)
        for (pair, crop) in zip(batch.pairs, batch.crops) {
            try cancellation.check()
            let pixels = try NativeImages.pair(inputURL: pair.input.url, targetURL: pair.target.url, transform: crop,
                mapKind: map.rawValue, normalConvention: pair.target.normalConvention ?? "OpenGL",
                exaggerate: settings.exaggerateFeatures, cancellation: cancellation, imageCache: imageCache)
            input.append(contentsOf: pixels.input.values); target.append(contentsOf: pixels.target.values)
        }
        return FactoryPreparedBatch(input: input, target: target,
                                    preparationSeconds: ProcessInfo.processInfo.systemUptime - start)
    }
}
