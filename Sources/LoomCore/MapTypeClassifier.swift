import CoreML
import CryptoKit
import Foundation
import ImageIO

/// A model prediction is a correctable guess, never a human map-type decision.
public struct MapTypeInference: Codable, Sendable, Equatable {
    public let role: MapRole
    public let confidence: Double
    public let accepted: Bool
    public let modelID: String
    public let sourceSignature: String
}

protocol MapTypePredictor: Sendable {
    var modelID: String { get }
    func probabilities(for file: URL) throws -> [String: Double]
}

enum MapTypeInferencePolicy {
    static let labels: [String: MapRole] = ["depth": .depth, "diffuse": .diffuse, "normal": .normal, "rough": .roughness]

    static func evaluate(_ probabilities: [String: Double], modelID: String, sourceSignature: String) -> MapTypeInference? {
        guard Set(probabilities.keys) == Set(labels.keys),
              probabilities.values.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
              abs(probabilities.values.reduce(0, +) - 1) < 0.01 else { return nil }
        let ranked = probabilities.sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }
        guard let first = ranked.first, let role = labels[first.key] else { return nil }
        // The supplied classifier particularly confuses scalar depth and roughness.
        // Confidence cannot establish that an image belongs to any of these classes.
        let scalar = role == .depth || role == .roughness
        let accepted = first.value >= (scalar ? 0.95 : 0.90)
            && first.value - ranked[1].value >= (scalar ? 0.50 : 0.30)
        return MapTypeInference(role: role, confidence: first.value, accepted: accepted,
                                modelID: modelID, sourceSignature: sourceSignature)
    }

    static func sourceSignature(file: URL, width: Int, height: Int) throws -> String {
        let attributes = try FileManager.default.attributesOfItem(atPath: file.resolvingSymlinksInPath().path)
        guard let size = attributes[.size] as? NSNumber, let modified = attributes[.modificationDate] as? Date else {
            throw LoomError("Could not inspect \(file.lastPathComponent) for map-type classification.")
        }
        let stamp = "\(file.lastPathComponent)|\(size)|\(modified.timeIntervalSince1970)|\(width)x\(height)"
        return SHA256.hash(data: Data(stamp.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

/// The source-folder actor calls this synchronously off the UI actor. The lock
/// also protects the model when a test or another caller uses the predictor.
final class NativeMapTypeClassifier: MapTypePredictor, @unchecked Sendable {
    let modelID: String
    private let model: MLModel
    private let inputName: String
    private let probabilitiesName: String
    private let width: Int
    private let height: Int
    private let lock = NSLock()

    static func bundled() -> NativeMapTypeClassifier? {
        guard let url = Bundle.main.url(forResource: "MatTypeClassifier", withExtension: "mlmodelc") else { return nil }
        return try? NativeMapTypeClassifier(modelURL: url)
    }

    init(modelURL: URL) throws {
        let compiled = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
        defer { if compiled != modelURL { try? FileManager.default.removeItem(at: compiled) } }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        model = try MLModel(contentsOf: compiled, configuration: configuration)
        let description = model.modelDescription
        guard Set(description.classLabels?.compactMap { $0 as? String } ?? []) == Set(MapTypeInferencePolicy.labels.keys),
              description.inputDescriptionsByName.count == 1,
              let input = description.inputDescriptionsByName.first,
              let constraint = input.value.imageConstraint,
              constraint.pixelsWide == 360, constraint.pixelsHigh == 360,
              let output = description.predictedProbabilitiesName,
              description.outputDescriptionsByName[output]?.type == .dictionary else {
            throw LoomError("The map-type classifier does not match the four-class RGB image contract.")
        }
        inputName = input.key; probabilitiesName = output
        width = constraint.pixelsWide; height = constraint.pixelsHigh
        modelID = "map-type-policy-v1:" + (try Self.identity(modelURL))
    }

    func probabilities(for file: URL) throws -> [String: Double] {
        lock.lock(); defer { lock.unlock() }
        return try autoreleasepool {
            guard let source = CGImageSourceCreateWithURL(file as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
                  let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceThumbnailMaxPixelSize: max(width, height),
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceShouldCacheImmediately: true
                  ] as CFDictionary) else {
                throw LoomError("Could not read \(file.lastPathComponent) for a map-type suggestion.")
            }
            // Only a disposable classifier view is converted. The original
            // image and the native training pixels are never rewritten.
            let value = try MLFeatureValue(cgImage: image, constraint: model.modelDescription.inputDescriptionsByName[inputName]!.imageConstraint!)
            let prediction = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: value]))
            guard let probabilities = prediction.featureValue(for: probabilitiesName), probabilities.type == .dictionary else {
                throw LoomError("The map-type classifier returned no probabilities.")
            }
            return Dictionary(uniqueKeysWithValues: probabilities.dictionaryValue.compactMap { key, value in
                (key as? String).map { ($0, value.doubleValue) }
            })
        }
    }

    private static func identity(_ url: URL) throws -> String {
        var digest = SHA256()
        if url.pathExtension != "mlmodelc" { digest.update(data: try Data(contentsOf: url)) }
        else {
            guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey]) else {
                throw LoomError("The map-type classifier resource could not be read.")
            }
            let files = try enumerator.compactMap { $0 as? URL }.filter { try $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true }
            for file in files.sorted(by: { $0.path < $1.path }) {
                digest.update(data: Data(file.path.dropFirst(url.path.count).utf8))
                digest.update(data: try Data(contentsOf: file))
            }
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
