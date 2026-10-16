import Foundation

/// Stable products describe pixel meaning independently from image map tags.
public enum NativeModelProduct: String, Codable, CaseIterable, Sendable {
    case depth, roughness, normal
    case photoRepair = "photo-repair-v1"
    case photoLighting = "photo-lighting-v1"
    public var isPhoto: Bool { self == .photoRepair || self == .photoLighting }
    public var channels: Int { isPhoto || self == .normal ? 3 : 1 }
    public var outputName: String { isPhoto ? "enhanced_rgb" : "material_map" }
    public var title: String {
        switch self {
        case .depth: "Depth"
        case .normal: "Normal"
        case .roughness: "Roughness"
        case .photoRepair: "Photo repair (noise / mild blur)"
        case .photoLighting: "Neutral lighting / exposure"
        }
    }
    public var constructors: [String] {
        NativeArchitectureConstructor.all.filter { $0.photo == isPhoto }.map(\.id)
    }
}

/// Immutable constructors, not an editable or generic graph description.
public struct NativeArchitectureConstructor: Sendable {
    public let id: String
    public let photo: Bool
    public static let all: [Self] = [
        .init(id: NativeArchitecture.identifier, photo: false),
        .init(id: NativeArchitecture.residualV2Identifier, photo: false),
        .init(id: NativeArchitecture.pyramidIdentifier, photo: false),
        .init(id: NativeArchitecture.pyramidV2Identifier, photo: false),
        .init(id: NativeArchitecture.compactIdentifier, photo: false),
        .init(id: NativeArchitecture.photoCompactIdentifier, photo: true)
    ]
}
