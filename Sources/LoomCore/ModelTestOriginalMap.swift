import Foundation

/// Resolves a comparison target from current image metadata, independently of
/// training eligibility, crop dimensions, approval, or review flags.
public struct ModelTestOriginalMap: Sendable, Equatable {
    public let image: SourceImage?
    public let issue: String?

    public init(image: SourceImage? = nil, issue: String? = nil) {
        self.image = image; self.issue = issue
    }

    public static func resolve(inputURL: URL?, map: MapRole?, images: [SourceImage]) -> Self {
        guard let inputURL, let map, MapRole.targets.contains(map) else { return .init(image: nil, issue: nil) }
        let path = inputURL.standardizedFileURL
        let input = images.first { $0.url.standardizedFileURL == path } ?? images.first {
            $0.url.standardizedFileURL.resolvingSymlinksInPath() == path.resolvingSymlinksInPath()
        }
        guard let input, input.role == .diffuse else { return .init(image: nil, issue: nil) }
        let candidates = images.filter {
            $0.pairingKey == input.pairingKey && $0.role == map &&
                $0.width == input.width && $0.height == input.height
        }
        guard candidates.count == 1 else {
            return .init(image: nil, issue: candidates.isEmpty
                ? "No original \(map.displayName.lowercased()) map is paired with this source image. You can still test it."
                : "Several original \(map.displayName.lowercased()) maps match this source image. Pair the intended map to compare it.")
        }
        return .init(image: candidates[0], issue: nil)
    }
}
