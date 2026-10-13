import Foundation

/// A temporary view of the current folder scan, never a dataset inventory.
public struct SourceMaterialFolder: Identifiable, Sendable, Equatable {
    public let id: String
    public let url: URL
    public let displayName: String
    public let resolutions: [SourceMaterialResolution]
    public var imageCount: Int { resolutions.reduce(0) { $0 + $1.images.count } }

    public static func group(images: [SourceImage], sourceFolder: URL) -> [SourceMaterialFolder] {
        let root = sourceFolder.standardizedFileURL
        let folders = Dictionary(grouping: images) { $0.url.deletingLastPathComponent().standardizedFileURL }
        return folders.map { folder, members in
            let resolutionGroups = Dictionary(grouping: members) { "\($0.width)x\($0.height)" }
            let resolutions = resolutionGroups.values.map { group in
                let first = group[0]
                return SourceMaterialResolution(id: folder.path + "|\(first.width)x\(first.height)",
                    width: first.width, height: first.height, images: group.sorted(by: imageOrder))
            }.sorted {
                let left = Int64($0.width) * Int64($0.height), right = Int64($1.width) * Int64($1.height)
                return left == right ? $0.id < $1.id : left < right
            }
            let relative = folder.path.hasPrefix(root.path + "/") ? String(folder.path.dropFirst(root.path.count + 1)) : folder.lastPathComponent
            let name = folder == root ? "Source images" : relative.split(separator: "/")
                .map { ImageMetadataStore.friendlyName(String($0)) }.joined(separator: " / ")
            return SourceMaterialFolder(id: folder.path, url: folder, displayName: name, resolutions: resolutions)
        }.sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }
    }
}

public struct SourceMaterialResolution: Identifiable, Sendable, Equatable {
    public let id: String
    public let width: Int
    public let height: Int
    public let images: [SourceImage]
    public var displayName: String { "\(width.formatted()) × \(height.formatted())" }
}

private func imageOrder(_ left: SourceImage, _ right: SourceImage) -> Bool {
    let roles: [MapRole] = [.diffuse, .depth, .bump, .roughness, .normal, .unknown]
    let a = roles.firstIndex(of: left.role) ?? roles.count
    let b = roles.firstIndex(of: right.role) ?? roles.count
    if a != b { return a < b }
    let order = left.displayName.localizedStandardCompare(right.displayName)
    return order == .orderedSame ? left.url.path < right.url.path : order == .orderedAscending
}
