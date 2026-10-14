import Foundation

extension LoomProject {
    /// Review applies to physical material folders, including maps hidden by a
    /// search or file filter. It is informational and never changes membership.
    public func setMaterialReviewed(imageIDs: [String], verified: Bool) throws {
        let requested = Set(imageIDs)
        guard !requested.isEmpty else { throw LoomError("Choose a material to review.") }
        let images = try scanImages()
        let selected = images.filter { requested.contains($0.id) }
        guard selected.count == requested.count else {
            throw LoomError("A selected image was removed. Refresh the material and try again.")
        }
        let folders = Set(selected.map { $0.url.deletingLastPathComponent().standardizedFileURL })
        let members = images.filter { folders.contains($0.url.deletingLastPathComponent().standardizedFileURL) }
        let originals: [(URL, ImageMetadata)] = try members.map { image in
            guard let metadata = try ImageMetadataStore.read(image.url), metadata.id == image.id else {
                throw LoomError("Image metadata changed. Refresh the material and try again.")
            }
            return (image.url, metadata)
        }
        var written: [(URL, ImageMetadata)] = []
        do {
            for (url, original) in originals where original.verified != verified {
                var updated = original
                updated.verified = verified
                try ImageMetadataStore.write(updated, to: url)
                written.append((url, original))
            }
        } catch {
            for (url, original) in written { try? ImageMetadataStore.write(original, to: url) }
            throw error
        }
    }
}
