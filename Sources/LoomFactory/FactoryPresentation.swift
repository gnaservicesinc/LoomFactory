import Darwin
import Foundation
import LoomCore

/// In-memory presentation of one physical scan. Never persisted as membership.
struct FactoryBrowserIndex: Sendable {
    let folders: [SourceMaterialFolder]
    let materialBrowserIndex: MaterialBrowserIndex
    let imagesByID: [String: SourceImage]
    private let entries: [(image: SourceImage, search: String)]

    init(snapshot: ProjectSnapshot) {
        folders = SourceMaterialFolder.group(images: snapshot.images, sourceFolder: snapshot.sourceURL)
        materialBrowserIndex = MaterialBrowserIndex(folders: folders)
        imagesByID = Dictionary(uniqueKeysWithValues: snapshot.images.map { ($0.id, $0) })
        entries = snapshot.images.map { image in
            (image, [image.displayName, image.url.lastPathComponent,
                ImageMetadataStore.friendlyName(image.url.deletingLastPathComponent().lastPathComponent),
                image.role.displayName, image.category ?? ""].joined(separator: "\n"))
        }
    }

    func matching(datasetID: String, datasetName: String?, search: String) -> [SourceImage] {
        entries.compactMap { entry in
            let image = entry.image
            let matchesDataset = datasetID.isEmpty ||
                (datasetID == "unassigned" ? image.category == nil : image.category == datasetName)
            return matchesDataset && (search.isEmpty || entry.search.localizedCaseInsensitiveContains(search)) ? image : nil
        }
    }
}

struct FactoryBrowserProjection {
    let revision: Int
    let datasetID: String
    let search: String
    let filter: MaterialBrowserFilter
    let matching: [SourceImage]
    let visible: [SourceImage]
    let folders: [SourceMaterialFolder]
}

struct FactoryTrainingPlanCache {
    let revision: Int
    let datasetID: String
    let map: MapRole
    let settings: TrainingSettings
    let plan: TrainingDataPlan?
    let issue: String?
}

/// All filesystem reads for adoption happen off the UI actor.
struct FactorySnapshotFiles: Sendable {
    let imageRevisions: [String: String]
    let exportedPackage: URL?
    let existingExportIsPresent: Bool

    init(snapshot: ProjectSnapshot, existingExport: URL?) {
        imageRevisions = Dictionary(uniqueKeysWithValues: snapshot.images.map { image in
            var info = stat()
            let revision = stat(image.url.path, &info) == 0
                ? "\(info.st_dev)|\(info.st_ino)|\(info.st_size)|\(info.st_mtimespec.tv_sec)|\(info.st_mtimespec.tv_nsec)|\(info.st_ctimespec.tv_sec)|\(info.st_ctimespec.tv_nsec)"
                : "unavailable"
            return (image.id, revision)
        })
        existingExportIsPresent = existingExport.map { FileManager.default.fileExists(atPath: $0.path) } ?? false
        let exports = snapshot.sourceURL.appendingPathComponent(".loom-factory/exports", isDirectory: true)
        let selectionDate = ["model-selection.json", "model-management.json"].compactMap { name in
            try? snapshot.sourceURL.appendingPathComponent(".loom-factory/\(name)")
                .resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        }.max() ?? .distantPast
        let files = (try? FileManager.default.contentsOfDirectory(at: exports,
            includingPropertiesForKeys: [.contentModificationDateKey], options: [.skipsHiddenFiles])) ?? []
        exportedPackage = files.compactMap { file -> (URL, Date)? in
            guard file.pathExtension == "textureloom" else { return nil }
            let modified = (try? file.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return modified >= selectionDate ? (file, modified) : nil
        }.max { $0.1 < $1.1 }?.0
    }
}
