import LoomCore
import SwiftUI

enum MaterialBrowserFilter: String, CaseIterable, Identifiable {
    case all, unreviewed, needsReview, ignored
    var id: String { rawValue }
    var title: String {
        switch self {
        case .all: "All materials"
        case .unreviewed: "Not reviewed"
        case .needsReview: "Needs map type"
        case .ignored: "Ignored files"
        }
    }
    func includes(_ image: SourceImage) -> Bool {
        switch self {
        case .all: true
        case .unreviewed: !image.verified
        case .needsReview: image.role == .unknown && !image.ignored
        case .ignored: image.ignored
        }
    }
}

/// Expansion is window state; it never changes project metadata or membership.
struct MaterialBrowserExpansion {
    var materials = Set<String>()
    var resolutions = Set<String>()

    mutating func reveal(selectedImageID: String?, in folders: [SourceMaterialFolder]) {
        guard let selectedImageID else { return }
        for folder in folders {
            if let resolution = folder.resolutions.first(where: { group in
                group.images.contains { $0.id == selectedImageID }
            }) {
                materials.insert(folder.id)
                resolutions.insert(resolution.id)
                return
            }
        }
    }

    mutating func retainExisting(in folders: [SourceMaterialFolder]) {
        materials.formIntersection(folders.map(\.id))
        resolutions.formIntersection(folders.flatMap(\.resolutions).map(\.id))
    }

    func rows(in folders: [SourceMaterialFolder]) -> [MaterialBrowserRow] {
        folders.flatMap { folder in
            var rows: [MaterialBrowserRow] = [.material(folder)]
            guard materials.contains(folder.id) else { return rows }
            for resolution in folder.resolutions {
                rows.append(.resolution(resolution, materialID: folder.id))
                guard resolutions.contains(resolution.id) else { continue }
                let counts = Dictionary(grouping: resolution.images, by: { $0.role.rawValue + "|" + ($0.normalConvention ?? "") })
                rows += resolution.images.map { image in
                    .image(image, resolutionID: resolution.id,
                        needsFilename: (counts[image.role.rawValue + "|" + (image.normalConvention ?? "")]?.count ?? 0) > 1)
                }
            }
            return rows
        }
    }

    func isExpanded(_ row: MaterialBrowserRow) -> Bool {
        switch row {
        case .material(let folder): materials.contains(folder.id)
        case .resolution(let resolution, _): resolutions.contains(resolution.id)
        case .image: false
        }
    }

    mutating func setExpanded(_ expanded: Bool, for row: MaterialBrowserRow) {
        switch row {
        case .material(let folder):
            if expanded { materials.insert(folder.id) } else { materials.remove(folder.id) }
        case .resolution(let resolution, _):
            if expanded { resolutions.insert(resolution.id) } else { resolutions.remove(resolution.id) }
        case .image: break
        }
    }

    mutating func activate(_ row: MaterialBrowserRow, in folders: [SourceMaterialFolder]) -> Bool {
        guard rows(in: folders).contains(where: { $0.id == row.id }) else { return false }
        setExpanded(!isExpanded(row), for: row)
        return true
    }

    mutating func move(_ direction: MoveCommandDirection, from rowID: String?, in folders: [SourceMaterialFolder]) -> MaterialBrowserRow? {
        let current = rows(in: folders)
        guard !current.isEmpty else { return nil }
        let index = current.firstIndex { $0.id == rowID } ?? 0
        let row = current[index]
        switch direction {
        case .up: return current[max(0, index - 1)]
        case .down: return current[min(current.count - 1, index + 1)]
        case .right:
            if row.level < 2 && !isExpanded(row) { setExpanded(true, for: row) }
            else if current.indices.contains(index + 1), current[index + 1].parentID == row.id { return current[index + 1] }
        case .left:
            if isExpanded(row) { setExpanded(false, for: row) }
            else if let parent = current.first(where: { $0.id == row.parentID }) { return parent }
        @unknown default: break
        }
        return row
    }
}

enum MaterialBrowserRow: Identifiable, Sendable {
    case material(SourceMaterialFolder)
    case resolution(SourceMaterialResolution, materialID: String)
    case image(SourceImage, resolutionID: String, needsFilename: Bool)

    var id: String {
        switch self {
        case .material(let folder): "material:" + folder.id
        case .resolution(let resolution, _): "resolution:" + resolution.id
        case .image(let image, _, _): "image:" + image.id
        }
    }

    var parentID: String? {
        switch self {
        case .material: nil
        case .resolution(_, let materialID): "material:" + materialID
        case .image(_, let resolutionID, _): "resolution:" + resolutionID
        }
    }

    var level: Int {
        switch self {
        case .material: 0
        case .resolution: 1
        case .image: 2
        }
    }
}

/// An in-memory index of the current scan. It is rebuilt with the folder
/// snapshot and never persisted as an image-membership inventory.
struct MaterialBrowserIndex: Sendable {
    private let rowsByID: [String: MaterialBrowserRow]
    private let foldersByImageID: [String: SourceMaterialFolder]

    init(folders: [SourceMaterialFolder] = []) {
        var rows: [String: MaterialBrowserRow] = [:]
        var owners: [String: SourceMaterialFolder] = [:]
        for folder in folders {
            let row = MaterialBrowserRow.material(folder)
            rows[row.id] = row
            for resolution in folder.resolutions {
                let resolutionRow = MaterialBrowserRow.resolution(resolution, materialID: folder.id)
                rows[resolutionRow.id] = resolutionRow
                let counts = Dictionary(grouping: resolution.images, by: { $0.role.rawValue + "|" + ($0.normalConvention ?? "") })
                for image in resolution.images {
                    let imageRow = MaterialBrowserRow.image(image, resolutionID: resolution.id,
                        needsFilename: (counts[image.role.rawValue + "|" + (image.normalConvention ?? "")]?.count ?? 0) > 1)
                    rows[imageRow.id] = imageRow
                    owners[image.id] = folder
                }
            }
        }
        rowsByID = rows
        foldersByImageID = owners
    }

    func row(id: String?) -> MaterialBrowserRow? { id.flatMap { rowsByID[$0] } }
    func folder(containingImageID id: String?) -> SourceMaterialFolder? { id.flatMap { foldersByImageID[$0] } }
}

/// Stable identities resolve selections without expanding the source tree.
enum MaterialBrowserSelection {
    static func row(id: String?, in folders: [SourceMaterialFolder]) -> MaterialBrowserRow? {
        guard let id else { return nil }
        for folder in folders {
            if id == "material:" + folder.id { return .material(folder) }
            for resolution in folder.resolutions {
                if id == "resolution:" + resolution.id { return .resolution(resolution, materialID: folder.id) }
                if let image = resolution.images.first(where: { id == "image:" + $0.id }) {
                    return .image(image, resolutionID: resolution.id, needsFilename: true)
                }
            }
        }
        return nil
    }

    static func representative(in images: [SourceImage]) -> SourceImage? {
        images.first { $0.role == .diffuse && !$0.ignored } ?? images.first { !$0.ignored } ?? images.first
    }
}

extension MaterialBrowserRow {
    var images: [SourceImage] {
        switch self {
        case .material(let folder): folder.resolutions.flatMap(\.images)
        case .resolution(let resolution, _): resolution.images
        case .image(let image, _, _): [image]
        }
    }

    var title: String {
        switch self {
        case .material(let folder): folder.displayName
        case .resolution(let resolution, _): resolution.displayName
        case .image(let image, _, _): image.displayName
        }
    }

    var reviewedCount: Int { images.filter(\.verified).count }
    var isReviewed: Bool { !images.isEmpty && images.allSatisfy(\.verified) }
}

struct MaterialBrowser: View {
    @Bindable var model: FactoryModel
    @State private var expansion = MaterialBrowserExpansion()
    @FocusState private var focusedRowID: String?

    var body: some View {
        let folders = model.visibleMaterialFolders
        let rows = expansion.rows(in: folders)
        let rowIDs = rows.map(\.id)
        let hierarchyIDs = model.materialFolders.flatMap { [$0.id] + $0.resolutions.map(\.id) }
        VStack(spacing: 6) {
            Picker("Show", selection: $model.fileFilter) {
                ForEach(MaterialBrowserFilter.allCases) { filter in
                    Text(filter.title).tag(filter)
                }
            }
            .font(.callout)
            .padding(.horizontal, 8)
            Text("Select to preview · arrows show individual files")
                .font(.caption).foregroundStyle(.secondary).padding(.horizontal, 8)
            // Keep selection keyed to file/folder identities. Native table drag
            // selection was not stable while folder polling changed row indexes.
            ScrollViewReader { scroll in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(rows) { row in rowView(row).id(row.id) }
                    }.padding(6)
                }
                .onMoveCommand(perform: moveFocus)
                .onChange(of: model.selectedImageID) {
                    guard let id = model.selectedImageID else { return }
                    let selected = MaterialBrowserSelection.row(id: model.selectedMaterialBrowserRowID, in: folders)
                    guard selected?.images.contains(where: { $0.id == id }) != true else { return }
                    model.selectedMaterialBrowserRowID = "image:" + id
                    expansion.reveal(selectedImageID: id, in: folders)
                }
                .task(id: model.selectedMaterialBrowserRowID) {
                    await Task.yield()
                    guard !Task.isCancelled, let id = model.selectedMaterialBrowserRowID,
                          rowIDs.contains(id) else { return }
                    scroll.scrollTo(id)
                }
                .onChange(of: focusedRowID) {
                    if let focusedRowID { scroll.scrollTo(focusedRowID) }
                }
                .onChange(of: rowIDs) {
                    if let focusedRowID, !rowIDs.contains(focusedRowID) { self.focusedRowID = nil }
                }
            }
            .overlay {
                if folders.isEmpty {
                    ContentUnavailableView(model.fileFilter == .all ? "No materials" : "No matching files", systemImage: "photo.stack",
                        description: Text(model.fileFilter == .needsReview ? "The visible files have recognized map types." : "Change the filter, dataset or search."))
                }
            }
            .onChange(of: hierarchyIDs) { expansion.retainExisting(in: model.materialFolders) }
        }
        .onChange(of: model.visibleImages.map(\.id)) { model.selectVisibleImageIfNeeded() }
    }

    private func rowView(_ row: MaterialBrowserRow) -> some View {
        let full = model.materialBrowserIndex.row(id: row.id) ?? row
        let directSelection = model.selectedMaterialBrowserRowID == row.id ||
            (model.selectedMaterialBrowserRowID == nil && row.id == model.selectedImageID.map { "image:" + $0 })
        let containsSelectedImage = model.selectedMaterialBrowserRowID?.hasPrefix("image:") == true &&
            full.images.contains { $0.id == model.selectedImageID }
        let selected = directSelection || (containsSelectedImage && row.level < 2 && !expansion.isExpanded(row))
        return HStack(spacing: 4) {
            if row.level < 2 {
                Button {
                    _ = expansion.activate(row, in: model.visibleMaterialFolders)
                } label: {
                    Image(systemName: expansion.isExpanded(row) ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        .frame(width: 14, height: 30).contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(expansion.isExpanded(row) ? "Hide files in \(row.title)" : "Show files in \(row.title)")
                .accessibilityIdentifier(row.id + ".expand")
            }
            Button { select(row) } label: { rowLabel(row, full: full) }
                .buttonStyle(.plain)
                .focused($focusedRowID, equals: row.id)
                .accessibilityIdentifier(row.id)
                .accessibilityValue(selected ? "Selected" : "")
                .accessibilityHint("Preview maps and review this material")
            if row.level == 0 {
                Button { model.setMaterialReviewed(imageIDs: full.images.map(\.id), verified: !full.isReviewed) } label: {
                    Label(full.isReviewed ? "Reviewed" : "Review", systemImage: full.isReviewed ? "checkmark.circle.fill" : "circle")
                        .font(.caption).foregroundStyle(full.isReviewed ? .green : .secondary)
                }
                .buttonStyle(.borderless)
                .disabled(model.busy)
                .help(full.isReviewed ? "Mark this material as not reviewed" : "Mark all maps in this material as reviewed")
                .accessibilityIdentifier(row.id + ".review")
            }
        }
        .padding(.vertical, 5)
        .padding(.leading, 6 + CGFloat(row.level) * 16)
        .padding(.trailing, 6)
        .background(selected ? FactoryStyle.accent.opacity(0.18) : .clear, in: RoundedRectangle(cornerRadius: 5))
    }

    @ViewBuilder private func rowLabel(_ row: MaterialBrowserRow, full: MaterialBrowserRow) -> some View {
        switch row {
        case .material(let folder):
            Label {
                VStack(alignment: .leading, spacing: 2) {
                    Text(folder.displayName).font(.callout.weight(.medium)).lineLimit(2)
                    Text("\(full.reviewedCount)/\(full.images.count) reviewed · \(folder.resolutions.count) resolutions")
                        .font(.caption).foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, alignment: .leading)
            } icon: { Image(systemName: "folder").foregroundStyle(FactoryStyle.accent) }
            .contentShape(Rectangle()).help(folder.url.path)
        case .resolution(let resolution, _):
            VStack(alignment: .leading, spacing: 2) {
                Text(resolution.displayName).font(.callout)
                Text("\(full.reviewedCount)/\(full.images.count) reviewed")
                    .font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
        case .image(let image, _, let needsFilename):
            MaterialMapRow(image: image, revision: model.imagePreviewRevision(image), needsFilename: needsFilename)
        }
    }

    private func select(_ row: MaterialBrowserRow) {
        guard MaterialBrowserSelection.row(id: row.id, in: model.visibleMaterialFolders) != nil else { return }
        focusedRowID = row.id
        model.selectedMaterialBrowserRowID = row.id
        model.selectedImageID = MaterialBrowserSelection.representative(in: row.images)?.id
    }

    private func moveFocus(_ direction: MoveCommandDirection) {
        let origin = focusedRowID ?? model.selectedMaterialBrowserRowID
        if let row = expansion.move(direction, from: origin, in: model.visibleMaterialFolders) { select(row) }
    }
}

private struct MaterialMapRow: View {
    let image: SourceImage
    let revision: String
    let needsFilename: Bool

    private var title: String { image.materialMapTitle }

    var body: some View {
        HStack(spacing: 8) {
            FactoryImage(url: image.url, revision: revision, maximumPixels: 100)
                .frame(width: 32, height: 32).clipShape(RoundedRectangle(cornerRadius: 4))
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.callout).lineLimit(1)
                if image.ignored {
                    Text("Ignored").font(.caption).foregroundStyle(.secondary)
                } else if needsFilename {
                    Text(image.url.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                }
            }
            Spacer(minLength: 0)
            if image.verified { Image(systemName: "checkmark.circle.fill").font(.caption).foregroundStyle(.green) }
        }
        .opacity(image.ignored ? 0.65 : 1).contentShape(Rectangle()).help(image.url.lastPathComponent)
        .accessibilityLabel("\(title), \(image.displayName), \(image.dimensions)\(image.ignored ? ", ignored" : "")")
    }
}

extension SourceImage {
    var materialMapTitle: String {
        if role == .normal {
            switch normalConvention {
            case "opengl": return "OpenGL Normal"
            case "directx": return "DirectX Normal"
            default: break
            }
        }
        return role.displayName
    }
}
