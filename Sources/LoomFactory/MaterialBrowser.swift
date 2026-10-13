import LoomCore
import SwiftUI

enum MaterialBrowserFilter: String, CaseIterable, Identifiable {
    case all, needsReview, ignored
    var id: String { rawValue }
    var title: String {
        switch self {
        case .all: "All map files"
        case .needsReview: "Needs map type"
        case .ignored: "Ignored files"
        }
    }
    func includes(_ image: SourceImage) -> Bool {
        switch self {
        case .all: true
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
}

struct MaterialBrowser: View {
    @Bindable var model: FactoryModel
    @State private var expansion = MaterialBrowserExpansion()

    private var hierarchyIDs: [String] {
        model.materialFolders.flatMap { [$0.id] + $0.resolutions.map(\.id) }
    }

    var body: some View {
        VStack(spacing: 4) {
            Picker("Show", selection: $model.fileFilter) {
                ForEach(MaterialBrowserFilter.allCases) { filter in
                    Text("\(filter.title) (\(model.matchingImages.filter(filter.includes).count.formatted()))")
                        .tag(filter)
                }
            }
            .font(.callout)
            .padding(.horizontal, 8)
            List(selection: $model.selectedImageID) {
                ForEach(model.visibleMaterialFolders) { folder in
                    DisclosureGroup(isExpanded: materialExpanded(folder.id)) {
                        ForEach(folder.resolutions) { resolution in
                            DisclosureGroup(isExpanded: resolutionExpanded(resolution.id)) {
                                ForEach(resolution.images) { image in
                                    MaterialMapRow(image: image, needsFilename: resolution.images.filter {
                                        $0.role == image.role && $0.normalConvention == image.normalConvention
                                    }.count > 1)
                                    .tag(image.id)
                                    .selectionDisabled(false)
                                }
                            } label: {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(resolution.displayName).font(.callout)
                                    Text("\(resolution.images.count.formatted()) \(resolution.images.count == 1 ? "map file" : "map files")")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                .padding(.vertical, 2)
                            }
                            .selectionDisabled()
                        }
                    } label: {
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(folder.displayName).font(.callout.weight(.medium)).lineLimit(2)
                                Text("\(folder.resolutions.count.formatted()) \(folder.resolutions.count == 1 ? "resolution" : "resolutions") · \(folder.imageCount.formatted()) \(folder.imageCount == 1 ? "map file" : "map files")")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        } icon: {
                            Image(systemName: "folder").foregroundStyle(FactoryStyle.accent)
                        }
                        .padding(.vertical, 3)
                        .help(folder.url.path)
                    }
                    .selectionDisabled()
                }
            }
            .listStyle(.sidebar)
            .scrollContentBackground(.hidden)
            .overlay {
                if model.visibleMaterialFolders.isEmpty {
                    ContentUnavailableView(model.fileFilter == .all ? "No materials" : "No matching files", systemImage: "photo.stack",
                                           description: Text(model.fileFilter == .needsReview ? "The visible files have recognized map types." : "Change the file filter, dataset or search."))
                }
            }
            .onAppear { expansion.reveal(selectedImageID: model.selectedImageID, in: model.materialFolders) }
            .onChange(of: model.selectedImageID) {
                expansion.reveal(selectedImageID: model.selectedImageID, in: model.materialFolders)
            }
            .onChange(of: hierarchyIDs) {
                expansion.retainExisting(in: model.materialFolders)
            }
        }
        .onChange(of: model.fileFilter) { model.selectVisibleImageIfNeeded() }
    }

    private func materialExpanded(_ id: String) -> Binding<Bool> {
        Binding(get: { expansion.materials.contains(id) }, set: { value in
            if value { expansion.materials.insert(id) } else { expansion.materials.remove(id) }
        })
    }

    private func resolutionExpanded(_ id: String) -> Binding<Bool> {
        Binding(get: { expansion.resolutions.contains(id) }, set: { value in
            if value { expansion.resolutions.insert(id) } else { expansion.resolutions.remove(id) }
        })
    }
}

private struct MaterialMapRow: View {
    let image: SourceImage
    let needsFilename: Bool

    private var title: String {
        if image.role == .normal {
            switch image.normalConvention {
            case "opengl": return "OpenGL Normal"
            case "directx": return "DirectX Normal"
            default: break
            }
        }
        return image.role.displayName
    }

    var body: some View {
        HStack(spacing: 8) {
            FactoryImage(url: image.url, revision: image.previewRevision, maximumPixels: 100)
                .frame(width: 32, height: 32)
                .clipShape(RoundedRectangle(cornerRadius: 4))
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
            if image.verified {
                Image(systemName: "checkmark.circle.fill").font(.caption).foregroundStyle(.green)
            }
        }
        .opacity(image.ignored ? 0.65 : 1)
        .contentShape(Rectangle())
        .help(image.url.lastPathComponent)
        .accessibilityLabel("\(title), \(image.displayName), \(image.dimensions)\(image.ignored ? ", ignored" : "")")
    }
}
