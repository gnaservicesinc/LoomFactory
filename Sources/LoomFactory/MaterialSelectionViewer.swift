import LoomCore
import SwiftUI

extension FactoryModel {
    var selectedMaterialBrowserRow: MaterialBrowserRow? {
        materialBrowserIndex.row(id: selectedMaterialBrowserRowID)
            ?? materialBrowserIndex.row(id: selectedImageID.map { "image:" + $0 })
    }
}

/// Selecting a material previews its maps immediately; disclosure remains an
/// optional way to navigate individual files in the source tree.
struct MaterialSelectionViewer: View {
    @Bindable var model: FactoryModel
    @State private var previewResolutionID = ""

    var body: some View {
        if let row = model.selectedMaterialBrowserRow {
            VStack(alignment: .leading, spacing: 12) {
                header(row)
                Divider()
                switch row {
                case .image(let image, _, _): imagePreview(image)
                case .material(let folder): materialPreview(folder)
                case .resolution(let resolution, _): mapGrid(resolution.images)
                }
            }
            .padding(.horizontal, 16)
            .frame(minWidth: 300, maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .accessibilityIdentifier("materials.selectionPreview")
        } else {
            ContentUnavailableView("Choose a material", systemImage: "photo.stack",
                description: Text("Select a folder to see its maps and review status."))
                .frame(maxWidth: .infinity)
        }
    }

    private func header(_ row: MaterialBrowserRow) -> some View {
        let folder = model.materialBrowserIndex.folder(containingImageID: row.images.first?.id)
        let reviewedImages = folder?.resolutions.flatMap(\.images) ?? row.images
        let reviewed = !reviewedImages.isEmpty && reviewedImages.allSatisfy(\.verified)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(row.title).font(.headline).lineLimit(2)
                    Text("\(reviewedImages.filter(\.verified).count)/\(reviewedImages.count) maps reviewed")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                Button {
                    model.setMaterialReviewed(imageIDs: reviewedImages.map(\.id), verified: !reviewed)
                } label: {
                    Label(reviewed ? "Reviewed" : "Mark Reviewed", systemImage: reviewed ? "checkmark.circle.fill" : "checkmark.circle")
                }
                .buttonStyle(.borderedProminent).tint(reviewed ? .green : FactoryStyle.accent)
                .disabled(model.busy)
                .help(reviewed ? "Mark this material as not reviewed" : "Mark every map in this material as reviewed")
                .accessibilityIdentifier("materials.markReviewed")
            }
            HStack {
                Text(folder?.resolutions.compactMap { $0.images.first?.category }.first ?? "Uncategorized")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                if let folder { Button("Show Folder") { model.reveal(folder.url) }.font(.caption) }
            }
        }
    }

    private func materialPreview(_ folder: SourceMaterialFolder) -> some View {
        let resolution = folder.resolutions.first { $0.id == previewResolutionID }
            ?? folder.resolutions.last { $0.width <= 1024 && $0.height <= 1024 }
            ?? folder.resolutions.first
        return VStack(alignment: .leading, spacing: 10) {
            if folder.resolutions.count > 1, let resolution {
                Picker("Preview resolution", selection: Binding(
                    get: { folder.resolutions.contains { $0.id == previewResolutionID } ? previewResolutionID : resolution.id },
                    set: { previewResolutionID = $0 }
                )) {
                    ForEach(folder.resolutions) { group in Text(group.displayName).tag(group.id) }
                }.frame(maxWidth: 290).accessibilityIdentifier("materials.previewResolution")
            }
            if let resolution { mapGrid(resolution.images) }
        }
    }

    private func mapGrid(_ images: [SourceImage]) -> some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 170, maximum: 300), spacing: 12)], spacing: 12) {
                ForEach(images) { image in
                    Button {
                        model.selectedMaterialBrowserRowID = "image:" + image.id
                        model.selectedImageID = image.id
                    } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            FactoryImage(url: image.url, revision: model.imagePreviewRevision(image), maximumPixels: 480)
                                .frame(height: 150).clipShape(RoundedRectangle(cornerRadius: 7))
                            HStack {
                                Text(image.materialMapTitle).font(.callout.weight(.medium))
                                Spacer(minLength: 4)
                                if image.verified { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                            }
                            Text(image.url.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                                .lineLimit(1).truncationMode(.middle)
                            Text("\(image.dimensions)\(image.ignored ? " · Ignored" : "")")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("materials.preview." + image.id)
                    .help("Inspect and edit \(image.url.lastPathComponent)")
                }
            }.padding(.bottom, 8)
        }.accessibilityIdentifier("materials.mapGrid")
    }

    private func imagePreview(_ image: SourceImage) -> some View {
        VStack(spacing: 10) {
            FactoryImage(url: image.url, revision: model.imagePreviewRevision(image), maximumPixels: 1400)
                .clipShape(RoundedRectangle(cornerRadius: 10)).frame(maxWidth: .infinity, maxHeight: .infinity)
            HStack {
                Text(image.materialMapTitle)
                Spacer()
                Text(image.dimensions)
                Button("Show File") { model.reveal(image.url) }
            }.font(.caption).foregroundStyle(.secondary)
        }
    }
}
