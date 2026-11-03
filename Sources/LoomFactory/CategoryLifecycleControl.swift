import Foundation
import LoomCore
import SwiftUI

struct CategoryLifecycleControl: View {
    @Bindable var model: FactoryModel
    let datasetID: String
    @State private var preview: CategoryClosePreview?
    @State private var confirmingReopen = false

    private var closed: ClosedCategory? { model.snapshot?.closedCategories.first { $0.id == datasetID } }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            if let closed {
                Label("Closed", systemImage: "archivebox.fill").font(.caption).foregroundStyle(.secondary)
                    .accessibilityIdentifier("coverage.closed.\(datasetID)")
                Button("Reopen Category…") { confirmingReopen = true }
                    .disabled(model.busy).accessibilityIdentifier("coverage.reopenCategory.\(datasetID)")
                Button("Show source list") { model.reveal(closed.manifestURL) }
                    .accessibilityIdentifier("coverage.categorySources.\(datasetID)")
            } else if model.snapshot?.canCloseCategory(datasetID) == true {
                Button("Close Category…") {
                    model.previewCategoryClose(datasetID: datasetID) { preview = $0 }
                }.disabled(model.busy).accessibilityIdentifier("coverage.closeCategory.\(datasetID)")
                    .help("Check downloadable sources and preview disk savings. Requires approved depth, normal and roughness models. Generated and unrecognized files stay on disk.")
            }
        }.controlSize(.small)
            .confirmationDialog("Close \(preview?.name ?? "category")?", isPresented: Binding(
                get: { preview != nil }, set: { if !$0 { preview = nil } }), titleVisibility: .visible) {
                Button("Close Category & Delete Sources", role: .destructive) {
                    if let preview { model.closeCategory(preview) }
                    preview = nil
                }.disabled(model.busy)
                Button("Cancel", role: .cancel) { preview = nil }
            } message: {
                Text("Delete \(preview?.fileCount ?? 0) downloadable source files from \(preview?.materialCount ?? 0) materials to free \(preview?.formattedSize ?? "0 bytes"). Keep \(preview?.preservedFileCount ?? 0) generated or other files and all models, approvals and checkpoints. A material list with Poly Haven / ambientCG links and review tags is saved first. Deletion is permanent. Reopen Category downloads current provider releases; internet access and provider availability are required.")
            }
            .confirmationDialog("Reopen \(closed?.name ?? "category")?", isPresented: $confirmingReopen, titleVisibility: .visible) {
                Button("Reopen & Download Sources") { if let closed { model.reopenCategory(closed) } }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Restores missing files from the saved source list (\(closed?.fileCount ?? 0) files, \(ByteCountFormatter.string(fromByteCount: closed?.sourceBytes ?? 0, countStyle: .file))). Existing files are checked and kept. If interrupted, press Reopen Category again to resume. No additional resolution tiers are requested.")
            }
            .onChange(of: model.snapshot?.projectID) { preview = nil; confirmingReopen = false }
    }
}
