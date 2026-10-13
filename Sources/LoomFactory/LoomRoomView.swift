import AppKit
import LoomCore
import SwiftUI

struct LoomRoomView: View {
    @Bindable var model: FactoryModel
    @State private var pairingSheet = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: FactoryStyle.spacing) {
                header
                if let issue = model.sourceIssue {
                    FactoryPanel("Source folder needs attention") {
                        Text(issue).foregroundStyle(.secondary).textSelection(.enabled)
                        if let folder = model.sourceFolder {
                            Text(folder.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                        Button("Try Again", systemImage: "arrow.clockwise") { model.refresh() }
                            .disabled(model.busy)
                    }
                }
                if let snapshot = model.snapshot {
                    sourcePanel(snapshot)
                    datasetPanel(snapshot)
                    HStack(alignment: .top, spacing: FactoryStyle.spacing) {
                        TrainingPanel(model: model).frame(maxWidth: .infinity)
                        VStack(spacing: FactoryStyle.spacing) {
                            CreateMLPanel(model: model)
                            PackagePanel(model: model)
                        }.frame(width: 390)
                    }
                    if model.busy || !model.events.isEmpty { RunPanel(model: model) }
                } else {
                    sourceWelcome
                }
            }
            .padding(28)
            .frame(maxWidth: 1600)
            .frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .toolbar {
            ToolbarItem(placement: .principal) { Text("Loom Factory").font(.headline) }
            ToolbarItemGroup(placement: .primaryAction) {
                if model.snapshot != nil {
                    Button { model.importImages() } label: { Label("Import Images", systemImage: "square.and.arrow.down") }
                        .disabled(model.busy)
                    Button { model.refresh() } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                        .disabled(model.busy)
                }
            }
        }
        .alert("Action could not finish", isPresented: Binding(
            get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK") { model.errorMessage = nil }
        } message: { Text(model.errorMessage ?? "") }
        .sheet(isPresented: $pairingSheet) { PairImagesSheet(model: model) }
        .onChange(of: model.settings) { model.scheduleSettingsSave() }
        .task {
            model.restoreLastFolder()
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(3)) } catch { break }
                await model.pollFolder()
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 14) {
            LoomMark().frame(width: 48, height: 48)
            VStack(alignment: .leading, spacing: 3) {
                Text("Loom Room").font(.system(size: 28, weight: .semibold))
                Text("Review your images, train your models, and build Texture Loom.")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if model.busy {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(model.activityTitle).font(.callout).lineLimit(1).frame(maxWidth: 340)
                    Button("Stop") { model.cancel() }
                }
            } else if model.snapshot != nil {
                FactoryStatus(text: "Project stored in source folder", symbol: "folder.badge.checkmark", color: .green)
            }
        }
        .padding(.bottom, 4)
    }

    private var sourceWelcome: some View {
        VStack(alignment: .center, spacing: 28) {
            VStack(spacing: 12) {
                Image(systemName: "folder.badge.plus").font(.system(size: 52)).foregroundStyle(FactoryStyle.accent)
                Text("Your source folder is your project").font(.title2.weight(.semibold))
                Text("Choose the folder that holds your material images. Your categories, review notes, training settings, and models stay with that folder.")
                    .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 620)
                Button("Choose Source Folder…", systemImage: "folder") { model.chooseSourceFolder() }
                    .buttonStyle(.borderedProminent).controlSize(.large).disabled(model.busy)
            }
            .padding(.vertical, 34)
            HStack(alignment: .top, spacing: 18) {
                welcomeStep("Review your datasets", "Images are categorized into datasets. Review the labels and correct them directly on the images.", "square.grid.2x2")
                welcomeStep("Train Texture Loom", "Train depth, roughness and normal models for your datasets.", "sparkles")
                welcomeStep("Export one package", "Bundle the classifier and your models for a custom Texture Studio build, or publish them together.", "shippingbox")
            }
        }
        .padding(32)
        .frame(maxWidth: .infinity)
        .background(.background, in: RoundedRectangle(cornerRadius: FactoryStyle.corner))
        .overlay(RoundedRectangle(cornerRadius: FactoryStyle.corner).strokeBorder(Color(nsColor: .separatorColor).opacity(0.55)))
    }

    private func welcomeStep(_ title: String, _ text: String, _ symbol: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: symbol).font(.headline)
            Text(text).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }.padding(20).frame(maxWidth: .infinity, alignment: .topLeading)
            .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
    }

    private func sourcePanel(_ snapshot: ProjectSnapshot) -> some View {
        let materials = model.materialFolders
        return HStack(spacing: 22) {
            Image(systemName: "folder.fill").font(.title2).foregroundStyle(FactoryStyle.accent)
            VStack(alignment: .leading, spacing: 4) {
                Text("Source folder").font(.headline)
                Text(snapshot.sourceURL.path).font(.callout).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle).textSelection(.enabled)
            }
            Spacer(minLength: 12)
            metric("Source materials", materials.count)
            metric("Map files", snapshot.images.count)
            metric("Datasets", snapshot.datasets.count)
            metric("Reviewed", materials.filter { folder in
                folder.resolutions.flatMap(\.images).allSatisfy(\.verified)
            }.count)
            Divider().frame(height: 36)
            Button("Show in Finder") { model.revealSource() }
            Button("Change Source Folder…") { model.chooseSourceFolder() }
        }
        .padding(18)
        .background(.background, in: RoundedRectangle(cornerRadius: FactoryStyle.corner))
        .overlay(RoundedRectangle(cornerRadius: FactoryStyle.corner).strokeBorder(Color(nsColor: .separatorColor).opacity(0.55)))
    }

    private func metric(_ name: String, _ count: Int) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(count, format: .number).font(.title3.weight(.semibold)).monospacedDigit()
            Text(name).font(.caption).foregroundStyle(.secondary)
        }.frame(minWidth: 58, alignment: .leading)
    }

    private func datasetPanel(_ snapshot: ProjectSnapshot) -> some View {
        let materials = model.visibleMaterialFolders
        let mapCount = materials.reduce(0) { $0 + $1.imageCount }
        let materialLabel = materials.count == 1 ? "source material" : "source materials"
        let mapLabel = mapCount == 1 ? "map file" : "map files"
        return FactoryPanel("Datasets", subtitle: "A dataset contains the images tagged with its category.") {
            HStack(spacing: 16) {
                Picker("Dataset", selection: $model.selectedDatasetID) {
                    Text("All materials").tag("")
                    Text("Uncategorized").tag("unassigned")
                    if !snapshot.datasets.isEmpty { Divider() }
                    ForEach(snapshot.datasets, id: \.id) { dataset in
                        Text(dataset.name).tag(dataset.id)
                    }
                }.frame(width: 270)
                Text("\(materials.count.formatted()) \(materialLabel) · \(mapCount.formatted()) \(mapLabel)")
                    .font(.callout).foregroundStyle(.secondary)
                Spacer()
                TextField("Find a material or map", text: $model.imageSearch).textFieldStyle(.roundedBorder).frame(width: 210)
                Button("Pair Images…") { pairingSheet = true }.disabled(model.busy || snapshot.images.count < 2)
                Button("Categorize", systemImage: "tag") { model.categorize() }.disabled(model.busy || snapshot.images.isEmpty)
                    .help("Categorize each material folder once using one diffuse image. Existing tags and reviewed images are preserved.")
            }
            HSplitView {
                MaterialBrowser(model: model).id(snapshot.sourceURL.path)
                    .frame(minWidth: 220, idealWidth: 285, maxWidth: 380)
                if let image = model.selectedImage {
                    VStack(spacing: 10) {
                        FactoryImage(url: image.url, revision: image.previewRevision)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                        HStack {
                            Text(image.role.displayName)
                                .help(image.url.lastPathComponent)
                            Spacer()
                            Text("\(image.width.formatted()) × \(image.height.formatted())")
                        }.font(.caption).foregroundStyle(.secondary)
                    }.padding(.horizontal, 16).frame(minWidth: 300)
                    ImageReviewPanel(model: model, image: image).frame(minWidth: 250, idealWidth: 290, maxWidth: 340)
                } else {
                    ContentUnavailableView("No image selected", systemImage: "photo", description: Text("Import images or choose another dataset."))
                        .frame(maxWidth: .infinity)
                }
            }.frame(height: 450)
        }
        .onChange(of: model.selectedDatasetID) {
            model.selectVisibleImageIfNeeded()
        }
        .onChange(of: model.imageSearch) {
            model.selectVisibleImageIfNeeded()
        }
    }
}

private struct ImageReviewPanel: View {
    @Bindable var model: FactoryModel
    let image: SourceImage
    @State private var category = ""
    @State private var note = ""
    @State private var verified = false
    @State private var ignored = false
    @State private var role: MapRole = .unknown
    @State private var normalConvention = "opengl"

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Image review").font(.headline)
                Text(image.displayName).font(.callout).foregroundStyle(.secondary).lineLimit(2)
            }
            VStack(alignment: .leading, spacing: 5) {
                Text("Dataset").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                HStack {
                    TextField("Category name", text: $category).textFieldStyle(.roundedBorder)
                    Menu {
                        Button("Uncategorized") { category = "" }
                        ForEach(model.snapshot?.datasets ?? [], id: \.id) { dataset in
                            Button(dataset.name) { category = dataset.name }
                        }
                    } label: { Image(systemName: "chevron.down") }.menuStyle(.borderlessButton).frame(width: 18)
                }
            }
            Picker("Map type", selection: $role) {
                ForEach(MapRole.allCases, id: \.self) { Text($0.displayName).tag($0) }
            }
            if role == .bump {
                Text("Bump maps are kept for inspection and are excluded from depth training.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if role == .normal {
                Picker("Normal format", selection: $normalConvention) {
                    Text("OpenGL").tag("opengl")
                    Text("DirectX").tag("directx")
                }
            }
            Toggle("Ignore this file", isOn: $ignored).toggleStyle(.checkbox)
                .help("Keep this image available for inspection and leave it out of categorization, Create ML export and model training.")
            Toggle("Reviewed and verified", isOn: $verified).toggleStyle(.checkbox)
            VStack(alignment: .leading, spacing: 5) {
                Text("Review note").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                TextEditor(text: $note).font(.callout).frame(minHeight: 60, maxHeight: 90)
                    .scrollContentBackground(.hidden).padding(6)
                    .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 7))
            }
            Text("Dataset and review apply to this material folder. Map type and Ignore apply to this file.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            HStack {
                Button("Show File") { model.reveal(image.url) }
                Spacer()
                Button("Save Review") { model.review(image: image, category: category, role: role, normalConvention: normalConvention, verified: verified, note: note, ignored: ignored) }
                    .buttonStyle(.borderedProminent)
            }
        }.padding(.leading, 16).disabled(model.busy)
            .task(id: "\(image.id)|\(image.category ?? "")|\(image.verified)|\(image.ignored)|\(image.role.rawValue)|\(image.normalConvention ?? "")|\(image.note)") {
                category = image.category ?? ""
                note = image.note
                verified = image.verified
                ignored = image.ignored
                role = image.role
                normalConvention = image.normalConvention ?? "opengl"
            }
    }
}

private struct PairImagesSheet: View {
    @Bindable var model: FactoryModel
    @Environment(\.dismiss) private var dismiss
    @State private var selected = Set<String>()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Pair material images").font(.title2.weight(.semibold))
            Text("Select a diffuse image and its matching maps. Set each map type in Image Review first.")
                .foregroundStyle(.secondary)
            List(model.snapshot?.images ?? [], id: \.id, selection: $selected) { image in
                HStack {
                    FactoryImage(url: image.url, revision: image.previewRevision, maximumPixels: 100)
                        .frame(width: 40, height: 40).clipShape(RoundedRectangle(cornerRadius: 6))
                    Text(image.displayName)
                    Spacer()
                    Text(image.role.displayName).foregroundStyle(.secondary)
                }.tag(image.id)
            }.frame(minHeight: 340)
            HStack {
                Text("Use ⌘ to select multiple images.").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Pair Selected") { model.pairImages(ids: selected.sorted()); dismiss() }
                    .buttonStyle(.borderedProminent).disabled(selected.count < 2)
            }
        }.padding(24).frame(width: 680)
    }
}

extension SourceImage {
    var previewRevision: String {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return "\(attributes?[.modificationDate] ?? "")|\(attributes?[.size] ?? "")"
    }
}
