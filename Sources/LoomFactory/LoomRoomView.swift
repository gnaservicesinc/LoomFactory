import AppKit
import LoomCore
import SwiftUI

struct LoomRoomView: View {
    @Bindable var model: FactoryModel
    @State private var pairingSheet = false
    @State private var syntheticPhotoSheet = false

    var body: some View {
        VStack(spacing: 0) {
            if model.snapshot != nil { workspaceNavigation }
            if model.workspaceSection == .materials, let snapshot = model.snapshot {
                materialsWorkspace(snapshot)
            } else if model.workspaceSection == .models, let snapshot = model.snapshot {
                modelsWorkspace(snapshot)
            } else if model.workspaceSection == .training, let snapshot = model.snapshot {
                trainingWorkspace(snapshot)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: FactoryStyle.spacing) {
                        header
                        sourceIssuePanel
                        if let snapshot = model.snapshot {
                            sourcePanel(snapshot)
                            switch model.workspaceSection {
                            case .materials:
                                datasetPanel(snapshot)
                                if model.busy || !model.events.isEmpty { RunPanel(model: model) }
                            case .models:
                                ModelCoverageView(model: model)
                            case .training:
                                EmptyView()
                            case .testing:
                                ModelTestView(model: model)
                            case .package:
                                HStack(alignment: .top, spacing: FactoryStyle.spacing) {
                                    CreateMLPanel(model: model).frame(maxWidth: .infinity)
                                    PackagePanel(model: model).frame(maxWidth: .infinity)
                                }
                                RunPanel(model: model)
                            }
                        } else {
                            sourceWelcome
                        }
                    }
                    .padding(28)
                    .frame(maxWidth: 1600)
                    .frame(maxWidth: .infinity)
                }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .toolbar {
            ToolbarItem(placement: .principal) { Text("Loom Factory").font(.headline) }
            ToolbarItemGroup(placement: .primaryAction) {
                if model.snapshot != nil {
                    Button { model.showModelTesting() } label: { Label("Test Model", systemImage: "play.rectangle") }
                        .accessibilityIdentifier("loom.testModel")
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
        .sheet(isPresented: $syntheticPhotoSheet) { SyntheticPhotoDataSheet(model: model) }
        .onChange(of: model.settings) { model.scheduleSettingsSave() }
        .task {
            model.restoreLastFolder()
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(3)) } catch { break }
                await model.pollFolder()
            }
        }
    }

    private var workspaceNavigation: some View {
        HStack {
            Picker("Loom Room workspace", selection: $model.workspaceSection) {
                ForEach(LoomWorkspaceSection.allCases) { section in
                    Text(section.displayName).tag(section)
                }
            }.pickerStyle(.segmented).labelsHidden().frame(width: 660, alignment: .leading)
                .accessibilityIdentifier("loom.workspaceSection")
            Spacer()
        }
        .padding(.horizontal, 28).padding(.vertical, 12)
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
    }

    @ViewBuilder private var sourceIssuePanel: some View {
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
    }

    private func trainingWorkspace(_ snapshot: ProjectSnapshot) -> some View {
        VStack(alignment: .leading, spacing: FactoryStyle.spacing) {
            header
            sourcePanel(snapshot)
            HStack(alignment: .top, spacing: FactoryStyle.spacing) {
                ScrollView {
                    TrainingPanel(model: model).frame(maxWidth: .infinity)
                }.accessibilityIdentifier("loom.trainingControls")
                ScrollView {
                    VStack(alignment: .leading, spacing: FactoryStyle.spacing) {
                        RunPanel(model: model)
                        TrainingOverviewPanel(model: model)
                    }
                }.frame(width: 330)
                    .accessibilityIdentifier("loom.trainingProgress")
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(28)
        .frame(maxWidth: 1600)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 14) {
            LoomMark().frame(width: 48, height: 48)
            VStack(alignment: .leading, spacing: 3) {
                Text("Loom Room").font(.system(size: 28, weight: .semibold))
                Text("Review your images, manage your models, and build Texture Loom.")
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

    private func compactSourceBar(_ snapshot: ProjectSnapshot) -> some View {
        HStack(spacing: 12) {
            Label(snapshot.sourceURL.lastPathComponent, systemImage: "folder")
                .font(.callout.weight(.semibold)).lineLimit(1).help(snapshot.sourceURL.path)
            Text("\(model.materialFolders.count.formatted()) materials · \(snapshot.images.count.formatted()) maps")
                .font(.caption).foregroundStyle(.secondary)
            Spacer()
            if model.busy {
                ProgressView().controlSize(.small)
                Text(model.activityTitle).font(.caption).lineLimit(1).frame(maxWidth: 300)
                Button(model.isTraining ? "Save & Stop" : "Stop") {
                    if model.isTraining { model.saveAndStopTraining() } else { model.cancel() }
                }
            }
            Button("Show Folder") { model.revealSource() }
            Button("Change Folder…") { model.chooseSourceFolder() }.disabled(model.busy)
        }.padding(.horizontal, 18).padding(.vertical, 10)
            .background(.bar)
    }

    private func materialsWorkspace(_ snapshot: ProjectSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            compactSourceBar(snapshot)
            sourceIssuePanel
            datasetPanel(snapshot)
        }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private func modelsWorkspace(_ snapshot: ProjectSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            compactSourceBar(snapshot)
            sourceIssuePanel
            ModelCoverageView(model: model).padding(16)
        }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private func datasetPanel(_ snapshot: ProjectSnapshot) -> some View {
        let materials = model.visibleMaterialFolders
        let mapCount = materials.reduce(0) { $0 + $1.imageCount }
        return VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Picker("Dataset", selection: $model.selectedDatasetID) {
                    Text("All materials").tag("")
                    Text("Uncategorized").tag("unassigned")
                    if !snapshot.datasets.isEmpty { Divider() }
                    ForEach(snapshot.datasets, id: \.id) { dataset in
                        Text(dataset.name).tag(dataset.id)
                    }
                }.frame(width: 260)
                Text("\(materials.count.formatted()) materials · \(mapCount.formatted()) maps")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                TextField("Find a material or map", text: $model.imageSearch).textFieldStyle(.roundedBorder).frame(width: 200)
                Button("Pair Images…") { pairingSheet = true }.disabled(model.busy || snapshot.images.count < 2)
                Button("Synthetic Photos…", systemImage: "camera.badge.ellipsis") { syntheticPhotoSheet = true }
                    .disabled(model.busy)
                    .accessibilityIdentifier("loom.syntheticPhotos")
                Button("Categorize", systemImage: "tag") { model.categorize() }.disabled(model.busy || snapshot.images.isEmpty)
                    .help("Categorize each material folder once using one diffuse image. Existing tags and reviewed images are preserved.")
            }
            HSplitView {
                MaterialBrowser(model: model).id(snapshot.sourceURL.path)
                    .frame(minWidth: 230, idealWidth: 285, maxWidth: 380)
                MaterialSelectionViewer(model: model).frame(minWidth: 340, maxWidth: .infinity, maxHeight: .infinity)
                if case .image(let image, _, _) = model.selectedMaterialBrowserRow {
                    ScrollView {
                        ImageReviewPanel(model: model, image: image)
                    }.frame(minWidth: 250, idealWidth: 285, maxWidth: 320)
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
        }.padding(16).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .onChange(of: model.selectedDatasetID) { model.selectVisibleImageIfNeeded() }
        .onChange(of: model.imageSearch) { model.selectVisibleImageIfNeeded() }
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
            if let inference = image.mapTypeInference, role == image.role {
                MapTypeInferenceNotice(inference: inference)
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
            Toggle("Reviewed", isOn: $verified).toggleStyle(.checkbox)
                .help("Optional record that you checked this material. Training uses its dataset tag and matching maps.")
            VStack(alignment: .leading, spacing: 5) {
                Text("Review note").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                TextEditor(text: $note).font(.callout).frame(minHeight: 60, maxHeight: 90)
                    .scrollContentBackground(.hidden).padding(6)
                    .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 7))
            }
            Text("Review is optional. Dataset and review apply to this material folder. Map type and Ignore apply to this file.")
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

struct MapTypeInferenceNotice: View {
    let inference: MapTypeInference

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("\(inference.accepted ? "Classifier guess" : "Classifier unsure"): \(inference.role.displayName) (\(inference.confidence.formatted(.percent.precision(.fractionLength(0))))).", systemImage: "sparkle.magnifyingglass")
            if !inference.accepted {
                Text("Choose the map type above.")
            } else if inference.role == .depth || inference.role == .roughness {
                Text("Depth and roughness can look similar. Check the map type.")
            } else {
                Text("Check the map type and correct it if needed.")
            }
        }
        .font(.caption).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("review.mapTypeInference")
    }
}

/// Temporary checkbox state. Only current physical members may be paired.
struct PairImagesSelection {
    var selectedImageIDs = Set<String>()

    mutating func select(_ id: String, isSelected: Bool, in images: [SourceImage]) {
        guard images.contains(where: { $0.id == id }) else {
            selectedImageIDs.remove(id)
            return
        }
        if isSelected { selectedImageIDs.insert(id) } else { selectedImageIDs.remove(id) }
    }

    mutating func retainExisting(in images: [SourceImage]) {
        selectedImageIDs.formIntersection(images.map(\.id))
    }

    func currentIDs(in images: [SourceImage]) -> [String] {
        selectedImageIDs.intersection(images.map(\.id)).sorted()
    }
}

struct PairImagesSheet: View {
    @Bindable var model: FactoryModel
    @Environment(\.dismiss) private var dismiss
    @State private var selection: PairImagesSelection

    init(model: FactoryModel, selectedImageIDs: Set<String> = []) {
        self.model = model
        _selection = State(initialValue: PairImagesSelection(selectedImageIDs: selectedImageIDs))
    }

    private var images: [SourceImage] { model.snapshot?.images ?? [] }
    private var currentSelection: [String] { selection.currentIDs(in: images) }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Pair material images").font(.title2.weight(.semibold))
            Text("Select all diffuse variants that share these maps, plus one image of each matching target type at the same resolution. Each diffuse becomes a separate training sample. Set map types in Image Review first.")
                .foregroundStyle(.secondary)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(images) { image in
                        Toggle(isOn: Binding(
                            get: { selection.selectedImageIDs.contains(image.id) },
                            set: { selection.select(image.id, isSelected: $0, in: images) }
                        )) {
                            HStack {
                                FactoryImage(url: image.url, revision: model.imagePreviewRevision(image), maximumPixels: 100)
                                    .frame(width: 40, height: 40).clipShape(RoundedRectangle(cornerRadius: 6))
                                Text(image.displayName).lineLimit(2)
                                Spacer()
                                Text(image.role.displayName).foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .toggleStyle(.checkbox)
                        .padding(4)
                        .accessibilityIdentifier("pair.image." + image.id)
                        .accessibilityLabel("\(image.displayName), \(image.role.displayName), \(image.dimensions)")
                        .accessibilityHint("Include this image in the material pair")
                    }
                }
                .padding(4)
            }
            .frame(minHeight: 340)
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Use the checkboxes to select matching images.")
                    Text("\(currentSelection.count.formatted()) \(currentSelection.count == 1 ? "image" : "images") selected")
                }.font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Pair Selected") {
                    let ids = selection.currentIDs(in: images)
                    guard ids.count >= 2, !model.busy else { return }
                    model.pairImages(ids: ids)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent).disabled(currentSelection.count < 2 || model.busy)
            }
        }.padding(24).frame(width: 680)
        .onChange(of: images.map(\.id), initial: true) { selection.retainExisting(in: images) }
    }
}

enum LoomWorkspaceSection: String, CaseIterable, Identifiable {
    case materials, models, training, testing, package
    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .materials: "Materials"
        case .models: "Models"
        case .training: "Training"
        case .testing: "Test Model"
        case .package: "Package"
        }
    }
}
