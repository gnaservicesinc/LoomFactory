import AppKit
import Foundation
import LoomCore
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class FactoryModel {
    private(set) var snapshot: ProjectSnapshot?
    var settings = ProjectSettings()
    var selectedDatasetID = ""
    var selectedImageID: String?
    var selectedMap: MapRole = .depth
    var imageSearch = ""
    var fileFilter: MaterialBrowserFilter = .all
    var workspaceSection: LoomWorkspaceSection = .materials
    private(set) var busy = false
    private(set) var activityTitle = ""
    private(set) var progress: Double?
    private(set) var events: [TrainingEvent] = []
    private(set) var exportedPackage: URL?
    private(set) var publishedURL: URL?
    private(set) var sourceIssue: String?
    var errorMessage: String?

    @ObservationIgnored private var project: LoomProject?
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var saveOperation: Task<Void, Never>?
    @ObservationIgnored private var generation = UUID()
    @ObservationIgnored private var polling = false
    @ObservationIgnored private let rememberSourceFolder: Bool

    var sourceFolder: URL? { project?.sourceFolder }

    var selectedDataset: DatasetCategory? {
        snapshot?.datasets.first { $0.id == selectedDatasetID }
    }

    var matchingImages: [SourceImage] {
        guard let snapshot else { return [] }
        let dataset = selectedDataset?.name
        return snapshot.images.filter { image in
            let matchesDataset = selectedDatasetID == "" ||
                (selectedDatasetID == "unassigned" ? image.category == nil : image.category == dataset)
            return matchesDataset && (imageSearch.isEmpty ||
                image.displayName.localizedCaseInsensitiveContains(imageSearch) ||
                image.url.lastPathComponent.localizedCaseInsensitiveContains(imageSearch) ||
                ImageMetadataStore.friendlyName(image.url.deletingLastPathComponent().lastPathComponent).localizedCaseInsensitiveContains(imageSearch) ||
                image.role.displayName.localizedCaseInsensitiveContains(imageSearch) ||
                image.category?.localizedCaseInsensitiveContains(imageSearch) == true)
        }
    }

    var visibleImages: [SourceImage] { matchingImages.filter(fileFilter.includes) }

    var materialFolders: [SourceMaterialFolder] {
        guard let snapshot else { return [] }
        return SourceMaterialFolder.group(images: snapshot.images, sourceFolder: snapshot.sourceURL)
    }

    var visibleMaterialFolders: [SourceMaterialFolder] {
        guard let snapshot else { return [] }
        return SourceMaterialFolder.group(images: visibleImages, sourceFolder: snapshot.sourceURL)
    }

    var selectedImage: SourceImage? {
        snapshot?.images.first { $0.id == selectedImageID }
    }

    var classifier: ModelArtifact? { snapshot?.artifacts.first { $0.kind == "classifier" } }
    var selectedArtifacts: [ModelArtifact] {
        snapshot?.artifacts.filter { $0.kind == "map" && $0.datasetID == selectedDatasetID } ?? []
    }
    var eligiblePairCount: Int {
        trainingPlan?.totalPairCount ?? 0
    }
    var selectedCheckpoint: TrainingCheckpoint? {
        snapshot?.checkpoints.first { $0.datasetID == selectedDatasetID && $0.map == selectedMap }
    }
    var effectiveTrainingSettings: TrainingSettings {
        var value = settings.training
        if value.continueLatestWeights == true, let checkpoint = selectedCheckpoint {
            value.modelWidth = checkpoint.modelWidth; value.modelBlocks = checkpoint.modelBlocks
            value.imageSize = checkpoint.imageSize; value.exaggerateFeatures = checkpoint.exaggerateFeatures
        }
        return value
    }
    var trainingOptionsIssue: String? {
        if let issue = effectiveTrainingSettings.validationIssue { return issue }
        do {
            if let plan = try snapshot?.trainingPlan(datasetID: selectedDatasetID, map: selectedMap, settings: effectiveTrainingSettings),
               effectiveTrainingSettings.earlyStoppingPatience > 0, plan.validationPairCount == 0 {
                return "Early stopping needs held-out pairs from another material folder. Disable early stopping or add another tagged material folder."
            }
        } catch { return error.localizedDescription }
        return nil
    }
    var trainingPlan: TrainingDataPlan? {
        guard effectiveTrainingSettings.validationIssue == nil else { return nil }
        return try? snapshot?.trainingPlan(datasetID: selectedDatasetID, map: selectedMap, settings: effectiveTrainingSettings)
    }
    var canContinueMap: Bool {
        selectedCheckpoint != nil
    }
    var canTrain: Bool {
        project != nil && selectedDataset != nil && trainingOptionsIssue == nil && (trainingPlan?.trainingPairCount ?? 0) > 0 && !busy &&
        (!(settings.training.continueLatestWeights ?? false) || canContinueMap)
    }

    init(rememberSourceFolder: Bool = true) {
        self.rememberSourceFolder = rememberSourceFolder
    }

    func restoreLastFolder() {
        guard snapshot == nil, !busy else { return }
        let arguments = ProcessInfo.processInfo.arguments
        if let index = arguments.firstIndex(of: "--source-folder"), arguments.indices.contains(index + 1) {
            openFolder(URL(fileURLWithPath: arguments[index + 1], isDirectory: true))
            return
        }
        guard rememberSourceFolder,
              let path = UserDefaults.standard.string(forKey: "recentSourceFolder"),
              FileManager.default.fileExists(atPath: path) else { return }
        openFolder(URL(fileURLWithPath: path, isDirectory: true))
    }

    func chooseSourceFolder() {
        let panel = NSOpenPanel()
        panel.title = "Choose your source folder"
        panel.prompt = "Open Source Folder"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = sourceFolder
        guard panel.runModal() == .OK, let url = panel.url else { return }
        openFolder(url)
    }

    func openFolder(_ url: URL) {
        operation?.cancel()
        saveOperation?.cancel()
        let oldProject = project
        generation = UUID()
        let current = generation
        project = nil
        snapshot = nil
        selectedImageID = nil
        selectedDatasetID = ""
        selectedMap = .depth
        imageSearch = ""
        fileFilter = .all
        workspaceSection = .materials
        settings = ProjectSettings()
        exportedPackage = nil
        publishedURL = nil
        events = []
        errorMessage = nil
        sourceIssue = nil
        busy = true
        activityTitle = "Opening source folder"
        progress = nil
        operation = Task {
            if let oldProject { await oldProject.cancelTraining() }
            do {
                let opened = try LoomProject(sourceFolder: url)
                let value = try await opened.snapshot()
                guard current == generation, !Task.isCancelled else { return }
                project = opened
                adopt(value, restoreSettings: true)
                if rememberSourceFolder { UserDefaults.standard.set(url.path, forKey: "recentSourceFolder") }
                activityTitle = "Source folder opened"
                busy = false
            } catch {
                guard current == generation else { return }
                fail(error)
            }
        }
    }

    func refresh() {
        perform("Refreshing images") { project in _ = try await project.snapshot() }
    }

    func pollFolder() async {
        guard let project, !busy, !polling else { return }
        polling = true
        defer { polling = false }
        let current = generation
        do {
            let value = try await project.snapshot()
            guard current == generation, !busy else { return }
            adopt(value, restoreSettings: false)
        } catch {
            guard current == generation, !busy else { return }
            invalidateSnapshot(error)
        }
    }

    func importImages() {
        guard snapshot != nil, !busy else { return }
        let panel = NSOpenPanel()
        panel.title = "Add images to your source folder"
        panel.prompt = "Import"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image, .folder]
        guard panel.runModal() == .OK else { return }
        let urls = panel.urls
        perform("Importing images") { project in try await project.importImages(urls) }
    }

    func review(image: SourceImage, category: String, role: MapRole, normalConvention: String, verified: Bool, note: String, ignored: Bool = false) {
        let normalized = category.trimmingCharacters(in: .whitespacesAndNewlines)
        perform("Saving image metadata") { project in
            try await project.setRole(id: image.id, role: role)
            try await project.setIgnored(id: image.id, ignored: ignored)
            if role == .normal { try await project.setNormalConvention(id: image.id, convention: normalConvention) }
            try await project.updateImage(id: image.id, category: normalized.isEmpty ? nil : normalized, verified: verified, note: note)
        }
    }

    func selectVisibleImageIfNeeded() {
        let members = visibleMaterialFolders.flatMap { $0.resolutions.flatMap(\.images) }
        guard !members.contains(where: { $0.id == selectedImageID }) else { return }
        selectedImageID = members.first(where: { $0.role == .diffuse && !$0.ignored })?.id ?? members.first?.id
    }

    func pairImages(ids: [String]) {
        perform("Pairing images") { project in try await project.pairImages(ids: ids) }
    }

    func scheduleSettingsSave() {
        guard let project, snapshot != nil, !busy else { return }
        saveOperation?.cancel()
        guard !settings.modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let value = settings
        let current = generation
        saveOperation = Task {
            do {
                try await Task.sleep(for: .milliseconds(350))
                try Task.checkCancellation()
                try await project.updateSettings(value)
            } catch is CancellationError {} catch {
                guard current == generation else { return }
                errorMessage = error.localizedDescription
            }
        }
    }

    func prepareForTermination() async -> Bool {
        guard !busy else { return false }
        saveOperation?.cancel()
        guard let project, snapshot != nil,
              !settings.modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return true }
        let current = generation
        do {
            try await project.updateSettings(settings)
            return current == generation && !busy
        } catch {
            errorMessage = error.localizedDescription
            return false
        }
    }

    func trainMap() {
        guard let dataset = selectedDataset, canTrain else { return }
        workspaceSection = .training
        let value = settings
        let map = selectedMap
        var training = effectiveTrainingSettings
        training.exaggerateFeatures = map == .depth && training.exaggerateFeatures
        let configuredTraining = training
        let onProgress = progressHandler()
        perform("Training \(dataset.name) · \(map.displayName)", clearEvents: true) { project in
            try await project.updateSettings(value)
            _ = try await project.trainMap(datasetID: dataset.id, map: map, settings: configuredTraining, progress: onProgress)
        }
    }

    func exportClassifierData() {
        guard snapshot != nil, !busy else { return }
        let panel = NSOpenPanel()
        panel.title = "Export Create ML Data"
        panel.message = "Choose where to create the category folders for Create ML."
        panel.prompt = "Export Data"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = sourceFolder?.appendingPathComponent(".loom-factory/exports", isDirectory: true)
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        let current = generation
        perform("Exporting Create ML data", clearEvents: true) { [weak self] project in
            let summary = try await project.exportClassifierData(to: destination)
            guard let self, self.generation == current else { return }
            self.events.append(TrainingEvent(type: "summary", message:
                "Exported \(summary.imageCount.formatted()) materials in \(summary.folderCount.formatted()) category folders for Create ML."))
            if !summary.skippedFolders.isEmpty {
                let names = summary.skippedFolders.map { ImageMetadataStore.friendlyName($0.lastPathComponent) }.joined(separator: ", ")
                self.events.append(TrainingEvent(type: "review", message:
                    "Folders needing review for Create ML: \(names). Check their category and native 1,024 × 1,024 diffuse image."))
            }
            NSWorkspace.shared.activateFileViewerSelecting([summary.directory])
        }
    }

    func importClassifier() {
        guard snapshot != nil, !busy else { return }
        let panel = NSOpenPanel()
        panel.title = "Import Create ML Classifier"
        panel.message = "Choose the Core ML classifier exported by Create ML."
        panel.prompt = "Import Classifier"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.treatsFilePackagesAsDirectories = false
        panel.allowedContentTypes = ["mlmodel", "mlpackage", "mlmodelc"].compactMap { UTType(filenameExtension: $0) } + [.folder]
        guard panel.runModal() == .OK, let modelURL = panel.url else { return }
        let current = generation
        perform("Importing Create ML classifier", clearEvents: true) { [weak self] project in
            _ = try await project.importClassifier(from: modelURL)
            guard let self, self.generation == current else { return }
            self.events.append(TrainingEvent(type: "summary", message: "Imported the Create ML classifier into this source project."))
        }
    }

    func categorize() {
        guard !busy, snapshot != nil else { return }
        let modelURL: URL?
        if let classifier, let path = classifier.coreMLPath {
            modelURL = URL(fileURLWithPath: path, relativeTo: classifier.directory)
        } else {
            modelURL = Bundle.main.url(forResource: "DTTC3", withExtension: "mlmodelc") ??
                Bundle.main.url(forResource: "DTTC3", withExtension: "mlmodel")
        }
        guard let modelURL else {
            errorMessage = "The dataset classifier is unavailable. Import a Create ML classifier first."
            return
        }
        let current = generation
        let onProgress: @Sendable (Double, String) -> Void = { [weak self] fraction, message in
            Task { @MainActor [weak self] in
                guard let self, self.generation == current else { return }
                self.progress = fraction
                self.activityTitle = message
            }
        }
        perform("Categorizing material folders", clearEvents: true) { [weak self] project in
            let summary = try await project.categorize(using: modelURL, progress: onProgress)
            guard let self, self.generation == current else { return }
            self.events.append(TrainingEvent(type: "summary", message:
                "Categorized \(summary.folderCount.formatted()) folders with \(summary.predictionCount.formatted()) classifier predictions; reused \(summary.reusedFolderCount.formatted()) existing folder categories."))
            if !summary.conflictingFolders.isEmpty {
                let names = summary.conflictingFolders.map { ImageMetadataStore.friendlyName($0.lastPathComponent) }.joined(separator: ", ")
                self.events.append(TrainingEvent(type: "review", message:
                    "Review folders with different existing categories: \(names). Existing tags were preserved."))
            }
        }
    }

    func exportPackage() {
        guard !busy, project != nil else { return }
        let value = settings
        let current = generation
        perform("Exporting Texture Loom", clearEvents: true) { [weak self] project in
            try await project.updateSettings(value)
            let url = try await project.exportPackage(name: value.modelName, version: value.modelVersion)
            guard let self, self.generation == current else { return }
            self.exportedPackage = url
            NSWorkspace.shared.activateFileViewerSelecting([url])
        }
    }

    func publishPackage() {
        guard let package = exportedPackage, !busy else { return }
        let repository = settings.huggingFaceRepository.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repository.isEmpty else { return }
        let current = generation
        let onProgress = progressHandler()
        perform("Publishing Texture Loom", clearEvents: true) { [weak self] project in
            let url = try await project.publishPackage(package: package, repository: repository, progress: onProgress)
            guard let self, self.generation == current else { return }
            self.publishedURL = url
        }
    }

    func cancel() {
        guard let project, busy else { return }
        operation?.cancel()
        Task { await project.cancelTraining() }
        activityTitle = "Stopping safely…"
    }

    func revealSource() {
        if let url = snapshot?.sourceURL { NSWorkspace.shared.open(url) }
    }

    func reveal(_ url: URL) { NSWorkspace.shared.activateFileViewerSelecting([url]) }

    private func perform(_ title: String, clearEvents: Bool = false,
                         action: @escaping @MainActor (LoomProject) async throws -> Void) {
        guard let project, !busy else { return }
        saveOperation?.cancel()
        let current = generation
        busy = true
        activityTitle = title
        progress = nil
        errorMessage = nil
        if clearEvents { events = [] }
        operation = Task {
            do {
                try Task.checkCancellation()
                try await action(project)
                let value = try await project.snapshot()
                guard generation == current else { return }
                adopt(value, restoreSettings: false)
                busy = false
                activityTitle = Task.isCancelled ? "Stopped" : "\(title) complete"
                progress = Task.isCancelled ? nil : 1
            } catch {
                guard generation == current else { return }
                if Task.isCancelled || error is CancellationError {
                    busy = false
                    activityTitle = "Stopped"
                    progress = nil
                    if let value = try? await project.snapshot() { adopt(value, restoreSettings: false) }
                } else {
                    let failure = error
                    do { adopt(try await project.snapshot(), restoreSettings: false) }
                    catch { invalidateSnapshot(error) }
                    fail(failure)
                }
            }
        }
    }

    private func progressHandler() -> @Sendable (TrainingEvent) -> Void {
        let current = generation
        return { [weak self] event in
            Task { @MainActor [weak self] in
                guard let self, self.generation == current else { return }
                self.events.append(event)
                if self.events.count > 300 { self.events.removeFirst(self.events.count - 300) }
                if self.operation?.isCancelled != true { self.activityTitle = event.message }
                if let progress = event.progress { self.progress = progress }
            }
        }
    }

    private func adopt(_ value: ProjectSnapshot, restoreSettings: Bool) {
        snapshot = value
        sourceIssue = nil
        if restoreSettings { settings = value.settings }
        if !selectedDatasetID.isEmpty, selectedDatasetID != "unassigned",
           !value.datasets.contains(where: { $0.id == selectedDatasetID }) { selectedDatasetID = "" }
        selectVisibleImageIfNeeded()
        if let url = exportedPackage, !FileManager.default.fileExists(atPath: url.path) { exportedPackage = nil }
        if exportedPackage == nil {
            let exports = value.sourceURL.appendingPathComponent(".loom-factory/exports", isDirectory: true)
            exportedPackage = (try? FileManager.default.contentsOfDirectory(at: exports,
                includingPropertiesForKeys: [.contentModificationDateKey], options: [.skipsHiddenFiles]))?
                .filter { $0.pathExtension == "textureloom" }
                .sorted {
                    let left = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                    let right = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                    return left > right
                }.first
        }
    }

    private func fail(_ error: Error) {
        busy = false
        progress = nil
        activityTitle = "Action could not finish"
        errorMessage = error.localizedDescription
    }

    private func invalidateSnapshot(_ error: Error) {
        snapshot = nil
        selectedImageID = nil
        sourceIssue = error.localizedDescription
    }

}
