import AppKit
import Foundation
import LoomCore
import Observation
import UniformTypeIdentifiers

enum FactoryTrainingStartMode: String, CaseIterable, Identifiable {
    case fresh, exportedModel, checkpoint
    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .fresh: "Replace with a fresh model"
        case .exportedModel: "Continue exported model weights"
        case .checkpoint: "Continue final checkpoint"
        }
    }
}

@MainActor @Observable
final class FactoryModel {
    private(set) var snapshot: ProjectSnapshot?
    var settings = ProjectSettings()
    var selectedDatasetID = "" {
        didSet {
            if selectedDatasetID != oldValue {
                selectedTrainingVersionID = ""; invalidateComparisonSelection(); resetModelTestSelection()
            }
        }
    }
    var selectedImageID: String?
    var selectedMap: MapRole = .depth {
        didSet {
            if selectedMap != oldValue {
                selectedTrainingVersionID = ""; invalidateComparisonSelection(); resetModelTestSelection()
            }
        }
    }
    var trainingStartMode: FactoryTrainingStartMode = .checkpoint
    var selectedTrainingVersionID = ""
    var runLabel = ""
    var comparisonFirstVersionID = "" {
        didSet { if comparisonFirstVersionID != oldValue { invalidateComparisonSelection() } }
    }
    var comparisonSecondVersionID = "" {
        didSet { if comparisonSecondVersionID != oldValue { invalidateComparisonSelection() } }
    }
    var comparisonPairID = "" {
        didSet { if comparisonPairID != oldValue { comparisonResult = nil } }
    }
    private(set) var comparisonPairs: [ModelComparisonPair] = []
    private(set) var comparisonResult: ModelComparisonResult?
    private(set) var comparisonIssue: String?
    private(set) var loadingComparisonPairs = false
    var selectedTestVersionID = "" {
        didSet { if selectedTestVersionID != oldValue { modelTestResult = nil; testWeightSource = nil } }
    }
    var testWeightSource: TrainingWeightSource? {
        didSet { if testWeightSource != oldValue { modelTestResult = nil } }
    }
    private(set) var testInputURL: URL?
    private(set) var modelTestResult: ModelTestResult?
    var imageSearch = ""
    var fileFilter: MaterialBrowserFilter = .all
    var workspaceSection: LoomWorkspaceSection = .materials
    private(set) var busy = false
    private(set) var activityTitle = ""
    private(set) var progress: Double?
    private(set) var events: [TrainingEvent] = []
    private(set) var trainingProgress: TrainingProgress?
    private(set) var exportedPackage: URL?
    private(set) var publishedURL: URL?
    private(set) var sourceIssue: String?
    var errorMessage: String?

    @ObservationIgnored private var project: LoomProject?
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var saveOperation: Task<Void, Never>?
    @ObservationIgnored private var generation = UUID()
    @ObservationIgnored private var polling = false
    @ObservationIgnored private var loadedComparisonKey = ""
    @ObservationIgnored private var packageNeedsExport = false
    @ObservationIgnored private var comparisonImageRevisions: [String: String] = [:]
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
    var selectedModelVersions: [ModelVersion] {
        snapshot?.modelVersions.filter { $0.datasetID == selectedDatasetID && $0.map == selectedMap } ?? []
    }
    var hasTrainingRuns: Bool { snapshot?.modelVersions.isEmpty == false }
    var trainingRunCount: Int { snapshot?.modelVersions.count ?? 0 }
    var exportedModelVersions: [ModelVersion] {
        selectedModelVersions.filter { $0.artifact?.weightsPath != nil }
    }
    var activeMapVersions: [ModelVersion] { snapshot?.modelVersions.filter { $0.isActive && $0.artifact != nil } ?? [] }
    var continuableModelVersions: [ModelVersion] {
        effectiveStartMode == .exportedModel ? exportedModelVersions : selectedModelVersions.filter { $0.checkpoint != nil }
    }
    var effectiveStartMode: FactoryTrainingStartMode {
        settings.training.continueLatestWeights == true ? trainingStartMode : .fresh
    }
    var selectedTrainingVersion: ModelVersion? {
        let versions = continuableModelVersions
        if !selectedTrainingVersionID.isEmpty { return versions.first { $0.id == selectedTrainingVersionID } }
        if effectiveStartMode == .exportedModel, let active = versions.first(where: \.isActive) { return active }
        return versions.first
    }
    var continuingTraining: Bool { effectiveStartMode != .fresh }
    var trainingStart: TrainingStart? {
        guard continuingTraining else { return .fresh }
        guard let version = selectedTrainingVersion else {
            if effectiveStartMode == .checkpoint, selectedTrainingVersionID.isEmpty, let checkpoint = selectedCheckpoint {
                return .checkpoint(runID: checkpoint.directory.lastPathComponent)
            }
            return nil
        }
        return effectiveStartMode == .exportedModel ? .model(runID: version.id) : .checkpoint(runID: version.id)
    }
    var firstComparisonVersion: ModelVersion? {
        exportedModelVersions.first { $0.id == comparisonFirstVersionID } ??
            exportedModelVersions.first(where: \.isActive) ?? exportedModelVersions.first
    }
    var secondComparisonVersion: ModelVersion? {
        let remaining = exportedModelVersions.filter { $0.id != firstComparisonVersion?.id }
        return remaining.first { $0.id == comparisonSecondVersionID } ?? remaining.first
    }
    var comparisonSelectionKey: String {
        var membership = Hasher()
        for image in snapshot?.images ?? [] where image.category == selectedDataset?.name {
            membership.combine(image.id); membership.combine(image.url); membership.combine(image.role)
            membership.combine(image.pairingKey); membership.combine(image.ignored)
            membership.combine(image.width); membership.combine(image.height)
            membership.combine(comparisonImageRevisions[image.id])
        }
        return [selectedDatasetID, selectedMap.rawValue, firstComparisonVersion?.id ?? "", secondComparisonVersion?.id ?? "",
                String(membership.finalize())].joined(separator: "|")
    }
    var selectedComparisonPair: ModelComparisonPair? {
        comparisonPairs.first { $0.id == comparisonPairID } ?? comparisonPairs.first
    }
    var canCompareModels: Bool {
        !busy && !loadingComparisonPairs && loadedComparisonKey == comparisonSelectionKey &&
        firstComparisonVersion?.artifact != nil && secondComparisonVersion?.artifact != nil && selectedComparisonPair != nil
    }
    var testableModelVersions: [ModelVersion] {
        snapshot?.modelVersions.filter { $0.artifact != nil || $0.checkpoint != nil } ?? []
    }
    var selectedTestVersion: ModelVersion? {
        if !selectedTestVersionID.isEmpty { return testableModelVersions.first { $0.id == selectedTestVersionID } }
        return testableModelVersions.first { $0.datasetID == selectedDatasetID && $0.map == selectedMap && $0.isActive } ??
            testableModelVersions.first { $0.datasetID == selectedDatasetID && $0.map == selectedMap } ??
            testableModelVersions.first(where: \.isActive) ?? testableModelVersions.first
    }
    var availableTestWeightSources: [TrainingWeightSource] {
        guard let version = selectedTestVersion else { return [] }
        return (version.artifact == nil ? [] : [.model]) + (version.checkpoint == nil ? [] : [.checkpoint])
    }
    var effectiveTestWeightSource: TrainingWeightSource? {
        testWeightSource ?? availableTestWeightSources.first
    }
    var testDiffuseImages: [SourceImage] {
        let dataset = selectedTestVersion?.datasetName ?? selectedDataset?.name
        return snapshot?.images.filter { $0.role == .diffuse && !$0.ignored &&
            (dataset == nil || $0.category == dataset) } ?? []
    }
    var modelTestIssue: String? {
        if !selectedTestVersionID.isEmpty, selectedTestVersion == nil {
            return "The selected run is no longer available. Choose another saved run."
        }
        guard selectedTestVersion != nil else { return "Train a model or open a source folder with a saved model or checkpoint." }
        if let source = effectiveTestWeightSource, !availableTestWeightSources.contains(source) {
            return "These saved weights are no longer available. Choose exported model weights or a checkpoint from this run."
        }
        guard let input = testInputURL else { return "Choose a diffuse image to see the model's output." }
        if !FileManager.default.fileExists(atPath: input.path) { return "The selected diffuse image is no longer available. Choose an image again." }
        return nil
    }
    var canTestModel: Bool { project != nil && !busy && modelTestIssue == nil && effectiveTestWeightSource != nil }
    var eligiblePairCount: Int {
        trainingPlan?.totalPairCount ?? 0
    }
    var selectedCheckpoint: TrainingCheckpoint? {
        guard effectiveStartMode != .exportedModel else { return nil }
        if !selectedTrainingVersionID.isEmpty { return selectedTrainingVersion?.checkpoint }
        if let checkpoint = selectedTrainingVersion?.checkpoint { return checkpoint }
        return snapshot?.checkpoints.first { $0.datasetID == selectedDatasetID && $0.map == selectedMap }
    }
    var effectiveTrainingSettings: TrainingSettings {
        var value = settings.training
        if continuingTraining {
            if effectiveStartMode == .exportedModel, let saved = selectedTrainingVersion?.settings {
                value.modelArchitecture = saved.modelArchitecture
                value.modelWidth = saved.modelWidth; value.modelBlocks = saved.modelBlocks
                value.imageSize = saved.imageSize; value.exaggerateFeatures = saved.exaggerateFeatures
            } else if effectiveStartMode == .checkpoint, let checkpoint = selectedCheckpoint {
                value.modelArchitecture = checkpoint.modelArchitecture
                value.modelWidth = checkpoint.modelWidth; value.modelBlocks = checkpoint.modelBlocks
                value.imageSize = checkpoint.imageSize; value.exaggerateFeatures = checkpoint.exaggerateFeatures
            }
        }
        return value
    }
    var trainingOptionsIssue: String? {
        if continuingTraining && !selectedTrainingVersionID.isEmpty && selectedTrainingVersion == nil {
            return "The selected saved version is no longer available. Choose another version or replace with a fresh model."
        }
        if effectiveStartMode == .exportedModel, let version = selectedTrainingVersion, version.settings == nil {
            return "This saved model has no readable architecture and training settings. Choose another version or replace with a fresh model."
        }
        if continuingTraining && !canContinueMap { return "Choose saved weights for this dataset and map, or replace with a fresh model." }
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
        if effectiveStartMode == .exportedModel { return selectedTrainingVersion?.artifact?.weightsPath != nil && selectedTrainingVersion?.settings != nil }
        return selectedCheckpoint != nil
    }
    var canTrain: Bool {
        project != nil && selectedDataset != nil && trainingOptionsIssue == nil && (trainingPlan?.trainingPairCount ?? 0) > 0 && !busy &&
        (!continuingTraining || canContinueMap)
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
        trainingStartMode = .checkpoint
        selectedTrainingVersionID = ""
        runLabel = ""
        comparisonFirstVersionID = ""
        comparisonSecondVersionID = ""
        comparisonPairID = ""
        comparisonPairs = []
        comparisonResult = nil
        comparisonIssue = nil
        loadingComparisonPairs = false
        loadedComparisonKey = ""
        packageNeedsExport = false
        comparisonImageRevisions = [:]
        resetModelTestSelection()
        testInputURL = nil
        imageSearch = ""
        fileFilter = .all
        workspaceSection = .materials
        settings = ProjectSettings()
        exportedPackage = nil
        publishedURL = nil
        events = []
        trainingProgress = nil
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
        let start = trainingStart
        let label = runLabel
        let onProgress = progressHandler()
        let current = generation
        perform("Training \(dataset.name) · \(map.displayName)", clearEvents: true) { [weak self] project in
            try await project.updateSettings(value)
            _ = try await project.trainMap(datasetID: dataset.id, map: map, settings: configuredTraining,
                                          start: start, runLabel: label, progress: onProgress)
            guard let self, self.generation == current else { return }
            self.invalidateExportedPackage()
        }
    }

    func setTrainingStartMode(_ mode: FactoryTrainingStartMode) {
        if mode == .fresh { settings.training.usePyramidForFreshRun(map: selectedMap) }
        trainingStartMode = mode == .fresh ? .checkpoint : mode
        settings.training.continueLatestWeights = mode != .fresh
        selectedTrainingVersionID = ""
    }

    func continueModelVersion(_ version: ModelVersion, source: FactoryTrainingStartMode) {
        setTrainingStartMode(source)
        selectedTrainingVersionID = version.id
    }

    func useModelVersionForExport(_ version: ModelVersion) {
        guard version.artifact != nil else { return }
        let current = generation
        perform("Selecting model for export") { [weak self] project in
            try await project.selectModelVersion(runID: version.id, datasetID: version.datasetID, map: version.map)
            guard let self, self.generation == current else { return }
            self.invalidateExportedPackage()
        }
    }

    func confirmDeleteTrainingRun(_ version: ModelVersion) {
        guard !busy else { return }
        let alert = NSAlert()
        alert.messageText = "Delete this training run?"
        alert.informativeText = "\(ModelVersionPresentation.title(version)) will move to Trash, including its saved model, checkpoint and history. Original images and exported packages remain available."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Move to Trash")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        deleteTrainingRuns([version.id])
    }

    func confirmDeleteAllTrainingRuns() {
        guard !busy, hasTrainingRuns else { return }
        let alert = NSAlert()
        alert.messageText = "Delete all \(trainingRunCount.formatted()) training runs?"
        alert.informativeText = "Every training run in this source folder will move to Trash, across all datasets and map types. This includes saved models, checkpoints and histories. Original images, the classifier and exported packages remain available."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Move All to Trash")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        deleteTrainingRuns(snapshot?.modelVersions.map(\.id) ?? [])
    }

    func deleteTrainingRuns(_ runIDs: [String]) {
        guard !busy, !runIDs.isEmpty else { return }
        let current = generation
        perform("Deleting training runs", clearEvents: true) { [weak self] project in
            let result = try await project.deleteTrainingRuns(runIDs: runIDs)
            guard let self, self.generation == current else { return }
            if result.runIDs.contains(self.selectedTrainingVersionID) { self.selectedTrainingVersionID = "" }
            if result.runIDs.contains(self.comparisonFirstVersionID) { self.comparisonFirstVersionID = "" }
            if result.runIDs.contains(self.comparisonSecondVersionID) { self.comparisonSecondVersionID = "" }
            self.invalidateComparisonSelection()
            if result.runIDs.contains(self.selectedTestVersionID) { self.resetModelTestSelection() }
            self.invalidateExportedPackage()
            if self.snapshot?.modelVersions.contains(where: { !result.runIDs.contains($0.id) &&
                $0.datasetID == self.selectedDatasetID && $0.map == self.selectedMap }) != true {
                self.setTrainingStartMode(.fresh)
                try await project.updateSettings(self.settings)
            }
            self.events.append(TrainingEvent(type: "summary", message:
                "Moved \(result.count.formatted()) training \(result.count == 1 ? "run" : "runs") to Trash."))
        }
    }

    func loadComparisonPairs() async {
        let key = comparisonSelectionKey
        guard key != loadedComparisonKey else { return }
        comparisonResult = nil
        comparisonIssue = nil
        comparisonPairs = []
        guard let project, let first = firstComparisonVersion?.artifact, let second = secondComparisonVersion?.artifact else {
            loadedComparisonKey = key
            loadingComparisonPairs = false
            return
        }
        let current = generation
        loadingComparisonPairs = true
        do {
            let pairs = try await project.comparePairOptions(datasetID: selectedDatasetID, map: selectedMap,
                                                             artifactIDs: [first.id, second.id])
            guard !Task.isCancelled, current == generation, key == comparisonSelectionKey else { return }
            comparisonPairs = pairs
            if !pairs.contains(where: { $0.id == comparisonPairID }) { comparisonPairID = pairs.first?.id ?? "" }
            if pairs.isEmpty { comparisonIssue = "No current paired material can supply a common crop for these versions." }
            loadedComparisonKey = key
            loadingComparisonPairs = false
        } catch {
            guard !Task.isCancelled, current == generation, key == comparisonSelectionKey else { return }
            comparisonIssue = error.localizedDescription
            loadedComparisonKey = key
            loadingComparisonPairs = false
        }
    }

    func compareModels() {
        guard canCompareModels, let first = firstComparisonVersion?.artifact, let second = secondComparisonVersion?.artifact,
              let pair = selectedComparisonPair else { return }
        let current = generation
        comparisonResult = nil
        let onProgress = progressHandler()
        perform("Comparing saved models", clearEvents: true) { [weak self] project in
            let result = try await project.compareModels(firstArtifactID: first.id, secondArtifactID: second.id,
                                                        pairID: pair.id, progress: onProgress)
            guard let self, self.generation == current else { return }
            self.comparisonResult = result
        }
    }

    private func invalidateComparisonSelection() {
        comparisonResult = nil
        comparisonPairs = []
        comparisonIssue = nil
        loadedComparisonKey = ""
    }

    func showModelTesting(_ version: ModelVersion? = nil) {
        guard !busy else { return }
        if let version {
            selectedTestVersionID = version.id
        } else if let version = selectedTestVersion {
            selectedTestVersionID = version.id
        }
        if testInputURL == nil, let image = selectedImage, image.role == .diffuse { setModelTestInput(image.url) }
        workspaceSection = .testing
    }

    func chooseModelTestInput() {
        guard !busy else { return }
        let panel = NSOpenPanel()
        panel.title = "Choose a diffuse image to test"
        panel.message = "Run the saved model on any diffuse image, including a material outside this source folder."
        panel.prompt = "Use Image"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.image]
        panel.directoryURL = testInputURL?.deletingLastPathComponent() ?? sourceFolder
        guard panel.runModal() == .OK, let url = panel.url else { return }
        setModelTestInput(url)
    }

    func setModelTestInput(_ url: URL) {
        guard !busy else { return }
        testInputURL = url
        modelTestResult = nil
    }

    func testModel() {
        guard canTestModel, let version = selectedTestVersion, let inputURL = testInputURL,
              let source = effectiveTestWeightSource else { return }
        selectedTestVersionID = version.id
        testWeightSource = source
        let current = generation
        let onProgress = progressHandler()
        modelTestResult = nil
        perform("Testing \(version.datasetName) · \(version.map.displayName)", clearEvents: true) { [weak self] project in
            let result = try await project.testModel(datasetID: version.datasetID, map: version.map, runID: version.id,
                                                     source: source, inputURL: inputURL, progress: onProgress)
            guard let self, self.generation == current else { return }
            self.modelTestResult = result
        }
    }

    func saveModelTestOutput() {
        guard let result = modelTestResult, !busy else { return }
        let panel = NSSavePanel()
        panel.title = "Save model output"
        panel.prompt = "Save Output"
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "\(result.inputURL.deletingPathExtension().lastPathComponent)_\(result.map.rawValue).png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try ModelTestOutputExport.save(result, to: url, sourceImageURLs: snapshot?.images.map(\.url) ?? [])
        } catch { errorMessage = error.localizedDescription }
    }

    private func resetModelTestSelection() {
        selectedTestVersionID = ""
        testWeightSource = nil
        modelTestResult = nil
    }

    private func invalidateExportedPackage() {
        exportedPackage = nil
        publishedURL = nil
        packageNeedsExport = true
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
            self.packageNeedsExport = false
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
        trainingProgress = nil
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
                if let value = event.trainingProgress { self.trainingProgress = value }
                if self.events.count > 300 { self.events.removeFirst(self.events.count - 300) }
                if self.operation?.isCancelled != true { self.activityTitle = event.message }
                if let progress = event.progress { self.progress = progress }
            }
        }
    }

    private func adopt(_ value: ProjectSnapshot, restoreSettings: Bool) {
        let previousModels = Set(snapshot?.modelVersions.filter { $0.isActive }.map(\.id) ?? [])
        let currentModels = Set(value.modelVersions.filter { $0.isActive }.map(\.id))
        if snapshot != nil, previousModels != currentModels { invalidateExportedPackage() }
        comparisonImageRevisions = Dictionary(uniqueKeysWithValues: value.images.map { image in
            let attributes = try? image.url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
            return (image.id, "\(attributes?.contentModificationDate?.timeIntervalSince1970 ?? 0)|\(attributes?.fileSize ?? 0)")
        })
        snapshot = value
        sourceIssue = nil
        if restoreSettings { settings = value.settings }
        if !selectedDatasetID.isEmpty, selectedDatasetID != "unassigned",
           !value.datasets.contains(where: { $0.id == selectedDatasetID }) { selectedDatasetID = "" }
        selectVisibleImageIfNeeded()
        if let url = exportedPackage, !FileManager.default.fileExists(atPath: url.path) { exportedPackage = nil }
        if exportedPackage == nil && !packageNeedsExport {
            let exports = value.sourceURL.appendingPathComponent(".loom-factory/exports", isDirectory: true)
            let selectionDate = (try? value.sourceURL.appendingPathComponent(".loom-factory/model-selection.json")
                .resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            exportedPackage = (try? FileManager.default.contentsOfDirectory(at: exports,
                includingPropertiesForKeys: [.contentModificationDateKey], options: [.skipsHiddenFiles]))?
                .filter {
                    $0.pathExtension == "textureloom" &&
                    ((try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast) >= selectionDate
                }
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
