import AppKit
import Foundation
import LoomCore
import LoomNative
import Observation
import UniformTypeIdentifiers

enum FactoryTrainingStartMode: String, CaseIterable, Identifiable {
    case fresh, exportedModel, checkpoint
    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .fresh: "Start a fresh model"
        case .exportedModel: "Continue exported model weights"
        case .checkpoint: "Continue saved checkpoint"
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
                restoreSelectedTrainingSettings()
            }
        }
    }
    var selectedImageID: String?
    var selectedMaterialBrowserRowID: String?
    var selectedMap: MapRole = .depth {
        didSet {
            if selectedMap != oldValue {
                selectedTrainingVersionID = ""; invalidateComparisonSelection(); resetModelTestSelection()
                restoreSelectedTrainingSettings()
            }
        }
    }
    var trainingStartMode: FactoryTrainingStartMode = .checkpoint
    var trainingScalarLossOverride: NativeScalarLoss?
    var selectedTrainingVersionID = "" {
        didSet {
            if selectedTrainingVersionID != oldValue { restoreSelectedTrainingSettings() }
        }
    }
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
        didSet {
            if selectedTestVersionID != oldValue {
                modelTestResult = nil; previousTestResult = nil; testWeightSource = nil; testCheckpointID = nil
                refreshTestOriginalMap()
            }
        }
    }
    var testWeightSource: TrainingWeightSource? {
        didSet { if testWeightSource != oldValue { preserveTestResultForComparison(); modelTestResult = nil; testCheckpointID = nil } }
    }
    private var modelTestDatasetID: String?
    private var modelTestMap: MapRole?
    private(set) var testCheckpointID: String?
    private(set) var testInputURL: URL?
    private(set) var modelTestResult: ModelTestResult?
    private(set) var previousTestResult: ModelTestResult?
    private(set) var testOriginalMap = ModelTestOriginalMap.resolve(inputURL: nil, map: nil, images: [])
    var imageSearch = ""
    var fileFilter: MaterialBrowserFilter = .all
    var workspaceSection: LoomWorkspaceSection = .materials
    private(set) var busy = false
    private(set) var activityTitle = ""
    private(set) var progress: Double?
    private(set) var events: [TrainingEvent] = []
    private(set) var latestTrainingMeasurements: TrainingEvent?
    var measurementEvents: [TrainingEvent] { latestTrainingMeasurements.map { [$0] } ?? [] }
    private(set) var trainingProgress: TrainingProgress?
    private(set) var isTraining = false
    private(set) var checkpointRequestPending = false
    private(set) var checkpointStatus: String?
    private(set) var lastCheckpointDirectory: URL?
    private(set) var exportedPackage: URL?
    private(set) var publishedURL: URL?
    private(set) var sourceIssue: String?
    var errorMessage: String?

    @ObservationIgnored private var project: LoomProject?
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var saveOperation: Task<Void, Never>?
    @ObservationIgnored private var checkpointOperation: Task<Bool, Never>?
    @ObservationIgnored private var originalMapOperation: Task<Void, Never>?
    @ObservationIgnored private var checkpointWillStop = false
    @ObservationIgnored private var generation = UUID()
    @ObservationIgnored private var polling = false
    @ObservationIgnored private var loadedComparisonKey = ""
    @ObservationIgnored private var packageNeedsExport = false
    @ObservationIgnored private var comparisonImageRevisions: [String: String] = [:]
    private(set) var previewRevisions: [String: String] = [:]
    @ObservationIgnored private var browserIndex: FactoryBrowserIndex?
    @ObservationIgnored private var browserProjection: FactoryBrowserProjection?
    @ObservationIgnored private var cachedTrainingPlan: FactoryTrainingPlanCache?
    @ObservationIgnored private var presentationRevision = 0
    @ObservationIgnored private var progressDeliverySerial: UInt64 = 0
    @ObservationIgnored private var adoptionSerial: UInt64 = 0
    @ObservationIgnored private let rememberSourceFolder: Bool

    var sourceFolder: URL? { project?.sourceFolder }

    var selectedDataset: DatasetCategory? {
        snapshot?.datasets.first { $0.id == selectedDatasetID }
    }

    var matchingImages: [SourceImage] {
        currentBrowserProjection()?.matching ?? []
    }

    var visibleImages: [SourceImage] { currentBrowserProjection()?.visible ?? [] }

    var materialFolders: [SourceMaterialFolder] {
        guard snapshot != nil else { return [] }
        return browserIndex?.folders ?? []
    }
    var materialBrowserIndex: MaterialBrowserIndex {
        guard snapshot != nil else { return .init() }
        return browserIndex?.materialBrowserIndex ?? .init()
    }

    var visibleMaterialFolders: [SourceMaterialFolder] {
        currentBrowserProjection()?.folders ?? []
    }

    var selectedImage: SourceImage? {
        guard snapshot != nil else { return nil }
        return selectedImageID.flatMap { browserIndex?.imagesByID[$0] }
    }

    func imagePreviewRevision(_ image: SourceImage) -> String {
        image.previewRevision + "|" + (previewRevisions[image.id] ?? "")
    }

    private func currentBrowserProjection() -> FactoryBrowserProjection? {
        guard let snapshot, let browserIndex else { return nil }
        if let value = browserProjection, value.revision == presentationRevision,
           value.datasetID == selectedDatasetID, value.search == imageSearch, value.filter == fileFilter { return value }
        let matching = browserIndex.matching(datasetID: selectedDatasetID, datasetName: selectedDataset?.name, search: imageSearch)
        let visible = matching.filter(fileFilter.includes)
        let folders = selectedDatasetID.isEmpty && imageSearch.isEmpty && fileFilter == .all ? browserIndex.folders :
            SourceMaterialFolder.group(images: visible, sourceFolder: snapshot.sourceURL)
        let value = FactoryBrowserProjection(revision: presentationRevision, datasetID: selectedDatasetID,
            search: imageSearch, filter: fileFilter, matching: matching, visible: visible, folders: folders)
        browserProjection = value
        return value
    }

    var modelCoverage: [ModelCoverageEntry] { snapshot?.modelCoverage ?? [] }
    var classifier: ModelArtifact? {
        if let entry = modelCoverage.first(where: { $0.slot.product == .classifier }) {
            return entry.candidates.first { $0.id == entry.effectiveRunID && $0.isAvailable }?.artifact
        }
        return snapshot?.artifacts.first { $0.kind == "classifier" }
    }
    var requiredModelCount: Int { modelCoverage.filter(\.isRequired).count }
    var availableRequiredModelCount: Int { modelCoverage.filter { $0.isRequired && $0.availableCount > 0 }.count }
    var approvedAssignmentCount: Int { modelCoverage.filter { $0.isRequired && $0.status == .approved }.count }
    var hasIncompleteCoverage: Bool {
        modelCoverage.contains { entry in
            entry.isRequired && entry.candidates.first(where: { $0.id == entry.effectiveRunID })?.isAvailable != true
        }
    }
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
    var testDatasetID: String {
        modelTestDatasetID ?? (!selectedDatasetID.isEmpty ? selectedDatasetID :
            snapshot?.modelVersions.first?.datasetID ?? snapshot?.datasets.first?.id ?? "")
    }
    var testMap: MapRole { modelTestMap ?? selectedMap }
    var testableModelVersions: [ModelVersion] {
        snapshot?.modelVersions.filter { $0.datasetID == testDatasetID && $0.map == testMap &&
            ($0.artifact != nil || !$0.checkpointVersions.isEmpty) } ?? []
    }
    var selectedTestVersion: ModelVersion? {
        if !selectedTestVersionID.isEmpty { return testableModelVersions.first { $0.id == selectedTestVersionID } }
        return testableModelVersions.first(where: \.isActive) ?? testableModelVersions.first
    }
    var testCheckpointVersions: [TrainingCheckpoint] { selectedTestVersion?.checkpointVersions ?? [] }
    var selectedTestCheckpoint: TrainingCheckpoint? {
        if let testCheckpointID { return testCheckpointVersions.first { $0.revisionID == testCheckpointID } }
        return selectedTestVersion?.checkpoint ?? testCheckpointVersions.first
    }
    var availableTestWeightSources: [TrainingWeightSource] {
        guard let version = selectedTestVersion else { return [] }
        return (version.artifact == nil ? [] : [.model]) + (version.checkpointVersions.isEmpty ? [] : [.checkpoint])
    }
    var effectiveTestWeightSource: TrainingWeightSource? {
        testWeightSource ?? availableTestWeightSources.first
    }
    var testDiffuseImages: [SourceImage] {
        guard let category = snapshot?.datasets.first(where: { $0.id == testDatasetID })?.name else { return [] }
        return snapshot?.images.filter { $0.role == .diffuse && $0.category == category && !$0.ignored } ?? []
    }
    var modelTestIssue: String? {
        if !selectedTestVersionID.isEmpty, selectedTestVersion == nil {
            return "The selected run is no longer available. Choose another saved run."
        }
        guard selectedTestVersion != nil else { return "Train a model or open a source folder with a saved model or checkpoint." }
        if let source = effectiveTestWeightSource, !availableTestWeightSources.contains(source) {
            return "These saved weights are no longer available. Choose exported model weights or a checkpoint from this run."
        }
        if effectiveTestWeightSource == .checkpoint, selectedTestCheckpoint == nil {
            return "The selected checkpoint is no longer available. Choose another saved checkpoint."
        }
        guard testInputURL != nil else { return "Choose a diffuse image to see the model's output." }
        return nil
    }
    var canTestModel: Bool { project != nil && !busy && modelTestIssue == nil && effectiveTestWeightSource != nil }
    var eligiblePairCount: Int {
        trainingPlan?.totalPairCount ?? 0
    }
    var selectedDatasetSourceCounts: (folders: Int, images: Int) {
        guard let dataset = selectedDataset else { return (0, 0) }
        let images = snapshot?.images.filter { $0.category == dataset.name } ?? []
        return (Set(images.map { $0.url.deletingLastPathComponent().standardizedFileURL }).count, images.count)
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
                value.scalarLoss = saved.scalarLoss
            } else if effectiveStartMode == .checkpoint, let checkpoint = selectedCheckpoint {
                value.modelArchitecture = checkpoint.modelArchitecture
                value.modelWidth = checkpoint.modelWidth; value.modelBlocks = checkpoint.modelBlocks
                value.scalarLoss = checkpoint.scalarLoss
            }
        }
        if value.modelArchitecture == NativeArchitecture.compactIdentifier {
            if let trainingScalarLossOverride, trainingScalarLossOverride.isCompactObjective {
                value.scalarLoss = trainingScalarLossOverride
            } else if !value.scalarLoss.isCompactObjective { value.scalarLoss = .linearDetailV1 }
        } else if selectedMap != .normal, let trainingScalarLossOverride {
            value.scalarLoss = trainingScalarLossOverride
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
        if let plan = trainingPlan,
           effectiveTrainingSettings.earlyStoppingPatience > 0, plan.validationPairCount == 0 {
            return "Early stopping needs held-out pairs from another material folder. Disable early stopping or add another tagged material folder."
        }
        if let issue = currentTrainingPlan().issue { return issue }
        return nil
    }
    var trainingPlan: TrainingDataPlan? {
        currentTrainingPlan().plan
    }
    private func currentTrainingPlan() -> FactoryTrainingPlanCache {
        let configured = effectiveTrainingSettings
        if let value = cachedTrainingPlan, value.revision == presentationRevision,
           value.datasetID == selectedDatasetID, value.map == selectedMap, value.settings == configured { return value }
        let plan: TrainingDataPlan?, issue: String?
        do { plan = try snapshot?.trainingPlan(datasetID: selectedDatasetID, map: selectedMap, settings: configured); issue = nil }
        catch { plan = nil; issue = error.localizedDescription }
        let value = FactoryTrainingPlanCache(revision: presentationRevision, datasetID: selectedDatasetID,
            map: selectedMap, settings: configured, plan: plan, issue: issue)
        cachedTrainingPlan = value
        return value
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
        guard !busy else {
            errorMessage = isTraining ? "Save & Stop this training run before opening another source folder." : "Wait for the current action to finish before opening another source folder."
            return
        }
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
        guard !busy else {
            errorMessage = isTraining ? "Save & Stop this training run before opening another source folder." : "Wait for the current action to finish before opening another source folder."
            return
        }
        operation?.cancel()
        originalMapOperation?.cancel()
        saveOperation?.cancel()
        let oldProject = project
        generation = UUID()
        let current = generation
        project = nil
        snapshot = nil
        selectedImageID = nil
        selectedMaterialBrowserRowID = nil
        browserIndex = nil; browserProjection = nil; cachedTrainingPlan = nil
        previewRevisions = [:]
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
        latestTrainingMeasurements = nil
        trainingProgress = nil
        isTraining = false; checkpointRequestPending = false; checkpointStatus = nil; lastCheckpointDirectory = nil
        errorMessage = nil
        sourceIssue = nil
        busy = true
        activityTitle = "Opening source folder"
        progress = nil
        operation = Task {
            if let oldProject { await oldProject.cancelTraining() }
            do {
                let opened = try await Task.detached(priority: .userInitiated) { try LoomProject(sourceFolder: url) }.value
                let value = try await opened.snapshot()
                guard current == generation, !Task.isCancelled else { return }
                project = opened
                await adopt(value, restoreSettings: true)
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
            await adopt(value, restoreSettings: false)
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

    func setMaterialReviewed(imageIDs: [String], verified: Bool) {
        perform(verified ? "Marking material reviewed" : "Marking material unreviewed") { project in
            try await project.setMaterialReviewed(imageIDs: imageIDs, verified: verified)
        }
    }

    func selectVisibleImageIfNeeded() {
        let folders = visibleMaterialFolders
        let members = folders.flatMap { $0.resolutions.flatMap(\.images) }
        guard !members.contains(where: { $0.id == selectedImageID }) else { return }
        if let row = MaterialBrowserSelection.row(id: selectedMaterialBrowserRowID, in: folders),
           let replacement = MaterialBrowserSelection.representative(in: row.images) {
            selectedImageID = replacement.id
            return
        }
        selectedImageID = members.first(where: { $0.role == .diffuse && !$0.ignored })?.id ?? members.first?.id
        selectedMaterialBrowserRowID = folders.first(where: { folder in
            folder.resolutions.contains { $0.images.contains { $0.id == selectedImageID } }
        }).map { "material:" + $0.id }
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
        let objective = selectedMap == .normal && training.modelArchitecture != NativeArchitecture.compactIdentifier ? nil : trainingScalarLossOverride
        let label = runLabel
        let onProgress = progressHandler()
        let current = generation
        perform("Training \(dataset.name) · \(map.displayName)", clearEvents: true, training: true) { [weak self] project in
            try await project.updateSettings(value)
            _ = try await project.trainMap(datasetID: dataset.id, map: map, settings: configuredTraining,
                                          start: start, runLabel: label, scalarLossOverride: objective, progress: onProgress)
            guard let self, self.generation == current else { return }
            self.invalidateExportedPackage()
        }
    }

    func setTrainingStartMode(_ mode: FactoryTrainingStartMode) {
        trainingScalarLossOverride = nil
        if mode == .fresh { settings.training.useFullResolutionForFreshRun(map: selectedMap) }
        trainingStartMode = mode == .fresh ? .checkpoint : mode
        settings.training.continueLatestWeights = mode != .fresh
        selectedTrainingVersionID = ""
        if mode != .fresh { restoreSelectedTrainingSettings() }
    }

    func continueModelVersion(_ version: ModelVersion, source: FactoryTrainingStartMode) {
        selectedDatasetID = version.datasetID
        selectedMap = version.map
        setTrainingStartMode(source)
        selectedTrainingVersionID = version.id
        restoreSelectedTrainingSettings()
        runLabel = version.label
    }

    func restoreSelectedTrainingSettings() {
        trainingScalarLossOverride = nil
        guard continuingTraining, let saved = selectedTrainingVersion?.savedSettings else { return }
        var restored = saved
        restored.seed = saved.requestedSeed ?? saved.seed
        restored.continueLatestWeights = true
        settings.training = restored
    }

    func updateModelVersionMetadata(_ version: ModelVersion, name: String, notes: String) {
        perform("Saving model name and notes") { project in
            try await project.updateModelVersionMetadata(runID: version.id, name: name, notes: notes)
        }
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

    func useCheckpointAsModel(_ version: ModelVersion, checkpointID: String? = nil) {
        guard !version.checkpointVersions.isEmpty else { return }
        let current = generation
        perform("Exporting selected checkpoint") { [weak self] project in
            _ = try await project.selectCheckpointModelVersion(runID: version.id, datasetID: version.datasetID, map: version.map,
                checkpointID: checkpointID)
            guard let self, self.generation == current else { return }
            self.invalidateExportedPackage()
        }
    }

    func useCoverageModel(artifact: ModelArtifact) {
        let current = generation
        perform("Choosing model to use") { [weak self] project in
            let runID = artifact.directory.lastPathComponent
            if artifact.kind == "classifier" {
                try await project.selectClassifierVersion(runID: runID)
            } else if let datasetID = artifact.datasetID, let map = artifact.map {
                try await project.selectModelVersion(runID: runID, datasetID: datasetID, map: map)
            } else { throw LoomError("Choose a model belonging to this group and product.") }
            guard let self, self.generation == current else { return }
            self.invalidateExportedPackage()
        }
    }

    func setModelApproval(runID: String, approved: Bool) {
        let current = generation
        perform(approved ? "Approving model" : "Removing model approval") { [weak self] project in
            try await project.setModelApproval(runID: runID, approved: approved)
            guard let self, self.generation == current else { return }
            self.invalidateExportedPackage()
        }
    }

    func approveAndUseCoverageModel(artifact: ModelArtifact) {
        let current = generation
        perform("Approving model for use") { [weak self] project in
            let runID = artifact.directory.lastPathComponent
            if artifact.kind == "classifier" {
                try await project.selectClassifierVersion(runID: runID)
            } else if let datasetID = artifact.datasetID, let map = artifact.map {
                try await project.selectModelVersion(runID: runID, datasetID: datasetID, map: map)
            } else { throw LoomError("Choose a model belonging to this group and product.") }
            try await project.setModelApproval(runID: runID, approved: true)
            guard let self, self.generation == current else { return }
            self.invalidateExportedPackage()
        }
    }

    func setModelRequirement(slot: ModelSlot, required: Bool) {
        let current = generation
        perform("Updating model requirements") { [weak self] project in
            try await project.setModelRequirement(slot: slot, required: required)
            guard let self, self.generation == current else { return }
            self.invalidateExportedPackage()
        }
    }

    func trainCoverageSlot(slot: ModelSlot) {
        guard !busy else { return }
        if let datasetID = slot.datasetID, let map = slot.product.map {
            selectedDatasetID = datasetID; selectedMap = map
            if !selectedModelVersions.contains(where: { $0.checkpoint != nil }) { setTrainingStartMode(.fresh) }
            workspaceSection = .training
        } else { workspaceSection = .package }
    }

    func testCoverageModel(artifact: ModelArtifact) {
        guard let version = snapshot?.modelVersions.first(where: { $0.directory == artifact.directory }) else { return }
        selectedDatasetID = version.datasetID; selectedMap = version.map
        showModelTesting(version)
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
        if let version {
            modelTestDatasetID = version.datasetID; modelTestMap = version.map
            selectedTestVersionID = version.id
        } else if let version = selectedTestVersion {
            selectedTestVersionID = version.id
        }
        if testInputURL == nil {
            let image = selectedImage.flatMap { image in testDiffuseImages.first { $0.id == image.id } } ?? testDiffuseImages.first
            if let image { setModelTestInput(image.url) }
        }
        workspaceSection = .testing
    }

    func chooseModelTestInput() {
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
        testInputURL = url
        modelTestResult = nil; previousTestResult = nil
        refreshTestOriginalMap()
    }

    func setModelTestDataset(_ id: String) {
        guard testDatasetID != id else { return }
        modelTestDatasetID = id
        clearModelTestRunSelection()
        if let input = testInputURL, snapshot?.images.contains(where: { $0.url == input }) == true {
            testInputURL = testDiffuseImages.first?.url
        }
        refreshTestOriginalMap()
    }

    func setModelTestMap(_ map: MapRole) {
        guard MapRole.targets.contains(map), testMap != map else { return }
        modelTestMap = map
        clearModelTestRunSelection()
        refreshTestOriginalMap()
    }

    func setModelTestCheckpoint(_ id: String?) {
        guard testCheckpointID != id else { return }
        preserveTestResultForComparison()
        testCheckpointID = id; modelTestResult = nil
    }

    private func preserveTestResultForComparison() {
        if let result = modelTestResult { previousTestResult = result }
    }

    private func refreshTestOriginalMap() {
        originalMapOperation?.cancel()
        let input = testInputURL, map = selectedTestVersion?.map
        let images = snapshot?.images ?? []
        let current = generation
        testOriginalMap = ModelTestOriginalMap.resolve(inputURL: nil, map: nil, images: [])
        guard input != nil, map != nil else { return }
        originalMapOperation = Task {
            let value = await Task.detached(priority: .userInitiated) {
                ModelTestOriginalMap.resolve(inputURL: input, map: map, images: images)
            }.value
            guard !Task.isCancelled, generation == current, testInputURL == input, selectedTestVersion?.map == map else { return }
            testOriginalMap = value
        }
    }

    func testModel() {
        guard canTestModel, let version = selectedTestVersion, let inputURL = testInputURL,
              let source = effectiveTestWeightSource else { return }
        let checkpointID = source == .checkpoint ? testCheckpointID : nil
        selectedTestVersionID = version.id
        testWeightSource = source
        testCheckpointID = checkpointID
        let current = generation
        let onProgress = progressHandler()
        preserveTestResultForComparison()
        modelTestResult = nil
        perform("Testing \(version.datasetName) · \(version.map.displayName)", clearEvents: true) { [weak self] project in
            let result = try await project.testModel(datasetID: version.datasetID, map: version.map, runID: version.id,
                                                     source: source, inputURL: inputURL, checkpointID: checkpointID, progress: onProgress)
            guard let self, self.generation == current, self.selectedTestVersion?.id == version.id,
                  self.testInputURL == inputURL, self.effectiveTestWeightSource == source,
                  (source != .checkpoint || self.testCheckpointID == checkpointID) else { return }
            self.recordModelTestResult(result)
        }
    }

    func recordModelTestResult(_ result: ModelTestResult) {
        if let previous = previousTestResult,
           previous.runID != result.runID || previous.inputURL != result.inputURL || previous.inputSHA256 != result.inputSHA256 || previous.map != result.map ||
           (previous.source == result.source && previous.checkpointID == result.checkpointID) {
            previousTestResult = nil
        }
        modelTestResult = result
    }

    func saveModelTestOutput(format: ModelTestExportFormat = .float32EXR) {
        guard let result = modelTestResult, !busy else { return }
        let panel = NSSavePanel()
        panel.title = "Save \(format.displayName) output"
        panel.prompt = format == .float32EXR ? "Save EXR" : "Export PNG"
        panel.allowedContentTypes = [format.contentType]
        panel.nameFieldStringValue = "\(result.inputURL.deletingPathExtension().lastPathComponent)_\(result.map.rawValue).\(format.fileExtension)"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let sourceURLs = snapshot?.images.map(\.url) ?? []
        perform("Saving model test output") { _ in
            try await Task.detached(priority: .userInitiated) {
                try ModelTestOutputExport.save(result, to: url, sourceImageURLs: sourceURLs, format: format)
            }.value
        }
    }

    private func resetModelTestSelection() {
        modelTestDatasetID = nil; modelTestMap = nil
        clearModelTestRunSelection()
    }

    private func clearModelTestRunSelection() {
        selectedTestVersionID = ""
        testWeightSource = nil
        testCheckpointID = nil
        modelTestResult = nil; previousTestResult = nil
        refreshTestOriginalMap()
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
        let current = generation
        let onProgress: @Sendable (Double, String) -> Void = { [weak self] fraction, message in
            Task { @MainActor [weak self] in
                guard let self, self.generation == current else { return }
                self.progress = fraction
                self.activityTitle = message
            }
        }
        perform("Categorizing material folders", clearEvents: true) { [weak self] project in
            // A classifier can tag an uncategorized source folder before that
            // folder has every category required by a deployment package.
            let chosen = try await project.selectedClassifierArtifact()
            let modelURL: URL?
            if let chosen, let path = chosen.coreMLPath {
                modelURL = URL(fileURLWithPath: path, relativeTo: chosen.directory)
            } else {
                modelURL = Bundle.main.url(forResource: "DTTC3", withExtension: "mlmodelc") ??
                    Bundle.main.url(forResource: "DTTC3", withExtension: "mlmodel")
            }
            guard let modelURL else { throw LoomError("The dataset classifier is unavailable. Import a Create ML classifier first.") }
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
        if isTraining { saveAndStopTraining(); return }
        operation?.cancel()
        Task { await project.cancelTraining() }
        activityTitle = "Stopping safely…"
    }

    func saveCheckpoint() { requestCheckpoint(stop: false) }

    func saveAndStopTraining() { requestCheckpoint(stop: true) }

    private func requestCheckpoint(stop: Bool) {
        guard let project, busy, isTraining, !checkpointRequestPending else { return }
        checkpointRequestPending = true
        checkpointWillStop = stop
        checkpointStatus = stop ? "Save & Stop queued; finishing the current optimizer update…" : "Checkpoint queued; finishing the current optimizer update…"
        errorMessage = nil
        let current = generation
        checkpointOperation = Task { [weak self] in
            do {
                let receipt = try await project.saveTrainingCheckpoint(stopAfterSaving: stop)
                guard let self, self.generation == current else { return false }
                self.lastCheckpointDirectory = receipt.runDirectory
                let result = stop ? (self.isTraining ? "; stopping…" : "; training stopped") : "; training continues"
                self.checkpointStatus = "Saved step \(receipt.completedUpdates.formatted()) at \(Date().formatted(date: .omitted, time: .shortened))\(result)"
                self.checkpointRequestPending = false
                if let value = try? await project.snapshot(), self.generation == current {
                    await self.adopt(value, restoreSettings: false)
                }
                return true
            } catch {
                guard let self, self.generation == current else { return false }
                self.checkpointRequestPending = false
                self.checkpointStatus = self.isTraining ? "Checkpoint was not saved; training continues." : "The requested checkpoint could not be saved. Check the latest saved version."
                self.errorMessage = error.localizedDescription
                return false
            }
        }
    }

    func waitForTrainingToFinish() async {
        await operation?.value
    }

    /// Quit waits for the save receipt before waiting for the training task.
    /// A failed checkpoint returns promptly and keeps the app and run alive.
    func saveTrainingAndPrepareForTermination() async -> Bool {
        if !isTraining { return await prepareForTermination() }
        if let checkpointOperation, checkpointRequestPending {
            let alreadyStopping = checkpointWillStop
            guard await checkpointOperation.value else { return false }
            if isTraining && !alreadyStopping { saveAndStopTraining() }
        } else { saveAndStopTraining() }
        guard let checkpointOperation, await checkpointOperation.value else { return false }
        await waitForTrainingToFinish()
        guard !busy, !isTraining else { return false }
        return await prepareForTermination()
    }

    func revealSource() {
        if let url = snapshot?.sourceURL { NSWorkspace.shared.open(url) }
    }

    func reveal(_ url: URL) { NSWorkspace.shared.activateFileViewerSelecting([url]) }

    private func perform(_ title: String, clearEvents: Bool = false, training: Bool = false,
                         action: @escaping @MainActor (LoomProject) async throws -> Void) {
        guard let project, !busy else { return }
        saveOperation?.cancel()
        let current = generation
        progressDeliverySerial &+= 1
        adoptionSerial &+= 1
        busy = true
        isTraining = training
        checkpointRequestPending = false; checkpointStatus = nil; lastCheckpointDirectory = nil
        activityTitle = title
        progress = nil
        errorMessage = nil
        if clearEvents { events = []; latestTrainingMeasurements = nil }
        trainingProgress = nil
        operation = Task {
            do {
                try Task.checkCancellation()
                try await action(project)
                guard generation == current else { return }
                do {
                    let value = try await project.snapshot()
                    guard generation == current else { return }
                    await adopt(value, restoreSettings: false)
                } catch {
                    guard generation == current else { return }
                    sourceIssue = "The action completed, but the source folder could not be refreshed: \(error.localizedDescription)"
                    events.append(.init(type: "warning", message: sourceIssue!))
                }
                busy = false
                isTraining = false
                activityTitle = Task.isCancelled ? "Stopped" : "\(title) complete"
                progress = Task.isCancelled ? nil : 1
            } catch {
                guard generation == current else { return }
                if Task.isCancelled || error is CancellationError || error is TrainingSavedStop {
                    busy = false
                    isTraining = false
                    activityTitle = error is TrainingSavedStop ? "Checkpoint saved · training stopped" : "Stopped"
                    if error is TrainingSavedStop, let status = checkpointStatus {
                        checkpointStatus = status.replacingOccurrences(of: "; stopping…", with: "; training stopped")
                    }
                    progress = nil
                    if let value = try? await project.snapshot() { await adopt(value, restoreSettings: false) }
                } else {
                    let failure = error
                    do { await adopt(try await project.snapshot(), restoreSettings: false) }
                    catch { invalidateSnapshot(error) }
                    fail(failure)
                }
            }
        }
    }

    private func progressHandler() -> @Sendable (TrainingEvent) -> Void {
        let current = generation
        let expectedSerial = progressDeliverySerial &+ 1
        let buffer = FactoryProgressBuffer()
        return { [weak self] event in
            buffer.append(event) { [weak self] batch, measurements in
                guard let self, self.generation == current, self.progressDeliverySerial == expectedSerial else { return }
                self.events.append(contentsOf: batch)
                self.latestTrainingMeasurements = measurements
                if let value = measurements?.trainingProgress { self.trainingProgress = value }
                if self.events.count > 300 { self.events.removeFirst(self.events.count - 300) }
                if self.busy, self.operation?.isCancelled != true, let latest = batch.last { self.activityTitle = latest.message }
                if self.busy, let progress = measurements?.progress { self.progress = progress }
            }
        }
    }

    private func adopt(_ value: ProjectSnapshot, restoreSettings: Bool) async {
        let current = generation
        adoptionSerial &+= 1
        let expectedAdoption = adoptionSerial
        let imagesChanged = snapshot?.images != value.images || snapshot?.sourceURL != value.sourceURL
        let oldIndex = browserIndex
        let existingExport = exportedPackage
        let prepared = await Task.detached(priority: .userInitiated) {
            (imagesChanged ? FactoryBrowserIndex(snapshot: value) : oldIndex,
             FactorySnapshotFiles(snapshot: value, existingExport: existingExport))
        }.value
        guard generation == current, adoptionSerial == expectedAdoption else { return }
        let previousModels = Set(snapshot?.modelVersions.filter { $0.isActive }.map(\.id) ?? [])
        let currentModels = Set(value.modelVersions.filter { $0.isActive }.map(\.id))
        if snapshot != nil, previousModels != currentModels || snapshot?.modelCoverage != value.modelCoverage { invalidateExportedPackage() }
        comparisonImageRevisions = prepared.1.imageRevisions
        if previewRevisions != prepared.1.imageRevisions { previewRevisions = prepared.1.imageRevisions }
        if imagesChanged {
            browserIndex = prepared.0
            presentationRevision += 1
            browserProjection = nil
            cachedTrainingPlan = nil
        }
        // An unchanged poll must not rebuild the mounted browser and previews.
        if snapshot?.sourceURL != value.sourceURL || snapshot?.settings != value.settings ||
           imagesChanged || snapshot?.datasets != value.datasets || snapshot?.artifacts != value.artifacts ||
           snapshot?.checkpoints != value.checkpoints || snapshot?.modelVersions != value.modelVersions ||
           snapshot?.modelCoverage != value.modelCoverage {
            snapshot = value
            refreshTestOriginalMap()
        }
        sourceIssue = nil
        if restoreSettings { settings = value.settings }
        if !selectedDatasetID.isEmpty, selectedDatasetID != "unassigned",
           !value.datasets.contains(where: { $0.id == selectedDatasetID }) { selectedDatasetID = "" }
        selectVisibleImageIfNeeded()
        if exportedPackage == existingExport, !prepared.1.existingExportIsPresent { exportedPackage = nil }
        let chosenFilesNeedAttention = value.modelCoverage.contains { entry in
            entry.status == .unavailable || entry.status == .stale ||
                entry.candidates.first(where: { $0.id == entry.effectiveRunID })?.approvalIsStale == true
        }
        if exportedPackage == nil && !packageNeedsExport && !chosenFilesNeedAttention {
            exportedPackage = prepared.1.exportedPackage
        }
    }

    private func fail(_ error: Error) {
        busy = false
        isTraining = false
        progress = nil
        activityTitle = "Action could not finish"
        errorMessage = error.localizedDescription
    }

    private func invalidateSnapshot(_ error: Error) {
        snapshot = nil
        selectedImageID = nil
        selectedMaterialBrowserRowID = nil
        browserIndex = nil; browserProjection = nil; cachedTrainingPlan = nil
        sourceIssue = error.localizedDescription
    }

}
