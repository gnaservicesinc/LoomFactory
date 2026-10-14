import LoomCore
import SwiftUI

enum ModelCoverageFilter: String, CaseIterable, Identifiable {
    case all, missing, needsApproval, needsSelection, ready
    var id: String { rawValue }
    var title: String {
        switch self {
        case .all: "All groups"
        case .missing: "Need models"
        case .needsApproval: "Need approval"
        case .needsSelection: "Need a choice"
        case .ready: "Ready for use"
        }
    }

    func includes(_ entry: ModelCoverageEntry) -> Bool {
        if self == .all { return true }
        guard entry.isRequired else { return false }
        let state = ModelCoveragePresentation(entry)
        switch self {
        case .all: return true
        case .missing: return entry.availableCount == 0
        case .needsApproval: return state.needsApproval
        case .needsSelection: return entry.availableCount > 0 && (entry.selectedRunID == nil || state.chosen?.isAvailable != true)
        case .ready: return state.isReady
        }
    }
}

/// Approval describes a reviewed artifact; selection describes what the system uses.
/// Keeping these facts separate also accommodates later products with several requirements.
struct ModelCoveragePresentation {
    let entry: ModelCoverageEntry

    init(_ entry: ModelCoverageEntry) { self.entry = entry }

    var chosen: ModelCoverageCandidate? {
        entry.candidates.first { $0.id == entry.selectedRunID }
    }
    var actionCandidate: ModelCoverageCandidate? {
        chosen ?? entry.candidates.first { $0.id == entry.effectiveRunID } ?? entry.candidates.first
    }
    var isReady: Bool {
        guard let chosen else { return false }
        return chosen.isAvailable && chosen.isApproved && !chosen.approvalIsStale
    }
    var needsApproval: Bool {
        guard entry.availableCount > 0 else { return false }
        if let chosen, chosen.isAvailable { return !chosen.isApproved || chosen.approvalIsStale }
        return entry.approvedCount == 0
    }
    var modelCounts: String {
        "\(entry.availableCount) completed \(entry.availableCount == 1 ? "model" : "models") · \(entry.approvedCount) approved"
    }
    var checkpointCounts: String? {
        guard entry.checkpointCount > 0 else { return nil }
        return "\(entry.checkpointCount) saved \(entry.checkpointCount == 1 ? "checkpoint" : "checkpoints")"
    }
    var detailCounts: String {
        [modelCounts, checkpointCounts].compactMap { $0 }.joined(separator: " · ")
    }
    var title: String {
        if entry.selectedRunID != nil && chosen?.isAvailable != true { return "Chosen model unavailable" }
        if isReady { return "Ready for use" }
        if chosen?.approvalIsStale == true { return "Review changed model" }
        if chosen != nil { return "Needs approval" }
        if entry.availableCount == 0 { return entry.checkpointCount > 0 ? "Training unfinished" : "Missing model" }
        return entry.approvedCount > 0 ? "Choose approved model" : "Needs approval"
    }
    var symbol: String {
        if isReady { return "checkmark.seal.fill" }
        if entry.selectedRunID != nil && chosen?.isAvailable != true { return "exclamationmark.triangle.fill" }
        if entry.availableCount == 0 { return entry.checkpointCount > 0 ? "pause.circle" : "circle.dashed" }
        return needsApproval ? "checkmark.seal" : "hand.point.up.left"
    }
    var color: Color {
        if isReady { return .green }
        if entry.availableCount == 0 || (entry.selectedRunID != nil && chosen?.isAvailable != true) { return .red }
        return needsApproval ? .orange : .accentColor
    }
}

struct ModelCoverageView: View {
    @Bindable var model: FactoryModel
    @State private var filter: ModelCoverageFilter = .all
    @State private var search = ""
    @State private var selectedEntryID: String?
    @State private var selectedCandidateRunID: String?
    @State private var comparingModels = false

    private var entries: [ModelCoverageEntry] { model.modelCoverage }
    private var requiredEntries: [ModelCoverageEntry] { entries.filter(\.isRequired) }
    private var matchingEntries: [ModelCoverageEntry] {
        entries.filter { entry in
            filter.includes(entry) && (search.isEmpty ||
                entry.groupName.localizedCaseInsensitiveContains(search) ||
                entry.slot.product.displayName.localizedCaseInsensitiveContains(search))
        }
    }
    private var visibleDatasetIDs: [String] {
        let matchingIDs = Set(matchingEntries.compactMap { $0.slot.datasetID })
        return model.snapshot?.datasets.filter { matchingIDs.contains($0.id) }.map(\.id) ?? []
    }
    private var selectedEntry: ModelCoverageEntry? {
        entries.first { $0.id == selectedEntryID } ??
            matchingEntries.first { $0.slot.datasetID == model.selectedDatasetID && $0.slot.product.map == model.selectedMap } ??
            matchingEntries.first
    }

    var body: some View {
        VStack(alignment: .leading, spacing: FactoryStyle.spacing) {
            FactoryPanel("Models", subtitle: "Approve & Use accepts the displayed version and chooses it for your package. Test output opens any saved version for inspection.") {
                if let selectedEntry { selectedModelActions(selectedEntry) }
                summary
                HStack(spacing: 16) {
                    Picker("Show groups", selection: $filter) {
                        ForEach(ModelCoverageFilter.allCases) { Text($0.title).tag($0) }
                    }.frame(width: 245).accessibilityIdentifier("coverage.filter")
                    Spacer()
                    TextField("Find a group or product", text: $search)
                        .textFieldStyle(.roundedBorder).frame(width: 240)
                        .accessibilityIdentifier("coverage.search")
                }
            }
            if matchingEntries.isEmpty {
                    ContentUnavailableView("No matching requirements", systemImage: "checklist",
                        description: Text(entries.isEmpty ? "Add categorized source images to see the models needed for each group." : "Change the filter to see other groups and products."))
                        .frame(maxWidth: .infinity).padding(.vertical, 24)
            } else {
                VSplitView {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            coverageMatrix
                            if let classifier = matchingEntries.first(where: { $0.slot.product == .classifier }) {
                                sharedClassifier(classifier)
                            }
                        }.padding(12)
                    }.frame(minHeight: 150, idealHeight: 210)
                        .accessibilityIdentifier("coverage.groups")
                    ScrollView {
                        if let selectedEntry { requirementDetail(selectedEntry).padding(12) }
                    }.frame(minHeight: 180).accessibilityIdentifier("coverage.inspector")
                    }
            }
        }
        .accessibilityIdentifier("loom.modelCoverage")
        .onChange(of: model.snapshot?.projectID) { selectedEntryID = nil; selectedCandidateRunID = nil }
        .onChange(of: filter) { selectedEntryID = nil; selectedCandidateRunID = nil }
        .onChange(of: search) { selectedEntryID = nil; selectedCandidateRunID = nil }
        .onChange(of: selectedEntryID) { selectedCandidateRunID = nil }
        .sheet(isPresented: $comparingModels) { ModelComparisonView(model: model) }
    }

    private var summary: some View {
        HStack(spacing: 0) {
            metric("Required models", count: requiredEntries.count, symbol: "checklist", color: .primary)
            metric("Need a model", count: requiredEntries.filter { $0.availableCount == 0 }.count, symbol: "circle.dashed", color: .red)
            metric("Need approval", count: requiredEntries.filter { ModelCoveragePresentation($0).needsApproval }.count, symbol: "checkmark.seal", color: .orange)
            metric("Ready for use", count: requiredEntries.filter { ModelCoveragePresentation($0).isReady }.count, symbol: "checkmark.seal.fill", color: .green)
        }.padding(.vertical, 6)
            .accessibilityIdentifier("coverage.summary")
    }

    private func metric(_ label: String, count: Int, symbol: String, color: Color) -> some View {
        HStack(spacing: 6) {
                Image(systemName: symbol).foregroundStyle(color)
                Text(count, format: .number).font(.callout.weight(.semibold)).monospacedDigit()
                Text(label).font(.caption).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private var coverageMatrix: some View {
        if !visibleDatasetIDs.isEmpty {
            LazyVStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 12) {
                    Text("Group").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        .frame(width: 165, alignment: .leading)
                    ForEach(ModelProductID.categoryProducts, id: \.self) { product in
                        Text(product.displayName).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                ForEach(visibleDatasetIDs, id: \.self) { datasetID in
                    HStack(alignment: .top, spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(groupName(datasetID)).font(.callout.weight(.semibold))
                            Text("\(groupImageCount(datasetID).formatted()) images").font(.caption).foregroundStyle(.secondary)
                        }.frame(width: 165, alignment: .leading)
                        ForEach(ModelProductID.categoryProducts, id: \.self) { product in
                            if let entry = entries.first(where: { $0.slot.datasetID == datasetID && $0.slot.product == product }) {
                                coverageCell(entry)
                            } else { Text("—").foregroundStyle(.secondary) }
                        }
                    }
                }
            }.accessibilityIdentifier("coverage.matrix")
        }
    }

    private func coverageCell(_ entry: ModelCoverageEntry) -> some View {
        let state = ModelCoveragePresentation(entry)
        let selected = selectedEntry?.id == entry.id
        return VStack(alignment: .leading, spacing: 8) {
            Button { selectedEntryID = entry.id } label: {
            VStack(alignment: .leading, spacing: 5) {
                Label(state.title, systemImage: state.symbol)
                    .font(.caption.weight(.semibold)).foregroundStyle(state.color)
                    .fixedSize(horizontal: false, vertical: true)
                Text(state.modelCounts)
                    .font(.caption).foregroundStyle(.secondary)
                if let checkpoints = state.checkpointCounts {
                    Text(checkpoints).font(.caption).foregroundStyle(.secondary)
                }
                Text(entry.isRequired ? (entry.usesLatestByDefault ? "Latest default" : (entry.selectedRunID == nil ? "Required" : "Model chosen")) : "Optional")
                    .font(.caption2).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, minHeight: 66, alignment: .leading)
                .contentShape(RoundedRectangle(cornerRadius: 9))
            }.buttonStyle(.plain)
            .help("Manage \(entry.groupName) · \(entry.slot.product.displayName)")
            .accessibilityLabel("\(entry.groupName), \(entry.slot.product.displayName), \(state.title), \(state.detailCounts)")
            .accessibilityIdentifier("coverage.slot.\(entry.id)")
            if let candidate = state.actionCandidate {
                Button(state.isReady ? "Approved & chosen" : "Approve & Use", systemImage: "checkmark.seal.fill") {
                    model.approveAndUseCoverageModel(artifact: candidate.artifact)
                }.buttonStyle(.borderedProminent).controlSize(.small)
                    .disabled(model.busy || state.isReady || !candidate.isAvailable || !candidate.canChoose)
                    .accessibilityIdentifier("coverage.quickApprove.\(entry.id)")
                if entry.slot.product != .classifier {
                    Button("Test output", systemImage: "play") { model.testCoverageModel(artifact: candidate.artifact) }
                        .controlSize(.small)
                        .accessibilityIdentifier("coverage.quickTest.\(entry.id)")
                }
            } else {
                Button("Train model…") { model.trainCoverageSlot(slot: entry.slot) }
                    .controlSize(.small).disabled(model.busy)
            }
        }.padding(11).frame(maxWidth: .infinity, alignment: .leading)
            .background(selected ? Color.accentColor.opacity(0.09) : Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 9))
            .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(selected ? Color.accentColor.opacity(0.6) : Color.primary.opacity(0.08)))
    }

    private func selectedModelActions(_ entry: ModelCoverageEntry) -> some View {
        let state = ModelCoveragePresentation(entry)
        let candidate = entry.candidates.first { $0.id == selectedCandidateRunID } ?? state.actionCandidate
        let ready = candidate.map { $0.isSelected && $0.isApproved && !$0.approvalIsStale && $0.isAvailable } ?? false
        return HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("\(entry.groupName) · \(entry.slot.product.displayName)").font(.headline)
                if let candidate {
                    Picker("Saved model", selection: Binding(
                        get: { selectedCandidateRunID ?? candidate.id }, set: { selectedCandidateRunID = $0 }
                    )) {
                        ForEach(entry.candidates) { value in
                            let version = model.snapshot?.modelVersions.first { $0.id == value.id }
                            Text(version.map(ModelVersionPresentation.title) ?? "Run \(value.id.prefix(8))").tag(value.id)
                        }
                    }.labelsHidden().controlSize(.small).accessibilityIdentifier("coverage.primaryVersion")
                } else { Text(state.title).font(.caption).foregroundStyle(state.color) }
            }.frame(maxWidth: .infinity, alignment: .leading)
            if let candidate {
                Button(ready ? "Approved & chosen" : "Approve & Use", systemImage: "checkmark.seal.fill") {
                    model.approveAndUseCoverageModel(artifact: candidate.artifact)
                }.buttonStyle(.borderedProminent)
                    .disabled(model.busy || ready || !candidate.isAvailable || !candidate.canChoose)
                    .accessibilityIdentifier("coverage.primaryApprove")
                if entry.slot.product != .classifier {
                    Button("Test output", systemImage: "play") { model.testCoverageModel(artifact: candidate.artifact) }
                        .accessibilityIdentifier("coverage.primaryTest")
                }
            }
        }.padding(.bottom, 6).accessibilityIdentifier("coverage.primaryActions")
    }

    private func sharedClassifier(_ entry: ModelCoverageEntry) -> some View {
        HStack(spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Label("Shared category classifier", systemImage: "arrow.triangle.branch").font(.callout.weight(.semibold))
                Text("Routes material images to these groups in the Texture Loom package.")
                    .font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading)
            coverageCell(entry).frame(width: 245)
        }.padding(.top, 6)
    }

    private func requirementDetail(_ entry: ModelCoverageEntry) -> some View {
        FactoryPanel("\(entry.groupName) · \(entry.slot.product.displayName)", subtitle: detailSubtitle(entry)) {
            HStack(spacing: 16) {
                if entry.slot.product != .classifier {
                    Toggle("Required in this system", isOn: Binding(
                        get: { entry.isRequired }, set: { model.setModelRequirement(slot: entry.slot, required: $0) }
                    )).toggleStyle(.checkbox).disabled(model.busy)
                        .accessibilityIdentifier("coverage.required.\(entry.id)")
                    Text("\(entry.pairCount.formatted()) paired teaching \(entry.pairCount == 1 ? "example" : "examples")")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Label("Required shared model", systemImage: "checkmark.circle").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button(entry.slot.product == .classifier ? "Set up classifier…" : (entry.availableCount == 0 ? "Train model…" : "Train another version…"), systemImage: "sparkles") {
                    model.trainCoverageSlot(slot: entry.slot)
                }.disabled(model.busy)
                    .accessibilityIdentifier("coverage.train.\(entry.id)")
                if comparisonVersions(entry).count >= 2 {
                    Button("A/B Compare", systemImage: "rectangle.split.2x1") {
                        selectTrainingContext(entry)
                        comparingModels = true
                    }.disabled(model.busy).accessibilityIdentifier("coverage.compare.\(entry.id)")
                }
            }
            if let detail = entry.detail {
                Label(detail, systemImage: "info.circle").font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if entry.slot.product == .classifier {
                Text("A classifier can categorize new images before all groups are ready. Package approval requires its labels to match the current groups.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if entry.usesLatestByDefault {
                Label("Using the latest available model by default. Choose a version to keep that choice when new models are trained.", systemImage: "clock.arrow.circlepath")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("coverage.latestDefault.\(entry.id)")
            }
            if entry.candidates.isEmpty {
                Text(entry.checkpointCount > 0 ? "No completed model is available. Continue a saved training run or train a new model." : "This group has no saved model for this product. Open training to create one.")
                    .font(.callout).foregroundStyle(.secondary).padding(.vertical, 10)
            } else {
                LazyVStack(spacing: 10) {
                    ForEach(entry.candidates) { candidate in
                        ModelCoverageCandidateRow(model: model, entry: entry, candidate: candidate)
                    }
                }
            }
            let checkpoints = unfinishedVersions(entry)
            if !checkpoints.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Saved checkpoints (\(checkpoints.count))").font(.callout.weight(.semibold))
                    Text("These saved training versions can be tested or continued. Complete training to create a model for approval and package use.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    ForEach(checkpoints) { version in
                        ModelCoverageCheckpointRow(model: model, version: version)
                    }
                }.accessibilityIdentifier("coverage.checkpoints.\(entry.id)")
            }
        }.accessibilityIdentifier("coverage.detail.\(entry.id)")
    }

    private func detailSubtitle(_ entry: ModelCoverageEntry) -> String {
        let state = ModelCoveragePresentation(entry)
        return "\(state.title) · \(state.detailCounts)"
    }
    private func groupName(_ datasetID: String) -> String {
        model.snapshot?.datasets.first { $0.id == datasetID }?.name ?? datasetID
    }
    private func groupImageCount(_ datasetID: String) -> Int {
        model.snapshot?.datasets.first { $0.id == datasetID }?.imageCount ?? 0
    }
    private func comparisonVersions(_ entry: ModelCoverageEntry) -> [ModelVersion] {
        model.snapshot?.modelVersions.filter {
            $0.datasetID == entry.slot.datasetID && $0.map == entry.slot.product.map && $0.artifact?.weightsPath != nil
        } ?? []
    }
    private func unfinishedVersions(_ entry: ModelCoverageEntry) -> [ModelVersion] {
        model.snapshot?.modelVersions.filter {
            $0.datasetID == entry.slot.datasetID && $0.map == entry.slot.product.map && $0.artifact == nil && $0.checkpoint != nil
        } ?? []
    }
    private func selectTrainingContext(_ entry: ModelCoverageEntry) {
        if let datasetID = entry.slot.datasetID { model.selectedDatasetID = datasetID }
        if let map = entry.slot.product.map { model.selectedMap = map }
    }
}

struct ModelCoverageCheckpointRow: View {
    @Bindable var model: FactoryModel
    let version: ModelVersion

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(ModelVersionPresentation.title(version)).font(.callout.weight(.semibold)).textSelection(.enabled)
                    Text("\(version.status.capitalized) · \((version.checkpoint?.totalEpochsCompleted ?? 0).formatted()) saved epochs")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                FactoryStatus(text: "Saved checkpoint", symbol: "pause.circle", color: .secondary)
            }
            if let settings = version.settings {
                Text(ModelVersionPresentation.summary(settings, map: version.map)).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ModelVersionMetadataEditor(model: model, version: version)
            HStack(spacing: 10) {
                Button("Continue checkpoint", action: continueTraining).disabled(model.busy)
                    .accessibilityIdentifier("coverage.continueCheckpoint.\(version.id)")
                Button("Test checkpoint") { model.showModelTesting(version) }
                    .accessibilityIdentifier("coverage.testCheckpoint.\(version.id)")
                Spacer(minLength: 0)
                Button("Show folder") { model.reveal(version.directory) }
                    .accessibilityIdentifier("coverage.revealCheckpoint.\(version.id)")
            }.controlSize(.small)
        }.padding(13).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 9))
            .accessibilityIdentifier("coverage.checkpoint.\(version.id)")
    }

    @MainActor func continueTraining() {
        model.selectedDatasetID = version.datasetID
        model.selectedMap = version.map
        model.continueModelVersion(version, source: .checkpoint)
        model.workspaceSection = .training
    }
}

struct ModelCoverageCandidateRow: View {
    @Bindable var model: FactoryModel
    let entry: ModelCoverageEntry
    let candidate: ModelCoverageCandidate

    private var version: ModelVersion? { model.snapshot?.modelVersions.first { $0.id == candidate.id } }
    private var removesApproval: Bool { candidate.isApproved || (candidate.approvalIsStale && !candidate.isAvailable) }
    private var title: String {
        if let version { return ModelVersionPresentation.title(version) }
        return "Run \(candidate.id.prefix(8)) · \(candidate.artifact.createdAt.formatted(date: .abbreviated, time: .shortened))"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.callout.weight(.semibold)).textSelection(.enabled)
                    Text("\((version?.exportedEpochsCompleted ?? candidate.artifact.epochs).formatted()) epochs · \(candidate.artifact.architecture)")
                        .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
                Spacer()
                if candidate.isSelected {
                    FactoryStatus(text: "Chosen for use", symbol: "checkmark.circle.fill", color: .accentColor)
                        .accessibilityIdentifier("coverage.chosen.\(candidate.id)")
                } else if candidate.id == entry.effectiveRunID && entry.usesLatestByDefault {
                    FactoryStatus(text: "Latest default", symbol: "clock", color: .secondary)
                }
                FactoryStatus(text: candidate.approvalIsStale ? "Review changed model" : (candidate.isApproved ? "Approved" : "Not approved"),
                    symbol: candidate.isApproved && !candidate.approvalIsStale ? "checkmark.seal.fill" : "checkmark.seal",
                    color: candidate.isApproved && !candidate.approvalIsStale ? .green : .orange)
                    .accessibilityIdentifier("coverage.approval.\(candidate.id)")
            }
            if let issue = candidate.availabilityIssue {
                Label(issue, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let settings = version?.settings {
                Text(ModelVersionPresentation.summary(settings, map: version?.map)).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let version { ModelVersionMetadataEditor(model: model, version: version) }
            HStack(spacing: 10) {
                Button(candidate.isApproved && candidate.isSelected && !candidate.approvalIsStale ? "Approved & chosen" : "Approve & Use", systemImage: "checkmark.seal.fill") {
                    model.approveAndUseCoverageModel(artifact: candidate.artifact)
                }.buttonStyle(.borderedProminent)
                    .disabled(model.busy || (candidate.isApproved && candidate.isSelected && !candidate.approvalIsStale) || !candidate.isAvailable || !candidate.canChoose)
                    .accessibilityIdentifier("coverage.approveAndUse.\(candidate.id)")
                if entry.slot.product != .classifier {
                    Button("Test output") { model.testCoverageModel(artifact: candidate.artifact) }
                        .accessibilityIdentifier("coverage.test.\(candidate.id)")
                }
                if let version, version.artifact?.weightsPath != nil {
                    Button("Continue training") {
                        model.selectedDatasetID = version.datasetID
                        model.selectedMap = version.map
                        model.continueModelVersion(version, source: .exportedModel)
                        model.workspaceSection = .training
                    }.disabled(model.busy)
                        .accessibilityIdentifier("coverage.continue.\(candidate.id)")
                }
                Spacer(minLength: 0)
                Button("Show folder") { model.reveal(candidate.artifact.directory) }
            }.controlSize(.small)
            HStack(spacing: 10) {
                Button(candidate.isSelected ? "Chosen for use" : "Use without changing approval", systemImage: "checkmark.circle") {
                    model.useCoverageModel(artifact: candidate.artifact)
                }.disabled(model.busy || candidate.isSelected || !candidate.canChoose)
                    .accessibilityIdentifier("coverage.use.\(candidate.id)")
                Button(removesApproval ? "Remove approval" : "Mark approved only", systemImage: "checkmark.seal") {
                    model.setModelApproval(runID: candidate.id, approved: !removesApproval)
                }.disabled(model.busy || (!candidate.isAvailable && !removesApproval))
                    .help("Records your review without changing which saved model is used.")
                    .accessibilityIdentifier("coverage.approve.\(candidate.id)")
            }.controlSize(.small)
        }.padding(13).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 9))
            .accessibilityIdentifier("coverage.candidate.\(candidate.id)")
    }
}
