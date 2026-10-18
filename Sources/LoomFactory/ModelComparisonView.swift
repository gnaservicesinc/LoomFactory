import AppKit
import LoomCore
import LoomNative
import SwiftUI

enum ModelVersionPresentation {
    static func title(_ version: ModelVersion) -> String {
        let label = version.label.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = label.isEmpty ? "Run \(version.id.prefix(8))" : label
        return "\(name) · \(version.id.prefix(8)) · \(clipping(version.settings)) · \(version.createdAt.formatted(date: .abbreviated, time: .shortened))"
    }

    static func clipping(_ settings: TrainingSettings?) -> String {
        guard let settings else { return "Clipping unknown" }
        guard settings.gradientClipPolicy != .off, settings.gradientClipNorm > 0 else { return "Clipping off" }
        if settings.gradientClipPolicy == .warmup {
            return "Clipping warmup \(number(settings.gradientClipWarmupStartNorm)) → \(number(settings.gradientClipNorm))"
        }
        return "Clipping max norm \(number(settings.gradientClipNorm))"
    }

    static func summary(_ settings: TrainingSettings, map: MapRole? = nil) -> String {
        let objective = map == .depth || map == .roughness
            ? " · \(settings.scalarLoss.displayName)" : ""
        return "\(settings.optimizer.displayName) · LR \(number(settings.learningRate)) · batch \(settings.batchSize) × \(settings.gradientAccumulationSteps) · \(settings.imageSize) px · \(settings.modelWidth) channels / \(settings.modelBlocks) blocks" + objective
    }

    static func number(_ value: Double) -> String {
        value.formatted(.number.precision(.significantDigits(1...6)))
    }
}

struct TrainingRunDeletionControl: View {
    @Bindable var model: FactoryModel

    var body: some View {
        if model.hasTrainingRuns {
            HStack {
                Text("\(model.trainingRunCount.formatted()) saved training \(model.trainingRunCount == 1 ? "run" : "runs") in this source folder")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Delete all runs…", role: .destructive) { model.confirmDeleteAllTrainingRuns() }
                    .disabled(model.busy)
                    .accessibilityIdentifier("training.deleteAllRuns")
            }.controlSize(.small)
        }
    }
}

struct ModelVersionRow: View {
    @Bindable var model: FactoryModel
    let version: ModelVersion

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(ModelVersionPresentation.title(version)).font(.callout.weight(.semibold)).textSelection(.enabled)
                    Text("\(version.status.capitalized) · \(epochs.formatted()) completed epochs")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if version.isActive {
                    Label("Used for export", systemImage: "checkmark.circle.fill")
                        .font(.caption.weight(.medium)).foregroundStyle(.green)
                        .accessibilityIdentifier("training.version.active.\(version.id)")
                }
            }
            if let settings = version.settings {
                Text(ModelVersionPresentation.summary(settings, map: version.map)).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ModelVersionMetadataEditor(model: model, version: version)
            if let metric = version.history.last(where: { $0.loss != nil || $0.validationLoss != nil }) {
                HStack(spacing: 14) {
                    Text("Epoch \(metric.epoch)")
                    if let loss = metric.loss { Text("Training loss \(ModelVersionPresentation.number(loss))") }
                    if let loss = metric.validationLoss { Text("Validation loss \(ModelVersionPresentation.number(loss))") }
                }.font(.caption).monospacedDigit().foregroundStyle(.secondary)
                if let norm = metric.meanGradientNorm {
                    Text("Mean global norm \(ModelVersionPresentation.number(norm))\(metric.peakGradientNorm.map { " · peak " + ModelVersionPresentation.number($0) } ?? "")")
                        .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
                if let clipped = metric.clippedUpdates {
                    Text("Saved run \(version.id.prefix(8)) · epoch \(metric.epoch): \(clipped)\(metric.optimizerUpdates.map { " of " + $0.formatted() } ?? "") updates clipped\(metric.gradientClipThreshold.map { " · applied cap " + ModelVersionPresentation.number($0) } ?? "")")
                        .font(.caption).monospacedDigit().foregroundStyle(.secondary).textSelection(.enabled)
                        .accessibilityIdentifier("training.version.clippingMeasurements.\(version.id)")
                }
            }
            if let lineage = version.lineage {
                Text("Continued \(lineage.source == .model ? "exported model" : "checkpoint") from run \(lineage.parentRunID.prefix(8))")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 9) {
                if version.artifact != nil || version.checkpoint != nil {
                    Button("Test model") { model.showModelTesting(version) }
                        .accessibilityIdentifier("training.version.test.\(version.id)")
                }
                if version.artifact?.weightsPath != nil {
                    Button("Load exported model") { model.continueModelVersion(version, source: .exportedModel) }
                        .accessibilityIdentifier("training.version.continueModel.\(version.id)")
                }
                if version.checkpoint != nil {
                    Button("Load checkpoint") { model.continueModelVersion(version, source: .checkpoint) }
                        .accessibilityIdentifier("training.version.continueCheckpoint.\(version.id)")
                }
                if version.artifact != nil {
                    Button("Use for export") { model.useModelVersionForExport(version) }.disabled(version.isActive)
                        .accessibilityIdentifier("training.version.useForExport.\(version.id)")
                }
                Spacer(minLength: 0)
                Button("Show folder") { model.reveal(version.directory) }
                    .accessibilityIdentifier("training.version.reveal.\(version.id)")
                Button("Delete run…", role: .destructive) { model.confirmDeleteTrainingRun(version) }
                    .disabled(model.busy)
                    .accessibilityIdentifier("training.version.delete.\(version.id)")
            }.controlSize(.small)
            if version.checkpoint != nil {
                Button("Use checkpoint as model", systemImage: "checkmark.circle") { model.useCheckpointAsModel(version) }
                    .disabled(model.busy).controlSize(.small)
                    .help("Exports this checkpoint as a separate selected version, preserving the original run.")
                    .accessibilityIdentifier("training.version.useCheckpoint.\(version.id)")
            }
            DisclosureGroup("Run settings and history") {
                VStack(alignment: .leading, spacing: 10) {
                    TrainingSettingsTable(first: version.settings)
                    if !version.history.isEmpty {
                        Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 5) {
                            GridRow {
                                Text("Epoch"); Text("Training loss"); Text("Validation loss"); Text("Learning rate"); Text("Objective")
                            }.foregroundStyle(.secondary)
                            ForEach(Array(version.history.enumerated()), id: \.offset) { _, metric in
                                GridRow {
                                    Text(metric.epoch.formatted())
                                    Text(metric.loss.map(ModelVersionPresentation.number) ?? "—")
                                    Text(metric.validationLoss.map(ModelVersionPresentation.number) ?? "—")
                                    Text(metric.learningRate.map(ModelVersionPresentation.number) ?? "—")
                                    Text(metric.scalarLoss?.displayName ?? (version.map == .normal ? "Normal angular" : "—"))
                                }
                            }
                        }.font(.caption).monospacedDigit()
                    }
                }.padding(.top, 7)
            }.font(.caption)
        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 9))
            .accessibilityIdentifier("training.version.\(version.id)")
    }

    private var epochs: Int { max(version.checkpoint?.totalEpochsCompleted ?? version.exportedEpochsCompleted, version.history.last?.epoch ?? 0) }
}

struct ModelComparisonView: View {
    @Bindable var model: FactoryModel
    @Environment(\.dismiss) private var dismiss
    @State private var showAllSettings = false
    @State private var imageZoom: ComparisonImageZoom = .fit

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Compare two runs · \(model.comparisonMap.displayName)").font(.title2.weight(.semibold))
                    Text("Both saved models receive the same exact crop from a current paired material.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction).disabled(model.busy)
            }
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 17) {
                    versionSelection
                    Text("Compare differences in structure, fine detail and artifacts across several materials. Different seeds are useful context when looking for an effect large enough to matter.")
                        .font(.callout).foregroundStyle(.secondary)
                        .accessibilityIdentifier("comparison.seedContext")
                    pairSelection
                    HStack {
                        Button("Compare same crop", systemImage: "rectangle.split.2x1") { model.compareModels() }
                            .buttonStyle(.borderedProminent).disabled(!model.canCompareModels)
                            .accessibilityIdentifier("comparison.start")
                        if model.busy {
                            ProgressView().controlSize(.small)
                            Text(model.activityTitle).font(.caption).foregroundStyle(.secondary)
                            Button("Stop") { model.cancel() }
                        }
                    }
                    if let issue = model.comparisonIssue {
                        Label(issue, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
                    }
                    if let result = model.comparisonResult { comparisonResult(result) }
                    if let first = model.firstComparisonVersion, let second = model.secondComparisonVersion {
                        Divider()
                        HStack {
                            Text("Settings differences").font(.headline)
                            Spacer()
                            Toggle("Show all settings", isOn: $showAllSettings).toggleStyle(.checkbox)
                        }
                        TrainingSettingsTable(first: first.settings, second: second.settings, differencesOnly: !showAllSettings)
                            .accessibilityIdentifier("comparison.settings")
                    }
                }.padding(.trailing, 8)
            }
        }.padding(24).frame(minWidth: 1050, idealWidth: 1160, minHeight: 700, idealHeight: 870)
            .background(.background)
            .interactiveDismissDisabled(model.busy)
            .task(id: model.comparisonSelectionKey) { await model.loadComparisonPairs() }
    }

    private var versionSelection: some View {
        HStack(alignment: .top, spacing: 20) {
            versionPicker("Model A", selection: Binding(get: { model.firstComparisonVersion?.id ?? "" }, set: { model.comparisonFirstVersionID = $0 }))
            versionPicker("Model B", selection: Binding(get: { model.secondComparisonVersion?.id ?? "" }, set: { model.comparisonSecondVersionID = $0 }), exclude: model.firstComparisonVersion?.id)
        }.disabled(model.busy)
    }

    private func versionPicker(_ title: String, selection: Binding<String>, exclude: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline.weight(.semibold))
            Picker(title, selection: selection) {
                if !selection.wrappedValue.isEmpty, !model.comparisonVersions.contains(where: { $0.id == selection.wrappedValue && $0.id != exclude }) {
                    Text("Selected run is unavailable; choose a run").tag(selection.wrappedValue)
                }
                ForEach(model.comparisonVersions.filter { $0.id != exclude }) { version in
                    Text(ModelVersionPresentation.title(version)).tag(version.id)
                }
            }.labelsHidden().accessibilityIdentifier(title == "Model A" ? "comparison.modelA" : "comparison.modelB")
            if let version = model.comparisonVersions.first(where: { $0.id == selection.wrappedValue }) {
                ModelComparisonRunDetails(version: version)
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private var pairSelection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Material pair").font(.subheadline.weight(.semibold))
            if model.loadingComparisonPairs {
                ProgressView("Checking current pairs…").controlSize(.small)
            } else {
                Picker("Material pair", selection: Binding(get: { model.selectedComparisonPair?.id ?? "" }, set: { model.comparisonPairID = $0 })) {
                    if model.comparisonPairs.isEmpty { Text("No common pair available").tag("") }
                    ForEach(model.comparisonPairs) { pair in
                        Text(pair.label).tag(pair.id)
                    }
                }.labelsHidden().disabled(model.busy).accessibilityIdentifier("comparison.pair")
                if let pair = model.selectedComparisonPair {
                    membership(pair)
                    Text("Input: \(pair.inputURL.lastPathComponent) · Target: \(pair.targetURL.lastPathComponent)")
                        .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
            }
        }
    }

    private func membership(_ pair: ModelComparisonPair) -> some View {
        HStack(spacing: 18) {
            membershipLabel("A", pair.seenByFirst)
            membershipLabel("B", pair.seenBySecond)
        }.accessibilityIdentifier("comparison.membership")
    }

    private func membershipLabel(_ modelName: String, _ membership: ModelComparisonMembership) -> some View {
        Label("\(modelName): \(membership.displayName)", systemImage: membership == .heldOut ? "checkmark.seal" : "info.circle")
            .font(.caption).foregroundStyle(membership == .heldOut ? Color.green : Color.secondary)
    }

    private func comparisonResult(_ result: ModelComparisonResult) -> some View {
        VStack(alignment: .leading, spacing: 13) {
            HStack {
                Text("\(result.cropSize) × \(result.cropSize) crop").font(.headline)
                Spacer()
                Button("Show comparison folder") { model.reveal(result.directory) }
            }
            membership(result.pair)
            Picker("Shared image zoom", selection: $imageZoom) {
                ForEach(ComparisonImageZoom.allCases) { Text($0.displayName).tag($0) }
            }.pickerStyle(.segmented).frame(width: 270).accessibilityIdentifier("comparison.zoom")
            HStack(spacing: 20) {
                imagePanel(outputTitle("A", id: result.firstArtifactID, mae: result.firstMAE), url: result.firstOutputURL, pixels: result.cropSize)
                imagePanel(outputTitle("B", id: result.secondArtifactID, mae: result.secondMAE), url: result.secondOutputURL, pixels: result.cropSize)
            }
            HStack(spacing: 20) {
                imagePanel("Diffuse input", url: result.inputURL, pixels: result.cropSize)
                imagePanel("Native target", url: result.targetURL, pixels: result.cropSize)
            }
            Text("MAE measures mean absolute error against the native target on this crop. A/B output difference: \(ModelVersionPresentation.number(result.outputDifferenceMAE)). Judge detail, relief and artifacts alongside error; lower error does not always look better.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text(result.targetNote).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text("Compare several held-out materials before drawing conclusions about model quality.")
                .font(.caption).foregroundStyle(.secondary)
        }.accessibilityIdentifier("comparison.result")
    }

    private func outputTitle(_ side: String, id: String, mae: Double) -> String {
        let version = model.comparisonVersions.first { $0.artifact?.id == id }
        let name = version.flatMap { $0.label.isEmpty ? nil : $0.label } ?? "Run"
        return "\(side) · \(name) · \(id.prefix(8)) · \(ModelVersionPresentation.clipping(version?.settings)) · MAE \(ModelVersionPresentation.number(mae))"
    }

    private func imagePanel(_ title: String, url: URL, pixels: Int) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline.weight(.medium)).monospacedDigit()
            if let scale = imageZoom.scale {
                ScrollView([.horizontal, .vertical]) {
                    FactoryImage(url: url, revision: url.path, maximumPixels: pixels, interpolation: .none)
                        .frame(width: CGFloat(pixels) * scale, height: CGFloat(pixels) * scale)
                }.frame(height: 340).clipShape(RoundedRectangle(cornerRadius: 7))
            } else {
                FactoryImage(url: url, revision: url.path, maximumPixels: pixels).frame(height: 340)
                    .clipShape(RoundedRectangle(cornerRadius: 7))
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct ModelComparisonRunDetails: View {
    let version: ModelVersion

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(ModelVersionPresentation.clipping(version.settings)).font(.callout.weight(.semibold))
            Text("Run \(version.id.prefix(8)) · \(version.exportedEpochsCompleted) exported epochs")
            if let settings = version.settings {
                Text("Seed \(settings.seed) · \(ModelVersionPresentation.summary(settings, map: version.map))")
                    .fixedSize(horizontal: false, vertical: true)
            }
        }.font(.caption).textSelection(.enabled)
    }
}

enum ComparisonImageZoom: String, CaseIterable, Identifiable {
    case fit, actual, double
    var id: String { rawValue }
    var displayName: String {
        switch self { case .fit: "Fit"; case .actual: "100%"; case .double: "200%" }
    }
    var scale: CGFloat? {
        switch self { case .fit: nil; case .actual: 1; case .double: 2 }
    }
}

struct TrainingSettingValue: Identifiable, Equatable {
    let id: String
    let label: String
    let value: String

    static func values(_ settings: TrainingSettings) -> [Self] {
        func item(_ id: String, _ label: String, _ value: String) -> Self { .init(id: id, label: label, value: value) }
        func number(_ value: Double) -> String { ModelVersionPresentation.number(value) }
        func yesNo(_ value: Bool) -> String { value ? "On" : "Off" }
        let clips = settings.gradientClipPolicy != .off && settings.gradientClipNorm > 0
        let warmsUp = clips && settings.gradientClipPolicy == .warmup
        var values: [Self] = [
            item("modelArchitecture", "Model architecture", TrainingNetwork(rawValue: settings.modelArchitecture)?.displayName ?? settings.modelArchitecture),
            item("scalarLoss", "Scalar training objective", settings.scalarLoss.displayName),
            item("imageSize", "Training dimensions", "\(settings.imageSize) × \(settings.imageSize)"),
            item("modelWidth", "Channel width", "\(settings.modelWidth)"),
            item("modelBlocks", "Residual blocks", "\(settings.modelBlocks)"),
            item("epochs", "Planned epochs", "\(settings.epochs)"),
            item("optimizer", "Optimizer", settings.optimizer.displayName),
            item("learningRate", "Learning rate", number(settings.learningRate)),
            item("batchSize", "Batch size", "\(settings.batchSize)"),
            item("gradientAccumulationSteps", "Gradient accumulation", "\(settings.gradientAccumulationSteps)"),
            item("gradientClipPolicy", "Gradient clipping", settings.gradientClipPolicy.displayName),
            item("effectiveGradientClipping", "Effective clipping", ModelVersionPresentation.clipping(settings)),
            item("gradientClipNorm", "Max norm", clips ? number(settings.gradientClipNorm) : "Not used (clipping off)"),
            item("gradientClipWarmupStartNorm", "Initial max norm", warmsUp ? number(settings.gradientClipWarmupStartNorm) : "Not used"),
            item("gradientClipWarmupUpdates", "Clipping warmup updates", warmsUp ? "\(settings.gradientClipWarmupUpdates)" : "Not used"),
            item("weightDecay", "Weight decay", number(settings.weightDecay)),
            item("beta1", "Beta 1", number(settings.beta1)),
            item("beta2", "Beta 2", number(settings.beta2)),
            item("momentum", "Momentum", number(settings.momentum)),
            item("learningRateSchedule", "Learning-rate schedule", settings.learningRateSchedule.displayName),
            item("warmupEpochs", "Learning-rate warmup epochs", "\(settings.warmupEpochs)"),
            item("minimumLearningRateRatio", "Final rate ratio", number(settings.minimumLearningRateRatio)),
            item("cropRefresh", "New crop positions", settings.cropRefresh.displayName),
            item("augmentationRefresh", "New flips and rotations", settings.augmentationRefresh.displayName),
            item("shuffleEachEpoch", "Randomize image order", yesNo(settings.shuffleEachEpoch)),
            item("flipCrops", "Random flips", yesNo(settings.flipCrops)),
            item("rotateCrops", "Random rotations", yesNo(settings.rotateCrops)),
            item("exaggerateFeatures", "Depth feature exaggeration", yesNo(settings.exaggerateFeatures)),
            item("seed", "Random seed", "\(settings.seed)"),
            item("validationFraction", "Validation holdout", "\(number(settings.validationFraction * 100))%"),
            item("validationSeed", "Validation seed", "\(settings.validationSeed)"),
            item("earlyStoppingPatience", "Early stopping patience", "\(settings.earlyStoppingPatience)"),
            item("exportBestWeights", "Export lowest validation loss", yesNo(settings.exportBestWeights)),
            item("checkpointEveryEpochs", "Checkpoint every N epochs", "\(settings.checkpointEveryEpochs)"),
            item("device", "Training device", settings.device.displayName)
        ]
        if let data = try? JSONEncoder().encode(settings),
           let encoded = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let levels = encoded["sourceResolutionLevels"] as? Int {
            values.insert(item("sourceResolutionLevels", "Source resolution levels", levels == 0 ? "All available" : "\(levels)"), at: 1)
        }
        return values
    }
}

struct TrainingSettingsTable: View {
    let first: TrainingSettings?
    var second: TrainingSettings?
    var differencesOnly = false

    var body: some View {
        if let first {
            let firstValues = TrainingSettingValue.values(first)
            let secondValues = second.map(TrainingSettingValue.values)
            let rows = firstValues.filter { row in
                !differencesOnly || secondValues?.first(where: { $0.id == row.id })?.value != row.value
            }
            if rows.isEmpty {
                Text("These versions have the same saved training settings.").font(.caption).foregroundStyle(.secondary)
            } else {
                Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 7) {
                    if second != nil {
                        GridRow { Text("Setting"); Text("Model A"); Text("Model B") }.fontWeight(.semibold)
                    }
                    ForEach(rows) { row in
                        GridRow(alignment: .top) {
                            Text(row.label).foregroundStyle(.secondary).frame(width: 190, alignment: .leading)
                            Text(row.value).textSelection(.enabled)
                            if let secondValues { Text(secondValues.first { $0.id == row.id }?.value ?? "—").textSelection(.enabled) }
                        }
                    }
                }.font(.caption).monospacedDigit()
            }
            if second == nil && differencesOnly {
                Text("The other version has no saved settings metadata.").font(.caption).foregroundStyle(.secondary)
            }
        } else {
            Text("Saved settings metadata is unavailable for this older version.").font(.caption).foregroundStyle(.secondary)
        }
    }
}
