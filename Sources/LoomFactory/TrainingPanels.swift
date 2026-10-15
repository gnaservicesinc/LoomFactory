import AppKit
import LoomCore
import SwiftUI

struct TrainingPanel: View {
    @Bindable var model: FactoryModel
    @State private var customArchitecture = false
    @State private var optimizerTuning = false
    @State private var comparingModels = false
    @State private var managedVersionID = ""

    init(model: FactoryModel, showTuning: Bool = false) {
        self.model = model
        _customArchitecture = State(initialValue: showTuning)
        _optimizerTuning = State(initialValue: showTuning)
    }

    var body: some View {
        FactoryPanel("Train a dataset", subtitle: "Train with native Swift and the Apple GPU from paired diffuse and material maps.") {
            datasetSelection
            trainedModels
            pairReadiness
            trainingSection("Model", symbol: "cube") { architectureSettings }
            trainingSection("Optimizer and batches", symbol: "slider.horizontal.3") { optimizerSettings }
            trainingSection("Run", symbol: "play.circle") { runSettings }
            trainingSection("Crops and image order", symbol: "crop.rotate") { cropSettings }
            trainingSection("Learning-rate schedule", symbol: "chart.xyaxis.line") { learningRateSettings }
            trainingSection("Validation and checkpoints", symbol: "checkmark.seal") { validationSettings }
            runSummary
            if let issue = model.trainingOptionsIssue {
                Label(issue, systemImage: "exclamationmark.triangle")
                    .font(.callout).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("training.optionsIssue")
            }
            FactoryActionButton(title: startButtonTitle, symbol: "play.fill", prominent: true) {
                model.trainMap()
            }.disabled(!model.canTrain)
                .accessibilityIdentifier("training.start")
            TrainingRunDeletionControl(model: model)
        }
        .disabled(model.busy)
        .sheet(isPresented: $comparingModels) { ModelComparisonView(model: model) }
    }

    private var datasetSelection: some View {
        HStack(alignment: .top, spacing: 20) {
            VStack(alignment: .leading, spacing: 7) {
                Text("Dataset").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                Picker("Dataset", selection: Binding(get: { model.selectedDataset?.id ?? "" }, set: { model.selectedDatasetID = $0 })) {
                    Text("Choose a dataset").tag("")
                    ForEach(model.snapshot?.datasets ?? [], id: \.id) { Text($0.name).tag($0.id) }
                }.labelsHidden().frame(maxWidth: .infinity)
                    .accessibilityIdentifier("training.dataset")
            }
            VStack(alignment: .leading, spacing: 7) {
                Text("Map to train").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                Picker("Map to train", selection: $model.selectedMap) {
                    ForEach(MapRole.targets) { Text($0.displayName).tag($0) }
                }.labelsHidden().frame(maxWidth: .infinity)
                    .accessibilityIdentifier("training.map")
            }
        }
    }

    private var pairReadiness: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 9) {
                Image(systemName: "square.stack.3d.up").foregroundStyle(FactoryStyle.accent)
                if let dataset = model.selectedDataset {
                    Text("\(model.eligiblePairCount.formatted()) \(model.selectedMap.displayName.lowercased()) \(model.eligiblePairCount == 1 ? "pair" : "pairs") ready in \(dataset.name)")
                        .font(.callout.weight(.medium))
                } else {
                    Text("Choose the dataset to train.").font(.callout).foregroundStyle(.secondary)
                }
            }
            Text("Uses current files tagged for this dataset with matching map types and native dimensions. Maps share each crop and transform; originals are preserved.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if let plan = model.trainingPlan {
                Divider().padding(.vertical, 4)
                TrainingSourceSummary(plan: plan)
            } else if model.selectedDataset != nil {
                let sources = model.selectedDatasetSourceCounts
                Text("\(countLabel(sources.folders, singular: "source material folder", plural: "source material folders")) · \(countLabel(sources.images, singular: "image file", plural: "image files"))")
                    .font(.caption).monospacedDigit().accessibilityIdentifier("training.sourceCounts")
            }
        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(FactoryStyle.accent.opacity(0.06), in: RoundedRectangle(cornerRadius: 9))
    }

    private var runSettings: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 11) {
            row("Training dimensions", detail: "Native pixels; larger maps supply random crops.") {
                Picker("Training dimensions", selection: imageSizeBinding) {
                    ForEach([256, 512, 1024, 2048, 4096], id: \.self) { value in
                        Text("\(value.formatted()) × \(value.formatted())").tag(value)
                    }
                }.labelsHidden().disabled(continuing).accessibilityIdentifier("training.imageSize")
            }
            row("Source resolution levels", detail: "Per material folder, start with the smallest usable original. Level 2 includes sources up to 2× that resolution; level 3 includes up to 4×.") {
                Picker("Source resolution levels", selection: $model.settings.training.sourceResolutionLevels) {
                    Text("1 (base)").tag(1)
                    Text("2 (up to 2×)").tag(2)
                    Text("3 (up to 4×)").tag(3)
                    Text("All available").tag(0)
                }.labelsHidden().accessibilityIdentifier("training.sourceResolutionLevels")
            }
            numberRow("Epochs", value: $model.settings.training.epochs, detail: "One pass through the training images.", identifier: "training.epochs")
            row("Training device", detail: "Native MPSGraph training uses the Apple GPU.") {
                Picker("Training device", selection: $model.settings.training.device) {
                    ForEach([TrainingDevice.automatic, .metal]) { Text($0.displayName).tag($0) }
                    if model.settings.training.device == .cpu { Text("CPU (unavailable for native training)").tag(TrainingDevice.cpu) }
                }.labelsHidden().accessibilityIdentifier("training.device")
            }
            numberRow("Random seed", value: $model.settings.training.seed, detail: "-1 uses a new system seed for a fresh run. Saved runs record their seed; use it to repeat crop, shuffle and initialization choices.", identifier: "training.seed")
        }
    }

    private var architectureSettings: some View {
        VStack(alignment: .leading, spacing: 11) {
            Picker("Start this run", selection: Binding(get: { model.effectiveStartMode }, set: { model.setTrainingStartMode($0) })) {
                ForEach(FactoryTrainingStartMode.allCases) { Text($0.displayName).tag($0) }
            }.accessibilityIdentifier("training.startMode")
            if continuing {
                Picker("Saved version", selection: Binding(
                    get: { model.selectedTrainingVersion?.id ?? "" }, set: { model.selectedTrainingVersionID = $0 }
                )) {
                    if model.continuableModelVersions.isEmpty { Text("No saved weights for this map").tag("") }
                    ForEach(model.continuableModelVersions) { version in
                        Text(ModelVersionPresentation.title(version)).tag(version.id)
                    }
                }.accessibilityIdentifier("training.continueVersion")
                Text(model.effectiveStartMode == .exportedModel
                     ? "Starts from the exported weights of this version, including lowest validation loss weights when that run exported them. A fresh optimizer uses the settings below."
                     : "Restores saved weights and compatible optimizer state. Unfinished checkpoints resume their saved image position when settings and sources match. Older checkpoints use a fresh optimizer; changing training settings starts a new epoch schedule.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack {
                    Text("All saved settings are loaded. Edit the controls below for this continuation.")
                    Spacer()
                    Button("Restore saved settings") { model.restoreSelectedTrainingSettings() }
                        .accessibilityIdentifier("training.restoreSavedSettings")
                }
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if model.selectedMap != .normal && model.effectiveTrainingSettings.scalarLoss == .absoluteV1 {
                    Text("This version keeps the original scalar training loss. If its output has saturated to white or black, choose Start a fresh model to use stable scalar training.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("training.legacyScalarLoss")
                }
            } else {
                Text("Starts new weights and keeps every previous version. A successful replacement becomes the model used for package export.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            VStack(alignment: .leading, spacing: 5) {
                Text("Model / checkpoint name").font(.caption.weight(.medium)).foregroundStyle(.secondary)
                TextField("Name this run, e.g. Bark depth · lower learning rate", text: $model.runLabel)
                    .textFieldStyle(.roundedBorder).accessibilityIdentifier("training.runLabel")
            }
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 11) {
                row("Network", detail: selectedNetwork.detail) {
                    Picker("Network", selection: networkBinding) {
                        ForEach(TrainingNetwork.allCases) { network in
                            Text(network.displayName).tag(network)
                        }
                    }.labelsHidden().accessibilityIdentifier("training.network")
                }
                row("Model size", detail: "Larger models use more memory and take longer to train.") {
                    Picker("Model size", selection: presetBinding) {
                        ForEach(TrainingModelPreset.allCases) { preset in
                            Text(preset.label(map: model.selectedMap, network: selectedNetwork)).tag(preset)
                        }
                    }.labelsHidden().accessibilityIdentifier("training.modelPreset")
                }
            }.disabled(continuing)
            HStack(spacing: 8) {
                Image(systemName: "number.square").foregroundStyle(FactoryStyle.accent)
                Text("\(model.effectiveTrainingSettings.parameterCount(map: model.selectedMap).formatted()) trainable parameters")
                    .font(.callout.weight(.semibold)).monospacedDigit()
                Spacer()
                Text("\(model.effectiveTrainingSettings.modelWidth) channels · \(model.effectiveTrainingSettings.modelBlocks) blocks")
                    .font(.caption).foregroundStyle(.secondary)
            }.accessibilityElement(children: .combine).accessibilityIdentifier("training.parameterCount")
            DisclosureGroup("Custom architecture", isExpanded: $customArchitecture) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 11) {
                    numberRow("Channel width", value: architectureWidthBinding, detail: "8–256 channels. Custom sizes are supported.", identifier: "training.modelWidth")
                    numberRow("Residual blocks", value: architectureBlocksBinding, detail: "1–12 blocks. More blocks add capacity and compute.", identifier: "training.modelBlocks")
                }.padding(.top, 8).disabled(continuing)
            }.font(.callout)
        }.toggleStyle(.checkbox)
    }

    private var optimizerSettings: some View {
        VStack(alignment: .leading, spacing: 11) {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 11) {
                row("Optimizer", detail: optimizerDescription) {
                    Picker("Optimizer", selection: $model.settings.training.optimizer) {
                        ForEach(TrainingOptimizer.allCases) { Text($0.displayName).tag($0) }
                    }.labelsHidden().accessibilityIdentifier("training.optimizer")
                }
                numberRow("Learning rate", value: $model.settings.training.learningRate, detail: model.settings.training.optimizer == .lion ? "For Lion, consider starting at 0.0001." : "The starting update size.", identifier: "training.learningRate")
                numberRow("Batch size", value: $model.settings.training.batchSize, detail: "Images processed together; raises memory use.", identifier: "training.batchSize")
                numberRow("Gradient accumulation", value: $model.settings.training.gradientAccumulationSteps, detail: "Batches per update; increases effective batch size without a larger image batch.", identifier: "training.gradientAccumulation")
                row("Gradient clipping", detail: "Clips the global L2 norm after all accumulated gradients are averaged, before the optimizer update.") {
                    Picker("Gradient clipping", selection: $model.settings.training.gradientClipPolicy) {
                        ForEach(TrainingGradientClipping.allCases) { Text($0.displayName).tag($0) }
                    }.labelsHidden().accessibilityIdentifier("training.gradientClipPolicy")
                }
                if model.settings.training.gradientClipPolicy != .off {
                    numberRow("Max norm", value: $model.settings.training.gradientClipNorm, detail: "Maximum global norm; warmup increases toward this cap. 0 disables clipping.", identifier: "training.gradientClipping")
                    if model.settings.training.gradientClipPolicy == .warmup {
                        numberRow("Initial max norm", value: $model.settings.training.gradientClipWarmupStartNorm, detail: "Smaller cap for the first update; must be at most Max norm.", identifier: "training.gradientClipWarmupStartNorm")
                        numberRow("Warmup updates", value: $model.settings.training.gradientClipWarmupUpdates, detail: "Optimizer updates until the cap reaches Max norm. Separate from learning-rate warmup.", identifier: "training.gradientClipWarmupUpdates")
                    }
                }
            }
            DisclosureGroup("Optimizer tuning", isExpanded: $optimizerTuning) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 11) {
                    numberRow("Weight decay", value: $model.settings.training.weightDecay, detail: "Regularization strength. 0 disables it.", identifier: "training.weightDecay")
                    if model.settings.training.optimizer == .sgd {
                        numberRow("Momentum", value: $model.settings.training.momentum, detail: "Carry part of the previous update forward.", identifier: "training.momentum")
                    } else {
                        numberRow("Beta 1", value: $model.settings.training.beta1, detail: "First-moment decay.", identifier: "training.beta1")
                        numberRow("Beta 2", value: $model.settings.training.beta2, detail: model.settings.training.optimizer == .lion ? "Momentum decay for Lion; 0.99 is a usual starting value." : "Second-moment decay.", identifier: "training.beta2")
                    }
                }.padding(.top, 8)
            }.font(.callout)
        }
    }

    private var learningRateSettings: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 11) {
            row("Schedule", detail: model.settings.training.learningRateSchedule == .constant ? "Keeps the rate steady after warmup." : "Gradually lowers the rate after warmup.") {
                Picker("Learning-rate schedule", selection: $model.settings.training.learningRateSchedule) {
                    ForEach(TrainingLearningRateSchedule.allCases) { Text($0.displayName).tag($0) }
                }.labelsHidden().accessibilityIdentifier("training.learningRateSchedule")
            }
            numberRow("Warmup epochs", value: $model.settings.training.warmupEpochs, detail: "Start with smaller updates. 0 skips warmup.", identifier: "training.warmupEpochs")
            if model.settings.training.learningRateSchedule == .cosine {
                numberRow("Final rate ratio", value: $model.settings.training.minimumLearningRateRatio, detail: "Fraction of the starting rate at the end; 0.1 means 10%.", identifier: "training.minimumLearningRateRatio")
            }
        }
    }

    private var cropSettings: some View {
        VStack(alignment: .leading, spacing: 12) {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 11) {
                row("New crop positions", detail: refreshDescription(model.settings.training.cropRefresh, transform: false)) {
                    refreshPicker("New crop positions", selection: $model.settings.training.cropRefresh)
                        .accessibilityIdentifier("training.cropRefresh")
                }
            }
            Toggle("Randomize image order each epoch", isOn: $model.settings.training.shuffleEachEpoch)
                .help("Off keeps the image order the same for every epoch.")
                .accessibilityIdentifier("training.shuffle")
            HStack(spacing: 24) {
                Toggle("Randomly flip crops", isOn: $model.settings.training.flipCrops)
                    .accessibilityIdentifier("training.flip")
                Toggle("Randomly rotate crops", isOn: $model.settings.training.rotateCrops)
                    .accessibilityIdentifier("training.rotate")
            }
            if model.settings.training.flipCrops || model.settings.training.rotateCrops {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 11) {
                    row("New flips and rotations", detail: refreshDescription(model.settings.training.augmentationRefresh, transform: true)) {
                        refreshPicker("New flips and rotations", selection: $model.settings.training.augmentationRefresh)
                            .accessibilityIdentifier("training.augmentationRefresh")
                    }
                }
                Text("Random flips choose horizontal and vertical directions. Rotations use quarter turns; normal-map vectors follow the same transforms.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if model.selectedMap == .depth {
                Toggle("Exaggerate features", isOn: exaggerationBinding).disabled(continuing)
                    .help("Apply nonlinear contrast to depth values during training. Original images are unchanged.")
                    .accessibilityIdentifier("training.exaggerateFeatures")
                if model.effectiveTrainingSettings.exaggerateFeatures {
                    Text("Nonlinear contrast emphasizes depth features while preserving the map’s range.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }.toggleStyle(.checkbox)
    }

    private var validationSettings: some View {
        VStack(alignment: .leading, spacing: 12) {
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 11) {
                row("Validation holdout", detail: "Percent of material folders reserved for validation. 0 uses all pairs for training.") {
                    HStack(spacing: 4) {
                        TextField("Validation holdout percent", value: validationPercentBinding, format: .number)
                            .textFieldStyle(.roundedBorder).accessibilityIdentifier("training.validationFraction")
                        Text("%").foregroundStyle(.secondary)
                    }
                }
                if model.settings.training.validationFraction > 0 {
                    numberRow("Validation seed", value: $model.settings.training.validationSeed, detail: "Keeps the same material folders in the holdout.", identifier: "training.validationSeed")
                }
                if model.settings.training.validationFraction > 0 || model.settings.training.earlyStoppingPatience > 0 {
                    numberRow("Early stopping", value: $model.settings.training.earlyStoppingPatience,
                              detail: model.settings.training.validationFraction > 0
                                ? "Stop after N epochs without validation improvement. 0 keeps all epochs."
                                : "Set to 0 to train without validation, or enable a validation holdout.",
                              identifier: "training.earlyStopping")
                }
                numberRow("Checkpoint every", value: $model.settings.training.checkpointEveryEpochs, detail: "Save weights every N epochs, and at the end.", identifier: "training.checkpointEveryEpochs")
            }
            if model.settings.training.validationFraction > 0 {
                Toggle("Export weights with the lowest validation loss", isOn: $model.settings.training.exportBestWeights)
                    .toggleStyle(.checkbox).accessibilityIdentifier("training.exportBest")
                    .help("Off exports the final training weights. Lowest loss does not always produce the best-looking material.")
            }
            if model.trainingPlan?.validationPairCount == 0 {
                Text("Without held-out validation, export uses the final weights. You can test and select a saved checkpoint if an earlier result looks better.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("training.finalWeightsWithoutValidation")
            }
        }
    }

    private var runSummary: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("This run").font(.subheadline.weight(.semibold))
            Text("\(model.settings.training.optimizer.displayName) · \(effectiveBatchDescription)")
                .font(.callout).monospacedDigit()
            if let plan = model.trainingPlan {
                Text("\(countLabel(plan.trainingPairCount, singular: "training pair", plural: "training pairs")) in \(countLabel(plan.trainingMaterialCount, singular: "material", plural: "materials")) · \(countLabel(plan.validationPairCount, singular: "validation pair", plural: "validation pairs")) in \(countLabel(plan.validationMaterialCount, singular: "material", plural: "materials"))")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                Text("Per epoch: \(countLabel(plan.batchesPerEpoch, singular: "batch", plural: "batches")) · \(countLabel(plan.optimizerStepsPerEpoch, singular: "optimizer update", plural: "optimizer updates"))")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                Text("Planned across \(model.effectiveTrainingSettings.epochs.formatted()) epochs: \(plan.trainingSampleVisits.formatted()) training pair uses · \(plan.validationSampleVisits.formatted()) validation pair uses · \(plan.totalBatches.formatted()) batches · \(plan.totalOptimizerUpdates.formatted()) optimizer updates")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("training.plannedWork")
                Text("Each pair supplies one crop per epoch. Early stopping or Stop can shorten the run.")
                    .font(.caption).foregroundStyle(.secondary)
                if model.settings.training.validationFraction > 0 && plan.validationPairCount == 0 {
                    Text("This dataset needs at least two source materials for a validation holdout. All available pairs will train.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("Crop positions: \(model.settings.training.cropRefresh.displayName.lowercased()) · Image order: \(model.settings.training.shuffleEachEpoch ? "reshuffled each epoch" : "same each epoch")")
                .font(.caption).foregroundStyle(.secondary)
            if model.settings.training.flipCrops || model.settings.training.rotateCrops {
                Text("Flips and rotations: \(model.settings.training.augmentationRefresh.displayName.lowercased())")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Validation reserves whole material folders so their maps stay together.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 9))
            .accessibilityIdentifier("training.summary")
    }

    @ViewBuilder private var trainedModels: some View {
        if !model.selectedModelVersions.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Saved checkpoints and models").font(.subheadline.weight(.semibold))
                    Spacer()
                    Button("A/B Compare", systemImage: "rectangle.split.2x1") { comparingModels = true }
                        .disabled(model.exportedModelVersions.count < 2).accessibilityIdentifier("training.compare")
                }
                Text("Select a saved run to rename, add notes, test, or load its checkpoint and settings.")
                    .font(.caption).foregroundStyle(.secondary)
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 3) {
                        ForEach(model.selectedModelVersions) { version in
                            Button {
                                managedVersionID = version.id
                            } label: {
                                HStack {
                                    Image(systemName: version.checkpoint == nil ? "cube" : "square.stack.3d.up")
                                    Text(ModelVersionPresentation.title(version)).lineLimit(1)
                                    Spacer()
                                    Text(version.status.capitalized).font(.caption).foregroundStyle(.secondary)
                                    if managedVersion?.id == version.id { Image(systemName: "checkmark") }
                                }.padding(7).frame(maxWidth: .infinity, alignment: .leading)
                                    .background(managedVersion?.id == version.id ? FactoryStyle.accent.opacity(0.12) : Color.clear,
                                        in: RoundedRectangle(cornerRadius: 6))
                            }.buttonStyle(.plain)
                                .accessibilityIdentifier("training.checkpoint.select.\(version.id)")
                        }
                    }
                }.frame(height: min(140, CGFloat(model.selectedModelVersions.count) * 34))
                if let version = managedVersion {
                    ModelVersionRow(model: model, version: version).id(version.id)
                }
            }.padding(.vertical, 8)
            Divider()
        }
    }

    private var managedVersion: ModelVersion? {
        model.selectedModelVersions.first { $0.id == managedVersionID }
            ?? model.selectedModelVersions.first { $0.id == model.selectedTrainingVersion?.id }
            ?? model.selectedModelVersions.first
    }

    private var startButtonTitle: String {
        if continuing { return "Continue \(model.selectedMap.displayName) Training" }
        return model.selectedModelVersions.isEmpty ? "Train \(model.selectedMap.displayName) Model" : "Train Replacement \(model.selectedMap.displayName) Model"
    }

    private var effectiveBatchDescription: String {
        let batch = model.settings.training.batchSize
        let accumulation = model.settings.training.gradientAccumulationSteps
        let (total, overflow) = batch.multipliedReportingOverflow(by: accumulation)
        guard batch > 0, accumulation > 0, !overflow else { return "Check batch and accumulation settings" }
        return "batch \(batch) × \(accumulation) accumulation = up to \(countLabel(total, singular: "image", plural: "images")) per update"
    }

    private func countLabel(_ count: Int, singular: String, plural: String) -> String {
        "\(count.formatted()) \(count == 1 ? singular : plural)"
    }

    private var continuing: Bool { model.continuingTraining }

    private var imageSizeBinding: Binding<Int> {
        Binding(get: { model.effectiveTrainingSettings.imageSize }, set: { model.settings.training.imageSize = $0 })
    }

    private var exaggerationBinding: Binding<Bool> {
        Binding(get: { model.effectiveTrainingSettings.exaggerateFeatures }, set: { model.settings.training.exaggerateFeatures = $0 })
    }

    private var selectedNetwork: TrainingNetwork {
        TrainingNetwork(rawValue: model.effectiveTrainingSettings.modelArchitecture) ?? .pyramid
    }

    private var networkBinding: Binding<TrainingNetwork> {
        Binding(get: { selectedNetwork }, set: { network in
            guard network != selectedNetwork else { return }
            let currentPreset = TrainingModelPreset.allCases.first {
                $0.width(network: selectedNetwork) == model.settings.training.modelWidth && model.settings.training.modelBlocks == 4
            }
            model.settings.training.modelArchitecture = network.rawValue
            if let width = currentPreset?.width(network: network) {
                model.settings.training.modelWidth = width
            } else {
                // Custom widths remain directly editable; changing networks
                // selects a practical size before the user adjusts them.
                model.settings.training.modelWidth = TrainingModelPreset.large.width(network: network)!
            }
        })
    }

    private var presetBinding: Binding<TrainingModelPreset> {
        Binding(get: {
            customArchitecture ? .custom : (TrainingModelPreset.allCases.first { $0.width(network: selectedNetwork) == model.effectiveTrainingSettings.modelWidth && model.effectiveTrainingSettings.modelBlocks == 4 } ?? .custom)
        }, set: { preset in
            guard let width = preset.width(network: selectedNetwork) else { customArchitecture = true; return }
            customArchitecture = false
            model.settings.training.modelWidth = width
            model.settings.training.modelBlocks = 4
        })
    }

    private var architectureWidthBinding: Binding<Int> {
        Binding(get: { model.effectiveTrainingSettings.modelWidth }, set: { model.settings.training.modelWidth = $0 })
    }

    private var architectureBlocksBinding: Binding<Int> {
        Binding(get: { model.effectiveTrainingSettings.modelBlocks }, set: { model.settings.training.modelBlocks = $0 })
    }

    private var validationPercentBinding: Binding<Double> {
        Binding(get: { model.settings.training.validationFraction * 100 }, set: { model.settings.training.validationFraction = $0 / 100 })
    }

    private var optimizerDescription: String {
        switch model.settings.training.optimizer {
        case .adamW: "Adaptive updates with decoupled weight decay."
        case .adam: "Adaptive updates with L2 regularization."
        case .sgd: "Momentum-based gradient descent."
        case .lion: "Sign-based updates with a smaller optimizer state."
        }
    }

    private func refreshPicker(_ title: String, selection: Binding<TrainingRefresh>) -> some View {
        Picker(title, selection: selection) {
            ForEach(TrainingRefresh.allCases) { Text($0.displayName).tag($0) }
        }.labelsHidden()
    }

    private func refreshDescription(_ refresh: TrainingRefresh, transform: Bool) -> String {
        let choices = transform ? "transforms" : "positions"
        let cadence = switch refresh {
        case .perSample: "Choose fresh \(choices) every time an image is used."
        case .perEpoch: "Choose \(choices) once per image each epoch."
        case .perRun: "Choose \(choices) once per image for this run."
        }
        return transform ? cadence : cadence + " All valid native pixel origins are eligible, including the image edges."
    }

    @ViewBuilder private func trainingSection<Content: View>(_ title: String, symbol: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: symbol).font(.subheadline.weight(.semibold))
            content()
        }.padding(.top, 5)
        Divider()
    }

    private func row<Control: View>(_ title: String, detail: String, @ViewBuilder control: () -> Control) -> some View {
        GridRow(alignment: .top) {
            Text(title).font(.callout).frame(width: 150, alignment: .leading).padding(.top, 4)
            control().frame(width: 165, alignment: .leading).font(.callout)
            Text(detail).font(.caption).foregroundStyle(.secondary)
                .frame(minWidth: 130, maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true).padding(.top, 4)
        }
    }

    private func numberRow(_ title: String, value: Binding<Int>, detail: String, identifier: String) -> some View {
        row(title, detail: detail) {
            TextField(title, value: value, format: .number).textFieldStyle(.roundedBorder)
                .accessibilityIdentifier(identifier)
        }
    }

    private func numberRow(_ title: String, value: Binding<Double>, detail: String, identifier: String) -> some View {
        row(title, detail: detail) {
            TextField(title, value: value, format: .number).textFieldStyle(.roundedBorder)
                .accessibilityIdentifier(identifier)
        }
    }
}

struct TrainingSourceSummary: View {
    let plan: TrainingDataPlan

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("\(countLabel(plan.sourceMaterialCount, singular: "source material folder", plural: "source material folders")) · \(countLabel(plan.sourceImageCount, singular: "image file", plural: "image files"))")
                .font(.callout.weight(.medium)).monospacedDigit()
                .accessibilityIdentifier("training.sourceCounts")
            Text("Will train \(countLabel(plan.trainingPairCount, singular: "pair", plural: "pairs")) from \(countLabel(plan.trainingMaterialCount, singular: "folder", plural: "folders")); validate \(countLabel(plan.validationPairCount, singular: "pair", plural: "pairs")) from \(countLabel(plan.validationMaterialCount, singular: "folder", plural: "folders")).")
                .font(.caption).monospacedDigit().fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("training.selectedCounts")
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 5) {
                GridRow {
                    Text("Original resolution")
                    Text("Usable pairs")
                    Text("Train")
                    Text("Validate")
                }.font(.caption.weight(.medium)).foregroundStyle(.secondary)
                ForEach(plan.resolutions) { resolution in
                    GridRow {
                        Text(resolution.displayName)
                        Text(resolution.eligiblePairCount.formatted())
                        Text(resolution.trainingPairCount.formatted())
                        Text(resolution.validationPairCount.formatted())
                    }.font(.caption).monospacedDigit()
                        .foregroundStyle(resolution.totalPairCount > 0 ? .primary : .secondary)
                        .accessibilityElement(children: .combine)
                        .accessibilityIdentifier("training.resolution.\(resolution.id)")
                }
            }
            if plan.excludedPairCount > 0 {
                Text("\(plan.excludedPairCount.formatted()) usable pairs excluded by the resolution-level setting.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if plan.unavailableMaterialCount > 0 {
                Text("\(plan.unavailableMaterialCount.formatted()) folders have no selected pair for this map and crop size.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Image files include diffuse inputs and maps. A usable pair needs both maps at matching dimensions, large enough for the training crop.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func countLabel(_ count: Int, singular: String, plural: String) -> String {
        "\(count.formatted()) \(count == 1 ? singular : plural)"
    }
}

enum TrainingModelPreset: String, CaseIterable, Identifiable {
    case compact, small, medium, large, extraLarge, huge, custom
    var id: String { rawValue }
    func width(network: TrainingNetwork) -> Int? {
        switch (network, self) {
        case (_, .custom): nil
        case (.pyramid, .compact): 8
        case (.pyramid, .small): 19
        case (.pyramid, .medium): 27
        case (.pyramid, .large): 38
        case (.pyramid, .extraLarge): 47
        case (.pyramid, .huge): 55
        case (.residual, .compact): 24
        case (.residual, .small): 84
        case (.residual, .medium): 116
        case (.residual, .large): 164
        case (.residual, .extraLarge): 204
        case (.residual, .huge): 236
        }
    }
    func label(map: MapRole, network: TrainingNetwork) -> String {
        guard let width = width(network: network) else { return "Custom" }
        let count = TrainingSettings(modelWidth: width, modelArchitecture: network.rawValue).parameterCount(map: map)
        return count.formatted(.number.notation(.compactName)) + " parameters"
    }
}

struct TrainingOverviewPanel: View {
    @Bindable var model: FactoryModel

    var body: some View {
        FactoryPanel("This run") {
            VStack(alignment: .leading, spacing: 6) {
                Text(model.selectedDataset?.name ?? "Choose a dataset").font(.headline)
                Text("\(model.selectedMap.displayName) model").font(.callout).foregroundStyle(.secondary)
            }
            Divider()
            if model.selectedDataset != nil {
                summary("Source folders", model.selectedDatasetSourceCounts.folders.formatted())
                summary("Image files", model.selectedDatasetSourceCounts.images.formatted())
            }
            summary("Parameters", model.effectiveTrainingSettings.parameterCount(map: model.selectedMap).formatted())
            summary("Optimizer", model.settings.training.optimizer.displayName)
            summary("Learning rate", model.settings.training.learningRate.formatted(.number.precision(.significantDigits(1...4))))
            summary("Effective batch", "Up to \(model.settings.training.effectiveBatchSize.formatted()) images")
            summary("Gradient clipping", model.settings.training.gradientClipPolicy.displayName)
            if model.continuingTraining, let version = model.selectedTrainingVersion {
                Text("Continuing \(model.effectiveStartMode == .exportedModel ? "exported model" : "saved checkpoint") · \(ModelVersionPresentation.title(version))")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if let plan = model.trainingPlan {
                Divider()
                summary("Resolution levels", model.effectiveTrainingSettings.sourceResolutionLevels == 0 ? "All available" : model.effectiveTrainingSettings.sourceResolutionLevels.formatted())
                summary("Training pairs", plan.trainingPairCount.formatted())
                summary("Validation pairs", plan.validationPairCount.formatted())
                summary("Training pair uses", plan.trainingSampleVisits.formatted())
                summary("Planned updates", plan.totalOptimizerUpdates.formatted())
                if model.settings.training.validationFraction > 0 && plan.validationPairCount == 0 {
                    Text("Validation needs at least two source materials.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("Crop positions: \(model.settings.training.cropRefresh.displayName.lowercased())")
                .font(.caption).foregroundStyle(.secondary)
            if model.settings.training.flipCrops || model.settings.training.rotateCrops {
                Text("Flips and rotations: \(model.settings.training.augmentationRefresh.displayName.lowercased())")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let issue = model.trainingOptionsIssue {
                Label(issue, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            FactoryActionButton(title: "Train \(model.selectedMap.displayName) Model", symbol: "play.fill", prominent: true) { model.trainMap() }
                .disabled(!model.canTrain).accessibilityIdentifier("training.overviewStart")
        }
    }

    private func summary(_ title: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).foregroundStyle(.secondary)
            Spacer()
            Text(value).monospacedDigit().multilineTextAlignment(.trailing)
        }.font(.callout)
    }
}

struct CreateMLPanel: View {
    @Bindable var model: FactoryModel

    var body: some View {
        FactoryPanel("Create ML", subtitle: "Prepare the images for your dataset classifier.") {
            Text("Export category folders with one native 1K diffuse image per material. Train the image classifier in Create ML, then import its Core ML model.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            FactoryActionButton(title: "Export Create ML Data…", symbol: "square.and.arrow.up") {
                model.exportClassifierData()
            }.disabled(model.busy || model.snapshot?.images.isEmpty != false)
            FactoryActionButton(title: "Import Classifier…", symbol: "square.and.arrow.down") {
                model.importClassifier()
            }.disabled(model.busy || model.snapshot == nil)
            if let classifier = model.classifier {
                HStack {
                    FactoryStatus(text: "Classifier ready", symbol: "checkmark.circle.fill", color: .green)
                    Spacer()
                    Button("Show Model") { model.reveal(classifier.directory) }
                }
                Text("Updated \(classifier.createdAt.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }.disabled(model.busy)
    }
}

struct PackagePanel: View {
    @Bindable var model: FactoryModel

    var body: some View {
        FactoryPanel("Texture Loom package", subtitle: "One package for a custom Texture Studio build.") {
            VStack(alignment: .leading, spacing: 12) {
                labeledField("Model name", text: $model.settings.modelName, prompt: "Texture Loom")
                labeledField("Version", text: $model.settings.modelVersion, prompt: "1.0.0")
            }
            if !model.modelCoverage.isEmpty {
                Text("\(model.availableRequiredModelCount) of \(model.requiredModelCount) required models available · \(model.approvedAssignmentCount) chosen and approved")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if model.hasIncompleteCoverage {
                    Text("This export has partial coverage. Review missing products in Models or mark intentionally omitted products optional.")
                        .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                }
                Button("Manage Models", systemImage: "checklist") { model.workspaceSection = .models }
                    .accessibilityIdentifier("package.manageModels")
            }
            HStack {
                Text("Included models").foregroundStyle(.secondary)
                Spacer()
                Text(model.activeMapVersions.count.formatted()).fontWeight(.semibold)
            }.font(.callout)
            ForEach(model.activeMapVersions) { version in
                Text("\(version.datasetName) · \(version.map.displayName) · \(ModelVersionPresentation.title(version))")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            FactoryActionButton(title: model.hasIncompleteCoverage ? "Export Partial Texture Loom" : "Export Texture Loom", symbol: "shippingbox", prominent: true) { model.exportPackage() }
                .disabled(model.busy || model.classifier == nil || (model.snapshot?.artifacts.filter { $0.kind == "map" }.isEmpty ?? true) ||
                          model.settings.modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.settings.modelVersion.isEmpty)
            if let package = model.exportedPackage {
                Button { model.reveal(package) } label: {
                    Label(package.lastPathComponent, systemImage: "checkmark.circle").font(.caption).lineLimit(2)
                }.buttonStyle(.plain).foregroundStyle(.green)
            }
            Divider()
            labeledField("Hugging Face repository", text: $model.settings.huggingFaceRepository, prompt: "Account/Texture-Loom")
            Button("Publish Exported Package", systemImage: "arrow.up.circle") { model.publishPackage() }
                .frame(maxWidth: .infinity).buttonStyle(.bordered).controlSize(.large)
                .disabled(model.busy || model.exportedPackage == nil || model.settings.huggingFaceRepository.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            if let url = model.publishedURL { Link("Open on Hugging Face", destination: url).font(.caption) }
        }.disabled(model.busy)
    }

    private func labeledField(_ title: String, text: Binding<String>, prompt: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption.weight(.medium)).foregroundStyle(.secondary)
            TextField(prompt, text: text).textFieldStyle(.roundedBorder)
        }
    }
}

struct RunPanel: View {
    @Bindable var model: FactoryModel
    @State private var showLog = false
    @State private var showMeasurements = true

    var body: some View {
        FactoryPanel("Run progress") {
            HStack(spacing: 14) {
                if model.busy { ProgressView().controlSize(.small) }
                Text(model.trainingProgress == nil || !model.isTraining ? model.activityTitle : "Training run")
                    .font(.callout).textSelection(.enabled)
                Spacer()
                if model.busy && !model.isTraining { Button("Stop") { model.cancel() }.accessibilityIdentifier("training.stop") }
            }
            if model.isTraining {
                HStack(spacing: 10) {
                    Button("Save Checkpoint", systemImage: "square.and.arrow.down") { model.saveCheckpoint() }
                        .accessibilityIdentifier("training.saveCheckpoint")
                    Button("Save & Stop", systemImage: "stop.fill") { model.saveAndStopTraining() }
                        .accessibilityIdentifier("training.saveAndStop")
                }
                .disabled(model.checkpointRequestPending)
                Text("Saves the latest completed optimizer update, including optimizer state and epoch position.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if let status = model.checkpointStatus {
                Text(status).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("training.checkpointStatus")
            }
            if let directory = model.lastCheckpointDirectory {
                Button("Show Checkpoint in Finder", systemImage: "folder") { model.reveal(directory.appendingPathComponent("checkpoint.json")) }
                    .font(.caption).accessibilityIdentifier("training.revealCheckpoint")
            }
            if let progress = model.progress, model.trainingProgress == nil {
                ProgressView(value: progress).progressViewStyle(.linear)
            }
            let metrics = TrainingRunMetrics(events: model.measurementEvents + model.events)
            if let progress = model.trainingProgress {
                TrainingProgressView(progress: progress, isRunning: model.busy)
            }
            DisclosureGroup("Training measurements", isExpanded: $showMeasurements) {
              VStack(alignment: .leading, spacing: 7) {
                if let epoch = metrics.epoch { metric("Epoch", epoch.formatted()) }
                if let loss = metrics.trainingLoss { metric("Training loss", loss.formatted(.number.precision(.significantDigits(5)))) }
                if let loss = metrics.validationLoss { metric("Validation loss", loss.formatted(.number.precision(.significantDigits(5)))) }
                if let update = metrics.update {
                    metric("Optimizer update", "\(update.formatted())\(metrics.totalUpdates.map { " / " + $0.formatted() } ?? "")")
                }
                if let rate = metrics.learningRate { metric("Learning rate", rate.formatted(.number.precision(.significantDigits(1...4)))) }
                if let norm = metrics.gradientNorm { metric("Global gradient norm", ModelVersionPresentation.number(norm)) }
                if let threshold = metrics.gradientClipThreshold { metric("Current clipping cap", ModelVersionPresentation.number(threshold)) }
                if let batch = metrics.effectiveBatchSize { metric("Images per update", "Up to \(batch.formatted())") }
                if let count = metrics.parameterCount { metric("Parameters", count.formatted()) }
                if let timing = model.latestTrainingMeasurements?.cycleTiming ?? model.events.last(where: { $0.cycleTiming != nil })?.cycleTiming {
                    metric("Image preparation", timing.preparationSeconds.formatted(.number.precision(.fractionLength(2))) + " s")
                    metric("Wait for images", timing.preparationWaitSeconds.formatted(.number.precision(.fractionLength(2))) + " s")
                    metric("Training", timing.trainingSeconds.formatted(.number.precision(.fractionLength(2))) + " s")
                    metric("Model update", timing.optimizerSeconds.formatted(.number.precision(.fractionLength(2))) + " s")
                }
              }
            }
            if !model.events.isEmpty {
                DisclosureGroup("Run details", isExpanded: $showLog) {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 7) {
                            ForEach(Array(model.events.enumerated()), id: \.offset) { _, event in
                                Text(event.message).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }.padding(.vertical, 8)
                    }.frame(height: 180)
                }
            }
        }
    }

    private func metric(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title)
            Spacer(minLength: 8)
            Text(value).monospacedDigit()
        }.font(.caption).foregroundStyle(.secondary)
    }

}

/// Status and checkpoint events keep the last reported measurements visible.
struct TrainingRunMetrics {
    let epoch: Int?
    let trainingLoss: Double?
    let validationLoss: Double?
    let learningRate: Double?
    let update: Int?
    let totalUpdates: Int?
    let parameterCount: Int?
    let effectiveBatchSize: Int?
    let gradientNorm: Double?
    let gradientClipThreshold: Double?

    init(events: [TrainingEvent]) {
        epoch = events.last { $0.epoch != nil }?.epoch
        trainingLoss = events.last { $0.loss != nil }?.loss
        validationLoss = events.last { $0.validationLoss != nil }?.validationLoss
        learningRate = events.last { $0.learningRate != nil }?.learningRate
        update = events.last { $0.update != nil }?.update
        totalUpdates = events.last { $0.totalUpdates != nil }?.totalUpdates
        parameterCount = events.last { $0.parameterCount != nil }?.parameterCount
        effectiveBatchSize = events.last { $0.effectiveBatchSize != nil }?.effectiveBatchSize
        gradientNorm = events.last { $0.gradientNorm != nil }?.gradientNorm
        gradientClipThreshold = events.last { $0.gradientClipThreshold != nil }?.gradientClipThreshold
    }
}
