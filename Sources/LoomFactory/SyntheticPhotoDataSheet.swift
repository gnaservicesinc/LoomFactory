import AppKit
import LoomCore
import SwiftUI
import UniformTypeIdentifiers

struct SyntheticPhotoDataSheet: View {
    @Bindable var model: FactoryModel
    @Environment(\.dismiss) private var dismiss
    @State private var recipe: SyntheticPhotoRecipe
    @State private var destination: URL?
    @State private var issue: String?

    init(model: FactoryModel, initialRecipe: SyntheticPhotoRecipe? = nil) {
        self.model = model
        _recipe = State(initialValue: initialRecipe ?? model.savedSyntheticPhotoRecipe())
    }
    private var plan: SyntheticPhotoPlan? { try? SyntheticPhotoPlan(images: model.snapshot?.images ?? [], recipe: recipe, sourceRoot: model.sourceFolder) }
    private var forecast: String { String(format: "%.2f", Double(plan?.estimatedMaximumBytes ?? 0)/1_073_741_824) }
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Generate Synthetic Photos").font(.title2.weight(.semibold))
            Text("Create degraded color photos paired with clean native 1K targets. Sources, metadata and material models stay untouched.")
                .foregroundStyle(.secondary)
            ScrollView {
                Form {
                    Section("Source materials") {
                        Picker("Category", selection: Binding(get: { recipe.categories.first ?? "" },
                            set: { recipe.categories = $0.isEmpty ? [] : [$0] })) {
                            Text("All categories").tag("")
                            ForEach(Array(Set((model.snapshot?.images ?? []).filter { !$0.ignored }.map { $0.category ?? "Uncategorized" })).sorted(), id: \.self) {
                                Text($0).tag($0)
                            }
                        }
                        Stepper(recipe.materialPercentage == 100 ? "All eligible materials (100%)" : "Random \(recipe.materialPercentage)% within each category",
                                value: $recipe.materialPercentage, in: 1...100, step: 5)
                        TextField("Random seed", value: $recipe.seed, format: .number)
                        Toggle("Include native 1K crops from higher-resolution color maps", isOn: $recipe.includeHigherResolutions)
                        Stepper("Crops per higher-resolution image: \(recipe.cropsPerResolution)", value: $recipe.cropsPerResolution, in: 1...8)
                        Text("Each material stays in one split across every resolution, crop and variant. Larger images are cropped, never resized. One color variant per native resolution is selected.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Section("Synthetic camera effects") {
                        HStack {
                            Text("Enhancer training presets")
                            Button("Repair") { var preset = PhotoEnhancerKind.repair.recipe; preset.seed = recipe.seed; preset.materialPercentage = recipe.materialPercentage; recipe = preset }
                            Button("Lighting") { var preset = PhotoEnhancerKind.lighting.recipe; preset.seed = recipe.seed; preset.materialPercentage = recipe.materialPercentage; recipe = preset }
                        }
                        Text("Use a stage preset when training an enhancer. Mixed lighting and blur/noise recipes remain useful experiments but cannot be imported as a single-purpose stage.")
                            .font(.caption).foregroundStyle(.secondary)
                        Toggle("Exposure (brighter and darker)", isOn: $recipe.exposure)
                        Toggle("Mild defocus blur", isOn: $recipe.blur)
                        Toggle("Approximate sensor noise", isOn: $recipe.noise)
                        Toggle("Uneven illumination and soft hotspots", isOn: $recipe.lighting)
                        Toggle("Perspective / reframing examples", isOn: $recipe.perspective)
                        LabeledContent("Effect strength") {
                            Slider(value: $recipe.strength, in: 0.05...1).frame(width: 220)
                            Text(recipe.strength, format: .percent.precision(.fractionLength(0))).monospacedDigit().frame(width: 42)
                        }
                        Stepper("Variants per crop and task: \(recipe.variantsPerCrop)", value: $recipe.variantsPerCrop, in: 1...16)
                        Toggle("Include an unchanged clean example per crop", isOn: $recipe.includeIdentity)
                        Text("Restoration and reframing are separate tasks. Reframing includes visibility masks. The saved seed and recipe reproduce camera transforms and effects. Clipped highlights may need generated detail.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Section("Output") {
                        Stepper("Maximum output: \(recipe.maximumOutputGiB) GiB", value: $recipe.maximumOutputGiB, in: 1...64)
                        LabeledContent("Folder") {
                            Text(destination?.path ?? "Source project / .loom-factory / synthetic-photo-data")
                                .font(.caption).lineLimit(2).textSelection(.enabled)
                            Button("Change…") { chooseDestination() }
                        }
                        Text("\(plan?.materials.count ?? 0) of \(plan?.eligibleMaterialCount ?? 0) eligible materials · \(plan?.cropCount ?? 0) native crops · \(plan?.pairCount ?? 0) pairs")
                            .font(.headline).accessibilityIdentifier("loom.syntheticPhotoForecast")
                        Text("Conservative disk forecast: \(forecast) GiB. Every run creates a new folder with recipe.json and manifest.json. No model training starts.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }.formStyle(.grouped).disabled(model.busy)
            }
            if let issue { Text(issue).foregroundStyle(.red).textSelection(.enabled) }
            if model.busy {
                HStack { ProgressView().controlSize(.small); Text(model.activityTitle).lineLimit(2); Spacer(); Button("Stop Generation") { model.cancel() } }
            }
            HStack {
                Button("Load Recipe…") { loadRecipe() }.disabled(model.busy)
                if let output = model.lastSyntheticPhotoDirectory {
                    Button("Show Output") { NSWorkspace.shared.activateFileViewerSelecting([output]) }
                    Button("Review Samples") { NSWorkspace.shared.open(output.appendingPathComponent("Review.html")) }
                        .disabled(model.busy)
                }
                Spacer()
                Button("Close") { dismiss() }.disabled(model.busy)
                Button("Generate Dataset") {
                    do { try recipe.validate(); issue = nil; model.generateSyntheticPhotoData(recipe, parent: destination) }
                    catch { issue = error.localizedDescription }
                }.buttonStyle(.borderedProminent)
                    .disabled(model.busy || (plan?.pairCount ?? 0) == 0 || (plan?.estimatedMaximumBytes ?? 0) > Int64(recipe.maximumOutputGiB)*1_073_741_824)
                    .accessibilityIdentifier("loom.generateSyntheticPhotoData")
            }
        }.padding(24).frame(width: 800, height: 780)
            .background(Color(nsColor: .windowBackgroundColor))
            .interactiveDismissDisabled(model.busy)
    }
    private func chooseDestination() {
        let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true
        panel.canCreateDirectories = true; panel.message = "Choose the parent folder for a new synthetic dataset."
        if panel.runModal() == .OK { destination = panel.url }
    }
    private func loadRecipe() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]; panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let value = try JSONDecoder().decode(SyntheticPhotoRecipe.self, from: Data(contentsOf: url))
            try value.validate(); recipe = value; issue = nil
        } catch { issue = "Could not load this recipe: " + error.localizedDescription }
    }
}
