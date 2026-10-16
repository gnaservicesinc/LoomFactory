import AppKit
import LoomCore
import SwiftUI
import UniformTypeIdentifiers

struct PhotoEnhancerSheet: View {
    @Bindable var model: FactoryModel
    @Environment(\.dismiss) private var dismiss
    @State private var settings = PhotoEnhancerSettings()
    @State private var dataset: URL?
    @State private var resume: URL?
    @State private var issue: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Train Photo Enhancers").font(.title2.bold())
            Text("Native RGB photo models, separate from height, normal and roughness. Start with repair, then lighting; generated microdetail and alpha are later stages.")
                .foregroundStyle(.secondary)
            Form {
                Section("1 · Choose a stage and native photo pairs") {
                    Picker("Enhancer", selection: $settings.kind) { ForEach(PhotoEnhancerKind.allCases) { Text($0.title).tag($0) } }
                        .disabled(resume != nil)
                    HStack {
                        Button("Import Pair Manifest…") { importPairs() }
                        Button("Use Imported Dataset…") { chooseDataset() }
                    }
                    Text(dataset?.path ?? "Generate stage-specific pairs in Materials → Synthetic Photos (Repair or Lighting preset), or import real paired 1K photographs.")
                        .font(.caption).textSelection(.enabled)
                    Text(model.photoDatasetSummary).font(.caption).foregroundStyle(.secondary)
                    Text("Whole physical surfaces stay in one split. Real noisy/clean pairs teach repair, not albedo. Synthetic lighting is a starting experiment; shadow removal is not yet a qualified capability.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("2 · Train a small residual RGB model") {
                    Picker("Compact NAF width", selection: $settings.width) { ForEach([16,22,32,46], id: \.self) { Text("\($0) channels").tag($0) } }
                        .disabled(resume != nil)
                    Text("\((try? settings.architecture.parameterCount) ?? 0) parameters · 1024 × 1024 native crops · physical batch 1 · AdamW, weight decay 0 · linear RGB/detail objective")
                        .font(.caption).foregroundStyle(.secondary)
                    TextField("Learning rate", value: $settings.learningRate, format: .number).disabled(resume != nil)
                    TextField("Seed", value: $settings.seed, format: .number).disabled(resume != nil)
                    Stepper("Total epochs: \(settings.epochs)", value: $settings.epochs, in: 1...10000)
                    Stepper("This session budget: \(settings.minutes) minutes", value: $settings.minutes, in: 1...480, step: 5)
                    HStack {
                        Button("Continue Checkpoint…") { chooseCheckpoint() }
                        if resume != nil { Button("Fresh Model") { resume = nil } }
                    }
                    if let resume { Text(resume.path).font(.caption).textSelection(.enabled) }
                    Text("Continuation restores the exact latest weights, optimizer and sampler in a new child run. Data, seed and optimizer semantics must match. Initial output is the unchanged photo; no pretrained weights are claimed.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("3 · Review and hand off") {
                    HStack {
                        Button("Test Saved Enhancer…") { testSaved() }
                        Button("Test Checkpoint…") { testCheckpoint() }
                    }
                    Text("Runs live in .loom-factory/photo-runs. Keep the latest checkpoint for continuation and the best export for A/B review. Best RGB error is a diagnostic; useful material appearance and clean-input preservation decide quality.")
                        .font(.caption)
                    if let output = model.lastPhotoEnhancerDirectory {
                        Button("Show Checkpoint & Core ML Export") { model.reveal(output) }
                    }
                    Text("The export contains enhancer.json, Enhancer.mlmodel and native weights, with product identity and checksums. No model is approved or selected automatically.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }.formStyle(.grouped).disabled(model.busy)
            if let issue { Text(issue).foregroundStyle(.red).textSelection(.enabled) }
            if model.busy {
                HStack { ProgressView().controlSize(.small); Text(model.activityTitle).lineLimit(2); Spacer()
                    if model.isTraining {
                        Button("Checkpoint") { model.saveCheckpoint() }.disabled(model.checkpointRequestPending)
                        Button("Save & Stop") { model.saveAndStopTraining() }.disabled(model.checkpointRequestPending)
                    }
                }
            }
            HStack { Button("Close") { dismiss() }.disabled(model.busy); Spacer()
                Button("Train Enhancer") {
                    guard let dataset else { return }
                    do { try settings.validate(); issue = nil; model.trainPhotoEnhancer(dataset: dataset, settings: settings, resume: resume) }
                    catch { issue = error.localizedDescription }
                }.buttonStyle(.borderedProminent).disabled(model.busy || dataset == nil)
                    .accessibilityIdentifier("loom.trainPhotoEnhancer")
            }
        }.padding(24).frame(width: 820, height: 820)
            .background(Color(nsColor: .windowBackgroundColor)).interactiveDismissDisabled(model.busy)
    }
    private func testCheckpoint() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]; panel.message = "Choose a photo checkpoint.json. You can inspect latest or best weights."
        guard panel.runModal() == .OK, let checkpoint = panel.url else { return }
        let choice = NSAlert(); choice.messageText = "Which saved weights?"; choice.informativeText = "Latest may look better even when validation ranks an earlier version higher. Both are unapproved candidates."
        choice.addButton(withTitle: "Latest"); choice.addButton(withTitle: "Best validation"); choice.addButton(withTitle: "Cancel")
        let response = choice.runModal(); guard response != .alertThirdButtonReturn else { return }
        let image = NSOpenPanel(); image.allowedContentTypes = [.png]; image.message = "Choose an opaque native 1K photo crop. CPU test only."
        guard image.runModal() == .OK, let input = image.url else { return }
        model.testPhotoCheckpoint(checkpoint: checkpoint, input: input, best: response == .alertSecondButtonReturn)
    }
    private func testSaved() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]; panel.message = "Choose enhancer.json in a saved photo-model export."
        guard panel.runModal() == .OK, let descriptor = panel.url else { return }
        let image = NSOpenPanel(); image.allowedContentTypes = [.png]; image.message = "Choose an opaque native 1024 × 1024 photo crop. Test runs on the CPU."
        guard image.runModal() == .OK, let input = image.url else { return }
        model.testPhotoEnhancer(export: descriptor.deletingLastPathComponent(), input: input)
    }
    private func importPairs() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]; panel.message = "Choose a synthetic manifest.json or a real-photo pair manifest. Originals are copied unchanged into a new project dataset."
        guard panel.runModal() == .OK, let file = panel.url else { return }
        resume = nil
        model.importPhotoPairs(manifest: file, kind: settings.kind) { dataset = $0 }
    }
    private func chooseDataset() {
        let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true
        panel.directoryURL = model.sourceFolder?.appendingPathComponent(".loom-factory/photo-datasets")
        if panel.runModal() == .OK { dataset = panel.url }
    }
    private func chooseCheckpoint() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]; panel.message = "Choose checkpoint.json in a photo-runs run (latest) or immutable checkpoint revision."
        panel.directoryURL = model.sourceFolder?.appendingPathComponent(".loom-factory/photo-runs")
        guard panel.runModal() == .OK, let file = panel.url else { return }
        do {
            let checkpoint = try JSONDecoder().decode(PhotoEnhancerCheckpoint.self, from: Data(contentsOf: file))
            guard checkpoint.schema == "loom.photo-checkpoint.v1" else { throw LoomError("Choose a photo enhancer checkpoint.") }
            settings = checkpoint.settings; resume = file; issue = nil
            var root = file.deletingLastPathComponent()
            if root.deletingLastPathComponent().lastPathComponent == "checkpoints" { root = root.deletingLastPathComponent().deletingLastPathComponent() }
            if let path = try? String(contentsOf: root.appendingPathComponent("dataset-location.txt"), encoding: .utf8) { dataset = URL(fileURLWithPath: path) }
        } catch { issue = error.localizedDescription }
    }
}
