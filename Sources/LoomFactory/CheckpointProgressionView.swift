import LoomCore
import SwiftUI

struct CheckpointProgressionView: View {
    let progression: CheckpointProgression
    let busy: Bool
    let viewOutput: (ModelTestResult) -> Void
    let reveal: (URL) -> Void
    var version: ModelVersion? = nil
    @State private var firstID = ""
    @State private var secondID = ""

    private var successful: [CheckpointTestOutput] { progression.outputs.filter { $0.result != nil } }
    private var first: CheckpointTestOutput? { successful.first { $0.id == firstID } ?? successful.first }
    private var second: CheckpointTestOutput? { successful.first { $0.id == secondID } ?? successful.last }

    var body: some View {
        FactoryPanel("One run · checkpoint progression", subtitle: "This compares checkpoints within one run. Use Compare Two Runs to compare training settings on the same input.") {
            if let version {
                Text(ModelVersionPresentation.title(version)).font(.headline).textSelection(.enabled)
            }
            Text("Previews share a fixed [0,1] scale; saved EXRs retain the full Float32 predictions.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Text("\(progression.results.count) of \(progression.checkpointCount) outputs saved\(progression.cancelled ? " · Stopped" : "")")
                    .font(.callout.weight(.medium))
                Spacer()
                Button("Show All Outputs", systemImage: "folder") { reveal(progression.directory) }
                    .accessibilityIdentifier("modelTest.showProgression")
            }
            Text("Each numbered folder contains the full-size EXR, PNG preview and test record. progression.json records checkpoint identities and any failures. Completed outputs remain available if you stop.")
                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            if successful.count > 1 {
                HStack {
                    checkpointPicker("A", selection: $firstID, fallback: first?.id ?? "")
                    checkpointPicker("B", selection: $secondID, fallback: second?.id ?? "")
                }
                HStack(alignment: .top, spacing: 18) {
                    if let first { output(first, height: 350) }
                    if let second { output(second, height: 350) }
                }.accessibilityIdentifier("modelTest.progressionComparison")
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 240), alignment: .top)], alignment: .leading, spacing: 18) {
                ForEach(progression.outputs) { entry in output(entry, height: 210) }
            }.accessibilityIdentifier("modelTest.progressionGallery")
        }.accessibilityIdentifier("modelTest.progression")
    }

    private func checkpointPicker(_ label: String, selection: Binding<String>, fallback: String) -> some View {
        Picker("Compare \(label)", selection: Binding(get: { selection.wrappedValue.isEmpty ? fallback : selection.wrappedValue },
                                                   set: { selection.wrappedValue = $0 })) {
            ForEach(successful) { entry in Text(title(entry)).tag(entry.id) }
        }.accessibilityIdentifier("modelTest.progression\(label)")
    }

    private func title(_ entry: CheckpointTestOutput) -> String {
        "Epoch \(entry.checkpoint.totalEpochsCompleted) · update \(entry.checkpoint.totalUpdatesCompleted) · \(entry.checkpoint.revisionID?.prefix(8) ?? "latest")"
    }

    private func output(_ entry: CheckpointTestOutput, height: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("Run \(entry.checkpoint.runID.prefix(8)) · \(ModelVersionPresentation.clipping(version?.settings))")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary).textSelection(.enabled)
            Text(title(entry)).font(.callout.weight(.medium)).textSelection(.enabled)
            if let result = entry.result {
                FactoryImage(url: result.previewURL, revision: result.directory.path, applyOrientationTransform: false)
                    .frame(height: height).clipShape(RoundedRectangle(cornerRadius: 7))
                HStack {
                    Button("Inspect Output") { viewOutput(result) }.disabled(busy)
                    Button("Show Files") { reveal(result.directory) }
                }.controlSize(.small)
            } else if let issue = entry.issue {
                Text(issue).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}
