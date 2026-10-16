import LoomCore
import SwiftUI

struct PhotoTrainingWorkspace: View {
    @Bindable var model: FactoryModel
    let train: (PhotoEnhancerKind) -> Void
    let generatePairs: (PhotoEnhancerKind) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: FactoryStyle.spacing) {
                Text("Photo Training").font(.title.bold())
                Text("Train RGB models to clean up camera noise, mild blur and uneven lighting before generating material maps.")
                    .foregroundStyle(.secondary)
                FactoryPanel("Repair · noise and mild blur") {
                    Text("Teach an aligned noisy/clean or mildly blurred/sharp photo pair. Include clean photos paired with themselves to check that real surface detail survives.")
                    actions(for: .repair)
                }
                FactoryPanel("Lighting · uneven illumination") {
                    Text("Teach unevenly lit photos against neutral references. Train separate weights after Repair; synthetic lighting pairs provide an initial experiment.")
                    actions(for: .lighting)
                }
                FactoryPanel("Prepare, train and compare") {
                    Text("1. Generate stage-specific pairs from your source materials, or use aligned real photo pairs.")
                    Text("2. Open the trainer and import the generated manifest or choose an imported dataset.")
                    Text("3. Train, save checkpoints and compare saved models on unfamiliar photos and clean inputs.")
                    Text("Experimental training paths are available. Pretrained cleanup weights, generated microdetail and alpha training are not included. Review useful texture detail before including an export in a Texture Studio build.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                if model.busy || !model.events.isEmpty { RunPanel(model: model) }
            }
            .padding(28)
            .frame(maxWidth: 1100, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .center)
        }
        .accessibilityIdentifier("loom.photoTrainingWorkspace")
    }

    private func actions(for kind: PhotoEnhancerKind) -> some View {
        HStack {
            Button(kind == .repair ? "Train Repair…" : "Train Lighting…") { train(kind) }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier("loom.trainPhoto.\(kind.rawValue)")
            Button("Generate Training Pairs…") { generatePairs(kind) }
                .accessibilityIdentifier("loom.generatePhotoPairs.\(kind.rawValue)")
        }
        .disabled(model.busy)
    }
}
