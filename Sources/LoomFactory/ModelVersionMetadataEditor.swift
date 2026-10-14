import LoomCore
import SwiftUI

struct ModelVersionMetadataEditor: View {
    @Bindable var model: FactoryModel
    let version: ModelVersion
    @State private var name: String
    @State private var notes: String

    init(model: FactoryModel, version: ModelVersion) {
        self.model = model; self.version = version
        _name = State(initialValue: version.label.isEmpty ? "Run \(version.id.prefix(8))" : version.label)
        _notes = State(initialValue: version.notes)
    }

    private var changed: Bool {
        name != (version.label.isEmpty ? "Run \(version.id.prefix(8))" : version.label) || notes != version.notes
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 12) {
                Text("Name").font(.caption).foregroundStyle(.secondary).frame(width: 40, alignment: .leading)
                TextField("Model / checkpoint name", text: $name)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("modelVersion.name.\(version.id)")
                Button("Save name & notes") {
                    model.updateModelVersionMetadata(version, name: name, notes: notes)
                }.disabled(model.busy || !changed || name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityIdentifier("modelVersion.saveMetadata.\(version.id)")
            }
            HStack(alignment: .top, spacing: 12) {
                Text("Notes").font(.caption).foregroundStyle(.secondary).frame(width: 40, alignment: .leading).padding(.top, 5)
                TextField("Settings, observations, or what to try next", text: $notes, axis: .vertical)
                .lineLimit(1...4).textFieldStyle(.roundedBorder)
                .accessibilityIdentifier("modelVersion.notes.\(version.id)")
            }
        }.onChange(of: version.label) { name = version.label.isEmpty ? "Run \(version.id.prefix(8))" : version.label }
            .onChange(of: version.notes) { notes = version.notes }
            .disabled(model.busy)
    }
}
