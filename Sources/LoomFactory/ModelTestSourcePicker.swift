import LoomCore
import SwiftUI

struct ModelTestSourcePicker: View {
    let images: [SourceImage]
    let select: (SourceImage) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var search = ""

    private var matches: [SourceImage] {
        guard !search.isEmpty else { return images }
        return images.filter {
            $0.url.path.localizedCaseInsensitiveContains(search) ||
                ($0.category?.localizedCaseInsensitiveContains(search) ?? false)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Use a source diffuse").font(.title2.weight(.semibold))
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            TextField("Search filename, material folder, or category", text: $search)
                .textFieldStyle(.roundedBorder).accessibilityIdentifier("modelTest.sourceSearch")
            Text("\(matches.count.formatted()) images · All categories and source resolutions are available for testing.")
                .font(.caption).foregroundStyle(.secondary)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(matches) { image in
                        Button {
                            select(image); dismiss()
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(image.url.lastPathComponent).font(.callout.weight(.medium))
                                    Text("\(image.category ?? "Uncategorized") · \(image.url.deletingLastPathComponent().lastPathComponent)")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Text(image.dimensions).font(.caption).foregroundStyle(.secondary).monospacedDigit()
                            }.padding(.vertical, 10).padding(.horizontal, 7).contentShape(Rectangle())
                        }.buttonStyle(.plain)
                            .accessibilityIdentifier("modelTest.source.\(image.id)")
                        Divider()
                    }
                }
            }
        }.padding(24).frame(minWidth: 600, idealWidth: 750, minHeight: 420, idealHeight: 560)
    }
}
