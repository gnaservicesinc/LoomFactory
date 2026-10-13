import SwiftUI

enum FactoryStyle {
    static let accent = Color(red: 0.35, green: 0.55, blue: 0.93)
    static let corner: CGFloat = 16
    static let spacing: CGFloat = 18
}

struct FactoryPanel<Content: View>: View {
    let title: String
    let subtitle: String?
    private let content: Content

    init(_ title: String, subtitle: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.subtitle = subtitle
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.headline)
                if let subtitle { Text(subtitle).font(.callout).foregroundStyle(.secondary) }
            }
            content
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background, in: RoundedRectangle(cornerRadius: FactoryStyle.corner))
        .overlay(RoundedRectangle(cornerRadius: FactoryStyle.corner).strokeBorder(Color(nsColor: .separatorColor).opacity(0.55)))
    }
}

struct FactoryStatus: View {
    let text: String
    let symbol: String
    var color: Color = .secondary

    var body: some View {
        Label(text, systemImage: symbol)
            .font(.caption.weight(.medium))
            .foregroundStyle(color)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(color.opacity(0.09), in: Capsule())
    }
}

struct LoomMark: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18).fill(FactoryStyle.accent.gradient)
            VStack(spacing: 7) {
                ForEach(0..<3) { row in
                    HStack(spacing: 6) {
                        ForEach(0..<3) { column in
                            RoundedRectangle(cornerRadius: 2)
                                .fill(.white.opacity(row == column ? 1 : 0.5))
                                .frame(width: 9, height: 9)
                        }
                    }
                }
            }
        }
        .accessibilityHidden(true)
    }
}

struct FactoryActionButton: View {
    let title: String
    let symbol: String
    var prominent = false
    let action: () -> Void

    var body: some View {
        if prominent {
            Button(action: action) { Label(title, systemImage: symbol).frame(maxWidth: .infinity) }
                .buttonStyle(.borderedProminent).controlSize(.large)
        } else {
            Button(action: action) { Label(title, systemImage: symbol).frame(maxWidth: .infinity) }
                .buttonStyle(.bordered).controlSize(.large)
        }
    }
}
