import AppKit
import SwiftUI

/// Small visual vocabulary shared by every panel: native colors and materials, consistent spacing and type.
enum Design {
    static let corner: CGFloat = 12
    static let innerCorner: CGFloat = 8
    static let gutter: CGFloat = 14
    static let panelPadding: CGFloat = 16

    static let cardBackground = Color(nsColor: .controlBackgroundColor)
    static let canvasBackground = Color(nsColor: .underPageBackgroundColor)
    static let hairline = Color(nsColor: .separatorColor)
}

/// A rounded card with a colored icon tile, title, optional subtitle and trailing accessory, like System Settings.
struct Card<Accessory: View, Content: View>: View {
    let title: String
    let symbol: String
    let tint: Color
    var subtitle: String? = nil
    @ViewBuilder var accessory: Accessory
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                IconTile(symbol: symbol, tint: tint)
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.headline)
                    if let subtitle {
                        Text(subtitle).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                Spacer(minLength: 8)
                accessory
            }
            content
        }
        .padding(Design.gutter)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Design.cardBackground, in: RoundedRectangle(cornerRadius: Design.corner, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Design.corner, style: .continuous)
                .strokeBorder(Design.hairline.opacity(0.6), lineWidth: 0.5))
    }
}

extension Card where Accessory == EmptyView {
    init(title: String, symbol: String, tint: Color, subtitle: String? = nil, @ViewBuilder content: () -> Content) {
        self.init(title: title, symbol: symbol, tint: tint, subtitle: subtitle, accessory: { EmptyView() }, content: content)
    }
}

/// White SF Symbol on a colored rounded square.
struct IconTile: View {
    let symbol: String
    let tint: Color
    var size: CGFloat = 26

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: size * 0.5, weight: .semibold))
            .foregroundStyle(.white)
            .frame(width: size, height: size)
            .background(tint.gradient, in: RoundedRectangle(cornerRadius: size * 0.26, style: .continuous))
    }
}

/// A labelled row in a parameter grid: fixed-width secondary label, then the control.
struct ParameterRow<Content: View>: View {
    let label: String
    var help: String? = nil
    @ViewBuilder var content: Content

    var body: some View {
        GridRow(alignment: .firstTextBaseline) {
            Text(label)
                .foregroundStyle(.secondary)
                .frame(width: 72, alignment: .leading)
                .help(help ?? "")
            content
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// Secondary explanatory text under a control.
struct Note: View {
    let text: String
    var tint: Color = .secondary
    var symbol: String? = nil

    init(_ text: String, tint: Color = .secondary, symbol: String? = nil) {
        self.text = text
        self.tint = tint
        self.symbol = symbol
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            if let symbol { Image(systemName: symbol) }
            Text(text).fixedSize(horizontal: false, vertical: true)
        }
        .font(.caption)
        .foregroundStyle(tint)
    }
}

/// Capsule toggle chip.
struct Chip: View {
    let title: String
    var symbol: String? = nil
    let selected: Bool
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if let symbol { Image(systemName: symbol) }
                Text(title)
            }
            .font(.callout)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(selected ? Color.accentColor.opacity(0.18) : Color.primary.opacity(0.05), in: Capsule())
            .overlay(Capsule().strokeBorder(selected ? Color.accentColor : .clear, lineWidth: 1))
            .foregroundStyle(selected ? Color.accentColor : .primary)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

/// Aspect-ratio chip that draws the ratio as a small rectangle.
struct AspectChip: View {
    let label: String
    let width: Int
    let height: Int
    let selected: Bool
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                let scale = 22 / CGFloat(max(width, height))
                RoundedRectangle(cornerRadius: 2.5, style: .continuous)
                    .strokeBorder(selected ? Color.accentColor : .secondary, lineWidth: 1.5)
                    .background(
                        (selected ? Color.accentColor.opacity(0.2) : .clear),
                        in: RoundedRectangle(cornerRadius: 2.5, style: .continuous)
                    )
                    .frame(width: CGFloat(width) * scale, height: CGFloat(height) * scale)
                    .frame(width: 26, height: 24)
                Text(label).font(.caption2).foregroundStyle(selected ? Color.accentColor : .secondary)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .background(
                selected ? Color.accentColor.opacity(0.08) : .clear,
                in: RoundedRectangle(cornerRadius: Design.innerCorner, style: .continuous)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("\(width) × \(height)")
    }
}

/// Small status pill with a colored dot.
struct StatusPill: View {
    let text: String
    let color: Color
    var busy = false

    var body: some View {
        HStack(spacing: 5) {
            if busy {
                ProgressView().controlSize(.mini)
            } else {
                Circle().fill(color).frame(width: 7, height: 7)
            }
            Text(text).font(.caption.weight(.medium))
        }
        .padding(.horizontal, 8).padding(.vertical, 3)
        .background(color.opacity(0.14), in: Capsule())
        .foregroundStyle(color == .secondary ? Color.secondary : color)
    }
}

/// Count badge for tabs.
struct Badge: View {
    let count: Int
    var tint: Color = .accentColor

    var body: some View {
        if count > 0 {
            Text("\(count)")
                .font(.caption2.weight(.bold)).monospacedDigit()
                .padding(.horizontal, 5).padding(.vertical, 1)
                .background(tint, in: Capsule())
                .foregroundStyle(.white)
        }
    }
}

/// Circular progress ring with a label in the middle.
struct ProgressRing: View {
    let fraction: Double
    var lineWidth: CGFloat = 4

    var body: some View {
        ZStack {
            Circle().stroke(Color.primary.opacity(0.1), lineWidth: lineWidth)
            Circle()
                .trim(from: 0, to: max(0.001, min(fraction, 1)))
                .stroke(Color.accentColor, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeOut(duration: 0.3), value: fraction)
        }
    }
}

/// Checkerboard used behind images with transparency.
struct Checkerboard: View {
    var body: some View {
        Canvas { context, size in
            let cell: CGFloat = 10
            for row in 0..<Int(size.height / cell) + 1 {
                for col in 0..<Int(size.width / cell) + 1 where (row + col) % 2 == 0 {
                    context.fill(
                        Path(CGRect(x: CGFloat(col) * cell, y: CGFloat(row) * cell, width: cell, height: cell)),
                        with: .color(Color.primary.opacity(0.05)))
                }
            }
        }
    }
}
