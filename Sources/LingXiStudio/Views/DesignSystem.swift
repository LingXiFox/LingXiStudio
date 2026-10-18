import AppKit
import SwiftUI

// MARK: - Tokens

/// LingXiFox brand tokens: fox orange for interaction and identity, warm charcoal and warm cream for every surface.
/// Each color resolves per appearance, so light and dark are tuned separately rather than inverted.
enum Brand {
    static let fox = Color(hex: 0xF28A42)
    static let caramel = Color(hex: 0xD96A2B)

    /// Window base: warm charcoal / warm cream.
    static let base = dynamic(light: 0xF5EFE5, dark: 0x1C1A18)
    /// Secondary surfaces: grouped tools, fields, tab tracks.
    static let surface = dynamic(light: 0xEBE2D5, dark: 0x292521)
    /// Raised writing surface for the prompt (still warm, never pure white).
    static let paper = dynamic(light: 0xFBF7F0, dark: 0x25221F)
    /// Neutral warm backdrop behind artwork.
    static let canvas = dynamic(light: 0xE7DED1, dark: 0x141210)
    /// Right column base, a step away from the creation side.
    static let side = dynamic(light: 0xF0E8DC, dark: 0x201D1B)

    static let text = dynamic(light: 0x302A25, dark: 0xF3EADF)
    static let textSecondary = dynamic(light: 0x695B4F, dark: 0xB9AC9E)
    static let textTertiary = dynamic(light: 0x6E6053, dark: 0xAC9D8D)
    static let hairline = dynamic(light: 0x302A25, dark: 0xF3EADF, alpha: 0.11)
    static let hover = dynamic(light: 0x302A25, dark: 0xF3EADF, alpha: 0.06)
    static let foxSoft = dynamic(light: 0xF28A42, dark: 0xF28A42, alpha: 0.15)
    /// Fox text on light backgrounds needs the deeper caramel for contrast.
    static let foxText = dynamic(light: 0xA44917, dark: 0xF5A066)
    static let onFox = Color(hex: 0x1C1A18)

    static let success = dynamic(light: 0x426C35, dark: 0x8DBF7A)
    static let warning = dynamic(light: 0x945315, dark: 0xF0AE5C)
    static let danger = dynamic(light: 0xC2473A, dark: 0xEE7B6B)

    static let primaryGradient = LinearGradient(colors: [fox, caramel], startPoint: .top, endPoint: .bottom)

    private static func dynamic(light: UInt32, dark: UInt32, alpha: CGFloat = 1) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            return NSColor(hex: isDark ? dark : light, alpha: alpha)
        })
    }
}

/// Spacing, radii and control metrics shared by every custom component.
enum Metrics {
    static let xs: CGFloat = 4
    static let s: CGFloat = 8
    static let m: CGFloat = 12
    static let l: CGFloat = 16
    static let xl: CGFloat = 24

    static let radiusS: CGFloat = 6
    static let radiusM: CGFloat = 9
    static let radiusL: CGFloat = 14

    static let control: CGFloat = 28
}

/// Type scale built on system text styles so it follows the user's accessibility settings.
enum TypeScale {
    static let section = Font.subheadline.weight(.semibold)
    static let label = Font.caption.weight(.medium)
    static let body = Font.body
    static let value = Font.body.monospacedDigit()
    static let prompt = Font.title3
    static let caption = Font.caption
}

/// Kept for the windows that still refer to the old names.
enum Design {
    static let corner = Metrics.radiusL
    static let innerCorner = Metrics.radiusM
    static let gutter = Metrics.m
    static let panelPadding = Metrics.l
    static let cardBackground = Brand.surface
    static let canvasBackground = Brand.canvas
    static let hairline = Brand.hairline
}

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(
            .sRGB, red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255, opacity: alpha)
    }
}

extension NSColor {
    convenience init(hex: UInt32, alpha: CGFloat = 1) {
        self.init(
            srgbRed: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255, alpha: alpha)
    }
}

extension View {
    /// Brand tint, base text color and window background for a top-level window.
    func brandWindow(_ background: Color = Brand.base) -> some View {
        self
            .tint(Brand.fox)
            .foregroundStyle(Brand.text)
            .background(background)
            .containerBackground(background, for: .window)
    }
}

// MARK: - Buttons

/// The fox-orange primary action: a quiet static gradient, darker while pressed.
struct PrimaryButtonStyle: ButtonStyle {
    var large = false
    @Environment(\.isEnabled) private var enabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(large ? .title3.weight(.semibold) : .body.weight(.semibold))
            .foregroundStyle(enabled ? Brand.onFox : Brand.textSecondary)
            .padding(.horizontal, large ? 22 : 14)
            .frame(height: large ? 40 : Metrics.control)
            .background {
                RoundedRectangle(cornerRadius: large ? 11 : Metrics.radiusM, style: .continuous)
                    .fill(enabled ? AnyShapeStyle(Brand.primaryGradient) : AnyShapeStyle(Brand.textTertiary.opacity(0.45)))
                    .overlay {
                        if configuration.isPressed {
                            RoundedRectangle(cornerRadius: large ? 11 : Metrics.radiusM, style: .continuous).fill(.black.opacity(0.14))
                        }
                    }
                    .shadow(color: enabled ? Brand.caramel.opacity(0.28) : .clear, radius: large ? 6 : 3, y: 2)
            }
            .contentShape(RoundedRectangle(cornerRadius: Metrics.radiusM, style: .continuous))
    }
}

/// Secondary action on a warm surface.
struct SecondaryButtonStyle: ButtonStyle {
    var destructive = false
    @Environment(\.isEnabled) private var enabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.body.weight(.medium))
            .foregroundStyle(destructive ? Brand.danger : Brand.text)
            .padding(.horizontal, 12)
            .frame(height: Metrics.control)
            .background(
                configuration.isPressed ? Brand.hairline : Brand.surface,
                in: RoundedRectangle(cornerRadius: Metrics.radiusM, style: .continuous))
            .opacity(enabled ? 1 : 0.45)
            .contentShape(RoundedRectangle(cornerRadius: Metrics.radiusM, style: .continuous))
    }
}

/// Borderless icon or text button with a hover wash.
struct GhostButtonStyle: ButtonStyle {
    var active = false

    func makeBody(configuration: Configuration) -> some View {
        GhostBody(configuration: configuration, active: active)
    }

    private struct GhostBody: View {
        let configuration: Configuration
        let active: Bool
        @State private var hovering = false
        @Environment(\.isEnabled) private var enabled

        var body: some View {
            configuration.label
                .foregroundStyle(active ? Brand.foxText : Brand.textSecondary)
                .padding(.horizontal, 7)
                .frame(minWidth: 26, minHeight: 26)
                .background(
                    active ? Brand.foxSoft : (hovering || configuration.isPressed ? Brand.hover : .clear),
                    in: RoundedRectangle(cornerRadius: Metrics.radiusS, style: .continuous))
                .opacity(enabled ? 1 : 0.4)
                .contentShape(Rectangle())
                .onHover { hovering = $0 }
        }
    }
}

extension ButtonStyle where Self == GhostButtonStyle {
    static var ghost: GhostButtonStyle { GhostButtonStyle() }
}

// MARK: - Segmented control

/// Compact capsule segmented control with a sliding fox selection.
struct BrandSegmented<Value: Hashable>: View {
    struct Option: Identifiable {
        let value: Value
        let title: String
        var symbol: String? = nil
        var help: String? = nil
        var enabled = true
        var id: String { title }
    }

    @Binding var selection: Value
    let options: [Option]
    var fill = false
    @Namespace private var namespace
    @FocusState private var focusedOption: Value?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        HStack(spacing: 2) {
            ForEach(options) { option in
                let selected = option.value == selection
                Button {
                    withAnimation(reduceMotion ? nil : .snappy(duration: 0.22)) { selection = option.value }
                } label: {
                    HStack(spacing: 5) {
                        if let symbol = option.symbol { Image(systemName: symbol).imageScale(.small) }
                        Text(option.title).lineLimit(1)
                    }
                    .font(.callout.weight(selected ? .semibold : .regular))
                    .foregroundStyle(selected ? Brand.text : Brand.textSecondary)
                    .padding(.horizontal, 11)
                    .frame(maxWidth: fill ? .infinity : nil, minHeight: Metrics.control - 4)
                    .background {
                        if selected {
                            RoundedRectangle(cornerRadius: Metrics.radiusS, style: .continuous)
                                .fill(Brand.paper)
                                .overlay(alignment: .bottom) {
                                    Capsule().fill(Brand.fox).frame(width: 14, height: 2.5).offset(y: -1)
                                }
                                .shadow(color: .black.opacity(0.08), radius: 1.5, y: 1)
                                .matchedGeometryEffect(id: "selection", in: namespace)
                        }
                    }
                    .overlay {
                        RoundedRectangle(cornerRadius: Metrics.radiusS, style: .continuous)
                            .strokeBorder(focusedOption == option.value ? Brand.fox : .clear, lineWidth: 1.5)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .focused($focusedOption, equals: option.value)
                .focusEffectDisabled()
                .disabled(!option.enabled)
                .opacity(option.enabled ? 1 : 0.4)
                .help(option.help ?? "")
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .padding(2)
        .background(Brand.surface, in: RoundedRectangle(cornerRadius: Metrics.radiusM, style: .continuous))
    }
}

// MARK: - Fields

/// A labelled compact parameter: small caption above, control inside a warm field.
struct ParamField<Content: View>: View {
    let label: String
    var trailing: String? = nil
    var help: String? = nil
    var plain = false
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 4) {
                Text(label).font(TypeScale.label).foregroundStyle(Brand.textSecondary)
                Spacer(minLength: 0)
                if let trailing { Text(trailing).font(TypeScale.caption).foregroundStyle(Brand.textTertiary) }
            }
            Group {
                if plain {
                    content
                } else {
                    content
                        .padding(.horizontal, 8)
                        .frame(height: Metrics.control)
                        .background(Brand.surface, in: RoundedRectangle(cornerRadius: Metrics.radiusM, style: .continuous))
                }
            }
        }
        .help(help ?? "")
    }
}

/// Integer field with − / + buttons inside one warm field.
struct StepperField: View {
    @Binding var value: Int
    let range: ClosedRange<Int>
    var step = 1
    var width: CGFloat = 44
    var suffix: String? = nil

    var body: some View {
        HStack(spacing: 2) {
            Button { value = max(range.lowerBound, value - step) } label: { Image(systemName: "minus") }
                .disabled(value <= range.lowerBound)
            TextField("", value: $value, format: .number.grouping(.never))
                .textFieldStyle(.plain)
                .multilineTextAlignment(.center)
                .font(TypeScale.value)
                .frame(minWidth: width)
                .onSubmit { value = min(max(value, range.lowerBound), range.upperBound) }
            if let suffix { Text(suffix).font(TypeScale.caption).foregroundStyle(Brand.textTertiary) }
            Button { value = min(range.upperBound, value + step) } label: { Image(systemName: "plus") }
                .disabled(value >= range.upperBound)
        }
        .buttonStyle(.ghost)
        .imageScale(.small)
    }
}

/// Section title used instead of card chrome: hierarchy comes from type and space.
struct SectionHeader<Accessory: View>: View {
    let title: String
    @ViewBuilder var accessory: Accessory

    var body: some View {
        HStack(spacing: 6) {
            Text(title).font(TypeScale.section).foregroundStyle(Brand.textSecondary)
            Spacer(minLength: 0)
            accessory
        }
    }
}

extension SectionHeader where Accessory == EmptyView {
    init(_ title: String) {
        self.init(title: title) { EmptyView() }
    }
}

/// A quiet grouped container for tool areas; used sparingly.
struct Panel<Content: View>: View {
    var padding: CGFloat = Metrics.m
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Brand.surface.opacity(0.55), in: RoundedRectangle(cornerRadius: Metrics.radiusL, style: .continuous))
    }
}

// MARK: - Small pieces

/// Secondary explanatory text.
struct Note: View {
    let text: String
    var tint: Color = Brand.textSecondary
    var symbol: String? = nil

    init(_ text: String, tint: Color = Brand.textSecondary, symbol: String? = nil) {
        self.text = text
        self.tint = tint
        self.symbol = symbol
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            if let symbol { Image(systemName: symbol) }
            Text(text).fixedSize(horizontal: false, vertical: true)
        }
        .font(TypeScale.caption)
        .foregroundStyle(tint)
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
                Circle().fill(color).frame(width: 6, height: 6)
            }
            Text(text).font(.caption.weight(.medium))
        }
        .padding(.horizontal, 8).padding(.vertical, 3)
        .background(color.opacity(0.14), in: Capsule())
        .foregroundStyle(color)
    }
}

/// Monochrome tag, e.g. "BF16" or "通用方法".
struct Tag: View {
    let text: String
    var tint: Color = Brand.textSecondary

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .foregroundStyle(tint)
            .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 4, style: .continuous))
    }
}

/// SF Symbol on a soft tinted square.
struct IconTile: View {
    let symbol: String
    var tint: Color = Brand.fox
    var size: CGFloat = 26

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: size * 0.48, weight: .semibold))
            .foregroundStyle(tint)
            .frame(width: size, height: size)
            .background(tint.opacity(0.16), in: RoundedRectangle(cornerRadius: size * 0.28, style: .continuous))
    }
}

/// Count badge for tabs.
struct Badge: View {
    let count: Int
    var tint: Color = Brand.foxText

    var body: some View {
        if count > 0 {
            Text("\(count)")
                .font(.caption2.weight(.bold)).monospacedDigit()
                .padding(.horizontal, 5).padding(.vertical, 1)
                .background(tint, in: Capsule())
                .foregroundStyle(Brand.base)
        }
    }
}

/// Thin determinate bar; animates only when the value changes.
struct ProgressBar: View {
    let fraction: Double
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Brand.hairline)
                Capsule().fill(Brand.primaryGradient)
                    .frame(width: max(geo.size.width * min(max(fraction, 0), 1), 4))
            }
        }
        .frame(height: 5)
        .animation(reduceMotion ? nil : .easeOut(duration: 0.3), value: fraction)
        .accessibilityLabel("生成进度")
        .accessibilityValue(Text(fraction, format: .percent.precision(.fractionLength(0))))
    }
}

/// Circular progress ring.
struct ProgressRing: View {
    let fraction: Double
    var lineWidth: CGFloat = 3
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            Circle().stroke(Brand.hairline, lineWidth: lineWidth)
            Circle()
                .trim(from: 0, to: max(0.001, min(fraction, 1)))
                .stroke(Brand.fox, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(reduceMotion ? nil : .easeOut(duration: 0.3), value: fraction)
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
                        with: .color(Brand.hairline))
                }
            }
        }
    }
}

/// Aspect-ratio glyph: the ratio drawn as a small rectangle.
struct AspectGlyph: View {
    let width: Int
    let height: Int
    var selected = false
    var box: CGFloat = 14

    var body: some View {
        let scale = box / CGFloat(max(width, height))
        RoundedRectangle(cornerRadius: 2, style: .continuous)
            .strokeBorder(selected ? Brand.fox : Brand.textSecondary, lineWidth: 1.3)
            .frame(width: CGFloat(width) * scale, height: CGFloat(height) * scale)
            .frame(width: box, height: box)
    }
}

/// Empty state in brand colors.
struct EmptyState: View {
    let symbol: String
    let title: String
    var message: String? = nil
    var compact = false

    var body: some View {
        VStack(spacing: compact ? 6 : 10) {
            Image(systemName: symbol)
                .font(.system(size: compact ? 22 : 34, weight: .light))
                .foregroundStyle(Brand.fox.opacity(0.8))
            Text(title).font(compact ? .callout.weight(.medium) : .title3.weight(.medium)).foregroundStyle(Brand.text)
            if let message {
                Text(message).font(.callout).foregroundStyle(Brand.textSecondary).multilineTextAlignment(.center)
                    .frame(maxWidth: 320)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
