import QwenImageCore
import SwiftUI

// MARK: - Common parameters

/// The everyday parameters in two compact rows; rules (Turbo locks the steps, sizes follow the input) show in the
/// controls' own state instead of explanatory text.
struct ParameterStrip: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        let p = model.parameters
        let turbo = p.preset == .turbo
        VStack(alignment: .leading, spacing: 7) {
            SectionHeader("参数")
            Grid(alignment: .leading, horizontalSpacing: Metrics.m, verticalSpacing: Metrics.m) {
                GridRow {
                    ParamField(label: "比例") { AspectPicker() }
                    ParamField(label: "尺寸", trailing: String(format: "%.2f MP", Double(p.width * p.height) / 1e6)) { sizeField }
                        .gridCellColumns(2)
                    ParamField(label: "调度", plain: true) {
                        BrandSegmented(
                            selection: $model.parameters.preset,
                            options: [
                                .init(value: .standard, title: "标准", help: "linear 网格 + 终端拉伸（官方默认）"),
                                .init(
                                    value: .turbo, title: "Turbo",
                                    help: model.turboAvailable
                                        ? "viggle_turbo 固定 6 个时间节点" : "需要在 LoRA 列表中启用一个「加速」用途的 LoRA",
                                    enabled: model.turboAvailable),
                            ],
                            fill: true)
                    }
                }
                GridRow {
                    ParamField(label: "步数", help: turbo ? "Turbo 固定 6 个时间节点" : nil) {
                        if turbo {
                            HStack(spacing: 5) {
                                Image(systemName: "lock.fill").imageScale(.small)
                                Text("6").font(TypeScale.value)
                                Spacer()
                                Text("Turbo").font(TypeScale.caption)
                            }
                            .foregroundStyle(Brand.textSecondary)
                        } else {
                            StepperField(value: $model.parameters.steps, range: 1...100)
                        }
                    }
                    ParamField(label: "Guidance", trailing: p.usesTrueCFG ? "真 CFG" : nil, help: guidanceHelp) {
                        HStack(spacing: 6) {
                            Slider(value: $model.parameters.guidance, in: 1...10, step: 0.5).controlSize(.small)
                            Text(String(format: "%.1f", p.guidance)).font(TypeScale.value).frame(width: 28, alignment: .trailing)
                        }
                    }
                    ParamField(label: "种子") { seedField }
                    ParamField(label: "张数") { StepperField(value: $model.parameters.count, range: 1...16, width: 28) }
                }
            }
            .onChange(of: model.turboAvailable) { _, available in
                if !available && model.parameters.preset == .turbo { model.parameters.preset = .standard }
            }
        }
    }

    private var guidanceHelp: String {
        let p = model.parameters
        if p.guidance <= 1 { return "1.0 = 不做引导（模型默认）" }
        return p.usesTrueCFG ? "真 CFG：每步正负两次前向，耗时约翻倍" : "需要填写负面提示词才会生效"
    }

    private var sizeField: some View {
        @Bindable var model = model
        let follows = model.mode != .textToImage && model.parameters.sizePolicy == .followImage
        return HStack(spacing: 4) {
            if follows {
                Image(systemName: "link").imageScale(.small).foregroundStyle(Brand.foxText)
                Text("\(model.parameters.width) × \(model.parameters.height)").font(TypeScale.value)
                Text("跟随输入图").font(TypeScale.caption).foregroundStyle(Brand.textTertiary)
                Spacer(minLength: 0)
                Button("手动") { model.parameters.sizePolicy = .manual }
                    .buttonStyle(.ghost).font(TypeScale.caption)
            } else {
                dimension($model.parameters.width)
                Text("×").foregroundStyle(Brand.textTertiary)
                dimension($model.parameters.height)
                Spacer(minLength: 0)
                if model.parameters.width % 32 != 0 || model.parameters.height % 32 != 0 {
                    Image(systemName: "exclamationmark.circle.fill").foregroundStyle(Brand.danger).help("宽高必须是 32 的倍数")
                }
                Button {
                    let w = model.parameters.width
                    model.parameters.width = model.parameters.height
                    model.parameters.height = w
                } label: {
                    Image(systemName: "arrow.left.arrow.right")
                }
                .buttonStyle(.ghost).help("横竖互换")
            }
        }
        .onChange(of: model.parameters.sizePolicy) { _, _ in model.applySizePolicy() }
    }

    private func dimension(_ value: Binding<Int>) -> some View {
        TextField("", value: value, format: .number.grouping(.never))
            .textFieldStyle(.plain)
            .font(TypeScale.value)
            .multilineTextAlignment(.center)
            .frame(width: 46)
            .onSubmit { value.wrappedValue = max(256, min(2048, (value.wrappedValue + 16) / 32 * 32)) }
    }

    private var seedField: some View {
        @Bindable var model = model
        let random = model.parameters.seedMode == .random
        return HStack(spacing: 4) {
            Button {
                model.parameters.seedMode = random ? .fixed : .random
            } label: {
                Image(systemName: random ? "shuffle" : "pin.fill")
            }
            .buttonStyle(GhostButtonStyle(active: !random))
            .help(random ? "随机种子；点击改为固定" : "固定种子；点击改为随机")
            if random {
                Text("随机").foregroundStyle(Brand.textSecondary)
                Spacer(minLength: 0)
            } else {
                TextField("", value: $model.parameters.seed, format: .number.grouping(.never))
                    .textFieldStyle(.plain).font(TypeScale.value)
                Button {
                    model.parameters.seed = UInt64.random(in: 0..<UInt64(UInt32.max))
                } label: {
                    Image(systemName: "dice")
                }
                .buttonStyle(.ghost).help("换一个种子")
            }
        }
    }
}

/// Aspect ratio as a glyph plus label, choosing from a small popover grid.
struct AspectPicker: View {
    @Environment(AppModel.self) private var model
    @State private var open = false

    var body: some View {
        let p = model.parameters
        let current = SizePreset.all.first { $0.width == p.width && $0.height == p.height }
        Button {
            open.toggle()
        } label: {
            HStack(spacing: 6) {
                AspectGlyph(width: p.width, height: p.height, selected: true)
                Text(current?.label ?? ratioLabel(p.width, p.height)).font(.body)
                Spacer(minLength: 0)
                Image(systemName: "chevron.up.chevron.down").imageScale(.small).foregroundStyle(Brand.textTertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .popover(isPresented: $open, arrowEdge: .bottom) { grid }
    }

    private var grid: some View {
        @Bindable var model = model
        return VStack(alignment: .leading, spacing: 10) {
            if model.mode != .textToImage {
                Toggle("跟随输入图比例", isOn: Binding(
                    get: { model.parameters.sizePolicy == .followImage },
                    set: { model.parameters.sizePolicy = $0 ? .followImage : .manual }))
                    .toggleStyle(.switch).controlSize(.small)
            }
            LazyVGrid(columns: Array(repeating: GridItem(.fixed(64), spacing: 6), count: 4), spacing: 6) {
                ForEach(SizePreset.all) { preset in
                    let selected = model.parameters.width == preset.width && model.parameters.height == preset.height
                    Button {
                        model.parameters.sizePolicy = .manual
                        model.parameters.width = preset.width
                        model.parameters.height = preset.height
                        open = false
                    } label: {
                        VStack(spacing: 5) {
                            AspectGlyph(width: preset.width, height: preset.height, selected: selected, box: 22)
                            Text(preset.label).font(.caption.weight(selected ? .semibold : .regular))
                            Text("\(preset.width)×\(preset.height)").font(.system(size: 9)).monospacedDigit()
                                .foregroundStyle(Brand.textTertiary)
                        }
                        .frame(width: 64, height: 64)
                        .background(selected ? Brand.foxSoft : Brand.surface,
                                    in: RoundedRectangle(cornerRadius: Metrics.radiusM, style: .continuous))
                        .foregroundStyle(selected ? Brand.foxText : Brand.text)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(12)
        .tint(Brand.fox)
    }

    private func ratioLabel(_ w: Int, _ h: Int) -> String {
        func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
        let g = max(gcd(w, h), 1)
        let (a, b) = (w / g, h / g)
        return a <= 32 && b <= 32 ? "\(a):\(b)" : "自定义"
    }
}

// MARK: - Advanced

/// Low-frequency options of this generation only; global paths and runtime policy live in Settings.
struct AdvancedSection: View {
    @Environment(AppModel.self) private var model
    @AppStorage("advancedExpanded") private var expanded = false

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation(.snappy(duration: 0.2)) { expanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.right").imageScale(.small).rotationEffect(.degrees(expanded ? 90 : 0))
                    Text("高级选项").font(TypeScale.section)
                    if !expanded { Text(summary).font(TypeScale.caption).foregroundStyle(Brand.textTertiary).lineLimit(1) }
                    Spacer()
                }
                .foregroundStyle(Brand.textSecondary)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if expanded {
                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: Metrics.l, verticalSpacing: 10) {
                    row("多张种子") {
                        BrandSegmented(
                            selection: $model.parameters.batchSeeds,
                            options: [
                                .init(value: .increment, title: "逐张 +1"),
                                .init(value: .random, title: "逐张随机"),
                            ])
                        .disabled(model.parameters.count <= 1)
                        .opacity(model.parameters.count <= 1 ? 0.5 : 1)
                    }
                    if let last = model.lastSeed {
                        row("上一张种子") {
                            HStack(spacing: 8) {
                                Text(String(last)).font(TypeScale.value).textSelection(.enabled)
                                Button("固定使用") {
                                    model.parameters.seedMode = .fixed
                                    model.parameters.seed = last
                                }
                                Button("复制") { ImageFileActions.copyText(String(last)) }
                            }
                            .buttonStyle(.ghost)
                        }
                    }
                    if model.mode.needsMask {
                        row("边缘羽化") {
                            HStack {
                                Slider(value: featherBinding, in: 0...48, step: 2).frame(maxWidth: 220).controlSize(.small)
                                Text("\(model.parameters.maskFeather) px").font(TypeScale.value)
                            }
                        }
                        row("蒙版") {
                            VStack(alignment: .leading, spacing: 6) {
                                Toggle("反选：涂抹区域保留，其余重绘", isOn: $model.parameters.maskInvert)
                                Toggle("选区外像素保持原样（按羽化蒙版贴回原图）", isOn: $model.parameters.keepUnmasked)
                            }
                            .toggleStyle(.checkbox)
                        }
                    }
                }
                .padding(.leading, 18)
                .transition(.opacity)
            }
        }
    }

    private func row<Content: View>(_ label: String, @ViewBuilder content: () -> Content) -> some View {
        GridRow {
            Text(label).font(TypeScale.label).foregroundStyle(Brand.textSecondary).gridColumnAlignment(.trailing)
            content()
        }
    }

    private var summary: String {
        var parts = [model.parameters.batchSeeds == .increment ? "多张种子逐张 +1" : "多张种子逐张随机"]
        if model.mode.needsMask { parts.append("羽化 \(model.parameters.maskFeather) px") }
        return parts.joined(separator: " · ")
    }

    private var featherBinding: Binding<Double> {
        Binding(get: { Double(model.parameters.maskFeather) }, set: { model.parameters.maskFeather = Int($0) })
    }
}

// MARK: - Generate bar

/// Pinned under the creation side: one line of status or blocking reason, then the fox generate button.
struct GenerateBar: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let issues = model.blockingIssues
        HStack(spacing: Metrics.m) {
            Group {
                if let issue = issues.first {
                    Label(issue, systemImage: "exclamationmark.circle.fill").foregroundStyle(Brand.danger)
                        .help(issues.joined(separator: "\n"))
                } else if let warning = model.warnings.first {
                    Label(warning, systemImage: "exclamationmark.triangle.fill").foregroundStyle(Brand.warning)
                        .help(model.warnings.joined(separator: "\n"))
                } else {
                    Text(summary).foregroundStyle(Brand.textSecondary)
                }
            }
            .font(.callout)
            .lineLimit(1)
            Spacer(minLength: Metrics.s)
            if model.isRunning {
                Button("停止", systemImage: "stop.fill") { model.stopCurrent() }
                    .buttonStyle(SecondaryButtonStyle())
                    .keyboardShortcut(".", modifiers: .command)
                    .help("停止当前任务（⌘.）")
            }
            Button {
                model.enqueue()
            } label: {
                HStack(spacing: 7) {
                    Image(systemName: model.isRunning ? "text.badge.plus" : "sparkles")
                    Text(model.isRunning ? "加入队列" : "生成")
                }
                .frame(minWidth: 96)
            }
            .buttonStyle(PrimaryButtonStyle(large: true))
            .disabled(!issues.isEmpty)
            .keyboardShortcut(.return, modifiers: .command)
            .help(issues.first ?? "生成（⌘↩）")
        }
        .padding(.horizontal, Metrics.xl)
        .padding(.vertical, Metrics.m)
        .background(Brand.base)
        .overlay(alignment: .top) { Rectangle().fill(Brand.hairline).frame(height: 1) }
    }

    private var summary: String {
        let p = model.parameters
        var parts = ["\(p.width)×\(p.height)", "\(p.effectiveSteps) 步"]
        if p.count > 1 { parts.append("\(p.count) 张") }
        if !model.activeLoRAs.isEmpty { parts.append("\(model.activeLoRAs.count) 个 LoRA") }
        if let rate = model.estimatedSecondsPerStep {
            let steps = model.mode.hasStrength ? Int((Float(p.effectiveSteps) * p.strength(for: model.mode)).rounded(.up)) : p.effectiveSteps
            parts.append("去噪约 \(Format.duration(rate * Double(steps) * Double(p.count)))")
        }
        if model.waitingCount > 0 { parts.append("排队 \(model.waitingCount)") }
        return parts.joined(separator: " · ") + "　⌘↩"
    }
}
