import QwenImageCore
import SwiftUI

// MARK: - Parameters

struct ParametersCard: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        let p = model.parameters
        Card(title: "生成参数", symbol: "slider.horizontal.3", tint: .blue, subtitle: summary) {
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 14) {
                ParameterRow(label: "尺寸") { sizeControls }
                ParameterRow(label: "调度") {
                    VStack(alignment: .leading, spacing: 4) {
                        Picker("调度", selection: $model.parameters.preset) {
                            ForEach(SchedulePreset.allCases) { preset in
                                Text(preset.title).tag(preset)
                            }
                        }
                        .labelsHidden().pickerStyle(.segmented).fixedSize()
                        .onChange(of: model.parameters.preset) { _, new in
                            if new == .turbo && !model.turboAvailable { model.parameters.preset = .standard }
                        }
                        if !model.turboAvailable {
                            Note("Turbo 需要在右下 LoRA 列表中启用一个「加速」用途的 LoRA（如 viggle turbo）", symbol: "lock")
                        } else {
                            Note(p.preset == .turbo ? "viggle_turbo 固定 6 个时间节点，步数锁定为 6" : "linear 网格 + 终端拉伸（官方默认）")
                        }
                    }
                }
                ParameterRow(label: "步数") {
                    HStack(spacing: 8) {
                        if p.preset == .turbo {
                            Label("6", systemImage: "lock.fill").monospacedDigit().foregroundStyle(.secondary)
                        } else {
                            TextField("步数", value: $model.parameters.steps, format: .number).frame(width: 56)
                                .textFieldStyle(.roundedBorder)
                            Stepper("", value: $model.parameters.steps, in: 1...100).labelsHidden()
                            ForEach([20, 30, 40, 50], id: \.self) { n in
                                Chip(title: "\(n)", selected: p.steps == n) { model.parameters.steps = n }
                            }
                        }
                        Spacer(minLength: 0)
                        estimate
                    }
                }
                ParameterRow(label: "Guidance") {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Slider(value: $model.parameters.guidance, in: 1...10, step: 0.5).frame(maxWidth: 300)
                            Text(String(format: "%.1f", p.guidance)).monospacedDigit().frame(width: 34)
                        }
                        Note(guidanceNote)
                    }
                }
                ParameterRow(label: "种子") { seedControls }
                ParameterRow(label: "张数") {
                    HStack {
                        Stepper(value: $model.parameters.count, in: 1...16) {
                            Text("\(p.count) 张").monospacedDigit()
                        }
                        .fixedSize()
                        if p.count > 1 {
                            Picker("种子规则", selection: $model.parameters.batchSeeds) {
                                Text("种子逐张 +1").tag(GenerationParameters.BatchSeeds.increment)
                                Text("逐张随机").tag(GenerationParameters.BatchSeeds.random)
                            }
                            .labelsHidden().fixedSize()
                        }
                    }
                }
            }
        }
    }

    private var summary: String {
        let p = model.parameters
        return "\(p.width)×\(p.height) · \(p.effectiveSteps) 步 · Guidance \(String(format: "%.1f", p.guidance))"
    }

    private var guidanceNote: String {
        let p = model.parameters
        if p.guidance <= 1 { return "1.0 = 不做引导（模型默认）" }
        return p.usesTrueCFG ? "真 CFG \(String(format: "%.1f", p.guidance))：每步正负两次前向，耗时约翻倍" : "需要填写负面提示词才会生效"
    }

    private var sizeControls: some View {
        @Bindable var model = model
        let followable = model.mode != .textToImage
        return VStack(alignment: .leading, spacing: 8) {
            if followable {
                Picker("尺寸来源", selection: $model.parameters.sizePolicy) {
                    Text("跟随\(model.mode == .reference ? "最后一张参考图" : model.mode == .imageToImage ? "初始图" : "主图")比例").tag(GenerationParameters.SizePolicy.followImage)
                    Text("手动").tag(GenerationParameters.SizePolicy.manual)
                }
                .labelsHidden().pickerStyle(.segmented).fixedSize()
                .onChange(of: model.parameters.sizePolicy) { _, _ in model.applySizePolicy() }
            }
            if !followable || model.parameters.sizePolicy == .manual {
                HStack(spacing: 2) {
                    ForEach(SizePreset.all) { preset in
                        AspectChip(
                            label: preset.label, width: preset.width, height: preset.height,
                            selected: model.parameters.width == preset.width && model.parameters.height == preset.height
                        ) {
                            model.parameters.width = preset.width
                            model.parameters.height = preset.height
                        }
                    }
                }
            }
            HStack(spacing: 6) {
                TextField("宽", value: $model.parameters.width, format: .number.grouping(.never)).frame(width: 64)
                    .textFieldStyle(.roundedBorder)
                Stepper("", value: $model.parameters.width, in: 256...2048, step: 32).labelsHidden()
                Image(systemName: "xmark").font(.caption).foregroundStyle(.secondary)
                TextField("高", value: $model.parameters.height, format: .number.grouping(.never)).frame(width: 64)
                    .textFieldStyle(.roundedBorder)
                Stepper("", value: $model.parameters.height, in: 256...2048, step: 32).labelsHidden()
                Button("横竖互换", systemImage: "arrow.left.arrow.right") {
                    let w = model.parameters.width
                    model.parameters.width = model.parameters.height
                    model.parameters.height = w
                }
                .labelStyle(.iconOnly).buttonStyle(.borderless)
                Text("\(model.parameters.imageTokens) token · \(String(format: "%.2f", Double(model.parameters.width * model.parameters.height) / 1e6)) MP")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .disabled(followable && model.parameters.sizePolicy == .followImage)
            if model.parameters.width % 32 != 0 || model.parameters.height % 32 != 0 {
                Note("宽高必须是 32 的倍数", tint: .red, symbol: "exclamationmark.circle")
            }
        }
    }

    private var seedControls: some View {
        @Bindable var model = model
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Picker("种子", selection: $model.parameters.seedMode) {
                    Text("随机").tag(GenerationParameters.SeedMode.random)
                    Text("固定").tag(GenerationParameters.SeedMode.fixed)
                }
                .labelsHidden().pickerStyle(.segmented).fixedSize()
                if model.parameters.seedMode == .fixed {
                    TextField("种子", value: $model.parameters.seed, format: .number.grouping(.never)).frame(width: 120)
                        .textFieldStyle(.roundedBorder)
                    Button("随机取一个", systemImage: "dice") {
                        model.parameters.seed = UInt64.random(in: 0..<UInt64(UInt32.max))
                    }
                    .labelStyle(.iconOnly).buttonStyle(.borderless)
                }
            }
            if let last = model.lastSeed {
                HStack(spacing: 6) {
                    Text("上一张实际种子 \(String(last))").font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    Button("固定使用") {
                        model.parameters.seedMode = .fixed
                        model.parameters.seed = last
                    }
                    .controlSize(.mini)
                    Button("复制") { ImageFileActions.copyText(String(last)) }.controlSize(.mini)
                }
            }
        }
    }

    @ViewBuilder
    private var estimate: some View {
        let p = model.parameters
        if let rate = model.estimatedSecondsPerStep {
            let steps = model.mode.hasStrength ? Int((Float(p.effectiveSteps) * p.strength(for: model.mode)).rounded(.up)) : p.effectiveSteps
            Label("约 \(Format.duration(rate * Double(steps) * Double(p.count)))", systemImage: "timer")
                .font(.caption).foregroundStyle(.secondary)
                .help("按本机实测 \(String(format: "%.2f", rate)) s/step 估算去噪耗时")
        } else {
            Label("首次运行后估算", systemImage: "timer").font(.caption).foregroundStyle(.tertiary)
        }
    }
}

// MARK: - Performance

struct PerformanceCard: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let m = model.metrics
        Card(title: "性能与内存", symbol: "memorychip.fill", tint: .green, subtitle: "32 GB 平衡档 · BF16 原始权重") {
            if let pressure = m.pressure {
                StatusPill(
                    text: "内存压力 \(pressure.label)",
                    color: pressure == .normal ? .green : pressure == .warning ? .orange : .red)
            }
        } content: {
            HStack(spacing: 8) {
                profile("32 GB 平衡", detail: "默认 · 已验证", available: true)
                profile("高画质优先", detail: "需更多内存，未验证", available: false)
                profile("省内存", detail: "需量化，未验证", available: false)
                profile("高速实验", detail: "需步骤缓存，未验证", available: false)
            }
            MemoryBar(used: m.appFootprint, estimate: model.estimatedPeakBytes, total: m.physicalMemory)
            HStack(alignment: .top, spacing: 28) {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                    InfoRow(label: "统一内存", value: Format.bytes(m.physicalMemory))
                    InfoRow(label: "应用占用", value: Format.bytes(m.appFootprint))
                    InfoRow(
                        label: "MLX 分配",
                        value: model.live.map { "\(Format.bytes($0.mlxActive))（峰值 \(Format.bytes($0.mlxPeak))）" } ?? "空闲")
                    InfoRow(label: "Swap", value: m.swapUsed.map { "\(Format.bytes($0)) / \(Format.bytes(m.swapTotal ?? 0))" } ?? "不可得")
                    InfoRow(
                        label: "可用内存", value: m.freePercent.map { "\($0)%" } ?? "不可得",
                        tint: m.pressure == .critical ? .red : m.pressure == .warning ? .orange : nil)
                    InfoRow(label: "热状态", value: m.thermal.label, tint: m.thermal.rawValue >= 2 ? .orange : nil)
                }
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                    let live = model.live
                    InfoRow(label: "阶段", value: live?.stageLabel ?? "空闲")
                    InfoRow(
                        label: "进度",
                        value: live.map { l in
                            l.stage == .denoising
                                ? "第 \(l.imageIndex + 1)/\(l.imageCount) 张 · 步 \(l.step)/\(l.totalSteps)"
                                : "第 \(l.imageIndex + 1)/\(l.imageCount) 张"
                        } ?? "–")
                    InfoRow(label: "速度", value: live?.secondsPerStep.map { String(format: "%.2f s/step", $0) } ?? "–")
                    InfoRow(label: "本张耗时", value: live.map { Format.duration($0.elapsed) } ?? "–")
                    InfoRow(label: "预计剩余", value: live?.remaining.map { Format.duration($0) } ?? "–")
                    InfoRow(label: "输出", value: "等待 \(model.waitingCount) · 失败 \(model.failedCount)")
                }
            }
            .font(.callout)
            Note("GPU 专用显存、硬件利用率等系统不提供的指标不显示；MLX 分配为运行时实际申请的统一内存。")
        }
    }

    private func profile(_ title: String, detail: String, available: Bool) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                if available { Image(systemName: "checkmark.circle.fill").foregroundStyle(.tint) }
                Text(title).font(.callout.weight(available ? .semibold : .regular))
            }
            Text(detail).font(.caption2).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10).padding(.vertical, 7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(available ? Color.accentColor.opacity(0.1) : Color.primary.opacity(0.03),
                    in: RoundedRectangle(cornerRadius: Design.innerCorner, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: Design.innerCorner, style: .continuous)
            .strokeBorder(available ? Color.accentColor.opacity(0.6) : Design.hairline.opacity(0.5)))
        .opacity(available ? 1 : 0.6)
        .help(available ? "当前档位" : "尚未验证，暂不可用")
    }
}

/// Unified-memory bar: current app footprint and the estimated peak of the configured task.
struct MemoryBar: View {
    let used: Int
    let estimate: Int64
    let total: Int64

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            GeometryReader { geo in
                let w = geo.size.width
                let estFrac = min(Double(estimate) / Double(max(total, 1)), 1)
                let usedFrac = min(Double(used) / Double(max(total, 1)), 1)
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.07))
                    Capsule().fill(color(estFrac).opacity(0.28)).frame(width: w * estFrac)
                    Capsule().fill(color(usedFrac).gradient).frame(width: max(w * usedFrac, 4))
                    Rectangle().fill(Color.red.opacity(0.6)).frame(width: 1.5).offset(x: w * 0.85)
                }
            }
            .frame(height: 8)
            HStack {
                Text("当前 \(Format.bytes(used))").foregroundStyle(.secondary)
                Spacer()
                Text("本任务预计峰值 \(Format.bytes(estimate)) / \(Format.bytes(total))")
                    .foregroundStyle(Double(estimate) > Double(total) * 0.85 ? .red : .secondary)
            }
            .font(.caption)
        }
    }

    private func color(_ fraction: Double) -> Color { fraction > 0.85 ? .red : fraction > 0.7 ? .orange : .green }
}

// MARK: - Advanced

struct AdvancedCard: View {
    @Environment(AppModel.self) private var model
    @State private var expanded = false

    var body: some View {
        Card(title: "高级与输出", symbol: "gearshape.2.fill", tint: .gray) {
            Button { withAnimation(.snappy(duration: 0.2)) { expanded.toggle() } } label: {
                Image(systemName: "chevron.down").rotationEffect(.degrees(expanded ? 180 : 0))
            }
            .buttonStyle(.borderless)
        } content: {
            if expanded {
                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 10) {
                    ParameterRow(label: "输出目录") {
                        HStack {
                            Text(model.outputDirectory.path).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                            Button("更改…") { if let url = ImageFileActions.chooseFolder() { model.outputDirectory = url } }
                                .controlSize(.small)
                            Button("打开", systemImage: "folder") { NSWorkspace.shared.open(model.outputDirectory) }
                                .labelStyle(.iconOnly).buttonStyle(.borderless)
                        }
                    }
                    ParameterRow(label: "保存") {
                        Note("无损 PNG（参数写入 PNG 描述字段）+ 同名 JSON 记录 + 输入图副本（inputs/）；命名：日期-时间-种子，不覆盖已有文件。导出格式在设置中选择。")
                    }
                    ParameterRow(label: "计算精度") {
                        Note("Transformer / 文本编码器：BF16 原始权重（fp32 残差流）· 视觉编码器与 VAE：真 FP32（关闭 TF32）。量化档位尚未验证，不提供。")
                    }
                    ParameterRow(label: "不可用") {
                        VStack(alignment: .leading, spacing: 3) {
                            Note("步骤缓存：会改变计算结果，执行器未实现，默认关闭", symbol: "nosign")
                            Note("VAE 分块：不需要，1024² 解码峰值约 12 GB", symbol: "nosign")
                            Note("提示词自动扩写 / 生成后自检：需要额外模型能力，未开放", symbol: "nosign")
                            Note("自动语义蒙版：本地没有可信的分割模型，未开放", symbol: "nosign")
                        }
                    }
                }
            } else {
                Text("输出目录、计算精度与暂不开放的功能说明").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Run bar

/// Generate / stop controls pinned under the parameter panel, with the live stage and blocking problems.
struct RunBar: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            if let live = model.live {
                ProgressRing(fraction: live.overallFraction).frame(width: 30, height: 30)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(live.stageLabel).font(.callout.weight(.semibold))
                        if live.stage == .denoising { Text("\(live.step)/\(live.totalSteps)").monospacedDigit() }
                        if live.imageCount > 1 { Text("· 第 \(live.imageIndex + 1)/\(live.imageCount) 张").foregroundStyle(.secondary) }
                    }
                    Text([live.secondsPerStep.map { String(format: "%.2f s/step", $0) }, live.remaining.map { "剩余约 \(Format.duration($0))" }]
                        .compactMap { $0 }.joined(separator: " · "))
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(model.blockingIssues.prefix(2), id: \.self) { issue in
                        Label(issue, systemImage: "exclamationmark.circle.fill").foregroundStyle(.red)
                    }
                    ForEach(model.warnings.prefix(model.blockingIssues.isEmpty ? 2 : 0), id: \.self) { warning in
                        Label(warning, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    }
                    if model.blockingIssues.isEmpty && model.warnings.isEmpty {
                        Label(summary, systemImage: model.mode.symbol).foregroundStyle(.secondary)
                    }
                }
                .font(.caption)
                .lineLimit(1)
            }
            Spacer(minLength: 8)
            if model.waitingCount > 0 {
                Text("排队 \(model.waitingCount)").font(.caption).foregroundStyle(.secondary)
            }
            Button(role: .cancel) {
                model.stopCurrent()
            } label: {
                Label("停止", systemImage: "stop.fill")
            }
            .controlSize(.large)
            .disabled(!model.isRunning)
            .keyboardShortcut(".", modifiers: .command)
            .help("停止当前任务（⌘.）")
            Button {
                model.enqueue()
            } label: {
                Label(model.isRunning ? "加入队列" : "生成", systemImage: model.isRunning ? "text.badge.plus" : "sparkles")
                    .frame(minWidth: 96)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!model.blockingIssues.isEmpty)
            .keyboardShortcut(.return, modifiers: .command)
            .help(model.blockingIssues.first ?? "加入队列并生成（⌘↩）")
        }
        .padding(.horizontal, Design.panelPadding)
        .padding(.vertical, 10)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }

    private var summary: String {
        let p = model.parameters
        return "\(model.mode.title) · \(p.width)×\(p.height) · \(p.effectiveSteps) 步 · "
            + (p.seedMode == .random ? "随机种子" : "种子 \(p.seed)") + " · \(model.activeLoRAs.count) 个 LoRA · ⌘↩ 生成"
    }
}
