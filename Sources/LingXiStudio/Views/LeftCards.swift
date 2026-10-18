import QwenImageCore
import SwiftUI

// MARK: - Model

struct ModelCard: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @State private var expanded = false

    var body: some View {
        Card(
            title: "模型", symbol: "cube.fill", tint: .indigo,
            subtitle: model.modelBytes > 0 ? "Qwen-Image 2.1 · BF16 · Safetensors · \(Format.bytes(model.modelBytes))" : model.modelName
        ) {
            HStack(spacing: 8) {
                Button { expanded.toggle() } label: {
                    Image(systemName: "chevron.down").rotationEffect(.degrees(expanded ? 180 : 0))
                }
                .buttonStyle(.borderless)
                .help(expanded ? "收起组件" : "展开组件")
            }
        } content: {
            HStack(spacing: 8) {
                ForEach(["text_encoder", "transformer", "vae"], id: \.self) { name in
                    ComponentChip(name: name)
                }
                Spacer()
                if case .error(let message) = model.modelStatus {
                    Label("模型不完整", systemImage: "exclamationmark.octagon.fill").foregroundStyle(.red).help(message)
                }
            }
            if expanded {
                VStack(alignment: .leading, spacing: 8) {
                    if let root = model.modelRoot {
                        Text(root.path).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                            .lineLimit(1).truncationMode(.middle)
                    }
                    if case .error(let message) = model.modelStatus {
                        Text(message).font(.callout).foregroundStyle(.red).textSelection(.enabled)
                    }
                    Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 5) {
                        ForEach(model.components, id: \.name) { c in
                            GridRow {
                                Text(Self.title(c.name))
                                Text(c.files > 0 ? "\(c.files) 个分片" : "配置文件").foregroundStyle(.secondary)
                                Text(c.bytes > 0 ? Format.bytes(c.bytes) : "–").monospacedDigit().foregroundStyle(.secondary)
                                Image(systemName: c.problems.isEmpty ? "checkmark.circle.fill" : "xmark.octagon.fill")
                                    .foregroundStyle(c.problems.isEmpty ? .green : .red)
                            }
                        }
                    }
                    .font(.callout)
                    HStack {
                        Button("模型库…", systemImage: "cube.box") { openWindow(id: "models") }
                        Button("重新校验", systemImage: "checkmark.shield") { model.validateModel() }
                            .disabled(model.modelRoot == nil || model.modelStatus == .validating)
                        Button("释放会话状态", systemImage: "eject") { model.unloadModel() }
                            .disabled(model.isRunning || model.modelRoot == nil)
                    }
                    .controlSize(.small)
                    Note("32 GB 平衡档：任务按 文本编码器 → (VAE 编码) → Transformer → VAE 解码 顺序加载，同一时刻只驻留一个大组件，用完即释放。")
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .animation(.snappy(duration: 0.2), value: expanded)
    }

    static func title(_ name: String) -> String {
        switch name {
        case "text_encoder": "文本编码器"
        case "transformer": "Transformer"
        case "vae": "VAE"
        case "processor": "分词器"
        case "scheduler": "调度配置"
        default: name
        }
    }
}

/// Component residency chip: shows the live state during a run.
struct ComponentChip: View {
    @Environment(AppModel.self) private var model
    let name: String

    var body: some View {
        let state = model.residency[name] ?? .notLoaded
        let component = model.components.first { $0.name == name }
        HStack(spacing: 5) {
            Group {
                switch state {
                case .loading: ProgressView().controlSize(.mini)
                case .resident: Image(systemName: "bolt.fill").foregroundStyle(.blue)
                case .released: Image(systemName: "arrow.uturn.down").foregroundStyle(.secondary)
                case .notLoaded:
                    Image(systemName: component?.problems.isEmpty == false ? "xmark.octagon.fill" : "circle.dashed")
                        .foregroundStyle(component?.problems.isEmpty == false ? .red : .secondary)
                }
            }
            .frame(width: 12)
            Text(ModelCard.title(name)).font(.caption.weight(.medium))
            Text(state.label).font(.caption2).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(state == .resident ? Color.blue.opacity(0.12) : Color.primary.opacity(0.05), in: Capsule())
    }
}

// MARK: - Prompt

struct PromptCard: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        let instruction = model.mode.usesInstruction
        Card(
            title: instruction ? "编辑指令" : "提示词", symbol: instruction ? "wand.and.stars" : "text.cursor",
            tint: .orange, subtitle: instruction ? "描述要怎样改动主图（与文生图提示词分开保存）" : "描述想要的画面，中英文均可"
        ) {
            HStack(spacing: 4) {
                PromptPresetsMenu()
                Menu {
                    let recent = recentPrompts
                    if recent.isEmpty { Text("暂无历史") }
                    ForEach(recent, id: \.self) { text in
                        Button(String(text.prefix(80))) { setText(text) }
                    }
                } label: {
                    Image(systemName: "clock.arrow.circlepath")
                }
                .menuStyle(.borderlessButton).fixedSize()
                .help("历史提示词")
            }
        } content: {
            PromptEditor(
                text: instruction ? $model.parameters.instruction : $model.parameters.prompt,
                placeholder: instruction ? "例如：把招牌颜色改成蓝色，其余保持不变" : "例如：雨夜街头的霓虹招牌，写着「QWEN IMAGE 2.1」，湿漉漉的地面倒影",
                minHeight: 96)
            HStack {
                Text("\(model.parameters.text(for: model.mode).count) 字").font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            DisclosureGroup {
                VStack(alignment: .leading, spacing: 6) {
                    PromptEditor(text: $model.parameters.negativePrompt, placeholder: "不希望出现的内容（可选）", minHeight: 44)
                        .disabled(model.parameters.guidance <= 1)
                        .opacity(model.parameters.guidance <= 1 ? 0.55 : 1)
                    if model.parameters.guidance <= 1 {
                        Note("Guidance ≤ 1 时不计算负面分支，这里的内容不会生效。把 Guidance 调到 1 以上即启用真 CFG。", symbol: "info.circle")
                    } else {
                        Note("真 CFG：每步额外计算一次负面分支，耗时约翻倍。", tint: .orange, symbol: "exclamationmark.triangle")
                    }
                }
                .padding(.top, 6)
            } label: {
                HStack(spacing: 6) {
                    Text("负面提示词").font(.callout)
                    if model.parameters.usesTrueCFG { StatusPill(text: "生效中", color: .orange) }
                }
            }
        }
    }

    private func setText(_ text: String) {
        if model.mode.usesInstruction { model.parameters.instruction = text } else { model.parameters.prompt = text }
    }

    private var recentPrompts: [String] {
        var seen = Set<String>()
        return model.tasks.compactMap { task -> String? in
            guard task.mode.usesInstruction == model.mode.usesInstruction else { return nil }
            let text = task.parameters.text(for: task.mode)
            return text.isEmpty || !seen.insert(text).inserted ? nil : text
        }.prefix(20).map { $0 }
    }
}

struct PromptPresetsMenu: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Menu {
            let mine = model.promptPresets.filter { $0.isInstruction == model.mode.usesInstruction }
            if mine.isEmpty { Text("还没有提示词预设") }
            ForEach(mine) { preset in
                Menu(preset.name) {
                    Button("替换为此预设") { set(preset.text) }
                    Button("追加到末尾") { set(current.isEmpty ? preset.text : current + "，" + preset.text) }
                    Divider()
                    Button("删除", role: .destructive) { model.promptPresets.removeAll { $0.id == preset.id } }
                }
            }
            Divider()
            Button("把当前内容存为预设…") {
                guard !current.isEmpty, let name = ImageFileActions.ask("提示词预设名称", initial: String(current.prefix(16))) else { return }
                model.promptPresets.append(PromptPreset(name: name, text: current, isInstruction: model.mode.usesInstruction))
            }
            .disabled(current.isEmpty)
        } label: {
            Image(systemName: "text.badge.star")
        }
        .menuStyle(.borderlessButton).fixedSize()
        .help("提示词预设")
    }

    private var current: String { model.parameters.text(for: model.mode) }

    private func set(_ text: String) {
        if model.mode.usesInstruction { model.parameters.instruction = text } else { model.parameters.prompt = text }
    }
}

// MARK: - Mode options

struct ModeOptionsCard: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        let mode = model.mode
        Card(title: mode.title, symbol: mode.symbol, tint: mode.isOfficial ? .teal : .purple, subtitle: mode.methodNote) {
            StatusPill(text: mode.isOfficial ? "官方管线" : "通用方法", color: mode.isOfficial ? .teal : .purple)
        } content: {
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 12) {
                if mode.needsConditions || mode.needsInitial {
                    ParameterRow(label: "输入图") {
                        HStack(spacing: 6) {
                            let images = mode.needsInitial ? (model.initialImage.map { [$0] } ?? []) : model.conditions
                            ForEach(Array(images.prefix(6).enumerated()), id: \.element.id) { i, image in
                                Thumbnail(url: image.url, size: 34, corner: 6)
                                    .overlay(alignment: .topLeading) {
                                        if i == 0 && (mode == .edit || mode == .inpaint) {
                                            Text("主").font(.system(size: 9, weight: .bold)).padding(2)
                                                .background(.tint, in: RoundedRectangle(cornerRadius: 3)).foregroundStyle(.white)
                                        }
                                    }
                            }
                            if images.count > 6 { Text("+\(images.count - 6)").font(.caption).foregroundStyle(.secondary) }
                            Button(images.isEmpty ? "添加…" : "管理…") { model.workspaceTab = .inputs }
                                .controlSize(.small)
                        }
                    }
                }
                if mode.hasStrength {
                    ParameterRow(label: mode == .imageToImage ? "重绘强度" : "编辑强度") {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Slider(value: strengthBinding, in: 0.05...1, step: 0.05).frame(maxWidth: 280)
                                Text(String(format: "%.2f", model.parameters.strength(for: mode))).monospacedDigit().frame(width: 40)
                            }
                            Note(strengthNote)
                        }
                    }
                }
                if mode.needsMask {
                    ParameterRow(label: "蒙版") {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(model.mask.isEmpty ? "未涂抹" : String(format: "已选 %.0f%%", model.mask.coverage * 100))
                                    .foregroundStyle(model.mask.isEmpty ? .orange : .primary)
                                Button("打开蒙版画布") { model.workspaceTab = .mask }.controlSize(.small)
                            }
                            Toggle("反选（涂抹区域保留，其余重绘）", isOn: $model.parameters.maskInvert)
                            HStack {
                                Text("边缘羽化").foregroundStyle(.secondary)
                                Slider(value: featherBinding, in: 0...48, step: 2).frame(maxWidth: 180)
                                Text("\(model.parameters.maskFeather) px").monospacedDigit().frame(width: 44)
                            }
                            Toggle("选区外像素保持原样（按羽化蒙版贴回原图）", isOn: $model.parameters.keepUnmasked)
                            Note("约定：白色 / 涂抹区域 = 待重绘区域。")
                        }
                    }
                }
                if mode.needsConditions {
                    ParameterRow(label: "参考策略") {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(mode == .reference
                                 ? "按「图 1、图 2…」顺序送入视觉编码器，可在提示词中按序号指代"
                                 : "第 1 张为主图（被编辑对象），其余为附加参考")
                            Note("每张参考图按 1024² 面积缩放；每张约增加 2.2 GB 内存与 1 s/步。界面上限 10 张，实际受内存预算限制。")
                        }
                    }
                }
            }
        }
    }

    private var strengthBinding: Binding<Double> {
        Binding(
            get: { Double(model.parameters.strength(for: model.mode)) },
            set: { value in
                switch model.mode {
                case .imageToImage: model.parameters.imageStrength = Float(value)
                case .edit: model.parameters.editStrength = Float(value)
                case .inpaint: model.parameters.inpaintStrength = Float(value)
                default: break
                }
            })
    }

    private var featherBinding: Binding<Double> {
        Binding(get: { Double(model.parameters.maskFeather) }, set: { model.parameters.maskFeather = Int($0) })
    }

    private var strengthNote: String {
        switch model.mode {
        case .imageToImage: "越低越接近初始图；只运行调度的后 \(Int((model.parameters.imageStrength * 100).rounded()))% 步数"
        case .edit: model.parameters.editStrength >= 0.999 ? "1.0 = 官方编辑（从纯噪声开始）" : "低于 1 时从加噪的主图起步（通用方法，更保守）"
        case .inpaint: "1.0 = 选区完全重绘；更低时选区保留更多原图结构"
        default: ""
        }
    }
}
