import QwenImageCore
import SwiftUI

/// Left side: mode, prompt (the visual focus), compact parameters, folded advanced options, pinned generate bar.
struct CreatePanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 0) {
            GeometryReader { geo in
                ScrollView {
                    VStack(alignment: .leading, spacing: Metrics.l + 4) {
                        ModeBar()
                        if model.mode != .textToImage { ModeOptions() }
                        PromptComposer()
                            .frame(minHeight: 200, maxHeight: .infinity)
                        ParameterStrip()
                        AdvancedSection()
                    }
                    .padding(.horizontal, Metrics.xl)
                    .padding(.top, Metrics.l)
                    .padding(.bottom, Metrics.l)
                    // The prompt grows into whatever height the other sections leave free.
                    .frame(maxWidth: .infinity, minHeight: geo.size.height, alignment: .top)
                }
                .scrollIndicators(.automatic)
            }
            GenerateBar()
        }
        .background(Brand.base)
    }
}

// MARK: - Mode

struct ModeBar: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: 7) {
            BrandSegmented(
                selection: $model.mode,
                options: CreationMode.allCases.enumerated().map { index, mode in
                    .init(value: mode, title: mode.title, symbol: mode.symbol, help: "\(mode.methodNote)（⌘\(index + 1)）")
                },
                fill: true)
            HStack(spacing: 6) {
                Tag(text: model.mode.isOfficial ? "官方管线" : "通用方法", tint: model.mode.isOfficial ? Brand.success : Brand.warning)
                Text(modeSummary)
                    .font(TypeScale.caption).foregroundStyle(Brand.textTertiary)
                    .lineLimit(1).truncationMode(.tail)
                    .help(model.mode.methodNote)
            }
        }
    }

    private var modeSummary: String {
        switch model.mode {
        case .textToImage: "用文字描述画面"
        case .reference: "参考输入图，创作新画面"
        case .imageToImage: "按强度重绘初始图"
        case .edit: "按指令编辑主图"
        case .inpaint: "涂抹选区，只改选中部分"
        }
    }
}

/// Inputs and strength for the image modes, kept to one compact row.
struct ModeOptions: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let mode = model.mode
        HStack(alignment: .bottom, spacing: Metrics.l) {
            ParamField(label: mode.inputsTitle, trailing: inputsTrailing, plain: true) {
                HStack(spacing: 6) {
                    ForEach(Array(images.prefix(6).enumerated()), id: \.element.id) { i, image in
                        Button { model.workspaceTab = .inputs } label: {
                            Thumbnail(url: image.url, size: 40, corner: 7)
                                .overlay(alignment: .bottomLeading) {
                                if i == 0 && (mode == .edit || mode == .inpaint) {
                                    Text("主").font(.system(size: 9, weight: .bold)).foregroundStyle(.white)
                                        .padding(.horizontal, 4).padding(.vertical, 1)
                                        .background(Brand.fox, in: RoundedRectangle(cornerRadius: 3)).padding(3)
                                }
                            }
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("查看输入图：\(image.name)")
                    }
                    if images.count > 6 {
                        Text("+\(images.count - 6)").font(TypeScale.caption).foregroundStyle(Brand.textSecondary)
                    }
                    Button {
                        addImages()
                    } label: {
                        Image(systemName: "plus")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(Brand.foxText)
                            .frame(width: 40, height: 40)
                            .background {
                                RoundedRectangle(cornerRadius: 7, style: .continuous)
                                    .strokeBorder(Brand.fox.opacity(0.55), style: StrokeStyle(lineWidth: 1.2, dash: [4, 3]))
                            }
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help(mode.needsInitial ? "选择初始图（也可拖入右侧画布或 ⌘V 粘贴）" : "添加图片（也可拖入右侧画布或 ⌘V 粘贴）")
                    if mode.needsMask {
                        Button {
                            model.workspaceTab = .mask
                        } label: {
                            Label(
                                model.mask.isEmpty ? "涂抹蒙版" : String(format: "蒙版 %.0f%%", model.mask.coverage * 100),
                                systemImage: "paintbrush.pointed")
                        }
                        .buttonStyle(SecondaryButtonStyle())
                        .foregroundStyle(model.mask.isEmpty ? Brand.warning : Brand.text)
                        .padding(.leading, 6)
                    }
                }
            }
            .layoutPriority(1)
            if mode.hasStrength {
                ParamField(
                    label: mode == .imageToImage ? "重绘强度" : mode == .edit ? "编辑强度" : "重绘强度",
                    trailing: String(format: "%.2f", model.parameters.strength(for: mode)),
                    help: strengthHelp
                ) {
                    Slider(value: strengthBinding, in: 0.05...1, step: 0.05).controlSize(.small)
                }
                .frame(minWidth: 160, maxWidth: 260)
            }
        }
    }

    private var images: [InputImage] {
        model.mode.needsInitial ? (model.initialImage.map { [$0] } ?? []) : model.conditions
    }

    private var inputsTrailing: String? {
        model.mode.needsInitial ? nil : "\(model.conditions.count)/10"
    }

    private func addImages() {
        model.workspaceTab = .inputs
        let urls = ImageFileActions.chooseImages(multiple: !model.mode.needsInitial)
        if model.mode.needsInitial, let first = urls.first { model.setInitial(url: first) } else { model.importConditions(urls: urls) }
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

    private var strengthHelp: String {
        switch model.mode {
        case .imageToImage: "越低越接近初始图；只运行调度的后 \(Int((model.parameters.imageStrength * 100).rounded()))% 步数"
        case .edit: "1.0 = 官方编辑（从纯噪声开始）；低于 1 时从加噪的主图起步（通用方法，更保守）"
        case .inpaint: "1.0 = 选区完全重绘；更低时选区保留更多原图结构"
        default: ""
        }
    }
}

// MARK: - Prompt

/// Parallel writing surfaces keep the description and exclusions visible together.
struct PromptComposer: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @FocusState private var focused: Field?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private enum Field: Hashable { case prompt, negative }

    var body: some View {
        @Bindable var model = model
        let instruction = model.mode.usesInstruction
        let text = instruction ? $model.parameters.instruction : $model.parameters.prompt
        VStack(spacing: Metrics.s) {
            HStack(alignment: .top, spacing: Metrics.m) {
                editor(instruction ? "编辑指令" : "提示词", text: text, field: .prompt,
                       placeholder: instruction ? "描述要怎样改动主图…" : "描述你想要的画面、主体、光线与构图…")
                editor("负面提示词", text: $model.parameters.negativePrompt, field: .negative,
                       placeholder: "不希望出现的内容…")
            }
            HStack(spacing: 6) {
                Button("预设", systemImage: "text.badge.star") { openWindow(id: "prompt-presets") }
                    .help("打开提示词预设窗口")
                Button("历史", systemImage: "clock.arrow.circlepath") { openWindow(id: "prompt-history") }
                    .help("打开提示词历史窗口")
                Spacer(minLength: 0)
                Text(model.parameters.usesTrueCFG ? "真 CFG · 耗时约翻倍" : "负面词在 Guidance > 1 时生效")
                    .font(TypeScale.caption)
                    .foregroundStyle(model.parameters.usesTrueCFG ? Brand.warning : Brand.textTertiary)
                    .lineLimit(1)
            }
            .buttonStyle(.ghost)
        }
    }

    private func editor(_ title: String, text: Binding<String>, field: Field, placeholder: String) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            SectionHeader(title: title) {
                Text("\(text.wrappedValue.count) 字").font(TypeScale.caption).monospacedDigit().foregroundStyle(Brand.textTertiary)
            }
            TextEditor(text: text)
                .accessibilityLabel(title)
                .font(TypeScale.prompt)
                .lineSpacing(4)
                .scrollContentBackground(.hidden)
                .focused($focused, equals: field)
                .padding(12)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay(alignment: .topLeading) {
                    if text.wrappedValue.isEmpty {
                        Text(placeholder).font(TypeScale.prompt).foregroundStyle(Brand.textTertiary)
                            .padding(.horizontal, 17).padding(.top, 12).allowsHitTesting(false)
                    }
                }
                .background(Brand.paper, in: RoundedRectangle(cornerRadius: Metrics.radiusL, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: Metrics.radiusL, style: .continuous)
                        .strokeBorder(focused == field ? Brand.fox.opacity(0.85) : Brand.hairline,
                                      lineWidth: focused == field ? 1.5 : 1)
                }
                .animation(reduceMotion ? nil : .easeOut(duration: 0.15), value: focused)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Searchable, independent picker windows; text is only applied after an explicit replace or append action.
struct PromptLibraryWindow: View {
    enum Source {
        case presets, history
        var windowID: String { self == .presets ? "prompt-presets" : "prompt-history" }
        var title: String { self == .presets ? "提示词预设" : "提示词历史" }
    }

    private struct Entry: Identifiable {
        let id: String
        let title: String
        let text: String
    }

    @Environment(AppModel.self) private var model
    @Environment(\.dismissWindow) private var dismissWindow
    let source: Source
    @State private var query = ""
    @State private var selection: String?
    @FocusState private var focusedEntry: String?
    @FocusState private var searchFocused: Bool

    var body: some View {
        HSplitView {
            VStack(spacing: Metrics.s) {
                TextField("搜索\(source.title)", text: $query)
                    .textFieldStyle(.plain).padding(8)
                    .background(Brand.paper, in: RoundedRectangle(cornerRadius: Metrics.radiusM))
                    .focused($searchFocused).focusEffectDisabled()
                    .overlay {
                        RoundedRectangle(cornerRadius: Metrics.radiusM)
                            .strokeBorder(searchFocused ? Brand.fox : Brand.hairline)
                    }
                    .padding([.horizontal, .top], Metrics.m)
                ScrollView {
                    LazyVStack(spacing: 6) {
                        ForEach(entries) { entry in
                            Button { selection = entry.id } label: {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(entry.title).font(.body.weight(.medium)).lineLimit(1)
                                    Text(entry.text).font(.caption).foregroundStyle(Brand.textSecondary).lineLimit(2)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading).padding(10)
                                .background(selection == entry.id ? Brand.foxSoft : Brand.surface.opacity(0.4),
                                            in: RoundedRectangle(cornerRadius: Metrics.radiusM))
                                .overlay {
                                    RoundedRectangle(cornerRadius: Metrics.radiusM)
                                        .strokeBorder(focusedEntry == entry.id ? Brand.fox : .clear, lineWidth: 1.5)
                                }
                            }
                            .buttonStyle(.plain)
                            .focused($focusedEntry, equals: entry.id).focusEffectDisabled()
                            .accessibilityAddTraits(selection == entry.id ? .isSelected : [])
                        }
                    }
                    .padding(.horizontal, Metrics.m)
                }
                if source == .presets {
                    Button("把当前提示词存为预设…", systemImage: "plus") { saveCurrent() }
                        .disabled(current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        .buttonStyle(.ghost).padding(.bottom, Metrics.s)
                }
            }
            .frame(minWidth: 230, idealWidth: 270, maxWidth: 340)
            VStack(alignment: .leading, spacing: Metrics.l) {
                Text(model.mode.usesInstruction ? "编辑指令" : "提示词").font(TypeScale.section).foregroundStyle(Brand.textSecondary)
                if let selected {
                    Text(selected.title).font(.title3.weight(.semibold)).lineLimit(2)
                    ScrollView {
                        Text(selected.text).font(.body).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    HStack {
                        if source == .presets {
                            Button("删除预设", systemImage: "trash", role: .destructive) {
                                model.promptPresets.removeAll { $0.id.uuidString == selected.id }
                                selection = nil
                            }
                            .buttonStyle(.ghost)
                        }
                        Spacer(minLength: 0)
                        Button("追加") { apply(selected.text, append: true) }.buttonStyle(SecondaryButtonStyle())
                        Button("使用") { apply(selected.text, append: false) }.buttonStyle(PrimaryButtonStyle())
                    }
                } else {
                    EmptyState(symbol: source == .presets ? "text.badge.star" : "clock.arrow.circlepath",
                               title: entries.isEmpty ? "暂无匹配内容" : "选择一条提示词", compact: true)
                }
            }
            .padding(Metrics.xl)
            .frame(minWidth: 340, maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: 620, minHeight: 380)
        .navigationTitle(source.title)
        .brandWindow()
    }

    private var current: String { model.parameters.text(for: model.mode) }
    private var selected: Entry? { entries.first { $0.id == selection } }

    private var entries: [Entry] {
        let candidates: [Entry]
        if source == .presets {
            candidates = model.promptPresets.filter { $0.isInstruction == model.mode.usesInstruction }
                .map { Entry(id: $0.id.uuidString, title: $0.name, text: $0.text) }
        } else {
            var seen = Set<String>()
            candidates = model.tasks.compactMap { task in
                guard task.mode.usesInstruction == model.mode.usesInstruction else { return nil }
                let text = task.parameters.text(for: task.mode)
                guard !text.isEmpty, seen.insert(text).inserted else { return nil }
                return Entry(id: text, title: String(text.prefix(60)), text: text)
            }
        }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return candidates.filter { q.isEmpty || $0.title.localizedCaseInsensitiveContains(q) || $0.text.localizedCaseInsensitiveContains(q) }
    }

    private func saveCurrent() {
        guard let name = ImageFileActions.ask("提示词预设名称", initial: String(current.prefix(16))) else { return }
        let preset = PromptPreset(name: name, text: current, isInstruction: model.mode.usesInstruction)
        model.promptPresets.append(preset)
        selection = preset.id.uuidString
    }

    private func apply(_ text: String, append: Bool) {
        let value = append && !current.isEmpty ? current + "，" + text : text
        if model.mode.usesInstruction { model.parameters.instruction = value } else { model.parameters.prompt = value }
        dismissWindow(id: source.windowID)
    }
}
