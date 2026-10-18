import QwenImageCore
import SwiftUI

/// Main window: the creation side (≈2/3) and the artwork side (≈1/3), the latter split into canvas and assistants.
struct WorkbenchView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        @Bindable var model = model
        ProportionalSplit(
            axis: .horizontal, fraction: $model.leftFraction, defaultFraction: 2.0 / 3.0, minFirst: 560, minSecond: 380
        ) {
            CreatePanel()
        } second: {
            ProportionalSplit(
                axis: .vertical, fraction: $model.rightTopFraction, defaultFraction: 0.5, minFirst: 260, minSecond: 240
            ) {
                ImageWorkspace()
            } second: {
                AssistantPanel()
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) { ModelMenu() }
            ToolbarItemGroup(placement: .primaryAction) {
                PresetMenu()
                Button("图库", systemImage: "photo.on.rectangle.angled") { openWindow(id: "gallery") }
                    .help("图库：筛选、比较、批量导出（⇧⌘G）")
            }
            ToolbarItem(placement: .primaryAction) { RuntimeStatusButton() }
        }
        .navigationTitle("")
        .brandWindow()
    }
}

/// Toolbar model switcher: name, precision and readiness only; details live in the menu and the inspector.
struct ModelMenu: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Menu {
            Section("模型") {
                ForEach(model.models.sorted { ($0.lastUsed ?? .distantPast) > ($1.lastUsed ?? .distantPast) }) { entry in
                    Button {
                        model.selectModel(entry)
                    } label: {
                        Label(
                            entry.name,
                            systemImage: entry.path == model.modelRoot?.path
                                ? "checkmark" : (entry.isComplete ? "cube" : "exclamationmark.triangle"))
                    }
                    .disabled(model.isRunning && entry.path != model.modelRoot?.path)
                }
            }
            Divider()
            Button("模型库…", systemImage: "cube.box") { openWindow(id: "models") }
            Button("组件与内存状态…", systemImage: "gauge.with.dots.needle.33percent") { openWindow(id: "inspector") }
            Button("重新校验", systemImage: "checkmark.shield") { model.validateModel() }
                .disabled(model.modelRoot == nil || model.modelStatus == .validating)
        } label: {
            HStack(spacing: 7) {
                Circle().fill(statusColor).frame(width: 7, height: 7)
                Text(model.modelName).fontWeight(.semibold)
                Text("BF16").font(.caption2.weight(.bold)).foregroundStyle(Brand.textSecondary)
            }
            .padding(.horizontal, 4)
        }
        .menuIndicator(.visible)
        .fixedSize()
        .help("\(model.modelStatus.label) · " + (model.isRunning ? "任务进行中，完成后才能切换模型" : "切换模型"))
    }

    private var statusColor: Color {
        switch model.modelStatus {
        case .ready: Brand.success
        case .inUse: Brand.fox
        case .validating: Brand.warning
        case .error: Brand.danger
        case .unselected: Brand.textTertiary
        }
    }
}

/// The single low-key status entry of the main window; opens the runtime inspector.
struct RuntimeStatusButton: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button {
            openWindow(id: "inspector")
        } label: {
            HStack(spacing: 6) {
                if let live = model.live {
                    ProgressRing(fraction: live.overallFraction, lineWidth: 2).frame(width: 12, height: 12)
                    Text(live.stage == .denoising ? "生成中 \(live.step)/\(live.totalSteps)" : "生成中")
                        .monospacedDigit()
                } else {
                    Image(systemName: symbol).foregroundStyle(color)
                    Text(text)
                }
            }
            .font(.callout)
            .padding(.horizontal, 6)
        }
        .help("Runtime Inspector：内存、组件、阶段与日志")
    }

    private var text: String {
        switch model.modelStatus {
        case .ready, .inUse: model.waitingCount > 0 ? "排队 \(model.waitingCount)" : "就绪"
        case .validating: "校验中"
        case .error: "模型不完整"
        case .unselected: "未选择模型"
        }
    }

    private var symbol: String {
        switch model.modelStatus {
        case .ready, .inUse: "checkmark.circle"
        case .validating: "hourglass"
        case .error, .unselected: "exclamationmark.triangle"
        }
    }

    private var color: Color {
        switch model.modelStatus {
        case .ready, .inUse: Brand.success
        case .validating: Brand.warning
        case .error, .unselected: Brand.danger
        }
    }
}

/// Toolbar menu for generation presets.
struct PresetMenu: View {
    @Environment(AppModel.self) private var model
    @State private var saving = false

    var body: some View {
        Menu {
            let sorted = model.presets.sorted { ($0.favorite ? 0 : 1, $0.name) < ($1.favorite ? 0 : 1, $1.name) }
            if sorted.isEmpty { Text("还没有预设") }
            ForEach(sorted) { preset in
                Menu {
                    Button("加载", systemImage: "arrow.down.doc") { model.apply(preset) }
                    if !preset.missingFiles.isEmpty {
                        Text("缺失 \(preset.missingFiles.count) 个文件，加载时会跳过并保留当前值")
                    }
                    Divider()
                    Button(preset.favorite ? "取消收藏" : "收藏", systemImage: "star") { update(preset) { $0.favorite.toggle() } }
                    Button(preset.isDefault ? "取消默认" : "设为默认", systemImage: "checkmark.seal") { model.setDefaultPreset(preset.id) }
                    Button("重命名…", systemImage: "pencil") {
                        if let name = ImageFileActions.ask("重命名预设", initial: preset.name) { update(preset) { $0.name = name } }
                    }
                    Button("删除", systemImage: "trash", role: .destructive) { model.presets.removeAll { $0.id == preset.id } }
                } label: {
                    Label(
                        preset.name + (preset.isDefault ? "（默认）" : ""),
                        systemImage: preset.favorite ? "star.fill" : (preset.missingFiles.isEmpty ? "slider.horizontal.3" : "exclamationmark.triangle"))
                }
            }
            Divider()
            Button("保存当前参数为预设…", systemImage: "plus") { saving = true }
        } label: {
            Label("预设", systemImage: "square.stack")
        }
        .sheet(isPresented: $saving) { SavePresetSheet() }
        .help("生成预设")
    }

    private func update(_ preset: GenerationPreset, _ change: (inout GenerationPreset) -> Void) {
        guard let i = model.presets.firstIndex(where: { $0.id == preset.id }) else { return }
        change(&model.presets[i])
    }
}

struct SavePresetSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var includes: Set<GenerationPreset.Part> = [.prompt, .loras]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("保存生成预设").font(.title3.weight(.semibold))
            TextField("预设名称", text: $name).textFieldStyle(.roundedBorder)
            Text("包含").font(TypeScale.section).foregroundStyle(Brand.textSecondary)
            ForEach(GenerationPreset.Part.allCases) { part in
                Toggle(part.label, isOn: Binding(
                    get: { includes.contains(part) },
                    set: { if $0 { includes.insert(part) } else { includes.remove(part) } }))
            }
            Note("始终包含模式、尺寸、步数、Guidance、种子、调度与强度。模型或 LoRA 文件失效时，加载会提示，不会悄悄替换。")
            HStack {
                Spacer()
                Button("取消") { dismiss() }.keyboardShortcut(.cancelAction).buttonStyle(SecondaryButtonStyle())
                Button("保存") {
                    model.savePreset(named: name.isEmpty ? "预设 \(model.presets.count + 1)" : name, includes: includes)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(PrimaryButtonStyle())
            }
        }
        .padding(20)
        .frame(width: 400)
        .brandWindow()
    }
}
