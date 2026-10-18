import QwenImageCore
import SwiftUI

/// Main window: parameters on the left (2/3), image workspace top right and management bottom right (1:1).
struct WorkbenchView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        @Bindable var model = model
        ProportionalSplit(
            axis: .horizontal, fraction: $model.leftFraction, defaultFraction: 2.0 / 3.0, minFirst: 600, minSecond: 360
        ) {
            ParameterPanel()
        } second: {
            ProportionalSplit(
                axis: .vertical, fraction: $model.rightTopFraction, defaultFraction: 0.5, minFirst: 240, minSecond: 260
            ) {
                ImageWorkspace()
            } second: {
                ManagementPanel()
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) { ModelMenu() }
            ToolbarItemGroup(placement: .primaryAction) {
                PresetMenu()
                Button("图库", systemImage: "photo.on.rectangle.angled") { openWindow(id: "gallery") }
                    .help("图库详细视图：筛选、比较、批量导出")
                Button("模型库", systemImage: "cube.box") { openWindow(id: "models") }
                    .help("模型库与下载管理")
                Button("恢复默认布局", systemImage: "rectangle.split.2x1") {
                    withAnimation(.snappy) {
                        model.leftFraction = 2.0 / 3.0
                        model.rightTopFraction = 0.5
                    }
                }
                .help("左 2/3 · 右 1/3，右栏上下 1:1")
            }
        }
        .navigationTitle("")
    }
}

/// Toolbar model switcher with readiness state.
struct ModelMenu: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Menu {
            ForEach(model.models.sorted { ($0.lastUsed ?? .distantPast) > ($1.lastUsed ?? .distantPast) }) { entry in
                Button {
                    model.selectModel(entry)
                } label: {
                    Label(entry.name, systemImage: entry.path == model.modelRoot?.path ? "checkmark" : (entry.isComplete ? "cube" : "exclamationmark.triangle"))
                }
                .disabled(model.isRunning && entry.path != model.modelRoot?.path)
            }
            Divider()
            Button("打开模型库…", systemImage: "cube.box") { openWindow(id: "models") }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "cube.fill").foregroundStyle(.tint)
                Text(model.modelName).fontWeight(.medium)
                StatusPill(text: model.modelStatus.label, color: statusColor, busy: model.modelStatus == .validating)
            }
        }
        .menuIndicator(.visible)
        .fixedSize()
        .help(model.isRunning ? "任务进行中，完成后才能切换模型" : "切换模型")
    }

    private var statusColor: Color {
        switch model.modelStatus {
        case .ready: .green
        case .inUse: .blue
        case .validating: .orange
        case .error: .red
        case .unselected: .secondary
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
            Text("包含").font(.headline)
            ForEach(GenerationPreset.Part.allCases) { part in
                Toggle(part.label, isOn: Binding(
                    get: { includes.contains(part) },
                    set: { if $0 { includes.insert(part) } else { includes.remove(part) } }))
            }
            Note("始终包含：模式、尺寸、步数、Guidance、种子设置、调度与强度。模型或 LoRA 路径失效时，加载会提示，不会悄悄替换。")
            HStack {
                Spacer()
                Button("取消") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("保存") {
                    model.savePreset(named: name.isEmpty ? "预设 \(model.presets.count + 1)" : name, includes: includes)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: 400)
    }
}

/// Left 2/3: model, prompt, mode options, parameters and performance, with the run controls pinned at the bottom.
struct ParameterPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: Design.gutter) {
                    ModelCard()
                    PromptCard()
                    if model.mode != .textToImage { ModeOptionsCard() }
                    ParametersCard()
                    PerformanceCard()
                    AdvancedCard()
                }
                .padding(Design.panelPadding)
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .scrollIndicators(.automatic)
            RunBar()
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}
