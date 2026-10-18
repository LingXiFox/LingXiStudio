import QwenImageCore
import SwiftUI
import UniformTypeIdentifiers

/// Right bottom: lightweight LoRA / history / task tabs.
struct AssistantPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: Metrics.l) {
                TabButton(title: "LoRA", tab: .lora, badge: model.activeLoRAs.count)
                TabButton(title: "历史", tab: .history, badge: 0)
                TabButton(
                    title: "任务", tab: .tasks, badge: model.waitingCount + (model.isRunning ? 1 : 0),
                    alert: model.failedCount > 0)
                Spacer()
                switch model.managementTab {
                case .lora: LoRAActions()
                case .history: HistoryActions()
                case .tasks: TaskActions()
                }
            }
            .padding(.horizontal, Metrics.l)
            .frame(height: 38)
            .overlay(alignment: .bottom) { Rectangle().fill(Brand.hairline).frame(height: 1) }
            Group {
                switch model.managementTab {
                case .lora: LoRAList()
                case .history: HistoryGrid()
                case .tasks: TaskList()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Brand.side)
    }
}

/// Underline tab: text weight and a fox rule mark the selection.
struct TabButton: View {
    @Environment(AppModel.self) private var model
    let title: String
    let tab: AppModel.ManagementTab
    let badge: Int
    var alert = false

    var body: some View {
        let selected = model.managementTab == tab
        Button {
            model.managementTab = tab
        } label: {
            HStack(spacing: 5) {
                Text(title)
                Badge(count: badge, tint: selected ? Brand.foxText : Brand.textTertiary)
                if alert { Circle().fill(Brand.danger).frame(width: 6, height: 6) }
            }
            .font(.callout.weight(selected ? .semibold : .regular))
            .foregroundStyle(selected ? Brand.text : Brand.textSecondary)
            .frame(maxHeight: .infinity)
            .overlay(alignment: .bottom) {
                Capsule().fill(selected ? Brand.fox : .clear).frame(height: 2)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

// MARK: - LoRA

struct LoRAActions: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @State private var importing = false

    var body: some View {
        HStack(spacing: 2) {
            Menu {
                let files = model.availableLoRAFiles
                if files.isEmpty { Text("LoRA 文件夹中没有新的文件") }
                ForEach(files, id: \.self) { url in
                    Button(url.deletingPathExtension().lastPathComponent) { model.importLoRAs([url]) }
                }
                Divider()
                Button("从文件导入…") { importing = true }
                Button("LoRA 库…") { openWindow(id: "loras") }
            } label: {
                Image(systemName: "plus")
            }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
            .help("添加 LoRA")
            Menu {
                if model.loraPresets.isEmpty { Text("还没有组合预设") }
                ForEach(model.loraPresets) { preset in
                    Button("\(preset.name)（\(preset.entries.count)）") { model.applyLoRAPreset(preset) }
                }
                Divider()
                Button("保存当前组合…") {
                    if let name = ImageFileActions.ask("LoRA 组合名称") { model.saveLoRAPreset(named: name) }
                }
                .disabled(model.loras.isEmpty)
                Divider()
                Button("全部停用") { for i in model.loras.indices { model.loras[i].enabled = false } }
                    .disabled(model.activeLoRAs.isEmpty)
                Button("清空列表", role: .destructive) { model.loras.removeAll() }.disabled(model.loras.isEmpty)
            } label: {
                Image(systemName: "ellipsis")
            }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
            .padding(.leading, 8)
            .help("组合预设与批量操作")
        }
        .foregroundStyle(Brand.textSecondary)
        .fileImporter(
            isPresented: $importing, allowedContentTypes: [UTType(filenameExtension: "safetensors") ?? .data],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result { model.importLoRAs(urls) }
        }
    }
}

/// Layer-style list: visibility, name and purpose, inline scale.
struct LoRAList: View {
    @Environment(AppModel.self) private var model
    @State private var dropTargeted = false

    var body: some View {
        @Bindable var model = model
        Group {
            if model.loras.isEmpty {
                EmptyState(
                    symbol: "square.stack.3d.up", title: "没有 LoRA",
                    message: "点「+」从 LoRA 文件夹添加，或把 .safetensors 拖到这里", compact: true)
            } else {
                List {
                    ForEach($model.loras) { $item in
                        LoRARow(item: $item)
                            .listRowInsets(EdgeInsets(top: 0, leading: 8, bottom: 0, trailing: 8))
                            .listRowSeparator(.hidden)
                            .listRowBackground(Color.clear)
                    }
                    .onMove { model.loras.move(fromOffsets: $0, toOffset: $1) }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .safeAreaInset(edge: .bottom) {
                    Text("按列表顺序叠加 · 修改在下一个任务生效")
                        .font(TypeScale.caption).foregroundStyle(Brand.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, Metrics.l).padding(.vertical, 6)
                }
            }
        }
        .overlay {
            if dropTargeted {
                RoundedRectangle(cornerRadius: Metrics.radiusL).strokeBorder(Brand.fox, lineWidth: 2).padding(4)
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            model.importLoRAs(urls)
            return true
        } isTargeted: { dropTargeted = $0 }
    }
}

struct LoRARow: View {
    @Environment(AppModel.self) private var model
    @Binding var item: LoRAItem
    @State private var hovering = false

    var body: some View {
        let active = item.enabled && item.isUsable
        HStack(alignment: .top, spacing: 8) {
            Button {
                item.enabled.toggle()
            } label: {
                Image(systemName: item.enabled ? "eye.fill" : "eye.slash")
                    .foregroundStyle(item.enabled ? Brand.foxText : Brand.textTertiary)
                    .frame(width: 22, height: 22)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(!item.isUsable)
            .help(item.enabled ? "停用" : "启用")
            .accessibilityLabel("\(item.enabled ? "停用" : "启用") LoRA：\(item.name)")
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(item.name).font(.callout.weight(.medium)).lineLimit(1).truncationMode(.middle)
                    Spacer(minLength: 0)
                    actions
                }
                if let problem = item.problems.first {
                    Text(problem).font(.caption2).foregroundStyle(Brand.danger).lineLimit(1).help(item.problems.joined(separator: "\n"))
                }
                HStack(spacing: 6) {
                    Tag(text: item.purpose.label, tint: item.purpose == .acceleration ? Brand.foxText : Brand.textSecondary)
                    if item.purpose == .acceleration && item.enabled && model.parameters.preset != .turbo {
                        Image(systemName: "exclamationmark.triangle.fill").font(.caption2).foregroundStyle(Brand.warning)
                            .help("加速 LoRA 已启用，但调度仍是标准；需要时在左侧选 Turbo")
                    }
                    Spacer(minLength: 4)
                    Text("强度").font(TypeScale.caption).foregroundStyle(Brand.textSecondary)
                    Slider(value: $item.scale, in: -1...2, step: 0.05)
                        .controlSize(.mini)
                        .frame(minWidth: 70, maxWidth: 120)
                        .disabled(!active)
                        .accessibilityLabel("\(item.name) 强度")
                    TextField("强度", value: $item.scale, format: .number.precision(.fractionLength(2)))
                        .textFieldStyle(.plain)
                        .font(.caption.monospacedDigit())
                        .multilineTextAlignment(.trailing)
                        .frame(width: 38)
                        .disabled(!active)
                        .accessibilityLabel("\(item.name) 强度数值")
                }
            }
            .help("\(item.name)\n\(info)\n\(item.path)")
        }
        .padding(8)
        .background(
            hovering ? Brand.hover : (active ? Brand.foxSoft.opacity(0.45) : .clear),
            in: RoundedRectangle(cornerRadius: Metrics.radiusM, style: .continuous))
        .opacity(item.enabled ? 1 : 0.62)
        .onHover { hovering = $0 }
    }

    private var actions: some View {
        Menu {
            Menu("用途") {
                ForEach(LoRAItem.Purpose.allCases, id: \.self) { purpose in
                    Button(purpose.label) { item.purpose = purpose }
                }
            }
            if !item.triggerWords.isEmpty {
                Button("插入触发词「\(item.triggerWords)」") {
                    if model.mode.usesInstruction { model.parameters.instruction += " " + item.triggerWords }
                    else { model.parameters.prompt += " " + item.triggerWords }
                }
            }
            Button("强度复位为 1.0") { item.scale = 1 }
            Divider()
            Section("文件信息") { Text(info) }
            Button("移除", role: .destructive) { model.loras.removeAll { $0.id == item.id } }
        } label: {
            Image(systemName: "ellipsis")
        }
        .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
        .foregroundStyle(Brand.textSecondary)
        .accessibilityLabel("\(item.name) 操作")
    }

    private var info: String {
        var parts = ["rank \(item.rank)", "\(item.targetCount) 层", Format.bytes(item.fileBytes)]
        if !item.coverage.isEmpty { parts.append(item.coverage.joined(separator: "/")) }
        return parts.joined(separator: " · ")
    }
}

// MARK: - History

struct HistoryActions: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        HStack(spacing: 2) {
            Button("刷新", systemImage: "arrow.clockwise") { model.reloadGallery() }.help("重新扫描输出目录")
            Button("图库", systemImage: "rectangle.grid.3x2") { openWindow(id: "gallery") }.help("打开图库详细视图")
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.ghost)
    }
}

struct HistoryGrid: View {
    enum Scope: Hashable { case all, favorites, mode }

    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @State private var text = ""
    @State private var scope: Scope = .all

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                HStack(spacing: 5) {
                    Image(systemName: "magnifyingglass").foregroundStyle(Brand.textTertiary).imageScale(.small)
                    TextField("提示词或种子", text: $text).textFieldStyle(.plain)
                }
                .padding(.horizontal, 8)
                .frame(height: 26)
                .background(Brand.surface, in: RoundedRectangle(cornerRadius: Metrics.radiusS, style: .continuous))
                BrandSegmented(
                    selection: $scope,
                    options: [
                        .init(value: .all, title: "全部"),
                        .init(value: .favorites, title: "收藏"),
                        .init(value: .mode, title: model.mode.title, help: "只看当前模式"),
                    ])
                    .fixedSize()
            }
            .padding(.horizontal, Metrics.m).padding(.vertical, Metrics.s)
            if items.isEmpty {
                EmptyState(
                    symbol: "photo.on.rectangle", title: model.artworks.isEmpty ? "还没有作品" : "没有匹配的作品",
                    message: model.artworks.isEmpty ? model.outputDirectory.path : nil, compact: true)
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 88), spacing: 8)], spacing: 8) {
                        ForEach(items) { item in HistoryCell(item: item) }
                    }
                    .padding(.horizontal, Metrics.m).padding(.bottom, Metrics.m)
                }
            }
        }
    }

    private var items: [Artwork] {
        let q = text.trimmingCharacters(in: .whitespaces).lowercased()
        return model.artworks.filter { item in
            switch scope {
            case .all: break
            case .favorites: if !model.favorites.contains(item.url.path) { return false }
            case .mode: if item.mode != model.mode { return false }
            }
            guard !q.isEmpty else { return true }
            return (item.prompt ?? "").lowercased().contains(q) || item.seed.map(String.init) == q
        }
    }
}

struct HistoryCell: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    let item: Artwork
    @State private var hovering = false

    var body: some View {
        let selected = item.url == model.selection
        Button {
            model.selection = item.url
            model.workspaceTab = .preview
        } label: {
            Thumbnail(url: item.url, size: 88)
            .overlay(
                RoundedRectangle(cornerRadius: Design.innerCorner, style: .continuous)
                    .strokeBorder(selected ? Brand.fox : .clear, lineWidth: 2.5))
            .overlay(alignment: .topTrailing) {
                if model.favorites.contains(item.url.path) {
                    Image(systemName: "star.fill").font(.caption2).foregroundStyle(Brand.fox)
                        .padding(4).background(.black.opacity(0.35), in: Circle()).padding(4)
                }
            }
        }
            .buttonStyle(.plain)
            .accessibilityLabel(item.prompt ?? item.url.lastPathComponent)
            .accessibilityAddTraits(selected ? .isSelected : [])
            .overlay(alignment: .bottom) {
                if hovering && item.isReusable {
                    Button {
                        model.reuse(item)
                    } label: {
                        Label("复用", systemImage: "arrow.uturn.backward").font(.caption2.weight(.semibold))
                            .padding(.horizontal, 7).padding(.vertical, 3)
                            .background(.black.opacity(0.55), in: Capsule())
                            .foregroundStyle(.white)
                    }
                    .buttonStyle(.plain)
                    .padding(5)
                    .help("复用全部参数")
                }
            }
            .onHover { hovering = $0 }
            .simultaneousGesture(TapGesture(count: 2).onEnded { openWindow(id: "viewer", value: item.url) })
            .contextMenu { ImageActionsMenu(item: item) }
            .help(item.prompt ?? "未记录参数")
    }
}

// MARK: - Tasks

struct TaskActions: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 2) {
            Button("清空等待") { model.clearWaiting() }.disabled(model.waitingCount == 0)
            Button("清除已结束") { model.clearFinished() }.disabled(!model.tasks.contains { $0.status.isFinished })
        }
        .buttonStyle(.ghost)
        .font(.caption)
    }
}

struct TaskList: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if model.tasks.isEmpty {
            EmptyState(symbol: "list.bullet.rectangle", title: "没有任务", message: "按 ⌘↩ 生成后，任务会出现在这里", compact: true)
        } else {
            List(model.tasks) { task in
                TaskRow(task: task)
                    .listRowInsets(EdgeInsets(top: 4, leading: 12, bottom: 4, trailing: 12))
                    .listRowSeparatorTint(Brand.hairline)
                    .listRowBackground(Color.clear)
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
        }
    }
}

struct TaskRow: View {
    @Environment(AppModel.self) private var model
    let task: TaskRecord
    @State private var showLog = false

    var body: some View {
        let live = model.live.flatMap { $0.taskID == task.id ? $0 : nil }
        HStack(alignment: .top, spacing: 10) {
            ZStack {
                if let first = task.outputs.last {
                    Thumbnail(url: URL(fileURLWithPath: first), size: 40, corner: 6)
                } else {
                    RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Brand.surface).frame(width: 40, height: 40)
                        .overlay { Image(systemName: task.mode.symbol).foregroundStyle(Brand.textTertiary) }
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    statusIcon
                    Text(task.title).lineLimit(1).font(.callout.weight(.medium))
                    Spacer(minLength: 4)
                    Text(task.createdAt, style: .time).font(.caption).foregroundStyle(Brand.textTertiary)
                }
                Text(detail).font(.caption).foregroundStyle(Brand.textSecondary).lineLimit(1).help(detail)
                if let live {
                    HStack(spacing: 8) {
                        ProgressBar(fraction: live.overallFraction)
                        Text(live.stage == .denoising ? "\(live.step)/\(live.totalSteps)" : live.stageLabel)
                            .font(.caption.monospacedDigit()).foregroundStyle(Brand.foxText).fixedSize()
                    }
                }
                if case .failed(let message) = task.status {
                    Text(message).font(.caption).foregroundStyle(Brand.danger).lineLimit(3).textSelection(.enabled)
                }
                HStack(spacing: 6) {
                    Text(task.status.label).font(.caption.weight(.medium)).foregroundStyle(statusColor)
                    if let s = task.startedAt, let f = task.finishedAt {
                        Text(Format.duration(f.timeIntervalSince(s))).font(.caption).foregroundStyle(Brand.textTertiary)
                    }
                    Spacer()
                    Group {
                        if task.status == .running || task.status == .waiting {
                            Button("取消") { model.cancel(task.id) }
                        } else {
                            Button("重新生成") { model.retry(task.id) }.help("保留参数重新生成")
                        }
                        Button(showLog ? "收起日志" : "日志") { showLog.toggle() }
                    }
                    .buttonStyle(.ghost)
                    .font(.caption)
                }
                if showLog {
                    Text(task.log.isEmpty ? "（无记录）" : task.log.joined(separator: "\n"))
                        .font(.caption.monospaced()).textSelection(.enabled)
                        .padding(8).frame(maxWidth: .infinity, alignment: .leading)
                        .background(Brand.surface, in: RoundedRectangle(cornerRadius: Metrics.radiusS, style: .continuous))
                }
            }
        }
        .padding(.vertical, 2)
    }

    private var detail: String {
        let p = task.parameters
        var parts = [task.mode.title, "\(p.width)×\(p.height)", "\(p.steps) 步", "Guidance \(String(format: "%.1f", p.guidance))"]
        if task.mode.hasStrength { parts.append("强度 \(String(format: "%.2f", p.strength(for: task.mode)))") }
        if !task.conditions.isEmpty { parts.append("\(task.conditions.count) 张输入图") }
        if p.count > 1 { parts.append("\(task.outputs.count)/\(p.count) 张") }
        if !task.seeds.isEmpty { parts.append("种子 " + task.seeds.map { String($0) }.joined(separator: ",")) }
        if !task.loras.isEmpty {
            parts.append(task.loras.map {
                "\(URL(fileURLWithPath: $0.path).deletingPathExtension().lastPathComponent)×\(String(format: "%.2f", $0.scale))"
            }.joined(separator: " + "))
        }
        parts.append(task.modelName)
        return parts.joined(separator: " · ")
    }

    private var statusColor: Color {
        switch task.status {
        case .waiting, .cancelled: Brand.textSecondary
        case .running: Brand.foxText
        case .done: Brand.success
        case .failed: Brand.danger
        case .interrupted: Brand.warning
        }
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch task.status {
        case .waiting: Image(systemName: "clock").foregroundStyle(Brand.textTertiary)
        case .running: ProgressView().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(Brand.success)
        case .failed: Image(systemName: "xmark.octagon.fill").foregroundStyle(Brand.danger)
        case .cancelled: Image(systemName: "minus.circle").foregroundStyle(Brand.textTertiary)
        case .interrupted: Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Brand.warning)
        }
    }
}
