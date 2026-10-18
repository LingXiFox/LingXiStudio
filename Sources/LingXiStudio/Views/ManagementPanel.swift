import QwenImageCore
import SwiftUI
import UniformTypeIdentifiers

/// Right bottom: the always-visible mode switcher, then LoRA / history / task tabs.
struct ManagementPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 0) {
            ModeSwitcher().padding(.horizontal, 10).padding(.top, 10).padding(.bottom, 8)
            HStack(spacing: 6) {
                TabButton(title: "LoRA", symbol: "square.stack.3d.up.fill", tab: .lora, badge: model.activeLoRAs.count)
                TabButton(title: "历史", symbol: "clock.fill", tab: .history, badge: 0)
                TabButton(
                    title: "任务", symbol: "list.bullet.rectangle.fill", tab: .tasks,
                    badge: model.waitingCount + (model.isRunning ? 1 : 0), alert: model.failedCount > 0)
                Spacer()
            }
            .padding(.horizontal, 10).padding(.bottom, 6)
            Divider()
            switch model.managementTab {
            case .lora: LoRAList()
            case .history: HistoryGrid()
            case .tasks: TaskList()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct TabButton: View {
    @Environment(AppModel.self) private var model
    let title: String
    let symbol: String
    let tab: AppModel.ManagementTab
    let badge: Int
    var alert = false

    var body: some View {
        let selected = model.managementTab == tab
        Button {
            model.managementTab = tab
        } label: {
            HStack(spacing: 5) {
                Image(systemName: symbol)
                Text(title)
                Badge(count: badge)
                if alert { Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.red) }
            }
            .font(.callout.weight(selected ? .semibold : .regular))
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(selected ? Color.accentColor.opacity(0.15) : .clear, in: Capsule())
            .foregroundStyle(selected ? Color.accentColor : .primary)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

struct ModeSwitcher: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 4) {
            ForEach(CreationMode.allCases) { mode in
                let selected = model.mode == mode
                Button {
                    withAnimation(.snappy(duration: 0.2)) { model.mode = mode }
                } label: {
                    VStack(spacing: 3) {
                        Image(systemName: mode.symbol).font(.system(size: 15, weight: .medium))
                        Text(mode.title).font(.caption2.weight(selected ? .semibold : .regular)).lineLimit(1)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
                    .background(selected ? Color.accentColor.opacity(0.16) : Color.primary.opacity(0.04),
                                in: RoundedRectangle(cornerRadius: Design.innerCorner, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: Design.innerCorner, style: .continuous)
                        .strokeBorder(selected ? Color.accentColor.opacity(0.7) : .clear))
                    .foregroundStyle(selected ? Color.accentColor : .primary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(mode.methodNote)
            }
        }
    }
}

// MARK: - LoRA

struct LoRAList: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @State private var importing = false
    @State private var dropTargeted = false

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Text("启用 \(model.activeLoRAs.count) / \(model.loras.count)").font(.caption).foregroundStyle(.secondary)
                Spacer()
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
                } label: {
                    Image(systemName: "square.stack")
                }
                .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                .help("LoRA 组合预设")
                Button("全部停用", systemImage: "pause.circle") {
                    for i in model.loras.indices { model.loras[i].enabled = false }
                }
                .disabled(model.activeLoRAs.isEmpty)
                Button("清空列表", systemImage: "trash") { model.loras.removeAll() }.disabled(model.loras.isEmpty)
            }
            .labelStyle(.iconOnly).buttonStyle(.borderless)
            .padding(.horizontal, 12).padding(.vertical, 6)

            if model.loras.isEmpty {
                ContentUnavailableView {
                    Label("没有 LoRA", systemImage: "square.stack.3d.up")
                } description: {
                    Text("点「+」从 LoRA 文件夹添加，或把 .safetensors 拖到这里。可同时启用多个，各自调整强度。")
                }
            } else {
                List {
                    ForEach($model.loras) { $item in LoRARow(item: $item) }
                        .onMove { model.loras.move(fromOffsets: $0, toOffset: $1) }
                }
                .listStyle(.inset)
                .scrollContentBackground(.hidden)
            }
            Note("修改在下一个加入队列的任务生效；已排队任务保留各自的组合。按列表顺序叠加，作品记录保存完整组合。", symbol: "info.circle")
                .padding(.horizontal, 12).padding(.vertical, 6)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxHeight: .infinity, alignment: .top)
        .overlay {
            if dropTargeted {
                RoundedRectangle(cornerRadius: Design.corner).strokeBorder(Color.accentColor, lineWidth: 2).padding(4)
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            model.importLoRAs(urls)
            return true
        } isTargeted: { dropTargeted = $0 }
        .fileImporter(
            isPresented: $importing, allowedContentTypes: [UTType(filenameExtension: "safetensors") ?? .data],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result { model.importLoRAs(urls) }
        }
    }
}

struct LoRARow: View {
    @Environment(AppModel.self) private var model
    @Binding var item: LoRAItem

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Toggle("", isOn: $item.enabled).labelsHidden().toggleStyle(.switch).controlSize(.mini)
                    .disabled(!item.isUsable)
                IconTile(symbol: item.purpose.symbol, tint: tint, size: 20)
                VStack(alignment: .leading, spacing: 1) {
                    Text(item.name).font(.callout.weight(.medium)).lineLimit(1).truncationMode(.middle).help(item.path)
                    if item.problems.isEmpty {
                        Text(info).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                Spacer()
                Menu(item.purpose.label) {
                    ForEach(LoRAItem.Purpose.allCases, id: \.self) { purpose in
                        Button(purpose.label) { item.purpose = purpose }
                    }
                }
                .menuStyle(.borderlessButton).fixedSize()
                .font(.caption)
                Button("移除", systemImage: "xmark") { model.loras.removeAll { $0.id == item.id } }
                    .labelStyle(.iconOnly).buttonStyle(.borderless).foregroundStyle(.secondary)
            }
            if item.enabled && item.isUsable {
                HStack(spacing: 8) {
                    Slider(value: $item.scale, in: -1...2, step: 0.05)
                    TextField("", value: $item.scale, format: .number.precision(.fractionLength(2)))
                        .frame(width: 48).textFieldStyle(.roundedBorder).font(.caption.monospacedDigit())
                }
            }
            ForEach(item.problems, id: \.self) { Note($0, tint: .red, symbol: "exclamationmark.octagon") }
            if !item.triggerWords.isEmpty {
                HStack(spacing: 4) {
                    Image(systemName: "quote.opening").font(.caption2)
                    Text(item.triggerWords).font(.caption2).lineLimit(1)
                    Button("插入") {
                        if model.mode.usesInstruction { model.parameters.instruction += " " + item.triggerWords }
                        else { model.parameters.prompt += " " + item.triggerWords }
                    }
                    .controlSize(.mini)
                }
                .foregroundStyle(.secondary)
            }
            if item.purpose == .acceleration && item.enabled && model.parameters.preset != .turbo {
                Note("加速 LoRA 已启用，但调度仍是标准；需要时在左侧选 Turbo", tint: .orange, symbol: "exclamationmark.triangle")
            }
        }
        .padding(.vertical, 4)
        .opacity(item.enabled ? 1 : 0.65)
    }

    private var tint: Color {
        switch item.purpose {
        case .acceleration: .orange
        case .style: .pink
        case .character: .blue
        case .other: .gray
        }
    }

    private var info: String {
        var parts = ["rank \(item.rank)", "\(item.targetCount) 层", Format.bytes(item.fileBytes)]
        if !item.coverage.isEmpty { parts.append(item.coverage.joined(separator: "/")) }
        return parts.joined(separator: " · ")
    }
}

// MARK: - History

struct HistoryGrid: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @State private var text = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("提示词或种子", text: $text).textFieldStyle(.plain)
                Button("刷新", systemImage: "arrow.clockwise") { model.reloadGallery() }
                Button("图库详细视图", systemImage: "rectangle.grid.3x2") { openWindow(id: "gallery") }
            }
            .labelStyle(.iconOnly).buttonStyle(.borderless)
            .padding(.horizontal, 12).padding(.vertical, 7)
            if items.isEmpty {
                ContentUnavailableView("没有作品", systemImage: "photo.on.rectangle", description: Text(model.outputDirectory.path))
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 84), spacing: 6)], spacing: 6) {
                        ForEach(items) { item in
                            Thumbnail(url: item.url, size: 84)
                                .overlay(
                                    RoundedRectangle(cornerRadius: Design.innerCorner)
                                        .strokeBorder(item.url == model.selection ? Color.accentColor : .clear, lineWidth: 2.5))
                                .overlay(alignment: .topTrailing) {
                                    if model.favorites.contains(item.url.path) {
                                        Image(systemName: "star.fill").font(.caption2).foregroundStyle(.yellow).padding(4)
                                    }
                                }
                                .onTapGesture(count: 2) { openWindow(id: "viewer", value: item.url) }
                                .onTapGesture {
                                    model.selection = item.url
                                    model.workspaceTab = .preview
                                }
                                .contextMenu { ImageActionsMenu(item: item) }
                                .help(item.prompt ?? "未记录参数")
                        }
                    }
                    .padding(10)
                }
            }
        }
    }

    private var items: [Artwork] {
        let q = text.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return model.artworks }
        return model.artworks.filter {
            ($0.prompt ?? "").lowercased().contains(q) || $0.seed.map(String.init) == q
        }
    }
}

// MARK: - Tasks

struct TaskList: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("一次只运行一个任务，其余排队").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("清空等待") { model.clearWaiting() }.disabled(model.waitingCount == 0)
                Button("清除已结束") { model.clearFinished() }.disabled(!model.tasks.contains { $0.status.isFinished })
            }
            .controlSize(.small)
            .padding(.horizontal, 12).padding(.vertical, 6)
            if model.tasks.isEmpty {
                ContentUnavailableView("没有任务", systemImage: "list.bullet.rectangle", description: Text("按 ⌘↩ 生成后，任务会出现在这里"))
            } else {
                List(model.tasks) { task in TaskRow(task: task) }
                    .listStyle(.inset)
                    .scrollContentBackground(.hidden)
            }
        }
    }
}

struct TaskRow: View {
    @Environment(AppModel.self) private var model
    let task: TaskRecord
    @State private var showLog = false

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            ZStack {
                if let first = task.outputs.last {
                    Thumbnail(url: URL(fileURLWithPath: first), size: 44, corner: 6)
                } else {
                    RoundedRectangle(cornerRadius: 6).fill(Color.primary.opacity(0.06)).frame(width: 44, height: 44)
                        .overlay { Image(systemName: task.mode.symbol).foregroundStyle(.secondary) }
                }
                if task.status == .running, let live = model.live, live.taskID == task.id {
                    ProgressRing(fraction: live.overallFraction, lineWidth: 3).frame(width: 26, height: 26)
                        .background(.regularMaterial, in: Circle())
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    statusIcon
                    Text(task.title).lineLimit(1).font(.callout.weight(.medium))
                    Spacer()
                    Text(task.createdAt, style: .time).font(.caption).foregroundStyle(.secondary)
                }
                Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                if task.status == .running, let live = model.live, live.taskID == task.id {
                    Text([live.stage == .denoising ? "\(live.stageLabel) \(live.step)/\(live.totalSteps)" : live.stageLabel,
                          live.secondsPerStep.map { String(format: "%.2f s/step", $0) },
                          live.remaining.map { "剩余约 \(Format.duration($0))" }].compactMap { $0 }.joined(separator: " · "))
                        .font(.caption.monospacedDigit()).foregroundStyle(.tint)
                }
                if case .failed(let message) = task.status {
                    Text(message).font(.caption).foregroundStyle(.red).lineLimit(3).textSelection(.enabled)
                }
                HStack(spacing: 8) {
                    Text(task.status.label).font(.caption.weight(.medium)).foregroundStyle(statusColor)
                    if let s = task.startedAt, let f = task.finishedAt {
                        Text(Format.duration(f.timeIntervalSince(s))).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if task.status == .running || task.status == .waiting {
                        Button("取消") { model.cancel(task.id) }
                    } else {
                        Button("保留参数重新生成") { model.retry(task.id) }
                    }
                    Button(showLog ? "收起" : "日志") { showLog.toggle() }
                }
                .controlSize(.small)
                if showLog {
                    Text(task.log.isEmpty ? "（无记录）" : task.log.joined(separator: "\n"))
                        .font(.caption.monospaced()).textSelection(.enabled)
                        .padding(6).frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var detail: String {
        let p = task.parameters
        var parts = [task.mode.title, task.modelName, "\(p.width)×\(p.height)", "\(p.steps) 步",
                     "Guidance \(String(format: "%.1f", p.guidance))"]
        if task.mode.hasStrength { parts.append("强度 \(String(format: "%.2f", p.strength(for: task.mode)))") }
        if !task.conditions.isEmpty { parts.append("\(task.conditions.count) 张输入图") }
        if p.count > 1 { parts.append("\(task.outputs.count)/\(p.count) 张") }
        if !task.seeds.isEmpty { parts.append("种子 " + task.seeds.map { String($0) }.joined(separator: ",")) }
        if !task.loras.isEmpty {
            parts.append(task.loras.map {
                "\(URL(fileURLWithPath: $0.path).deletingPathExtension().lastPathComponent)×\(String(format: "%.2f", $0.scale))"
            }.joined(separator: " + "))
        }
        return parts.joined(separator: " · ")
    }

    private var statusColor: Color {
        switch task.status {
        case .waiting: .secondary
        case .running: .blue
        case .done: .green
        case .failed: .red
        case .cancelled: .secondary
        case .interrupted: .orange
        }
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch task.status {
        case .waiting: Image(systemName: "clock").foregroundStyle(.secondary)
        case .running: ProgressView().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed: Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
        case .cancelled: Image(systemName: "minus.circle").foregroundStyle(.secondary)
        case .interrupted: Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
        }
    }
}
