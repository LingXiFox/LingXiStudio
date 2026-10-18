import QwenImageCore
import SwiftUI
import UniformTypeIdentifiers

/// Right bottom: the always-visible mode switcher, then LoRA / history / task tabs.
struct ManagementPanel: View {
    enum Tab: Hashable { case lora, history, tasks }

    @Environment(AppModel.self) private var model
    @State private var tab: Tab = .lora

    var body: some View {
        VStack(spacing: 0) {
            ModeSwitcher().padding(8)
            Divider()
            Picker("", selection: $tab) {
                Text("LoRA（\(model.activeLoRAs.count)）").tag(Tab.lora)
                Text("历史").tag(Tab.history)
                Text(taskTabTitle).tag(Tab.tasks)
            }
            .pickerStyle(.segmented).labelsHidden()
            .padding(8)
            switch tab {
            case .lora: LoRAList()
            case .history: HistoryGrid()
            case .tasks: TaskList()
            }
        }
    }

    private var taskTabTitle: String {
        var title = "任务"
        if model.isRunning || model.waitingCount > 0 { title += "（\(model.waitingCount + (model.isRunning ? 1 : 0))）" }
        if model.failedCount > 0 { title += " ⚠︎" }
        return title
    }
}

struct ModeSwitcher: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 4) {
            ForEach(CreationMode.allCases) { mode in
                let selected = model.mode == mode
                Button {
                    model.mode = mode
                } label: {
                    Label(mode.title, systemImage: mode.symbol)
                        .labelStyle(.titleAndIcon)
                        .font(.callout)
                        .lineLimit(1)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(selected ? .accentColor : nil)
                .disabled(mode.unavailableReason != nil)
                .help(mode.unavailableReason ?? mode.title)
            }
        }
        .controlSize(.small)
    }
}

// MARK: - LoRA

struct LoRAList: View {
    @Environment(AppModel.self) private var model
    @State private var importing = false
    @State private var dropTargeted = false

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 0) {
            HStack {
                Text("启用 \(model.activeLoRAs.count) / \(model.loras.count)").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("导入…", systemImage: "plus") { importing = true }
                Button("全部停用", systemImage: "pause.circle") {
                    for i in model.loras.indices { model.loras[i].enabled = false }
                }
                .disabled(model.activeLoRAs.isEmpty)
                Button("清空列表", systemImage: "trash") { model.loras.removeAll() }
                    .disabled(model.loras.isEmpty)
            }
            .labelStyle(.iconOnly).buttonStyle(.borderless)
            .padding(.horizontal, 10)

            if model.loras.isEmpty {
                ContentUnavailableView(
                    "没有 LoRA", systemImage: "square.stack.3d.up",
                    description: Text("点「+」或把 .safetensors 拖到这里。可同时启用多个，各自调整强度。"))
            } else {
                List {
                    ForEach($model.loras) { $item in
                        LoRARow(item: $item)
                    }
                    .onMove { model.loras.move(fromOffsets: $0, toOffset: $1) }
                }
                .listStyle(.inset)
            }
            Text("改动在下一个加入队列的任务生效；已排队任务保留各自的 LoRA 组合。按列表顺序叠加。")
                .font(.caption2).foregroundStyle(.secondary)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .overlay {
            if dropTargeted { RoundedRectangle(cornerRadius: 8).stroke(Color.accentColor, lineWidth: 2).padding(4) }
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
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Toggle("", isOn: $item.enabled).labelsHidden().toggleStyle(.checkbox)
                    .disabled(!item.isUsable)
                Text(item.name).lineLimit(1).truncationMode(.middle).help(item.path)
                Menu(item.purpose.label) {
                    ForEach(LoRAItem.Purpose.allCases, id: \.self) { purpose in
                        Button(purpose.label) { item.purpose = purpose }
                    }
                }
                .menuStyle(.borderlessButton).fixedSize()
                .font(.caption)
                .padding(.horizontal, 4)
                .background(Color.accentColor.opacity(0.12), in: Capsule())
                Spacer()
                Button("移除", systemImage: "xmark") { model.loras.removeAll { $0.id == item.id } }
                    .labelStyle(.iconOnly).buttonStyle(.borderless)
            }
            HStack {
                Text("强度").font(.caption).foregroundStyle(.secondary)
                Slider(value: $item.scale, in: -1...2, step: 0.05)
                TextField("", value: $item.scale, format: .number.precision(.fractionLength(2))).frame(width: 52)
            }
            .disabled(!item.enabled || !item.isUsable)
            if item.problems.isEmpty {
                Text(info).font(.caption2).foregroundStyle(.secondary)
            } else {
                ForEach(item.problems, id: \.self) { Text($0).font(.caption2).foregroundStyle(.red) }
            }
            if item.purpose == .acceleration && item.enabled && model.parameters.preset != .turbo {
                Text("加速 LoRA 已启用，但调度仍是标准 40 步；需要时在左侧选 Turbo").font(.caption2).foregroundStyle(.orange)
            }
        }
        .padding(.vertical, 2)
        .opacity(item.enabled ? 1 : 0.6)
    }

    private var info: String {
        var parts = ["rank \(item.rank)", "\(item.targetCount) 层", Format.bytes(item.fileBytes)]
        if !item.coverage.isEmpty { parts.append(item.coverage.joined(separator: "/")) }
        parts.append("适配 Qwen Image 2.1")
        return parts.joined(separator: " · ")
    }
}

// MARK: - History

struct HistoryGrid: View {
    @Environment(AppModel.self) private var model
    @State private var filter = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                TextField("按提示词或种子筛选", text: $filter).textFieldStyle(.roundedBorder)
                Button("刷新", systemImage: "arrow.clockwise") { model.reloadHistory() }
                    .labelStyle(.iconOnly).buttonStyle(.borderless)
                Button("在 Finder 中打开输出目录", systemImage: "folder") {
                    NSWorkspace.shared.open(model.outputDirectory)
                }
                .labelStyle(.iconOnly).buttonStyle(.borderless)
            }
            .padding(.horizontal, 10).padding(.bottom, 6)
            if items.isEmpty {
                ContentUnavailableView("没有作品", systemImage: "photo.on.rectangle", description: Text(model.outputDirectory.path))
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 88), spacing: 6)], spacing: 6) {
                        ForEach(items) { item in
                            Thumbnail(url: item.url, size: 88)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 6)
                                        .stroke(item.url == model.selection ? Color.accentColor : .clear, lineWidth: 2))
                                .onTapGesture { model.selection = item.url }
                                .onTapGesture(count: 2) { model.reuse(item) }
                                .contextMenu { ImageActionsMenu(item: item) }
                                .help(item.metadata?.request.prompt ?? "未记录参数")
                        }
                    }
                    .padding(8)
                }
            }
        }
    }

    private var items: [HistoryItem] {
        let query = filter.trimmingCharacters(in: .whitespaces).lowercased()
        guard !query.isEmpty else { return model.history }
        return model.history.filter { item in
            guard let r = item.metadata?.request else { return false }
            return r.prompt.lowercased().contains(query) || String(r.seed) == query
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
                Button("清除已结束") { model.clearFinished() }
                    .disabled(!model.tasks.contains { $0.status.isFinished })
            }
            .controlSize(.small)
            .padding(.horizontal, 10)
            if model.tasks.isEmpty {
                ContentUnavailableView("没有任务", systemImage: "list.bullet.rectangle")
            } else {
                List(model.tasks) { task in TaskRow(task: task) }
                    .listStyle(.inset)
            }
        }
    }
}

struct TaskRow: View {
    @Environment(AppModel.self) private var model
    let task: TaskRecord
    @State private var showLog = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                statusIcon
                Text(task.title).lineLimit(1)
                Spacer()
                Text(task.createdAt, style: .time).font(.caption).foregroundStyle(.secondary)
            }
            Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            if task.status == .running, let live = model.live, live.taskID == task.id {
                ProgressView(value: Double(live.step), total: Double(max(live.totalSteps, 1))) {
                    Text(live.stage == .denoising ? "\(live.stageLabel) \(live.step)/\(live.totalSteps)" : live.stageLabel)
                        .font(.caption)
                }
            }
            if case .failed(let message) = task.status {
                Text(message).font(.caption).foregroundStyle(.red).lineLimit(3).textSelection(.enabled)
            }
            HStack {
                Text(task.status.label).font(.caption.weight(.medium))
                Spacer()
                if task.status == .running || task.status == .waiting {
                    Button("取消") { model.cancel(task.id) }
                } else {
                    Button("保留参数重新生成") { model.retry(task.id) }
                }
                Button(showLog ? "收起日志" : "日志") { showLog.toggle() }
            }
            .controlSize(.small)
            if showLog {
                Text(task.log.isEmpty ? "（无记录）" : task.log.joined(separator: "\n"))
                    .font(.caption.monospaced()).textSelection(.enabled)
                    .padding(6).frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 4))
            }
        }
        .padding(.vertical, 4)
    }

    private var detail: String {
        let p = task.parameters
        var parts = [
            CreationMode.textToImage.title, task.modelName, "\(p.width)×\(p.height)", "\(p.steps) 步",
            "Guidance \(String(format: "%.1f", p.guidance))",
        ]
        if p.count > 1 { parts.append("\(task.outputs.count)/\(p.count) 张") }
        if !task.seeds.isEmpty { parts.append("种子 " + task.seeds.map { String($0) }.joined(separator: ",")) }
        if !task.loras.isEmpty {
            parts.append(task.loras.map { "\(URL(fileURLWithPath: $0.path).deletingPathExtension().lastPathComponent)×\(String(format: "%.2f", $0.scale))" }.joined(separator: " + "))
        }
        return parts.joined(separator: " · ")
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch task.status {
        case .waiting: Image(systemName: "clock").foregroundStyle(.secondary)
        case .running: ProgressView().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed: Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
        case .cancelled: Image(systemName: "minus.circle").foregroundStyle(.secondary)
        case .interrupted: Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
        }
    }
}
