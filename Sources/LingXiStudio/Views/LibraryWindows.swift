import QwenImageCore
import SwiftUI

/// Model library: known folders, scan, Hugging Face cache, validation, and the download manager.
struct ModelLibraryWindow: View {
    @Environment(AppModel.self) private var model
    @State private var tab = 0
    @State private var note: String?

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $tab) {
                Text("模型").tag(0)
                Text("下载").tag(1)
            }
            .pickerStyle(.segmented).fixedSize().padding(10)
            Divider()
            if tab == 0 { models } else { DownloadsView() }
        }
        .navigationTitle("模型库")
        .frame(minWidth: 720, minHeight: 460)
    }

    private var models: some View {
        VStack(spacing: 0) {
            HStack {
                Button("添加模型文件夹…", systemImage: "folder.badge.plus") {
                    if let url = ImageFileActions.chooseFolder(prompt: "添加") { model.addModelFolder(url) }
                }
                Button("扫描模型根目录…", systemImage: "magnifyingglass") {
                    if let url = ImageFileActions.chooseFolder(prompt: "扫描") {
                        let before = model.models.count
                        model.addModelFolder(url)
                        note = "找到 \(model.models.count - before) 个新模型"
                    }
                }
                Button("识别 Hugging Face 缓存", systemImage: "shippingbox") {
                    note = "从 Hugging Face 缓存识别到 \(model.scanHuggingFaceCache()) 个新模型"
                }
                Button("全部重新校验", systemImage: "checkmark.shield") { model.revalidateModels() }
                Spacer()
                if let note { Text(note).font(.caption).foregroundStyle(.secondary) }
            }
            .controlSize(.small).padding(10)
            List {
                ForEach(model.models.sorted { ($0.favorite ? 0 : 1, -($0.lastUsed?.timeIntervalSince1970 ?? 0)) < ($1.favorite ? 0 : 1, -($1.lastUsed?.timeIntervalSince1970 ?? 0)) }) { entry in
                    ModelRow(entry: entry)
                }
            }
            .listStyle(.inset(alternatesRowBackgrounds: true))
            Note("只识别 Qwen-Image-2.1 官方 diffusers 目录（transformer / text_encoder / vae，Safetensors）。GGUF 等格式未经适配验证，不会被当作可加载模型。", symbol: "info.circle")
                .padding(10).frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct ModelRow: View {
    @Environment(AppModel.self) private var model
    let entry: ModelEntry

    var body: some View {
        HStack(spacing: 12) {
            IconTile(symbol: "cube.fill", tint: entry.path == model.modelRoot?.path ? Brand.fox : Brand.textTertiary, size: 30)
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(entry.name).font(.headline)
                    if entry.path == model.modelRoot?.path { StatusPill(text: "当前", color: Brand.foxText) }
                    if entry.favorite { Image(systemName: "star.fill").foregroundStyle(.yellow) }
                }
                Text(entry.path).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                HStack(spacing: 10) {
                    Text(entry.source)
                    if entry.bytes > 0 { Text(Format.bytes(entry.bytes)) }
                    if let used = entry.lastUsed { Text("最近使用 \(used.formatted(date: .abbreviated, time: .shortened))") }
                }
                .font(.caption).foregroundStyle(.secondary)
                if !entry.problems.isEmpty {
                    Text(entry.problems.prefix(3).joined(separator: "；")).font(.caption).foregroundStyle(.red).lineLimit(2)
                }
            }
            Spacer()
            if entry.validatedAt == nil {
                StatusPill(text: "未校验", color: .secondary)
            } else {
                StatusPill(text: entry.isComplete ? "完整" : "不完整", color: entry.isComplete ? .green : .red)
            }
            Button("使用") { model.selectModel(entry) }
                .disabled(entry.path == model.modelRoot?.path || model.isRunning || !entry.isComplete)
            Menu {
                Button(entry.favorite ? "取消收藏" : "收藏") { update { $0.favorite.toggle() } }
                Button("重新校验") { update { $0 = ModelScanner.validate($0) } }
                Button("在 Finder 中显示") { NSWorkspace.shared.activateFileViewerSelecting([entry.url]) }
                Divider()
                Button("从列表移除（不删除文件）", role: .destructive) { model.models.removeAll { $0.path == entry.path } }
                    .disabled(entry.path == model.modelRoot?.path)
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
        }
        .padding(.vertical, 4)
    }

    private func update(_ change: (inout ModelEntry) -> Void) {
        guard let i = model.models.firstIndex(where: { $0.path == entry.path }) else { return }
        change(&model.models[i])
    }
}

/// Hugging Face downloads: list files, choose, download with resume / retry / verification.
struct DownloadsView: View {
    @Environment(AppModel.self) private var model
    @State private var repoInput = "Qwen/Qwen-Image-2.1"
    @State private var revision = "main"
    @State private var destination: URL = FileManager.default.homeDirectoryForCurrentUser.appending(path: "Models")
    @State private var files: [DownloadManager.RemoteFile] = []
    @State private var chosen: Set<String> = []
    @State private var listing = false
    @State private var error: String?

    var body: some View {
        let downloads = model.downloads
        VStack(alignment: .leading, spacing: 10) {
            if model.offline {
                Note("已开启离线模式：下载不可用。可在设置中关闭。", tint: .orange, symbol: "wifi.slash")
            }
            HStack {
                TextField("Hugging Face 仓库 ID 或链接，例如 Qwen/Qwen-Image-2.1", text: $repoInput).textFieldStyle(.roundedBorder)
                TextField("版本", text: $revision).textFieldStyle(.roundedBorder).frame(width: 80)
                Button("列出文件") { list() }.disabled(listing || model.offline)
                if listing { ProgressView().controlSize(.small) }
            }
            HStack {
                Text("保存到").foregroundStyle(.secondary)
                Text(targetFolder.path).lineLimit(1).truncationMode(.middle).font(.callout.monospaced())
                Button("更改…") { if let url = ImageFileActions.chooseFolder() { destination = url } }.controlSize(.small)
            }
            if let error { Note(error, tint: .red, symbol: "exclamationmark.octagon") }
            if !files.isEmpty {
                HStack {
                    Text("\(files.count) 个文件，选中 \(chosen.count) 个 · \(Format.bytes(files.filter { chosen.contains($0.path) }.reduce(0) { $0 + $1.size }))")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("全选") { chosen = Set(files.map(\.path)) }
                    Button("全不选") { chosen.removeAll() }
                    Button("开始下载") {
                        guard let repo = DownloadManager.repoID(from: repoInput) else { return }
                        downloads.start(repo: repo, revision: revision, files: files.filter { chosen.contains($0.path) }, destination: targetFolder)
                        files = []
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(chosen.isEmpty)
                }
                .controlSize(.small)
                List(files) { file in
                    Toggle(isOn: Binding(get: { chosen.contains(file.path) }, set: { if $0 { chosen.insert(file.path) } else { chosen.remove(file.path) } })) {
                        HStack {
                            Text(file.path).font(.callout.monospaced())
                            Spacer()
                            Text(Format.bytes(file.size)).foregroundStyle(.secondary).monospacedDigit()
                            if file.sha256 != nil { Image(systemName: "checkmark.seal").foregroundStyle(.secondary).help("下载后校验 SHA-256") }
                        }
                    }
                }
                .frame(minHeight: 140)
            }
            Divider()
            Text("下载任务").font(.headline)
            if downloads.jobs.isEmpty {
                Text("还没有下载任务。已存在且大小一致的文件会自动跳过；中断后重新开始会从断点续传。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            ScrollView {
                VStack(spacing: 10) {
                    ForEach(downloads.jobs) { job in DownloadJobRow(job: job) }
                }
            }
        }
        .padding(12)
    }

    private var targetFolder: URL {
        let name = DownloadManager.repoID(from: repoInput)?.split(separator: "/").last.map(String.init) ?? "model"
        return destination.lastPathComponent == name ? destination : destination.appending(path: name)
    }

    private func list() {
        guard let repo = DownloadManager.repoID(from: repoInput) else {
            error = "无法识别仓库 ID"
            return
        }
        listing = true
        error = nil
        Task {
            do {
                files = try await model.downloads.listFiles(repo: repo, revision: revision)
                chosen = Set(files.map(\.path))
            } catch {
                self.error = "\(error)"
            }
            listing = false
        }
    }
}

struct DownloadJobRow: View {
    @Environment(AppModel.self) private var model
    let job: DownloadManager.Job
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(job.repo).font(.callout.weight(.semibold))
                Text("→ \(job.destination.path)").font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                Spacer()
                if job.isActive {
                    Button("取消") { model.downloads.cancel(job.id) }
                } else if job.files.contains(where: { if case .failed = $0.state { true } else { $0.state == .cancelled } }) {
                    Button("重试") { model.downloads.retry(job.id) }
                }
                Button(expanded ? "收起" : "详情") { expanded.toggle() }
            }
            .controlSize(.small)
            ProgressView(value: Double(job.receivedBytes), total: Double(max(job.totalBytes, 1)))
            HStack {
                Text("\(Format.bytes(job.receivedBytes)) / \(Format.bytes(job.totalBytes))")
                let done = job.files.filter { $0.state == .done || $0.state == .skipped }.count
                Text("· 文件 \(done)/\(job.files.count)")
                if let error = job.error { Text("· \(error)").foregroundStyle(.red) }
                if !job.isActive && job.error == nil { Text("· 已结束").foregroundStyle(.secondary) }
            }
            .font(.caption).foregroundStyle(.secondary)
            if expanded {
                ForEach(job.files) { f in
                    HStack {
                        Text(f.file.path).font(.caption.monospaced()).lineLimit(1)
                        Spacer()
                        Text(f.state.label).font(.caption).foregroundStyle(color(f.state))
                    }
                }
            }
        }
        .padding(10)
        .background(Design.cardBackground, in: RoundedRectangle(cornerRadius: Design.innerCorner))
    }

    private func color(_ state: DownloadManager.FileState) -> Color {
        switch state {
        case .done, .skipped: .green
        case .failed: .red
        case .downloading, .verifying: .blue
        default: .secondary
        }
    }
}

/// LoRA library: everything in the LoRA folder plus imported files, with search, purpose, trigger words and combos.
struct LoRALibraryWindow: View {
    @Environment(AppModel.self) private var model
    @State private var search = ""
    @State private var purpose: LoRAItem.Purpose?
    @State private var repoInput = ""
    @State private var hfFiles: [DownloadManager.RemoteFile] = []
    @State private var hfError: String?

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                TextField("搜索名称", text: $search).textFieldStyle(.roundedBorder).frame(maxWidth: 240)
                Picker("用途", selection: $purpose) {
                    Text("全部用途").tag(LoRAItem.Purpose?.none)
                    ForEach(LoRAItem.Purpose.allCases, id: \.self) { Text($0.label).tag(LoRAItem.Purpose?.some($0)) }
                }
                .fixedSize()
                Spacer()
                Text("LoRA 文件夹：\(model.loraFolder.path)").font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                Button("更改…") { if let url = ImageFileActions.chooseFolder() { model.loraFolder = url } }.controlSize(.small)
            }
            List {
                Section("工作台列表（\(model.loras.count)）") {
                    ForEach($model.loras) { $item in
                        if matches(item) {
                            HStack(spacing: 10) {
                                Toggle("启用", isOn: $item.enabled).labelsHidden().toggleStyle(.checkbox)
                                    .disabled(!item.canToggle).accessibilityLabel("启用 LoRA：\(item.name)")
                                IconTile(symbol: item.purpose.symbol, size: 22)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(item.name).font(.callout.weight(.medium))
                                    Text("rank \(item.rank) · \(item.precision) · \(item.targetCount) 层 · \(Format.bytes(item.fileBytes)) · \(item.source)")
                                        .font(.caption).foregroundStyle(.secondary)
                                    if !item.problems.isEmpty { Text(item.problems.joined(separator: "；")).font(.caption).foregroundStyle(.red) }
                                }
                                Spacer()
                                TextField("触发词（可选）", text: $item.triggerWords).textFieldStyle(.roundedBorder).frame(width: 180)
                                Picker("", selection: $item.purpose) {
                                    ForEach(LoRAItem.Purpose.allCases, id: \.self) { Text($0.label).tag($0) }
                                }
                                .labelsHidden().fixedSize()
                                Button("重新检查") { item.refresh() }.controlSize(.small)
                            }
                        }
                    }
                }
                Section("LoRA 文件夹中未添加的文件") {
                    ForEach(model.availableLoRAFiles.filter { search.isEmpty || $0.lastPathComponent.localizedCaseInsensitiveContains(search) }, id: \.self) { url in
                        HStack {
                            Text(url.deletingPathExtension().lastPathComponent)
                            Spacer()
                            Button("添加") { model.importLoRAs([url]) }.controlSize(.small)
                        }
                    }
                }
                Section("组合预设（\(model.loraPresets.count)）") {
                    ForEach(model.loraPresets) { preset in
                        HStack {
                            Text(preset.name).font(.callout.weight(.medium))
                            Text(preset.entries.map { "\(URL(fileURLWithPath: $0.path).deletingPathExtension().lastPathComponent)×\(String(format: "%.2f", $0.scale))" }.joined(separator: " + "))
                                .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                            Spacer()
                            Button("应用") { model.applyLoRAPreset(preset) }.controlSize(.small)
                            Button("删除", role: .destructive) { model.loraPresets.removeAll { $0.id == preset.id } }.controlSize(.small)
                        }
                    }
                }
            }
            .listStyle(.inset)
            GroupBox("从 Hugging Face 导入") {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        TextField("仓库 ID 或链接", text: $repoInput).textFieldStyle(.roundedBorder)
                        Button("列出 LoRA 文件") { listHF() }.disabled(model.offline)
                    }
                    if let hfError { Note(hfError, tint: .red) }
                    ForEach(hfFiles) { file in
                        HStack {
                            Text(file.path).font(.caption.monospaced())
                            Spacer()
                            Text(Format.bytes(file.size)).font(.caption).foregroundStyle(.secondary)
                            Button("下载到 LoRA 文件夹") {
                                guard let repo = DownloadManager.repoID(from: repoInput) else { return }
                                model.downloads.start(repo: repo, files: [file], destination: model.loraFolder)
                            }
                            .controlSize(.small)
                        }
                    }
                    Note("下载进度在「模型库 → 下载」中查看；完成后从上方「未添加的文件」加入。导入前会检查是否适配 Qwen-Image-2.1。")
                }
                .padding(4)
            }
        }
        .padding(12)
        .navigationTitle("LoRA 库")
        .frame(minWidth: 760, minHeight: 520)
    }

    private func matches(_ item: LoRAItem) -> Bool {
        (search.isEmpty || item.name.localizedCaseInsensitiveContains(search)) && (purpose == nil || item.purpose == purpose)
    }

    private func listHF() {
        guard let repo = DownloadManager.repoID(from: repoInput) else {
            hfError = "无法识别仓库 ID"
            return
        }
        hfError = nil
        Task {
            do {
                hfFiles = try await model.downloads.listFiles(repo: repo).filter { $0.path.hasSuffix(".safetensors") }
                if hfFiles.isEmpty { hfError = "仓库中没有 .safetensors 文件" }
            } catch {
                hfError = "\(error)"
            }
        }
    }
}

/// Settings: global paths, export, runtime policy, appearance and diagnostics. Nothing here changes a single
/// generation; those parameters stay on the main window.
struct SettingsView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        TabView {
            GeneralSettings().tabItem { Label("通用", systemImage: "gearshape") }
            ResourceSettings().tabItem { Label("模型与资源", systemImage: "cube.box") }
            PerformanceSettings().tabItem { Label("性能", systemImage: "gauge.with.dots.needle.67percent") }
            AppearanceSettings().tabItem { Label("外观", systemImage: "circle.lefthalf.filled") }
            DiagnosticsView().tabItem { Label("高级", systemImage: "stethoscope") }
        }
        .frame(width: 640, height: 500)
        .brandWindow()
    }
}

/// A path row: the folder, a chooser and a Finder shortcut.
struct PathRow: View {
    let title: String
    let url: URL?
    let choose: (URL) -> Void

    var body: some View {
        LabeledContent(title) {
            HStack(spacing: 6) {
                Text(url?.path ?? "未选择").lineLimit(1).truncationMode(.middle).foregroundStyle(Brand.textSecondary)
                    .textSelection(.enabled)
                Button("选择…") { if let picked = ImageFileActions.chooseFolder() { choose(picked) } }
                if let url {
                    Button("在 Finder 中显示", systemImage: "folder") { NSWorkspace.shared.open(url) }
                        .labelStyle(.iconOnly).buttonStyle(.borderless)
                }
            }
        }
    }
}

struct GeneralSettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        Form {
            Section("保存") {
                PathRow(title: "输出目录", url: model.outputDirectory) { model.outputDirectory = $0 }
                LabeledContent("自动保存", value: "无损 PNG + 同名 JSON 记录 + 输入图副本")
                LabeledContent("命名", value: "日期-时间-种子，不覆盖已有文件")
            }
            Section("导出（另存为与批量导出）") {
                Picker("格式", selection: $model.exportSettings.format) {
                    ForEach(ExportSettings.Format.available) { Text($0.label).tag($0) }
                }
                if model.exportSettings.format.isLossy {
                    LabeledContent("质量") {
                        HStack {
                            Slider(value: $model.exportSettings.quality, in: 0.5...1)
                            Text("\(Int(model.exportSettings.quality * 100))").monospacedDigit().frame(width: 30)
                        }
                    }
                }
                TextField("文件名模板", text: $model.exportSettings.template)
                Text("可用：{date} {time} {seed} {mode} {width} {height} {index} {name}。本机 ImageIO 不能写 WebP，所以不提供。")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("同时导出参数记录（JSON）", isOn: $model.exportSettings.includeRecord)
            }
            Section("网络") {
                Toggle("离线模式（禁用所有下载）", isOn: $model.offline)
                Text("推理始终在本机离线进行，提示词和图片从不上传。下载只在你明确操作时发生。")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Brand.base)
    }
}

struct ResourceSettings: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        @Bindable var model = model
        Form {
            Section("模型") {
                PathRow(title: "当前模型", url: model.modelRoot) { url in
                    model.addModelFolder(url)
                    model.modelRoot = url
                }
                LabeledContent("状态", value: model.modelStatus.label)
                HStack {
                    Button("模型库与下载…") { openWindow(id: "models") }
                    Button("重新校验") { model.validateModel() }
                        .disabled(model.modelRoot == nil || model.modelStatus == .validating)
                }
            }
            Section("LoRA") {
                PathRow(title: "LoRA 文件夹", url: model.loraFolder) { model.loraFolder = $0 }
                Button("LoRA 库…") { openWindow(id: "loras") }
            }
            Section("图库") {
                ForEach(model.galleryFolders, id: \.self) { path in
                    LabeledContent(URL(fileURLWithPath: path).lastPathComponent) {
                        HStack {
                            Text(path).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                            Button("移除") { model.galleryFolders.removeAll { $0 == path } }
                        }
                    }
                }
                HStack {
                    Button("添加图库文件夹…") {
                        if let url = ImageFileActions.chooseFolder(prompt: "添加") { model.galleryFolders.append(url.path) }
                    }
                    if let legacy = model.legacyGalleryCandidate {
                        Button("导入旧版 WebUI 图库") { model.galleryFolders.append(legacy.path) }.help(legacy.path)
                    }
                }
                Text("额外文件夹只读取，不会移动或改写其中的文件。").font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Brand.base)
    }
}

/// Runtime policy. Only settings the runtime really honours are editable; process-level MLX options apply at the next
/// launch because MLX reads them once when it creates the Metal device.
struct PerformanceSettings: View {
    @AppStorage(RuntimeOverrides.opsKey) private var ops = MLXRuntime.commandBufferOps
    @AppStorage(RuntimeOverrides.mbKey) private var mb = MLXRuntime.commandBufferMB

    var body: some View {
        Form {
            Section("内存策略") {
                LabeledContent("档位", value: "32 GB 平衡（已验证）")
                LabeledContent("模型驻留", value: "按阶段加载，同一时刻只驻留一个大组件")
                LabeledContent("缓存", value: "每个阶段结束后同步并清空 MLX 缓存")
                Text("文本编码器（16 GB）与 Transformer（14 GB）无法同时常驻 32 GB 统一内存，所以任务之间不复用已加载的权重；任务结束后进程内存回到约 0.4 GB。其他档位需要量化或步骤缓存，尚未验证，不提供。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Metal 命令缓冲（下次启动生效）") {
                Stepper(value: $ops, in: 10...1000, step: 10) {
                    LabeledContent("每个缓冲最多算子", value: "\(ops)")
                }
                Stepper(value: $mb, in: 40...2000, step: 20) {
                    LabeledContent("每个缓冲最多 MB", value: "\(mb)")
                }
                LabeledContent("本次进程实际值", value: "\(RuntimeOverrides.current("MLX_MAX_OPS_PER_BUFFER")) op / \(RuntimeOverrides.current("MLX_MAX_MB_PER_BUFFER")) MB")
                HStack {
                    Button("恢复默认（\(MLXRuntime.commandBufferOps) / \(MLXRuntime.commandBufferMB)）") {
                        ops = MLXRuntime.commandBufferOps
                        mb = MLXRuntime.commandBufferMB
                    }
                    .disabled(ops == MLXRuntime.commandBufferOps && mb == MLXRuntime.commandBufferMB)
                    Button("重新启动 LingXiStudio") { RuntimeOverrides.relaunch() }
                        .disabled(!RuntimeOverrides.pendingRestart)
                }
                Text("MLX 只在首次创建 Metal 设备时读取这两个值。默认值是本机实测最快的组合；只影响提交批次，不改变计算结果。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("计算精度（固定）") {
                LabeledContent("Transformer / 文本编码器", value: "BF16 原始权重")
                LabeledContent("视觉编码器 / VAE", value: MLXRuntime.tf32 ? "TF32" : "真 FP32（TF32 关闭）")
                Text("精度属于已验收的官方语义对齐范围，GUI 不提供降精度或跳步加速。").font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Brand.base)
    }
}

/// User overrides of process-level MLX options, applied before `MLXRuntime.configure()` on launch.
enum RuntimeOverrides {
    static let opsKey = "runtime.commandBufferOps"
    static let mbKey = "runtime.commandBufferMB"

    /// Exports the stored overrides unless the environment already sets them; `MLXRuntime.configure()` fills the rest.
    static func apply() {
        let defaults = AppPaths.defaults
        if let ops = defaults.object(forKey: opsKey) as? Int { setenv("MLX_MAX_OPS_PER_BUFFER", String(ops), 0) }
        if let mb = defaults.object(forKey: mbKey) as? Int { setenv("MLX_MAX_MB_PER_BUFFER", String(mb), 0) }
    }

    static func current(_ name: String) -> String { getenv(name).map { String(cString: $0) } ?? "MLX 默认" }

    static var pendingRestart: Bool {
        let defaults = AppPaths.defaults
        let ops = defaults.object(forKey: opsKey) as? Int ?? MLXRuntime.commandBufferOps
        let mb = defaults.object(forKey: mbKey) as? Int ?? MLXRuntime.commandBufferMB
        return current("MLX_MAX_OPS_PER_BUFFER") != String(ops) || current("MLX_MAX_MB_PER_BUFFER") != String(mb)
    }

    @MainActor
    static func relaunch() {
        let config = NSWorkspace.OpenConfiguration()
        config.createsNewApplicationInstance = true
        NSWorkspace.shared.openApplication(at: Bundle.main.bundleURL, configuration: config) { _, _ in
            DispatchQueue.main.async { NSApp.terminate(nil) }
        }
    }
}

struct AppearanceSettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        Form {
            Section("主题") {
                Picker("外观", selection: $model.appearance) {
                    Text("跟随系统").tag(AppModel.Appearance.system)
                    Text("浅色 · 奶白").tag(AppModel.Appearance.light)
                    Text("深色 · 炭黑").tag(AppModel.Appearance.dark)
                }
                .pickerStyle(.inline)
            }
            Section("界面") {
                Button("恢复默认布局（左 2/3 · 右 1/3，右栏上下 1:1）") {
                    model.leftFraction = 2.0 / 3.0
                    model.rightTopFraction = 0.5
                }
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Brand.base)
    }
}

struct DiagnosticsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @State private var log = ""

    var body: some View {
        Form {
            Section("运行时") {
                LabeledContent("推理", value: "原生 Swift + MLX，完全离线")
                LabeledContent("系统", value: ProcessInfo.processInfo.operatingSystemVersionString)
                LabeledContent("统一内存", value: Format.bytes(model.physicalMemory))
                LabeledContent("应用数据", value: AppPaths.support.path)
                Button("打开 Runtime Inspector") { openWindow(id: "inspector") }
            }
            Section("今日日志") {
                ScrollView {
                    Text(log.isEmpty ? "（空）" : log).font(.caption.monospaced()).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: 140)
                HStack {
                    Button("刷新") { load() }
                    Button("在 Finder 中显示日志") { NSWorkspace.shared.activateFileViewerSelecting([AppLog.todayFile]) }
                    Button("复制诊断信息") {
                        ImageFileActions.copyText(
                            "LingXiStudio · \(ProcessInfo.processInfo.operatingSystemVersionString) · \(Format.bytes(model.physicalMemory))\n\(log)")
                    }
                }
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Brand.base)
        .onAppear(perform: load)
    }

    private func load() {
        log = (try? String(contentsOf: AppLog.todayFile, encoding: .utf8)).map { String($0.suffix(6000)) } ?? ""
    }
}

// MARK: - Runtime inspector

/// On-demand technical monitor: model components and residency, memory, the live stage and step timing.
struct RuntimeInspector: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let m = model.metrics
        let live = model.live
        Form {
            Section("模型") {
                LabeledContent("目录") {
                    Text(model.modelRoot?.path ?? "未选择").lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                }
                LabeledContent("状态", value: model.modelStatus.label)
                if case .error(let message) = model.modelStatus {
                    Text(message).font(.caption).foregroundStyle(Brand.danger).textSelection(.enabled)
                }
                ForEach(model.components, id: \.name) { c in
                    LabeledContent(Self.title(c.name)) {
                        HStack(spacing: 8) {
                            Text(c.bytes > 0 ? Format.bytes(c.bytes) : "配置").monospacedDigit().foregroundStyle(.secondary)
                            Text((model.residency[c.name] ?? .notLoaded).label)
                                .foregroundStyle(model.residency[c.name] == .resident ? Brand.foxText : .secondary)
                            Image(systemName: c.problems.isEmpty ? "checkmark.circle.fill" : "xmark.octagon.fill")
                                .foregroundStyle(c.problems.isEmpty ? Brand.success : Brand.danger)
                        }
                    }
                }
                HStack {
                    Button("重新校验") { model.validateModel() }
                        .disabled(model.modelRoot == nil || model.modelStatus == .validating)
                    Button("释放会话状态") { model.unloadModel() }.disabled(model.isRunning || model.modelRoot == nil)
                }
            }
            Section("内存") {
                MemoryBar(used: m.appFootprint, estimate: model.estimatedPeakBytes, total: model.physicalMemory)
                LabeledContent("应用占用", value: Format.bytes(m.appFootprint))
                LabeledContent("MLX 分配", value: live.map { "\(Format.bytes($0.mlxActive))（峰值 \(Format.bytes($0.mlxPeak))）" } ?? "空闲")
                LabeledContent("本任务预计峰值", value: Format.bytes(model.estimatedPeakBytes))
                LabeledContent("Swap", value: m.swapUsed.map { "\(Format.bytes($0)) / \(Format.bytes(m.swapTotal ?? 0))" } ?? "不可得")
                LabeledContent("可用内存", value: m.freePercent.map { "\($0)%" } ?? "不可得")
                LabeledContent("内存压力", value: m.pressure?.label ?? "不可得")
                LabeledContent("热状态", value: m.thermal.label)
            }
            Section("当前任务") {
                LabeledContent("阶段", value: live?.stageLabel ?? "空闲")
                LabeledContent("进度", value: live.map { l in
                    l.stage == .denoising ? "第 \(l.imageIndex + 1)/\(l.imageCount) 张 · 步 \(l.step)/\(l.totalSteps)" : "第 \(l.imageIndex + 1)/\(l.imageCount) 张"
                } ?? "–")
                LabeledContent("速度", value: live?.secondsPerStep.map { String(format: "%.2f s/step", $0) } ?? "–")
                LabeledContent("本张耗时", value: live.map { Format.duration($0.elapsed) } ?? "–")
                LabeledContent("预计剩余", value: live?.remaining.map { Format.duration($0) } ?? "–")
                LabeledContent("队列", value: "等待 \(model.waitingCount) · 失败 \(model.failedCount)")
            }
            Section("Runtime") {
                LabeledContent("Metal 命令缓冲", value: "\(RuntimeOverrides.current("MLX_MAX_OPS_PER_BUFFER")) op / \(RuntimeOverrides.current("MLX_MAX_MB_PER_BUFFER")) MB")
                LabeledContent("FP32", value: RuntimeOverrides.current("MLX_ENABLE_TF32") == "0" ? "真 FP32（TF32 关闭）" : "TF32")
                Text("GPU 利用率、专用显存等系统不提供的指标不显示；MLX 分配为运行时实际申请的统一内存。")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Brand.base)
        .tint(Brand.fox)
        .frame(minWidth: 460, minHeight: 520)
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
                    Capsule().fill(Brand.hairline)
                    Capsule().fill(color(estFrac).opacity(0.3)).frame(width: w * estFrac)
                    Capsule().fill(color(usedFrac)).frame(width: max(w * usedFrac, 4))
                    Rectangle().fill(Brand.danger.opacity(0.7)).frame(width: 1.5).offset(x: w * 0.85)
                }
            }
            .frame(height: 8)
            HStack {
                Text("当前 \(Format.bytes(used))")
                Spacer()
                Text("预计峰值 \(Format.bytes(estimate)) / \(Format.bytes(total))")
                    .foregroundStyle(Double(estimate) > Double(total) * 0.85 ? Brand.danger : .secondary)
            }
            .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func color(_ fraction: Double) -> Color { fraction > 0.85 ? Brand.danger : fraction > 0.7 ? Brand.warning : Brand.success }
}
