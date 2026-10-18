import AppKit
import Foundation
import Observation
import QwenImageCore

/// Workbench state shared by every panel. Owns the task queue and talks to the generation engine.
@MainActor
@Observable
final class AppModel {
    enum WorkspaceTab: String, CaseIterable { case preview, inputs, mask }
    enum ManagementTab: String, CaseIterable { case lora, history, tasks }
    enum Appearance: String, CaseIterable { case system, light, dark }

    /// Brief outcome of the last task that did not produce an image.
    struct CanvasNotice: Equatable, Identifiable {
        let id = UUID()
        var failed: Bool
        var title: String
        var detail: String?
    }

    // MARK: Settings

    var modelRoot: URL? { didSet { save(); validateModel() } }
    var outputDirectory: URL { didSet { save(); reloadGallery() } }
    var exportSettings = ExportSettings() { didSet { save() } }
    var offline = false { didSet { save(); downloads.offline = offline } }
    var appearance: Appearance = .system { didSet { save(); applyAppearance() } }
    /// Extra folders shown in the gallery, e.g. the previous WebUI's output folder.
    var galleryFolders: [String] = [] { didSet { save(); reloadGallery() } }
    var loraFolder: URL { didSet { save() } }

    // MARK: Model library

    var models: [ModelEntry] = [] { didSet { save() } }
    var modelStatus: ModelStatus = .unselected
    var components: [ModelDirectory.Component] = []
    var residency: [String: ComponentResidency] = [:]
    var modelName: String { modelRoot?.lastPathComponent ?? "未选择模型" }
    var modelBytes: Int64 { components.reduce(0) { $0 + $1.bytes } }

    // MARK: Creation context

    var mode: CreationMode = .textToImage {
        didSet {
            save()
            if mode.needsMask { workspaceTab = .mask } else if mode == .textToImage { workspaceTab = .preview }
            else if workspaceTab == .mask { workspaceTab = .inputs }
            syncMaskCanvas()
            applySizePolicy()
        }
    }
    var parameters = GenerationParameters() { didSet { save() } }
    var loras: [LoRAItem] = [] { didSet { save() } }
    var loraPresets: [LoRAPreset] = [] { didSet { Store.save(loraPresets, "lora-presets.json") } }
    var promptPresets: [PromptPreset] = [] { didSet { Store.save(promptPresets, "prompt-presets.json") } }
    var presets: [GenerationPreset] = [] { didSet { Store.save(presets, "presets.json") } }
    /// Condition images (reference / edit / inpaint; the first is the main image for edit and inpaint).
    var conditions: [InputImage] = [] { didSet { save(); syncMaskCanvas(); applySizePolicy() } }
    /// Start image for image-to-image.
    var initialImage: InputImage? { didSet { save(); applySizePolicy() } }
    let mask = MaskDocument()
    /// Seed of the most recent image, shown next to the seed field.
    var lastSeed: UInt64?

    // MARK: Tasks and results

    var tasks: [TaskRecord] = [] { didSet { saveTasks() } }
    var live: LiveProgress?
    private(set) var cancellationRequested = false
    var canvasNotice: CanvasNotice?
    var artworks: [Artwork] = []
    var favorites: Set<String> = [] { didSet { save() } }
    var selection: URL?
    var galleryFilter = GalleryFilter()
    var metrics = SystemMetrics.sample()
    /// Measured steady-state seconds per step by resolution and cost class, for time estimates.
    var stepRates: [String: Double] = [:] { didSet { save() } }

    // MARK: Layout

    var leftFraction: Double = 2.0 / 3.0
    var rightTopFraction: Double = 0.5
    var workspaceTab: WorkspaceTab = .preview
    var managementTab: ManagementTab = .lora

    let downloads = DownloadManager()
    private let engine = GenerationEngine()
    private var runningTaskID: UUID?
    private var metricsTimer: Timer?
    private let defaults = UserDefaults.standard
    private var loading = true

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        outputDirectory = home.appending(path: "Pictures/LingXiStudio")
        loraFolder = home.appending(path: "Models/Qwen-Image-2.1-LoRA")
        if let path = defaults.string(forKey: "outputDirectory") { outputDirectory = URL(fileURLWithPath: path) }
        if let path = defaults.string(forKey: "loraFolder") { loraFolder = URL(fileURLWithPath: path) }
        if let path = defaults.string(forKey: "modelRoot") {
            modelRoot = URL(fileURLWithPath: path)
        } else {
            let fallback = home.appending(path: "Models/Qwen-Image-2.1-BF16")
            if FileManager.default.fileExists(atPath: fallback.path) { modelRoot = fallback }
        }
        func decode<T: Decodable>(_ type: T.Type, _ key: String) -> T? {
            defaults.data(forKey: key).flatMap { try? JSONDecoder().decode(type, from: $0) }
        }
        parameters = decode(GenerationParameters.self, "parameters") ?? GenerationParameters()
        loras = (decode([LoRAItem].self, "loras") ?? []).map { var item = $0; item.refresh(); return item }
        stepRates = decode([String: Double].self, "stepRates") ?? [:]
        exportSettings = decode(ExportSettings.self, "exportSettings") ?? ExportSettings()
        models = decode([ModelEntry].self, "models") ?? []
        favorites = Set(defaults.stringArray(forKey: "favorites") ?? [])
        galleryFolders = defaults.stringArray(forKey: "galleryFolders") ?? []
        offline = defaults.bool(forKey: "offline")
        appearance = Appearance(rawValue: defaults.string(forKey: "appearance") ?? "") ?? .system
        mode = CreationMode(rawValue: defaults.string(forKey: "mode") ?? "") ?? .textToImage
        conditions = (decode([InputImage].self, "conditions") ?? []).filter { FileManager.default.fileExists(atPath: $0.storedPath) }
        initialImage = decode(InputImage.self, "initialImage").flatMap {
            FileManager.default.fileExists(atPath: $0.storedPath) ? $0 : nil
        }
        presets = Store.load([GenerationPreset].self, "presets.json") ?? []
        loraPresets = Store.load([LoRAPreset].self, "lora-presets.json") ?? []
        promptPresets = Store.load([PromptPreset].self, "prompt-presets.json") ?? []
        tasks = (Store.load([TaskRecord].self, "tasks.json") ?? []).map { task in
            var task = task
            if !task.status.isFinished { task.status = .interrupted }
            return task
        }
        if let root = modelRoot, !models.contains(where: { $0.path == root.path }) {
            models.append(ModelEntry(path: root.path, lastUsed: Date()))
        }
        downloads.offline = offline
        loading = false
        if let preset = presets.first(where: \.isDefault), parameters == GenerationParameters() { apply(preset) }
        validateModel()
        reloadGallery()
        syncMaskCanvas()
        applyAppearance()
        // Memory and thermal figures are cheap sysctl reads; two seconds is enough for warnings and the inspector.
        metricsTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.metrics = SystemMetrics.sample() }
        }
        AppLog.write("app started; model \(modelRoot?.path ?? "none")")
        AppLog.write("runtime Metal=\(RuntimeOverrides.current("MLX_MAX_OPS_PER_BUFFER"))ops/\(RuntimeOverrides.current("MLX_MAX_MB_PER_BUFFER"))MB TF32=\(RuntimeOverrides.current("MLX_ENABLE_TF32"))")
    }

    func applyAppearance() {
        switch appearance {
        case .system: NSApp?.appearance = nil
        case .light: NSApp?.appearance = NSAppearance(named: .aqua)
        case .dark: NSApp?.appearance = NSAppearance(named: .darkAqua)
        }
    }

    // MARK: - Model lifecycle

    func validateModel() {
        guard let root = modelRoot else {
            modelStatus = .unselected
            components = []
            return
        }
        modelStatus = .validating
        Task.detached {
            let result = Result { try ModelDirectory(root: root).validate() }
            await MainActor.run {
                guard self.modelRoot == root else { return }
                switch result {
                case .success(let components):
                    self.components = components
                    self.modelStatus = .ready
                case .failure(let error):
                    self.components = []
                    self.modelStatus = .error("\(error)")
                }
                self.residency = [:]
                if let i = self.models.firstIndex(where: { $0.path == root.path }) {
                    self.models[i] = ModelScanner.validate(self.models[i])
                }
            }
        }
    }

    func selectModel(_ entry: ModelEntry) {
        guard !isRunning || entry.path == modelRoot?.path else { return }
        if let i = models.firstIndex(where: { $0.path == entry.path }) { models[i].lastUsed = Date() }
        modelRoot = entry.url
    }

    func addModelFolder(_ url: URL, source: String = "本地文件夹") {
        let found = ModelScanner.scan(root: url)
        let candidates = found.isEmpty ? [url] : found
        for dir in candidates where !models.contains(where: { $0.path == dir.path }) {
            models.append(ModelScanner.validate(ModelEntry(path: dir.path, source: source)))
        }
        if modelRoot == nil, let first = candidates.first { modelRoot = first }
    }

    func scanHuggingFaceCache() -> Int {
        let found = ModelScanner.huggingFaceCache()
        var added = 0
        for dir in found where !models.contains(where: { $0.path == dir.path }) {
            models.append(ModelScanner.validate(ModelEntry(path: dir.path, source: "Hugging Face 缓存")))
            added += 1
        }
        return added
    }

    func revalidateModels() {
        let snapshot = models
        Task.detached {
            let validated = snapshot.map(ModelScanner.validate)
            await MainActor.run { self.models = validated }
        }
    }

    /// Releases per-model state kept between tasks (tokenizer, schedule config). Weights never stay resident.
    func unloadModel() {
        guard let root = modelRoot, runningTaskID == nil else { return }
        engine.forget(model: root)
        residency = [:]
    }

    // MARK: - Input images

    func addConditions(_ images: [InputImage]) {
        let room = max(0, 10 - conditions.count)
        conditions += images.prefix(room).filter { image in !conditions.contains { $0.sha256 == image.sha256 } }
    }

    func importConditions(urls: [URL]) {
        addConditions(urls.compactMap { try? InputStore.importFile($0) })
    }

    func pasteConditions() { addConditions(InputStore.fromPasteboard()) }

    func setInitial(url: URL) {
        if let image = try? InputStore.importFile(url) { initialImage = image }
    }

    func makeMain(_ id: UUID) {
        guard let i = conditions.firstIndex(where: { $0.id == id }), i > 0 else { return }
        let item = conditions.remove(at: i)
        conditions.insert(item, at: 0)
    }

    func removeCondition(_ id: UUID) { conditions.removeAll { $0.id == id } }

    func replaceCondition(_ id: UUID, with url: URL) {
        guard let i = conditions.firstIndex(where: { $0.id == id }), let image = try? InputStore.importFile(url) else { return }
        conditions[i] = image
    }

    /// The image the mask is painted on (edit / inpaint main image).
    var mainImage: InputImage? { conditions.first }

    private func syncMaskCanvas() {
        mask.prepare(width: mainImage?.width ?? 0, height: mainImage?.height ?? 0, imageID: mainImage?.sha256)
    }

    /// Follows the main / last / start image's aspect ratio at 1024² (the official default) unless set manually.
    func applySizePolicy() {
        guard !loading, parameters.sizePolicy == .followImage else { return }
        let source: (Int, Int)? =
            switch mode {
            case .textToImage: nil
            case .reference: conditions.last.map { ($0.width, $0.height) }
            case .edit, .inpaint: conditions.first.map { ($0.width, $0.height) }
            case .imageToImage: initialImage.map { ($0.width, $0.height) }
            }
        guard let (w, h) = source else { return }
        let size = conditionDimensions(ratio: Double(w) / Double(h))
        if parameters.width != size.width || parameters.height != size.height {
            parameters.width = size.width
            parameters.height = size.height
        }
    }

    // MARK: - LoRA

    func importLoRAs(_ urls: [URL]) {
        for url in urls where url.pathExtension == "safetensors" {
            guard !loras.contains(where: { $0.path == url.path }) else { continue }
            var item = LoRAItem(path: url.path)
            item.refresh()
            loras.append(item)
        }
    }

    /// LoRA files in the LoRA folder that are not in the list yet.
    var availableLoRAFiles: [URL] {
        let files = (try? FileManager.default.contentsOfDirectory(at: loraFolder, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "safetensors" && !loras.map(\.path).contains($0.path) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    var activeLoRAs: [LoRAItem] { loras.filter { $0.enabled && $0.isUsable && $0.scale != 0 } }

    /// The turbo schedule only makes sense with an acceleration LoRA that is switched on.
    var turboAvailable: Bool { activeLoRAs.contains { $0.purpose == .acceleration } }

    func saveLoRAPreset(named name: String) {
        loraPresets.append(LoRAPreset(name: name, entries: loras.map(\.spec)))
    }

    func applyLoRAPreset(_ preset: LoRAPreset) { restoreLoRAs(preset.entries) }

    // MARK: - Validation and estimates

    /// Estimated peak process memory of the denoising stage (measured on M5 32 GB: 19.3 GB for 1024² text-to-image
    /// with a rank-256 LoRA, +2.2 GB of cached keys/values per 1024² condition image, doubled by true CFG).
    var estimatedPeakBytes: Int64 {
        let gb: Double = 1_073_741_824
        let conditionsUsed = mode.needsConditions ? conditions.count : 0
        let pixelsScale = Double(parameters.width * parameters.height) / Double(1024 * 1024)
        var estimate = 17.5 + 1.8 * pixelsScale
        estimate += 2.2 * Double(conditionsUsed) * (parameters.usesTrueCFG ? 2 : 1)
        if parameters.usesTrueCFG { estimate += 0.3 }
        return Int64(estimate * gb)
    }

    let physicalMemory = Int64(ProcessInfo.processInfo.physicalMemory)
    var memoryBudget: Int64 { Int64(Double(physicalMemory) * 0.85) }

    /// Problems that block queuing the current configuration.
    var blockingIssues: [String] {
        var issues: [String] = []
        if modelStatus != .ready && modelStatus != .inUse { issues.append("模型未就绪") }
        if parameters.text(for: mode).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append(mode.usesInstruction ? "编辑指令为空" : "提示词为空")
        }
        if parameters.width % 32 != 0 || parameters.height % 32 != 0 { issues.append("宽高必须是 32 的倍数") }
        if parameters.preset == .turbo && !turboAvailable { issues.append("Turbo 需要启用一个加速 LoRA") }
        for item in loras where item.enabled && !item.isUsable { issues.append("LoRA「\(item.name)」不可用") }
        switch mode {
        case .reference: if conditions.isEmpty { issues.append("至少需要一张参考图") }
        case .edit: if conditions.isEmpty { issues.append("需要一张要编辑的主图") }
        case .inpaint:
            if conditions.isEmpty { issues.append("需要一张要重绘的主图") }
            if mask.isEmpty { issues.append("蒙版为空：请在蒙版画布上涂抹要重绘的区域") }
        case .imageToImage: if initialImage == nil { issues.append("需要一张初始图") }
        case .textToImage: break
        }
        if estimatedPeakBytes > memoryBudget {
            issues.append("预计峰值内存 \(Format.bytes(estimatedPeakBytes)) 超出本机安全范围，请减少参考图或关闭真 CFG")
        }
        return issues
    }

    /// Non-blocking risks shown next to the generate button.
    var warnings: [String] {
        var notes: [String] = []
        let pixels = parameters.width * parameters.height
        if pixels > 1536 * 1536 { notes.append("分辨率超过 1536²，可能出现 Swap") }
        if parameters.usesTrueCFG { notes.append("真 CFG 每步计算两次，耗时约翻倍") }
        if parameters.guidance > 1 && !parameters.usesTrueCFG { notes.append("Guidance > 1 但负面提示词为空，引导不会生效") }
        if activeLoRAs.contains(where: { $0.scale < 0 || $0.scale > 1.5 }) { notes.append("LoRA 强度超出常用范围（0–1.5）") }
        if estimatedPeakBytes > Int64(Double(physicalMemory) * 0.7) && estimatedPeakBytes <= memoryBudget {
            notes.append("预计峰值内存 \(Format.bytes(estimatedPeakBytes))，接近本机上限")
        }
        if metrics.pressure == .critical { notes.append("系统内存压力严重") }
        if mode == .imageToImage && parameters.imageStrength >= 0.99 { notes.append("强度 1.0 等同于从纯噪声生成（不保留初始图）") }
        return notes
    }

    private func rateKey(for p: GenerationParameters, loras: [LoRASpec], conditions: Int) -> String {
        "\(p.width)x\(p.height)-lora\(loras.isEmpty ? 0 : 1)-cfg\(p.usesTrueCFG ? 1 : 0)-cond\(conditions)"
    }

    var estimatedSecondsPerStep: Double? {
        stepRates[rateKey(for: parameters, loras: activeLoRAs.map(\.spec), conditions: mode.needsConditions ? conditions.count : 0)]
    }

    // MARK: - Queue

    var isRunning: Bool { runningTaskID != nil }
    var waitingCount: Int { tasks.filter { $0.status == .waiting }.count }
    var failedCount: Int { tasks.filter { if case .failed = $0.status { true } else { false } }.count }

    func enqueue() {
        guard blockingIssues.isEmpty, let root = modelRoot else { return }
        canvasNotice = nil
        var params = parameters
        params.steps = params.effectiveSteps
        var maskInput: InputImage?
        if mode.needsMask, let rgba = mask.rgbaImage() {
            maskInput = try? InputStore.importRGBA(rgba, name: "mask")
        }
        let task = TaskRecord(
            mode: mode, modelName: root.lastPathComponent, modelPath: root.path, parameters: params,
            loras: activeLoRAs.map(\.spec), conditions: mode.needsConditions ? conditions : [],
            initial: mode.needsInitial ? initialImage : nil, mask: maskInput)
        tasks.insert(task, at: 0)
        AppLog.write("queued \(task.id) \(mode.rawValue) \(params.width)x\(params.height) \(params.steps) steps")
        startNextIfIdle()
    }

    func retry(_ id: UUID) {
        guard let old = tasks.first(where: { $0.id == id }) else { return }
        let task = TaskRecord(
            mode: old.mode, modelName: old.modelName, modelPath: old.modelPath, parameters: old.parameters,
            loras: old.loras, conditions: old.conditions, initial: old.initial, mask: old.mask)
        tasks.insert(task, at: 0)
        startNextIfIdle()
    }

    func cancel(_ id: UUID) {
        guard let index = tasks.firstIndex(where: { $0.id == id }) else { return }
        if id == runningTaskID {
            guard !cancellationRequested else { return }
            cancellationRequested = true
            AppLog.write("task \(id) cancellation requested")
            engine.cancel()
        } else if tasks[index].status == .waiting {
            tasks[index].status = .cancelled
        }
    }

    func stopCurrent() {
        if let id = runningTaskID { cancel(id) }
    }

    func clearWaiting() {
        for i in tasks.indices where tasks[i].status == .waiting { tasks[i].status = .cancelled }
    }

    func clearFinished() {
        tasks.removeAll { $0.status.isFinished }
    }

    private func startNextIfIdle() {
        guard runningTaskID == nil, let index = tasks.lastIndex(where: { $0.status == .waiting }) else { return }
        var task = tasks[index]
        task.status = .running
        task.startedAt = Date()
        let p = task.parameters
        let base = p.seedMode == .fixed ? p.seed : UInt64.random(in: 0..<UInt64(UInt32.max))
        let seeds = (0..<max(1, p.count)).map { i -> UInt64 in
            i == 0 || p.batchSeeds == .increment ? base &+ UInt64(i) : UInt64.random(in: 0..<UInt64(UInt32.max))
        }
        tasks[index] = task
        runningTaskID = task.id
        cancellationRequested = false
        canvasNotice = nil
        workspaceTab = .preview
        modelStatus = .inUse
        residency = [:]
        var records = task.conditions.map { $0.record(role: .condition) }
        if let initial = task.initial { records.append(initial.record(role: .initial)) }
        if let mask = task.mask { records.append(mask.record(role: .mask)) }
        let request = GenerationRequest(
            prompt: p.text(for: task.mode), negativePrompt: p.usesTrueCFG ? p.negativePrompt : nil, width: p.width,
            height: p.height, steps: p.steps, trueCFGScale: p.guidance, seed: seeds[0], scheduler: p.preset.schedule,
            loras: task.loras, mode: task.mode, inputs: records, strength: p.strength(for: task.mode),
            maskFeather: p.maskFeather, maskInvert: p.maskInvert, keepUnmaskedPixels: p.keepUnmasked)
        let job = GenerationEngine.Job(
            taskID: task.id, modelRoot: URL(fileURLWithPath: task.modelPath), request: request, seeds: seeds,
            outputDirectory: outputDirectory)
        let id = task.id
        let key = rateKey(for: p, loras: task.loras, conditions: task.conditions.count)
        engine.submit(job) { event in
            Task { @MainActor in self.handle(event, taskID: id, imageCount: seeds.count, rateKey: key) }
        }
    }

    private func handle(_ event: GenerationEngine.Event, taskID: UUID, imageCount: Int, rateKey: String) {
        guard let index = tasks.firstIndex(where: { $0.id == taskID }) else { return }
        switch event {
        case .started(let imageIndex, let seed):
            tasks[index].seeds.append(seed)
            tasks[index].log.append("第 \(imageIndex + 1)/\(imageCount) 张开始，seed \(seed)")
            lastSeed = seed
        case .progress(let imageIndex, let p):
            if live?.stage != p.stage {
                AppLog.write("task \(taskID) image \(imageIndex) stage=\(p.stage.rawValue) elapsed=\(p.elapsed) conditioningCacheHit=\(p.conditioningCacheHit)")
            }
            live = LiveProgress(
                taskID: taskID, imageIndex: imageIndex, imageCount: imageCount, stage: p.stage, fraction: p.fraction,
                step: p.step, totalSteps: p.totalSteps, secondsPerStep: p.secondsPerStep, elapsed: p.elapsed,
                mlxActive: p.memory.activeBytes, mlxPeak: p.memory.peakBytes, conditioningCacheHit: p.conditioningCacheHit)
            updateResidency(p.stage, conditioningCacheHit: p.conditioningCacheHit)
        case .saved(_, let url, let seconds, let stepSeconds):
            tasks[index].outputs.append(url.path)
            tasks[index].log.append("已保存 \(url.lastPathComponent)（\(Format.duration(seconds))）")
            if stepSeconds.count > 1 {
                let steady = stepSeconds.dropFirst()
                stepRates[rateKey] = steady.reduce(0, +) / Double(steady.count)
            }
            insertArtwork(url)
            selection = url
            workspaceTab = .preview
        case .finished(let result):
            AppLog.write("task \(taskID) finished: \(result)")
            switch result {
            case .success: tasks[index].status = .done
            case .failure(PipelineError.cancelled):
                tasks[index].status = .cancelled
                selection = nil
                canvasNotice = CanvasNotice(
                    failed: false, title: "已取消",
                    detail: tasks[index].outputs.isEmpty ? nil : "已完成的 \(tasks[index].outputs.count) 张已保存在图库")
            case .failure(let error):
                tasks[index].status = .failed(Self.explain(error))
                canvasNotice = CanvasNotice(failed: true, title: "生成失败", detail: Self.explain(error))
                tasks[index].log.append("错误：\(error)")
                AppLog.write("task \(taskID) failed: \(error)")
            }
            tasks[index].finishedAt = Date()
            runningTaskID = nil
            cancellationRequested = false
            live = nil
            residency = residency.mapValues { $0 == .notLoaded ? .notLoaded : .released }
            if modelStatus == .inUse { modelStatus = .ready }
            startNextIfIdle()
        }
    }

    /// Maps an error to the component it concerns (model files, LoRA, inputs, parameters or memory).
    static func explain(_ error: Error) -> String {
        switch error {
        case let e as ModelDirectory.ValidationError: "模型文件：\(e)"
        case let e as WeightError: "模型组件：\(e)"
        case let e as LoRAError: "LoRA 兼容性：\(e)"
        case let e as ConditioningError: "输入图像：\(e)"
        case let e as PipelineError: "参数：\(e)"
        default:
            "\(error)".localizedCaseInsensitiveContains("memory") ? "内存压力：\(error)" : "\(error)"
        }
    }

    /// The balanced profile keeps one large component resident at a time.
    private func updateResidency(_ stage: GenerationStage, conditioningCacheHit: Bool) {
        func set(_ name: String, _ state: ComponentResidency) {
            if residency[name] != state { residency[name] = state }
        }
        switch stage {
        case .validating: break
        case .loadingTextEncoder: set("text_encoder", .loading)
        case .encodingPrompt: if !conditioningCacheHit { set("text_encoder", .resident) }
        case .encodingImages:
            if !conditioningCacheHit { set("text_encoder", .released) }
            set("vae", .resident)
        case .loadingTransformer:
            if !conditioningCacheHit { set("text_encoder", .released) }
            if residency["vae"] == .resident { set("vae", .released) }
            set("transformer", .loading)
        case .prefill, .denoising: set("transformer", .resident)
        case .loadingVAE:
            set("transformer", .released)
            set("vae", .loading)
        case .decoding: set("vae", .resident)
        case .done: set("vae", .released)
        }
    }

    // MARK: - Gallery

    func reloadGallery() {
        let folders = [(outputDirectory, Artwork.Origin.native)]
            + galleryFolders.map { (URL(fileURLWithPath: $0), Artwork.Origin.legacy) }
        Task.detached {
            let items = folders.flatMap { GalleryScanner.scan($0.0, origin: $0.1) }.sorted { $0.created > $1.created }
            await MainActor.run {
                self.artworks = items
                if self.selection == nil || !items.contains(where: { $0.url == self.selection }) {
                    self.selection = items.first?.url
                }
            }
        }
    }

    private func insertArtwork(_ url: URL) {
        artworks.removeAll { $0.url == url }
        if let item = GalleryScanner.scan(url.deletingLastPathComponent(), origin: .native).first(where: { $0.url == url }) {
            artworks.insert(item, at: 0)
        }
    }

    var filteredArtworks: [Artwork] { artworks.filter { galleryFilter.matches($0, favorites: favorites) } }

    var selectedArtwork: Artwork? { artworks.first { $0.url == selection } }

    func toggleFavorite(_ item: Artwork) {
        if favorites.contains(item.url.path) { favorites.remove(item.url.path) } else { favorites.insert(item.url.path) }
    }

    /// Restores every recorded parameter of a past image: mode, prompt, size, steps, guidance, seed, schedule,
    /// strengths, input images and the full LoRA combination. Fields the record lacks keep their current values.
    func reuse(_ item: Artwork, sameSeed: Bool = true) {
        if let meta = item.metadata {
            let r = meta.request
            var p = parameters
            if r.mode.usesInstruction { p.instruction = r.prompt } else { p.prompt = r.prompt }
            p.negativePrompt = r.negativePrompt ?? ""
            p.width = r.width
            p.height = r.height
            p.sizePolicy = .manual
            p.steps = r.steps
            p.guidance = r.trueCFGScale
            p.seedMode = sameSeed ? .fixed : .random
            p.seed = r.seed
            p.count = 1
            p.preset = r.scheduler == .viggleTurbo ? .turbo : .standard
            switch r.mode {
            case .imageToImage: p.imageStrength = r.strength
            case .edit: p.editStrength = r.strength
            case .inpaint: p.inpaintStrength = r.strength
            default: break
            }
            p.maskFeather = r.maskFeather
            p.maskInvert = r.maskInvert
            p.keepUnmasked = r.keepUnmaskedPixels
            parameters = p
            let images = r.inputs.compactMap { record -> (InputImageRecord.Role, InputImage)? in
                guard FileManager.default.fileExists(atPath: record.path) else { return nil }
                return (record.role, InputImage(
                    storedPath: record.path, sourcePath: record.source, sha256: record.sha256, width: record.width,
                    height: record.height))
            }
            conditions = images.filter { $0.0 == .condition }.map(\.1)
            if let start = images.first(where: { $0.0 == .initial }) { initialImage = start.1 }
            mode = r.mode
            if let maskImage = images.first(where: { $0.0 == .mask }),
                let rgba = try? RGBAImage(contentsOf: maskImage.1.url)
            {
                syncMaskCanvas()
                mask.load(rgba)
            }
            restoreLoRAs(r.loras)
        } else if let legacy = item.legacy {
            var p = parameters
            if let prompt = legacy.prompt { p.prompt = prompt }
            p.negativePrompt = legacy.negativePrompt ?? p.negativePrompt
            if let steps = legacy.steps { p.steps = steps }
            if let guidance = legacy.guidance { p.guidance = guidance }
            if let seed = legacy.seed, sameSeed {
                p.seed = seed
                p.seedMode = .fixed
            }
            p.width = item.pixelWidth / 32 * 32
            p.height = item.pixelHeight / 32 * 32
            p.sizePolicy = .manual
            p.preset = legacy.scheduler == "viggle_turbo" ? .turbo : .standard
            parameters = p
            mode = .textToImage
            if !legacy.loras.isEmpty { restoreLoRAs(legacy.loras) }
        }
    }

    private func restoreLoRAs(_ specs: [LoRASpec]) {
        var restored: [LoRAItem] = []
        for spec in specs {
            var item = loras.first { $0.path == spec.path } ?? LoRAItem(path: spec.path)
            item.scale = spec.scale
            item.enabled = spec.enabled
            item.refresh()
            restored.append(item)
        }
        for item in loras where !restored.contains(where: { $0.path == item.path }) {
            var item = item
            item.enabled = false
            restored.append(item)
        }
        loras = restored
    }

    /// Sends a gallery image into the creation context.
    enum Destination { case reference, initial, edit, inpaint }

    func send(_ item: Artwork, to destination: Destination) {
        guard let image = try? InputStore.importFile(item.url) else { return }
        switch destination {
        case .reference:
            addConditions([image])
            mode = .reference
        case .initial:
            initialImage = image
            mode = .imageToImage
        case .edit:
            conditions.removeAll { $0.sha256 == image.sha256 }
            conditions.insert(image, at: 0)
            mode = .edit
        case .inpaint:
            conditions.removeAll { $0.sha256 == image.sha256 }
            conditions.insert(image, at: 0)
            mode = .inpaint
            mask.clear()
        }
        workspaceTab = destination == .inpaint ? .mask : .inputs
    }

    func exportArtworks(_ items: [Artwork], to folder: URL) -> (Int, [String]) {
        var errors: [String] = []
        var done = 0
        for (i, item) in items.enumerated() {
            do {
                _ = try exportSettings.export(item, index: i + 1, to: folder)
                done += 1
            } catch {
                errors.append("\(item.url.lastPathComponent)：\(error)")
            }
        }
        return (done, errors)
    }

    private func sidecars(_ item: Artwork) -> [URL] {
        [item.url, item.url.deletingPathExtension().appendingPathExtension("json"),
         item.url.deletingPathExtension().appendingPathExtension("metadata.json")]
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }

    func moveArtworks(_ items: [Artwork], to folder: URL) {
        for item in items {
            for url in sidecars(item) {
                let target = folder.appending(path: url.lastPathComponent)
                if !FileManager.default.fileExists(atPath: target.path) {
                    try? FileManager.default.moveItem(at: url, to: target)
                }
            }
        }
        reloadGallery()
    }

    func trashArtworks(_ items: [Artwork]) {
        for item in items {
            for url in sidecars(item) { try? FileManager.default.trashItem(at: url, resultingItemURL: nil) }
            favorites.remove(item.url.path)
        }
        if let selection, items.contains(where: { $0.url == selection }) { self.selection = nil }
        reloadGallery()
    }

    /// The previous WebUI's default output folder, when present and not added yet.
    var legacyGalleryCandidate: URL? {
        let url = FileManager.default.homeDirectoryForCurrentUser.appending(path: "Pictures/Qwen-Image-mflux")
        return FileManager.default.fileExists(atPath: url.path) && !galleryFolders.contains(url.path) ? url : nil
    }

    // MARK: - Presets

    func savePreset(named name: String, includes: Set<GenerationPreset.Part>) {
        var preset = GenerationPreset(name: name, includes: includes, mode: mode, parameters: parameters)
        if includes.contains(.inputs) {
            preset.conditions = conditions
            preset.initial = initialImage
        }
        if includes.contains(.model) { preset.modelPath = modelRoot?.path }
        if includes.contains(.loras) { preset.loras = loras.map(\.spec) }
        if includes.contains(.output) { preset.outputDirectory = outputDirectory.path }
        if !includes.contains(.prompt) {
            preset.parameters.prompt = ""
            preset.parameters.instruction = ""
            preset.parameters.negativePrompt = ""
        }
        presets.append(preset)
    }

    func apply(_ preset: GenerationPreset) {
        var p = preset.parameters
        if !preset.includes.contains(.prompt) {
            p.prompt = parameters.prompt
            p.instruction = parameters.instruction
            p.negativePrompt = parameters.negativePrompt
        }
        parameters = p
        mode = preset.mode
        if preset.includes.contains(.inputs) {
            conditions = preset.conditions.filter { FileManager.default.fileExists(atPath: $0.storedPath) }
            initialImage = preset.initial.flatMap { FileManager.default.fileExists(atPath: $0.storedPath) ? $0 : nil }
        }
        if preset.includes.contains(.model), let path = preset.modelPath, FileManager.default.fileExists(atPath: path) {
            modelRoot = URL(fileURLWithPath: path)
        }
        if preset.includes.contains(.loras) { restoreLoRAs(preset.loras) }
        if preset.includes.contains(.output), let path = preset.outputDirectory { outputDirectory = URL(fileURLWithPath: path) }
    }

    func setDefaultPreset(_ id: UUID) {
        for i in presets.indices { presets[i].isDefault = presets[i].id == id && !presets[i].isDefault }
    }

    // MARK: - Persistence

    private func save() {
        guard !loading else { return }
        let encoder = JSONEncoder()
        defaults.set(modelRoot?.path, forKey: "modelRoot")
        defaults.set(outputDirectory.path, forKey: "outputDirectory")
        defaults.set(loraFolder.path, forKey: "loraFolder")
        defaults.set(try? encoder.encode(parameters), forKey: "parameters")
        defaults.set(try? encoder.encode(loras), forKey: "loras")
        defaults.set(try? encoder.encode(stepRates), forKey: "stepRates")
        defaults.set(try? encoder.encode(exportSettings), forKey: "exportSettings")
        defaults.set(try? encoder.encode(models), forKey: "models")
        defaults.set(try? encoder.encode(conditions), forKey: "conditions")
        defaults.set(try? encoder.encode(initialImage), forKey: "initialImage")
        defaults.set(Array(favorites), forKey: "favorites")
        defaults.set(galleryFolders, forKey: "galleryFolders")
        defaults.set(offline, forKey: "offline")
        defaults.set(appearance.rawValue, forKey: "appearance")
        defaults.set(mode.rawValue, forKey: "mode")
    }

    private func saveTasks() {
        guard !loading else { return }
        Store.save(Array(tasks.prefix(500)), "tasks.json")
    }
}
