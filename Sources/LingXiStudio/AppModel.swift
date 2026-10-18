import AppKit
import Foundation
import Observation
import QwenImageCore

/// Workbench state shared by every panel. Owns the task queue and talks to the generation engine.
@MainActor
@Observable
final class AppModel {
    // MARK: Settings

    var modelRoot: URL? { didSet { save(); validateModel() } }
    var outputDirectory: URL { didSet { save(); reloadHistory() } }

    // MARK: Model

    var modelStatus: ModelStatus = .unselected
    var components: [ModelDirectory.Component] = []
    var residency: [String: ComponentResidency] = [:]
    var modelName: String { modelRoot?.lastPathComponent ?? "未选择模型" }
    var modelBytes: Int64 { components.reduce(0) { $0 + $1.bytes } }

    // MARK: Parameters

    var mode: CreationMode = .textToImage
    var parameters = GenerationParameters() { didSet { save() } }
    var loras: [LoRAItem] = [] { didSet { save() } }
    /// Seed of the most recent image, shown next to the seed field.
    var lastSeed: UInt64?

    // MARK: Tasks and results

    var tasks: [TaskRecord] = [] { didSet { saveTasks() } }
    var live: LiveProgress?
    var history: [HistoryItem] = []
    var selection: URL?
    var metrics = SystemMetrics.sample()
    /// Measured steady-state seconds per step by resolution and LoRA/CFG cost class, for time estimates.
    var stepRates: [String: Double] = [:] { didSet { save() } }

    // MARK: Layout

    var leftFraction: Double = 2.0 / 3.0
    var rightTopFraction: Double = 0.5

    private let engine = GenerationEngine()
    private var runningTaskID: UUID?
    private var metricsTimer: Timer?
    private let defaults = UserDefaults.standard
    private var loading = true

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        outputDirectory = home.appending(path: "Pictures/LingXiStudio")
        if let path = defaults.string(forKey: "outputDirectory") { outputDirectory = URL(fileURLWithPath: path) }
        if let path = defaults.string(forKey: "modelRoot") {
            modelRoot = URL(fileURLWithPath: path)
        } else {
            let fallback = home.appending(path: "Models/Qwen-Image-2.1-BF16")
            if FileManager.default.fileExists(atPath: fallback.path) { modelRoot = fallback }
        }
        if let data = defaults.data(forKey: "parameters"),
            let value = try? JSONDecoder().decode(GenerationParameters.self, from: data)
        {
            parameters = value
        }
        if let data = defaults.data(forKey: "loras"), let value = try? JSONDecoder().decode([LoRAItem].self, from: data) {
            loras = value.map { var item = $0; item.refresh(); return item }
        }
        if let data = defaults.data(forKey: "stepRates"), let value = try? JSONDecoder().decode([String: Double].self, from: data) {
            stepRates = value
        }
        tasks = loadTasks().map { task in
            var task = task
            if !task.status.isFinished { task.status = .interrupted }
            return task
        }
        loading = false
        validateModel()
        reloadHistory()
        metricsTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.metrics = SystemMetrics.sample() }
        }
    }

    // MARK: Model lifecycle

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
            }
        }
    }

    /// Releases per-model state kept between tasks (tokenizer, schedule config). Weights never stay resident.
    func unloadModel() {
        guard let root = modelRoot, runningTaskID == nil else { return }
        engine.forget(model: root)
        residency = [:]
    }

    // MARK: LoRA

    func importLoRAs(_ urls: [URL]) {
        for url in urls where url.pathExtension == "safetensors" {
            guard !loras.contains(where: { $0.path == url.path }) else { continue }
            var item = LoRAItem(path: url.path)
            item.refresh()
            loras.append(item)
        }
    }

    var activeLoRAs: [LoRAItem] { loras.filter { $0.enabled && $0.isUsable && $0.scale != 0 } }

    /// The turbo schedule only makes sense with an acceleration LoRA that is switched on.
    var turboAvailable: Bool { activeLoRAs.contains { $0.purpose == .acceleration } }

    // MARK: Validation of the current parameters

    /// Problems that block queuing the current configuration.
    var blockingIssues: [String] {
        var issues: [String] = []
        if mode.unavailableReason != nil { issues.append("当前模式尚未实现") }
        if modelStatus != .ready && modelStatus != .inUse { issues.append("模型未就绪") }
        if parameters.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { issues.append("提示词为空") }
        if parameters.width % 32 != 0 || parameters.height % 32 != 0 { issues.append("宽高必须是 32 的倍数") }
        if parameters.preset == .turbo && !turboAvailable { issues.append("Turbo 需要启用一个加速 LoRA") }
        for item in loras where item.enabled && !item.isUsable { issues.append("LoRA「\(item.name)」不可用") }
        return issues
    }

    /// Non-blocking risks shown next to the generate button.
    var warnings: [String] {
        var notes: [String] = []
        let pixels = parameters.width * parameters.height
        if pixels > 1536 * 1536 { notes.append("分辨率超过 1536²，32 GB 机器上可能出现 Swap") }
        if parameters.usesTrueCFG { notes.append("真 CFG 每步计算两次，耗时约翻倍") }
        if parameters.guidance > 1 && !parameters.usesTrueCFG { notes.append("Guidance > 1 但负面提示词为空，引导不会生效") }
        let scaleOut = activeLoRAs.filter { $0.scale < 0 || $0.scale > 1.5 }
        if !scaleOut.isEmpty { notes.append("LoRA 强度超出常用范围（0–1.5）") }
        if metrics.pressure == .critical { notes.append("系统内存压力严重") }
        return notes
    }

    var estimatedSecondsPerStep: Double? { stepRates[rateKey(for: parameters, loras: activeLoRAs.map(\.spec))] }

    // MARK: Queue

    var isRunning: Bool { runningTaskID != nil }
    var waitingCount: Int { tasks.filter { $0.status == .waiting }.count }
    var failedCount: Int { tasks.filter { if case .failed = $0.status { true } else { false } }.count }

    func enqueue() {
        guard blockingIssues.isEmpty, let root = modelRoot else { return }
        var params = parameters
        if params.preset == .turbo { params.steps = FlowMatchSchedule.viggleTurboNodes.count }
        let task = TaskRecord(
            modelName: root.lastPathComponent, modelPath: root.path, parameters: params,
            loras: activeLoRAs.map(\.spec))
        tasks.insert(task, at: 0)
        startNextIfIdle()
    }

    func retry(_ id: UUID) {
        guard let old = tasks.first(where: { $0.id == id }) else { return }
        let task = TaskRecord(
            mode: old.mode, modelName: old.modelName, modelPath: old.modelPath, parameters: old.parameters,
            loras: old.loras)
        tasks.insert(task, at: 0)
        startNextIfIdle()
    }

    func cancel(_ id: UUID) {
        guard let index = tasks.firstIndex(where: { $0.id == id }) else { return }
        if id == runningTaskID {
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
        guard runningTaskID == nil,
            let index = tasks.lastIndex(where: { $0.status == .waiting })
        else { return }
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
        modelStatus = .inUse
        residency = [:]
        let request = GenerationRequest(
            prompt: p.prompt, negativePrompt: p.usesTrueCFG ? p.negativePrompt : nil, width: p.width,
            height: p.height, steps: p.steps, trueCFGScale: p.guidance, seed: seeds[0],
            scheduler: p.preset.schedule, loras: task.loras)
        let job = GenerationEngine.Job(
            taskID: task.id, modelRoot: URL(fileURLWithPath: task.modelPath), request: request, seeds: seeds,
            outputDirectory: outputDirectory)
        let id = task.id
        let key = rateKey(for: p, loras: task.loras)
        engine.submit(job) { event in
            Task { @MainActor in self.handle(event, taskID: id, imageCount: seeds.count, rateKey: key) }
        }
    }

    private func rateKey(for p: GenerationParameters, loras: [LoRASpec]) -> String {
        "\(p.width)x\(p.height)-lora\(loras.isEmpty ? 0 : 1)-cfg\(p.usesTrueCFG ? 1 : 0)"
    }

    private func handle(_ event: GenerationEngine.Event, taskID: UUID, imageCount: Int, rateKey: String) {
        guard let index = tasks.firstIndex(where: { $0.id == taskID }) else { return }
        switch event {
        case .started(let imageIndex, let seed):
            tasks[index].seeds.append(seed)
            tasks[index].log.append("第 \(imageIndex + 1)/\(imageCount) 张开始，seed \(seed)")
            lastSeed = seed
        case .progress(let imageIndex, let p):
            live = LiveProgress(
                taskID: taskID, imageIndex: imageIndex, imageCount: imageCount, stage: p.stage, fraction: p.fraction,
                step: p.step, totalSteps: p.totalSteps, secondsPerStep: p.secondsPerStep, elapsed: p.elapsed,
                mlxActive: p.memory.activeBytes, mlxPeak: p.memory.peakBytes)
            updateResidency(p.stage)
        case .saved(_, let url, let seconds, let stepSeconds):
            tasks[index].outputs.append(url.path)
            tasks[index].log.append("已保存 \(url.lastPathComponent)（\(Format.duration(seconds))）")
            if stepSeconds.count > 1 {
                let steady = stepSeconds.dropFirst()
                stepRates[rateKey] = steady.reduce(0, +) / Double(steady.count)
            }
            insertHistory(url)
            selection = url
        case .finished(let result):
            switch result {
            case .success: tasks[index].status = .done
            case .failure(PipelineError.cancelled): tasks[index].status = .cancelled
            case .failure(let error):
                tasks[index].status = .failed("\(error)")
                tasks[index].log.append("错误：\(error)")
            }
            tasks[index].finishedAt = Date()
            runningTaskID = nil
            live = nil
            residency = residency.mapValues { $0 == .notLoaded ? .notLoaded : .released }
            if modelStatus == .inUse { modelStatus = .ready }
            startNextIfIdle()
        }
    }

    /// The balanced profile keeps one large component resident at a time.
    private func updateResidency(_ stage: GenerationStage) {
        func set(_ name: String, _ state: ComponentResidency) {
            if residency[name] != state { residency[name] = state }
        }
        switch stage {
        case .validating: break
        case .loadingTextEncoder: set("text_encoder", .loading)
        case .encodingPrompt: set("text_encoder", .resident)
        case .loadingTransformer:
            set("text_encoder", .released)
            set("transformer", .loading)
        case .prefill, .denoising: set("transformer", .resident)
        case .loadingVAE:
            set("transformer", .released)
            set("vae", .loading)
        case .decoding: set("vae", .resident)
        case .done: set("vae", .released)
        }
    }

    // MARK: History

    func reloadHistory() {
        let dir = outputDirectory
        Task.detached {
            let urls = (try? FileManager.default.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: [.creationDateKey], options: [.skipsHiddenFiles])) ?? []
            let items = urls.filter { $0.pathExtension.lowercased() == "png" }.map { url in
                let created = (try? url.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                return HistoryItem(url: url, metadata: try? GenerationMetadata.read(sidecarOf: url), created: created)
            }.sorted { $0.created > $1.created }
            await MainActor.run {
                self.history = items
                if self.selection == nil { self.selection = items.first?.url }
            }
        }
    }

    private func insertHistory(_ url: URL) {
        history.removeAll { $0.url == url }
        history.insert(HistoryItem(url: url, metadata: try? GenerationMetadata.read(sidecarOf: url), created: Date()), at: 0)
    }

    var selectedItem: HistoryItem? { history.first { $0.url == selection } }

    /// Restores every recorded parameter of a past image (prompt, size, steps, guidance, seed, schedule, LoRAs).
    func reuse(_ item: HistoryItem, sameSeed: Bool = true) {
        guard let meta = item.metadata else { return }
        let r = meta.request
        var p = parameters
        p.prompt = r.prompt
        p.negativePrompt = r.negativePrompt ?? ""
        p.width = r.width
        p.height = r.height
        p.steps = r.steps
        p.guidance = r.trueCFGScale
        p.seedMode = sameSeed ? .fixed : .random
        p.seed = r.seed
        p.count = 1
        p.preset = r.scheduler == .viggleTurbo ? .turbo : .standard
        parameters = p
        mode = .textToImage
        // The recorded LoRA set replaces the current one; files that moved stay listed and flagged.
        var restored: [LoRAItem] = []
        for spec in r.loras {
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

    // MARK: Persistence

    private func save() {
        guard !loading else { return }
        defaults.set(modelRoot?.path, forKey: "modelRoot")
        defaults.set(outputDirectory.path, forKey: "outputDirectory")
        defaults.set(try? JSONEncoder().encode(parameters), forKey: "parameters")
        defaults.set(try? JSONEncoder().encode(loras), forKey: "loras")
        defaults.set(try? JSONEncoder().encode(stepRates), forKey: "stepRates")
    }

    private var tasksURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "LingXiStudio")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appending(path: "tasks.json")
    }

    private func saveTasks() {
        guard !loading else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try? encoder.encode(Array(tasks.prefix(500))).write(to: tasksURL, options: .atomic)
    }

    private func loadTasks() -> [TaskRecord] {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let data = try? Data(contentsOf: tasksURL) else { return [] }
        return (try? decoder.decode([TaskRecord].self, from: data)) ?? []
    }
}
