import Foundation
import QwenImageCore

typealias CreationMode = GenerationMode

extension GenerationMode: Identifiable {
    public var id: String { rawValue }

    var title: String {
        switch self {
        case .textToImage: "文生图"
        case .reference: "参考图创作"
        case .imageToImage: "图生图"
        case .edit: "图像编辑"
        case .inpaint: "局部重绘"
        }
    }

    var symbol: String {
        switch self {
        case .textToImage: "text.below.photo"
        case .reference: "photo.stack"
        case .imageToImage: "photo.on.rectangle.angled"
        case .edit: "wand.and.stars"
        case .inpaint: "paintbrush.pointed"
        }
    }

    /// Whether the mode is the official Qwen-Image-2.1 pipeline or a generic method layered on it.
    var isOfficial: Bool { self == .textToImage || self == .reference || self == .edit }

    var methodNote: String {
        switch self {
        case .textToImage: "官方管线：提示词 → 全新图像"
        case .reference: "官方管线：参考图作为视觉条件（图像不作为初始状态）"
        case .edit: "官方管线：主图作为视觉条件，按指令整体编辑"
        case .imageToImage: "通用方法（SDEdit）：初始图加噪后按强度重绘，Qwen-Image-2.1 无官方图生图管线"
        case .inpaint: "编辑条件 + 逐步蒙版潜变量混合（与 diffusers Edit-Inpaint 同法），选区外像素按羽化蒙版保留"
        }
    }

    /// The prompt field reads as a description or as an edit instruction.
    var usesInstruction: Bool { self == .edit || self == .inpaint }

    var needsConditions: Bool { self == .reference || self == .edit || self == .inpaint }
    var needsInitial: Bool { self == .imageToImage }
    var needsMask: Bool { self == .inpaint }
    var hasStrength: Bool { self == .imageToImage || self == .edit || self == .inpaint }

    var inputsTitle: String {
        switch self {
        case .reference: "参考图"
        case .edit, .inpaint: "主图与附加参考"
        case .imageToImage: "初始图"
        case .textToImage: "参考图"
        }
    }
}

/// Sampling schedules the executor has verified, each with the step count it requires.
enum SchedulePreset: String, CaseIterable, Identifiable, Codable {
    case standard, turbo

    var id: String { rawValue }
    var title: String { self == .standard ? "标准" : "Turbo" }
    var detail: String { self == .standard ? "linear 网格 + 终端拉伸" : "viggle_turbo 6 节点" }
    var schedule: SamplerSchedule { self == .standard ? .linear : .viggleTurbo }
}

/// Every user-editable generation parameter, persisted between launches and recorded with each task.
struct GenerationParameters: Codable, Equatable {
    enum SeedMode: String, Codable, CaseIterable { case random, fixed }
    enum BatchSeeds: String, Codable, CaseIterable { case increment, random }
    enum SizePolicy: String, Codable, CaseIterable { case followImage, manual }

    var prompt = ""
    /// Edit instructions are kept apart from descriptions, so switching modes never loses either.
    var instruction = ""
    var negativePrompt = ""
    var width = 1024
    var height = 1024
    var sizePolicy: SizePolicy = .followImage
    var steps = 40
    var guidance: Float = 1.0
    var seedMode: SeedMode = .random
    var seed: UInt64 = 42
    var count = 1
    var batchSeeds: BatchSeeds = .increment
    var preset: SchedulePreset = .standard
    var imageStrength: Float = 0.6
    var editStrength: Float = 1.0
    var inpaintStrength: Float = 1.0
    var maskFeather = 8
    var maskInvert = false
    var keepUnmasked = true

    init() {}

    init(from decoder: Decoder) throws {
        // Every field is optional on disk so older saved parameters keep loading.
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = GenerationParameters()
        func v<T: Decodable>(_ key: CodingKeys, _ fallback: T) -> T { (try? c.decodeIfPresent(T.self, forKey: key)) ?? fallback }
        prompt = v(.prompt, d.prompt)
        instruction = v(.instruction, d.instruction)
        negativePrompt = v(.negativePrompt, d.negativePrompt)
        width = v(.width, d.width)
        height = v(.height, d.height)
        sizePolicy = v(.sizePolicy, d.sizePolicy)
        steps = v(.steps, d.steps)
        guidance = v(.guidance, d.guidance)
        seedMode = v(.seedMode, d.seedMode)
        seed = v(.seed, d.seed)
        count = v(.count, d.count)
        batchSeeds = v(.batchSeeds, d.batchSeeds)
        preset = v(.preset, d.preset)
        imageStrength = v(.imageStrength, d.imageStrength)
        editStrength = v(.editStrength, d.editStrength)
        inpaintStrength = v(.inpaintStrength, d.inpaintStrength)
        maskFeather = v(.maskFeather, d.maskFeather)
        maskInvert = v(.maskInvert, d.maskInvert)
        keepUnmasked = v(.keepUnmasked, d.keepUnmasked)
    }

    func text(for mode: GenerationMode) -> String { mode.usesInstruction ? instruction : prompt }

    func strength(for mode: GenerationMode) -> Float {
        switch mode {
        case .imageToImage: imageStrength
        case .edit: editStrength
        case .inpaint: inpaintStrength
        default: 1
        }
    }

    /// True CFG runs a second, negative branch every step; it only exists above 1 with a negative prompt.
    var usesTrueCFG: Bool { guidance > 1 && !negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var imageTokens: Int { (width / 16) * (height / 16) }
    var effectiveSteps: Int { preset == .turbo ? FlowMatchSchedule.viggleTurboNodes.count : steps }
}

struct SizePreset: Identifiable, Hashable {
    let label: String
    let width: Int
    let height: Int
    var id: String { "\(width)x\(height)" }

    static let all: [SizePreset] = [
        .init(label: "1:1", width: 1024, height: 1024),
        .init(label: "4:3", width: 1152, height: 864),
        .init(label: "3:4", width: 864, height: 1152),
        .init(label: "3:2", width: 1216, height: 800),
        .init(label: "2:3", width: 800, height: 1216),
        .init(label: "16:9", width: 1344, height: 768),
        .init(label: "9:16", width: 768, height: 1344),
        .init(label: "大 1:1", width: 1328, height: 1328),
    ]
}

/// An input image the user added. The app keeps its own copy (content-addressed), so later edits or moves of the
/// original never change what a queued task or a recorded result refers to.
struct InputImage: Identifiable, Codable, Equatable, Hashable {
    var id = UUID()
    var storedPath: String
    var sourcePath: String?
    var sha256: String
    var width: Int
    var height: Int

    var name: String {
        sourcePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? URL(fileURLWithPath: storedPath).lastPathComponent
    }
    var url: URL { URL(fileURLWithPath: storedPath) }

    func record(role: InputImageRecord.Role) -> InputImageRecord {
        InputImageRecord(role: role, path: storedPath, source: sourcePath, sha256: sha256, width: width, height: height)
    }
}

/// A LoRA entry in the workbench list. Changes apply to tasks queued afterwards; a queued task keeps its snapshot.
struct LoRAItem: Identifiable, Codable, Equatable {
    enum Purpose: String, Codable, CaseIterable {
        case acceleration, style, character, other

        var label: String {
            switch self {
            case .acceleration: "加速"
            case .style: "风格"
            case .character: "角色"
            case .other: "其他"
            }
        }

        var symbol: String {
            switch self {
            case .acceleration: "hare.fill"
            case .style: "paintpalette.fill"
            case .character: "person.fill"
            case .other: "square.stack.3d.up.fill"
            }
        }
    }

    var id = UUID()
    var path: String
    var scale: Float = 1.0
    var enabled = true
    var purpose: Purpose
    var triggerWords = ""
    var source = "本地文件"
    // Inspection results; refreshed on launch and on import.
    var fileBytes: Int64 = 0
    var rank = 0
    var targetCount = 0
    var coverage: [String] = []
    var problems: [String] = []

    var name: String { URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent }
    var isUsable: Bool { problems.isEmpty && FileManager.default.fileExists(atPath: path) }

    init(path: String) {
        self.path = path
        let lower = URL(fileURLWithPath: path).lastPathComponent.lowercased()
        purpose = ["turbo", "lightning", "step", "fast", "hyper"].contains(where: lower.contains) ? .acceleration : .other
    }

    init(from decoder: Decoder) throws {
        // Fields added later default instead of failing the whole saved list.
        let c = try decoder.container(keyedBy: CodingKeys.self)
        path = try c.decode(String.self, forKey: .path)
        id = (try? c.decodeIfPresent(UUID.self, forKey: .id)) ?? UUID()
        scale = (try? c.decodeIfPresent(Float.self, forKey: .scale)) ?? 1
        enabled = (try? c.decodeIfPresent(Bool.self, forKey: .enabled)) ?? true
        purpose = (try? c.decodeIfPresent(Purpose.self, forKey: .purpose)) ?? .other
        triggerWords = (try? c.decodeIfPresent(String.self, forKey: .triggerWords)) ?? ""
        source = (try? c.decodeIfPresent(String.self, forKey: .source)) ?? "本地文件"
    }

    mutating func refresh() {
        guard FileManager.default.fileExists(atPath: path) else {
            problems = ["文件不存在（已移动或删除）"]
            return
        }
        do {
            let info = try LoRAAdapter.inspect(path: path)
            fileBytes = info.fileBytes
            rank = info.rank
            targetCount = info.targetCount
            coverage = info.coverage
            problems = info.problems
        } catch {
            problems = ["\(error)"]
        }
    }

    var spec: LoRASpec { LoRASpec(path: path, scale: scale, enabled: enabled) }
}

/// Saved LoRA combination.
struct LoRAPreset: Identifiable, Codable, Equatable {
    var id = UUID()
    var name: String
    var entries: [LoRASpec]
}

/// One queued or finished generation task (`count` images with one parameter snapshot).
struct TaskRecord: Identifiable, Codable {
    enum Status: Codable, Equatable {
        case waiting
        case running
        case done
        case failed(String)
        case cancelled
        case interrupted

        var label: String {
            switch self {
            case .waiting: "等待"
            case .running: "运行中"
            case .done: "完成"
            case .failed: "失败"
            case .cancelled: "已取消"
            case .interrupted: "已中断（应用退出）"
            }
        }

        var isFinished: Bool { self != .waiting && self != .running }
    }

    var id = UUID()
    var createdAt = Date()
    var mode: CreationMode = .textToImage
    var modelName: String
    var modelPath: String
    var parameters: GenerationParameters
    var loras: [LoRASpec]
    var conditions: [InputImage] = []
    var initial: InputImage?
    var mask: InputImage?
    var status: Status = .waiting
    var outputs: [String] = []
    /// Seeds actually used, one per finished or running image.
    var seeds: [UInt64] = []
    var log: [String] = []
    var startedAt: Date?
    var finishedAt: Date?

    init(mode: CreationMode, modelName: String, modelPath: String, parameters: GenerationParameters, loras: [LoRASpec],
         conditions: [InputImage], initial: InputImage?, mask: InputImage?) {
        self.mode = mode
        self.modelName = modelName
        self.modelPath = modelPath
        self.parameters = parameters
        self.loras = loras
        self.conditions = conditions
        self.initial = initial
        self.mask = mask
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        mode = (try? c.decodeIfPresent(CreationMode.self, forKey: .mode)) ?? .textToImage
        modelName = try c.decode(String.self, forKey: .modelName)
        modelPath = try c.decode(String.self, forKey: .modelPath)
        parameters = try c.decode(GenerationParameters.self, forKey: .parameters)
        loras = try c.decode([LoRASpec].self, forKey: .loras)
        conditions = (try? c.decodeIfPresent([InputImage].self, forKey: .conditions)) ?? []
        initial = try? c.decodeIfPresent(InputImage.self, forKey: .initial)
        mask = try? c.decodeIfPresent(InputImage.self, forKey: .mask)
        status = try c.decode(Status.self, forKey: .status)
        outputs = try c.decode([String].self, forKey: .outputs)
        seeds = try c.decode([UInt64].self, forKey: .seeds)
        log = try c.decode([String].self, forKey: .log)
        startedAt = try c.decodeIfPresent(Date.self, forKey: .startedAt)
        finishedAt = try c.decodeIfPresent(Date.self, forKey: .finishedAt)
    }

    var title: String {
        let text = parameters.text(for: mode).trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? "（空提示词）" : String(text.prefix(60))
    }
}

/// Live state of the running image, published by the engine.
struct LiveProgress {
    var taskID: UUID
    var imageIndex: Int
    var imageCount: Int
    var stage: GenerationStage
    var fraction: Double
    var step: Int
    var totalSteps: Int
    var secondsPerStep: Double?
    var elapsed: TimeInterval
    var mlxActive: Int
    var mlxPeak: Int

    var stageLabel: String { stage.label }

    var remaining: Double? {
        guard let rate = secondsPerStep else { return nil }
        let thisImage = Double(totalSteps - step) * rate
        let later = Double(imageCount - imageIndex - 1) * Double(totalSteps) * rate
        return thisImage + later
    }

    /// Denoising dominates the run time, so the bar follows the step count and reserves a small share for the rest.
    var overallFraction: Double {
        let perImage: Double =
            switch stage {
            case .validating, .loadingTextEncoder, .encodingPrompt: 0.02
            case .encodingImages: 0.04
            case .loadingTransformer, .prefill: 0.06
            case .denoising: 0.06 + 0.86 * Double(step) / Double(max(totalSteps, 1))
            case .loadingVAE, .decoding: 0.93
            case .done: 1
            }
        return (Double(imageIndex) + perImage) / Double(max(imageCount, 1))
    }
}

extension GenerationStage {
    var label: String {
        switch self {
        case .validating: "校验"
        case .loadingTextEncoder: "加载文本编码器"
        case .encodingPrompt: "提示词编码"
        case .encodingImages: "图像编码"
        case .loadingTransformer: "加载 Transformer"
        case .prefill: "条件前缀缓存"
        case .denoising: "去噪"
        case .loadingVAE: "加载 VAE"
        case .decoding: "VAE 解码"
        case .done: "保存"
        }
    }
}

/// Residency of a model component during the staged (32 GB balanced) run.
enum ComponentResidency: Equatable {
    case notLoaded, loading, resident, released

    var label: String {
        switch self {
        case .notLoaded: "未驻留"
        case .loading: "加载中"
        case .resident: "驻留"
        case .released: "已释放"
        }
    }
}

enum ModelStatus: Equatable {
    case unselected
    case validating
    case ready
    case inUse
    case error(String)

    var label: String {
        switch self {
        case .unselected: "未选择"
        case .validating: "校验中"
        case .ready: "已就绪"
        case .inUse: "正在推理"
        case .error: "错误"
        }
    }
}
