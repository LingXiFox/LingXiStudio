import Foundation
import QwenImageCore

/// The five creation modes of the product spec. Only text-to-image has an executor path today; the others stay
/// visible so the workbench layout is final, but cannot be selected until their model path exists.
enum CreationMode: String, CaseIterable, Identifiable, Codable {
    case textToImage, reference, imageToImage, edit, inpaint

    var id: String { rawValue }

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

    /// Why the mode cannot run yet, or `nil` when it can.
    var unavailableReason: String? {
        switch self {
        case .textToImage: nil
        case .reference, .edit:
            "需要 Qwen Image 2.1 的图像条件/编辑路径，原生执行器尚未实现"
        case .imageToImage: "需要 VAE 编码器与初始潜变量加噪，原生执行器尚未实现"
        case .inpaint: "需要编辑路径与蒙版条件，原生执行器尚未实现（规划于 P2）"
        }
    }
}

/// Sampling schedules the executor has verified, each with the step count it requires.
enum SchedulePreset: String, CaseIterable, Identifiable, Codable {
    case standard, turbo

    var id: String { rawValue }
    var title: String { self == .standard ? "标准（linear）" : "Turbo（viggle_turbo · 6 步）" }
    var schedule: SamplerSchedule { self == .standard ? .linear : .viggleTurbo }
}

/// Every user-editable generation parameter, persisted between launches and recorded with each task.
struct GenerationParameters: Codable, Equatable {
    enum SeedMode: String, Codable, CaseIterable { case random, fixed }
    enum BatchSeeds: String, Codable, CaseIterable { case increment, random }

    var prompt = ""
    var negativePrompt = ""
    var width = 1024
    var height = 1024
    var steps = 40
    var guidance: Float = 1.0
    var seedMode: SeedMode = .random
    var seed: UInt64 = 42
    var count = 1
    var batchSeeds: BatchSeeds = .increment
    var preset: SchedulePreset = .standard

    /// True CFG runs a second, negative branch every step; it only exists above 1 with a negative prompt.
    var usesTrueCFG: Bool { guidance > 1 && !negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var imageTokens: Int { (width / 16) * (height / 16) }
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
        .init(label: "1:1 大", width: 1328, height: 1328),
    ]
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
    }

    var id = UUID()
    var path: String
    var scale: Float = 1.0
    var enabled = true
    var purpose: Purpose
    var triggerWords = ""
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

    mutating func refresh() {
        guard FileManager.default.fileExists(atPath: path) else {
            problems = ["文件不存在"]
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
    var status: Status = .waiting
    var outputs: [String] = []
    /// Seeds actually used, one per finished or running image.
    var seeds: [UInt64] = []
    var log: [String] = []
    var startedAt: Date?
    var finishedAt: Date?

    var title: String {
        let text = parameters.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
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
}

extension GenerationStage {
    var label: String {
        switch self {
        case .validating: "校验"
        case .loadingTextEncoder: "加载文本编码器"
        case .encodingPrompt: "提示词编码"
        case .loadingTransformer: "加载 Transformer"
        case .prefill: "文本前缀缓存"
        case .denoising: "去噪"
        case .loadingVAE: "加载 VAE"
        case .decoding: "VAE 解码"
        case .done: "保存"
        }
    }
}

/// A finished image on disk with its generation record.
struct HistoryItem: Identifiable, Hashable {
    var url: URL
    var metadata: GenerationMetadata?
    var created: Date

    var id: URL { url }

    static func == (a: HistoryItem, b: HistoryItem) -> Bool { a.url == b.url }
    func hash(into hasher: inout Hasher) { hasher.combine(url) }
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
