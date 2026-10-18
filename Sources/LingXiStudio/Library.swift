import Foundation
import ImageIO
import QwenImageCore
import UniformTypeIdentifiers

// MARK: - Generation presets

/// A saved parameter set. `includes` selects which parts are applied when it is loaded.
struct GenerationPreset: Identifiable, Codable, Equatable {
    enum Part: String, Codable, CaseIterable, Identifiable {
        case prompt, inputs, model, loras, output

        var id: String { rawValue }
        var label: String {
            switch self {
            case .prompt: "提示词"
            case .inputs: "参考图 / 输入图"
            case .model: "模型"
            case .loras: "LoRA 组合"
            case .output: "输出目录"
            }
        }
    }

    var id = UUID()
    var name: String
    var favorite = false
    var isDefault = false
    var includes: Set<Part>
    var mode: GenerationMode
    var parameters: GenerationParameters
    var conditions: [InputImage] = []
    var initial: InputImage?
    var modelPath: String?
    var loras: [LoRASpec] = []
    var outputDirectory: String?
    var created = Date()

    /// Paths this preset refers to that no longer exist.
    var missingFiles: [String] {
        var paths: [String] = []
        if includes.contains(.model), let modelPath { paths.append(modelPath) }
        if includes.contains(.loras) { paths += loras.map(\.path) }
        if includes.contains(.inputs) { paths += (conditions + (initial.map { [$0] } ?? [])).map(\.storedPath) }
        return paths.filter { !FileManager.default.fileExists(atPath: $0) }
    }

    var summary: String {
        var parts = [mode.title, "\(parameters.width)×\(parameters.height)", "\(parameters.effectiveSteps) 步"]
        if parameters.seedMode == .fixed { parts.append("种子 \(parameters.seed)") }
        parts.append(includes.contains(.prompt) ? "含提示词" : "不含提示词")
        if includes.contains(.loras), !loras.isEmpty { parts.append("\(loras.count) 个 LoRA") }
        return parts.joined(separator: " · ")
    }
}

/// Reusable prompt snippet.
struct PromptPreset: Identifiable, Codable, Equatable {
    var id = UUID()
    var name: String
    var text: String
    var isInstruction = false
}

/// JSON files under Application Support.
enum Store {
    static func load<T: Decodable>(_ type: T.Type, _ name: String) -> T? {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let data = try? Data(contentsOf: AppPaths.support.appending(path: name)) else { return nil }
        return try? decoder.decode(type, from: data)
    }

    static func save<T: Encodable>(_ value: T, _ name: String) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(value).write(to: AppPaths.support.appending(path: name), options: .atomic)
    }
}

// MARK: - Model library

/// A known model folder.
struct ModelEntry: Identifiable, Codable, Equatable {
    var path: String
    var favorite = false
    var lastUsed: Date?
    var source: String = "本地文件夹"
    /// Last validation: total weight bytes and problems (empty = complete).
    var bytes: Int64 = 0
    var problems: [String] = []
    var validatedAt: Date?

    var id: String { path }
    var name: String { URL(fileURLWithPath: path).lastPathComponent }
    var url: URL { URL(fileURLWithPath: path) }
    var isComplete: Bool { validatedAt != nil && problems.isEmpty }
}

enum ModelScanner {
    /// Folders under `root` (two levels deep) that hold a Qwen-Image-2.1 diffusers model.
    static func scan(root: URL) -> [URL] {
        var found: [URL] = []
        let fm = FileManager.default
        func check(_ dir: URL) {
            let index = dir.appending(path: "model_index.json")
            guard let data = try? Data(contentsOf: index),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                json["_class_name"] as? String == "QwenImage21Pipeline"
            else { return }
            found.append(dir)
        }
        check(root)
        for level1 in (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? [] {
            check(level1)
            for level2 in (try? fm.contentsOfDirectory(at: level1, includingPropertiesForKeys: nil)) ?? [] {
                check(level2)
            }
        }
        return found
    }

    /// Hugging Face hub cache snapshots that hold a Qwen-Image-2.1 model.
    static func huggingFaceCache() -> [URL] {
        let env = ProcessInfo.processInfo.environment
        let hub = env["HF_HUB_CACHE"].map { URL(fileURLWithPath: $0) }
            ?? env["HF_HOME"].map { URL(fileURLWithPath: $0).appending(path: "hub") }
            ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/hub")
        let repos = (try? FileManager.default.contentsOfDirectory(at: hub, includingPropertiesForKeys: nil)) ?? []
        return repos.filter { $0.lastPathComponent.hasPrefix("models--") }.flatMap { repo -> [URL] in
            let snapshots = repo.appending(path: "snapshots")
            let dirs = (try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil)) ?? []
            return dirs.flatMap(scan)
        }
    }

    static func validate(_ entry: ModelEntry) -> ModelEntry {
        var entry = entry
        do {
            let components = try ModelDirectory(root: entry.url).validate()
            entry.bytes = components.reduce(0) { $0 + $1.bytes }
            entry.problems = []
        } catch let error as ModelDirectory.ValidationError {
            if case .incomplete(let problems) = error { entry.problems = problems }
        } catch {
            entry.problems = ["\(error)"]
        }
        entry.validatedAt = Date()
        return entry
    }
}

// MARK: - Export

struct ExportSettings: Codable, Equatable {
    enum Format: String, Codable, CaseIterable, Identifiable {
        case png, jpeg, heic, avif, tiff

        var id: String { rawValue }
        var label: String {
            switch self {
            case .png: "PNG（无损）"
            case .jpeg: "JPEG"
            case .heic: "HEIC"
            case .avif: "AVIF"
            case .tiff: "TIFF（无损）"
            }
        }
        var type: UTType {
            switch self {
            case .png: .png
            case .jpeg: .jpeg
            case .heic: .heic
            case .avif: UTType("public.avif") ?? .png
            case .tiff: .tiff
            }
        }
        var isLossy: Bool { self == .jpeg || self == .heic || self == .avif }
        var fileExtension: String { self == .jpeg ? "jpg" : rawValue }

        /// Formats ImageIO on this system can write (WebP is read-only in ImageIO, so it is not offered).
        static var available: [Format] {
            let writable = Set((CGImageDestinationCopyTypeIdentifiers() as? [String]) ?? [])
            return allCases.filter { writable.contains($0.type.identifier) }
        }
    }

    var format: Format = .png
    var quality: Double = 0.92
    var template = "{date}-{time}-{seed}"
    var includeRecord = true

    /// `{date} {time} {seed} {mode} {width} {height} {index} {name}`.
    func fileName(for item: Artwork, index: Int) -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd"
        let t = DateFormatter()
        t.dateFormat = "HHmmss"
        var name = template
        let values: [String: String] = [
            "{date}": f.string(from: item.created), "{time}": t.string(from: item.created),
            "{seed}": item.seed.map(String.init) ?? "noseed", "{mode}": item.mode?.rawValue ?? "unknown",
            "{width}": "\(item.pixelWidth)", "{height}": "\(item.pixelHeight)", "{index}": String(format: "%03d", index),
            "{name}": item.url.deletingPathExtension().lastPathComponent,
        ]
        for (key, value) in values { name = name.replacingOccurrences(of: key, with: value) }
        let cleaned = name.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-")
        return (cleaned.isEmpty ? "image" : cleaned) + "." + format.fileExtension
    }

    /// Writes one image in the chosen format, with its generation record next to it. Never overwrites: a numeric
    /// suffix is added when the name exists.
    func export(_ item: Artwork, index: Int, to folder: URL) throws -> URL {
        var target = folder.appending(path: fileName(for: item, index: index))
        var n = 2
        while FileManager.default.fileExists(atPath: target.path) {
            let base = target.deletingPathExtension().lastPathComponent
            target = folder.appending(path: "\(base)-\(n).\(format.fileExtension)")
            n += 1
        }
        if format == .png, item.url.pathExtension.lowercased() == "png" {
            try FileManager.default.copyItem(at: item.url, to: target)
        } else {
            guard let source = CGImageSourceCreateWithURL(item.url as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
                let dest = CGImageDestinationCreateWithURL(target as CFURL, format.type.identifier as CFString, 1, nil)
            else { throw ImageWriterError.encodeFailed(target.path) }
            let options: [CFString: Any] = format.isLossy ? [kCGImageDestinationLossyCompressionQuality: quality] : [:]
            CGImageDestinationAddImage(dest, image, options as CFDictionary)
            guard CGImageDestinationFinalize(dest) else { throw ImageWriterError.encodeFailed(target.path) }
        }
        if includeRecord {
            for suffix in ["json", "metadata.json"] {
                let sidecar = item.url.deletingPathExtension().appendingPathExtension(suffix)
                if FileManager.default.fileExists(atPath: sidecar.path) {
                    let out = target.deletingPathExtension().appendingPathExtension(suffix)
                    if !FileManager.default.fileExists(atPath: out.path) { try FileManager.default.copyItem(at: sidecar, to: out) }
                }
            }
        }
        return target
    }
}
