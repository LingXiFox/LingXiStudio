import Foundation
import ImageIO
import QwenImageCore

/// A finished image on disk with whatever generation record exists for it.
struct Artwork: Identifiable, Hashable {
    enum Origin: Hashable { case native, legacy }

    var url: URL
    var created: Date
    var origin: Origin
    /// Record written by this app (`<name>.json`).
    var metadata: GenerationMetadata?
    /// Fields recovered from the old Gradio/MFLUX WebUI (`<name>.metadata.json`); missing ones stay nil.
    var legacy: LegacyRecord?
    var pixelWidth: Int
    var pixelHeight: Int

    var id: URL { url }

    static func == (a: Artwork, b: Artwork) -> Bool { a.url == b.url }
    func hash(into hasher: inout Hasher) { hasher.combine(url) }

    var prompt: String? { metadata?.request.prompt ?? legacy?.prompt }
    var seed: UInt64? { metadata?.request.seed ?? legacy?.seed }
    var steps: Int? { metadata?.request.steps ?? legacy?.steps }
    var guidance: Float? { metadata?.request.trueCFGScale ?? legacy?.guidance }
    var mode: GenerationMode? { metadata?.request.mode ?? (legacy != nil ? legacy?.mode : nil) }
    var modelName: String? { metadata?.model ?? legacy?.model }
    var loras: [LoRASpec] { metadata?.request.loras ?? legacy?.loras ?? [] }
    var hasInputImages: Bool { !(metadata?.request.inputs.isEmpty ?? true) || legacy?.hadReferenceImages == true }
    var isReusable: Bool { metadata != nil || legacy?.prompt != nil }
}

/// Generation fields of an image made by the previous WebUI (MFLUX sidecar `<name>.metadata.json`).
struct LegacyRecord: Hashable {
    var prompt: String?
    var negativePrompt: String?
    var seed: UInt64?
    var steps: Int?
    var guidance: Float?
    var scheduler: String?
    var loras: [LoRASpec]
    var model: String?
    var mode: GenerationMode?
    var hadReferenceImages: Bool

    init?(json: [String: Any]) {
        func str(_ k: String) -> String? { json[k] as? String }
        func num(_ k: String) -> Double? { (json[k] as? NSNumber)?.doubleValue }
        prompt = str("prompt")
        negativePrompt = str("negative_prompt") ?? str("negative")
        seed = num("seed").map { UInt64(max($0, 0)) }
        steps = num("steps").map { Int($0) }
        guidance = num("guidance").map { Float($0) }
        scheduler = str("scheduler")
        model = str("model") ?? str("model_path").map { URL(fileURLWithPath: $0).lastPathComponent }
        let paths = json["lora_paths"] as? [String] ?? []
        let scales = json["lora_scales"] as? [Double] ?? []
        loras = paths.enumerated().map { i, p in LoRASpec(path: p, scale: Float(i < scales.count ? scales[i] : 1)) }
        let images = (json["image_paths"] as? [String]) ?? (json["image_path"] as? String).map { [$0] } ?? []
        hadReferenceImages = !images.isEmpty
        mode = images.isEmpty ? .textToImage : nil
        if prompt == nil, seed == nil, steps == nil { return nil }
    }
}

enum GalleryScanner {
    static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "avif", "tif", "tiff"]

    /// Lists images of a folder with their records. Runs off the main thread.
    static func scan(_ folder: URL, origin: Artwork.Origin) -> [Artwork] {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: [.creationDateKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles])) ?? []
        return urls.filter { imageExtensions.contains($0.pathExtension.lowercased()) }.map { url in
            let values = try? url.resourceValues(forKeys: [.creationDateKey, .contentModificationDateKey])
            var size = (0, 0)
            if let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
            {
                size = ((props[kCGImagePropertyPixelWidth] as? Int) ?? 0, (props[kCGImagePropertyPixelHeight] as? Int) ?? 0)
            }
            let metadata = try? GenerationMetadata.read(sidecarOf: url)
            var legacy: LegacyRecord?
            if metadata == nil {
                let sidecar = url.deletingPathExtension().appendingPathExtension("metadata.json")
                if let data = try? Data(contentsOf: sidecar),
                    let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                {
                    legacy = LegacyRecord(json: json)
                }
            }
            return Artwork(
                url: url, created: values?.creationDate ?? values?.contentModificationDate ?? .distantPast,
                origin: origin, metadata: metadata, legacy: legacy, pixelWidth: size.0, pixelHeight: size.1)
        }
    }
}

/// Filter state of the gallery views.
struct GalleryFilter: Equatable {
    enum DateRange: String, CaseIterable { case all, today, week, month }

    var text = ""
    var mode: GenerationMode?
    var dateRange: DateRange = .all
    var favoritesOnly = false
    var withInputsOnly = false
    var size: String?  // "WxH"
    var model: String?

    func matches(_ item: Artwork, favorites: Set<String>) -> Bool {
        if favoritesOnly && !favorites.contains(item.url.path) { return false }
        if withInputsOnly && !item.hasInputImages { return false }
        if let mode, item.mode != mode { return false }
        if let size, "\(item.pixelWidth)x\(item.pixelHeight)" != size { return false }
        if let model, item.modelName != model { return false }
        switch dateRange {
        case .all: break
        case .today: if !Calendar.current.isDateInToday(item.created) { return false }
        case .week: if item.created < Date().addingTimeInterval(-7 * 86400) { return false }
        case .month: if item.created < Date().addingTimeInterval(-30 * 86400) { return false }
        }
        let q = text.trimmingCharacters(in: .whitespaces).lowercased()
        if !q.isEmpty {
            let hay = [(item.prompt ?? ""), item.seed.map(String.init) ?? "", item.url.lastPathComponent].joined(separator: " ")
                .lowercased()
            if !hay.contains(q) { return false }
        }
        return true
    }
}
