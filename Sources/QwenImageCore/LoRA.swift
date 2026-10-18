import Foundation
import MLX

/// A LoRA adapter file, the strength it is applied with, and whether it is active.
public struct LoRASpec: Codable, Sendable, Hashable {
    public var path: String
    public var scale: Float
    public var enabled: Bool

    public init(path: String, scale: Float = 1.0, enabled: Bool = true) {
        self.path = path
        self.scale = scale
        self.enabled = enabled
    }

    public var name: String { URL(fileURLWithPath: path).lastPathComponent }
}

public enum LoRAError: Error, CustomStringConvertible {
    case unreadable(String)
    case malformed(file: String, reason: String)
    case unmatched(file: String, keys: [String])
    case shapeMismatch(file: String, key: String, base: [Int], adapter: [Int])
    case unknownAdapter(UUID)

    public var description: String {
        switch self {
        case .malformed(let file, let reason): "Invalid LoRA \(file): \(reason)"
        case .unreadable(let file): "LoRA file unreadable: \(file)"
        case .unmatched(let file, let keys):
            "LoRA \(file) targets \(keys.count) layer(s) this model does not have, e.g. \(keys.prefix(3).joined(separator: ", "))"
        case .shapeMismatch(let file, let key, let base, let adapter):
            "LoRA \(file) does not fit \(key): base weight \(base), adapter factors \(adapter)"
        case .unknownAdapter(let id): "No loaded LoRA with id \(id)"
        }
    }
}

/// One adapter loaded from disk, kept separate from the base model. Its factors are never merged into base weights.
public final class LoRAAdapter: @unchecked Sendable {
    /// Low-rank factors of one target layer, stored exactly as in the file (no rescaling, no dtype change).
    struct Factors {
        let a: MLXArray  // [r, in]
        let b: MLXArray  // [out, r]
        /// PEFT `lora_alpha / r` for this layer.
        let alphaOverRank: Float
    }

    public let path: String
    /// Factors keyed by the base weight they adapt, e.g. `transformer_blocks.3.attn.to_q`.
    let targets: [String: Factors]

    public var name: String { URL(fileURLWithPath: path).lastPathComponent }
    public var targetCount: Int { targets.count }
    public var rank: Int { targets.values.first?.a.dim(0) ?? 0 }

    /// Loads PEFT/diffusers keys (`<prefix><module>.lora_A.weight`, `.lora_B.weight`, optional `.alpha`).
    public init(path: String, prefix: String = "transformer.") throws {
        self.path = path
        guard let (arrays, metadata) = try? loadArraysAndMetadata(url: URL(fileURLWithPath: path)) else {
            throw LoRAError.unreadable(path)
        }
        let peftAlpha = Self.peftAlpha(metadata)
        var targets: [String: Factors] = [:]
        let factorKeys = arrays.keys.filter { $0.hasSuffix(".lora_A.weight") || $0.hasSuffix(".lora_B.weight") }
        guard !factorKeys.isEmpty else { throw LoRAError.malformed(file: path, reason: "No PEFT LoRA factors found") }
        let knownKeys = Set(factorKeys + factorKeys.filter { $0.hasSuffix(".lora_A.weight") }.map {
            String($0.dropLast(".lora_A.weight".count)) + ".alpha"
        })
        let unsupported = arrays.keys.filter { !knownKeys.contains($0) }.sorted()
        guard unsupported.isEmpty else { throw LoRAError.malformed(file: path, reason: "Unsupported tensors: \(unsupported.prefix(3))") }
        for key in factorKeys.sorted() where key.hasSuffix(".lora_A.weight") {
            let module = String(key.dropLast(".lora_A.weight".count))
            guard let a = arrays[key], let b = arrays[module + ".lora_B.weight"],
                a.ndim == 2, b.ndim == 2, a.dim(0) > 0, a.dim(0) == b.dim(1),
                a.dim(1) > 0, b.dim(0) > 0,
                [DType.float16, .bfloat16, .float32].contains(a.dtype), a.dtype == b.dtype
            else { throw LoRAError.malformed(file: path, reason: "Missing or invalid factors for \(module)") }
            let rank = a.dim(0)
            if let alpha = arrays[module + ".alpha"], alpha.size != 1 {
                throw LoRAError.malformed(file: path, reason: "Non-scalar alpha for \(module)")
            }
            let alpha = arrays[module + ".alpha"]?.item(Float.self) ?? peftAlpha ?? Float(rank)
            guard alpha.isFinite else { throw LoRAError.malformed(file: path, reason: "Non-finite alpha for \(module)") }
            let sourcePrefix = [prefix, "diffusion_model."].first { module.hasPrefix($0) }
            let target = sourcePrefix.map { String(module.dropFirst($0.count)) } ?? module
            var mapped = [(target, b)]
            if target.hasSuffix(".img_mlp.gate_up") {
                guard b.dim(0).isMultiple(of: 2) else {
                    throw LoRAError.malformed(file: path, reason: "Odd gate_up output size for \(module)")
                }
                let parent = String(target.dropLast("gate_up".count))
                let half = b.dim(0) / 2
                // ComfyUI gate_up stores gate rows first; splitting B shares A without changing factor bytes.
                mapped = [(parent + "gate_layer", b[..<half]), (parent + "proj", b[half...])]
            }
            for (name, factorB) in mapped {
                guard targets[name] == nil else { throw LoRAError.malformed(file: path, reason: "Duplicate target \(name)") }
                targets[name] = Factors(a: a, b: factorB, alphaOverRank: alpha / Float(rank))
            }
        }
        for key in factorKeys where key.hasSuffix(".lora_B.weight") {
            let module = String(key.dropLast(".lora_B.weight".count))
            guard arrays[module + ".lora_A.weight"] != nil else {
                throw LoRAError.malformed(file: path, reason: "Missing A factor for \(module)")
            }
        }
        self.targets = targets
    }

    /// `lora_alpha` from diffusers/PEFT adapter metadata, when the file carries it.
    private static func peftAlpha(_ metadata: [String: String]) -> Float? {
        guard let raw = metadata["lora_adapter_metadata"],
            let json = try? JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any],
            let alpha = json.first(where: { $0.key.hasSuffix("lora_alpha") })?.value as? Double
        else { return nil }
        return Float(alpha)
    }
}

/// What a LoRA file contains and whether it fits the Qwen-Image-2.1 transformer, read without loading the model.
public struct LoRAInspection: Sendable {
    public var path: String
    public var fileBytes: Int64
    public var rank: Int
    /// PEFT `lora_alpha` (or the per-layer `.alpha`), when the file records one.
    public var alpha: Float?
    public var targetCount: Int
    public var dtype: String
    /// Short labels for what the adapter touches, e.g. "attention", "MLP", "timestep".
    public var coverage: [String]
    /// Reasons the adapter cannot be applied; empty when it fits.
    public var problems: [String]

    public var isCompatible: Bool { problems.isEmpty }
    public var name: String { URL(fileURLWithPath: path).lastPathComponent }
}

extension LoRAAdapter {
    /// Reads the adapter's header and checks every target against the transformer layout (names and shapes).
    public static func inspect(path: String) throws -> LoRAInspection {
        let adapter = try LoRAAdapter(path: path)
        let bytes = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int64) ?? 0
        var shapes: [String: [Int]] = [:]
        for item in QwenImageTransformer.loraLayout() {
            for (target, shape) in zip(item.targets, item.shapes) { shapes[target] = shape }
        }
        var problems: [String] = []
        if adapter.targets.isEmpty { problems.append("No PEFT LoRA layers (lora_A / lora_B) found") }
        let unmatched = adapter.targets.keys.filter { shapes[$0] == nil }.sorted()
        if !unmatched.isEmpty {
            problems.append("\(unmatched.count) layer(s) are not in Qwen-Image-2.1, e.g. \(unmatched.prefix(2).joined(separator: ", "))")
        }
        for (target, f) in adapter.targets.sorted(by: { $0.key < $1.key }) {
            guard let base = shapes[target] else { continue }
            if f.a.dim(1) != base[1] || f.b.dim(0) != base[0] || f.a.dim(0) != f.b.dim(1) {
                problems.append("\(target): factors \(f.b.dim(0))x\(f.a.dim(1)) do not fit \(base[0])x\(base[1])")
                break
            }
        }
        let keys = adapter.targets.keys
        var coverage: [String] = []
        if keys.contains(where: { $0.contains(".attn.") }) { coverage.append("attention") }
        if keys.contains(where: { $0.contains(".img_mlp.") }) { coverage.append("MLP") }
        if keys.contains(where: { $0.hasPrefix("modulation") }) { coverage.append("modulation") }
        if keys.contains(where: { $0.hasPrefix("time_text_embed") }) { coverage.append("timestep") }
        let first = adapter.targets.values.first
        return LoRAInspection(
            path: path, fileBytes: bytes, rank: adapter.rank,
            alpha: first.map { $0.alphaOverRank * Float($0.a.dim(0)) }, targetCount: adapter.targetCount,
            dtype: Set(adapter.targets.values.map { "\($0.a.dtype)" }).sorted().joined(separator: "/"),
            coverage: coverage, problems: problems)
    }
}

/// Ordered set of loaded adapters with per-adapter enable state and strength.
///
/// Loading, unloading, enabling, disabling and rescaling only change this list; base weights are never touched.
/// A transformer picks the active composition up through `QwenImageTransformer.bind(_:)`.
public final class LoRAStack: @unchecked Sendable {
    public struct Entry: Sendable {
        public let id: UUID
        public let adapter: LoRAAdapter
        public var scale: Float
        public var enabled: Bool
    }

    public private(set) var entries: [Entry] = []
    /// Bumped on every change so bound transformers know to rebuild their LoRA bindings.
    public private(set) var revision = 0

    public init() {}

    @discardableResult
    public func load(_ spec: LoRASpec) throws -> UUID {
        let entry = Entry(id: UUID(), adapter: try LoRAAdapter(path: spec.path), scale: spec.scale, enabled: spec.enabled)
        entries.append(entry)
        revision += 1
        return entry.id
    }

    public func unload(_ id: UUID) throws {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { throw LoRAError.unknownAdapter(id) }
        entries.remove(at: index)
        revision += 1
    }

    public func setEnabled(_ id: UUID, _ enabled: Bool) throws {
        try update(id) { $0.enabled = enabled }
    }

    public func setScale(_ id: UUID, _ scale: Float) throws {
        try update(id) { $0.scale = scale }
    }

    /// Adapters that contribute to the forward pass, in application order.
    public var active: [Entry] { entries.filter { $0.enabled && $0.scale != 0 } }

    public var specs: [LoRASpec] {
        entries.map { LoRASpec(path: $0.adapter.path, scale: $0.scale, enabled: $0.enabled) }
    }

    private func update(_ id: UUID, _ change: (inout Entry) -> Void) throws {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { throw LoRAError.unknownAdapter(id) }
        change(&entries[index])
        revision += 1
    }
}

/// Active LoRA for a group of projections that read the same input (e.g. Q/K/V).
///
/// Each target's update is added exactly as PEFT does: `y + (B (A x)) * scaling` with `scaling = scale * alpha / r`.
/// The update stays a separate branch: `addMM` onto the base output measured 3x slower on M5, and one fused
/// down-projection per group was no faster in the full model than one per target.
struct LoRAGroup {
    struct Part {
        let a: MLXArray  // [r, in]
        let b: MLXArray  // [out, r]
        let scale: Float
    }

    /// Per target in the group (indexed like the group's targets), the adapters touching it, in stack order.
    let parts: [[Part]]

    var arrays: [MLXArray] { parts.flatMap { $0.flatMap { [$0.a, $0.b] } } }

    /// `base[i]` are the base outputs of each target for input `x`.
    func apply(_ x: MLXArray, to base: [MLXArray]) -> [MLXArray] {
        zip(base, parts).map { y, parts in
            // Mixed native factors may promote the adapter branch; keep the base projection's output dtype.
            parts.reduce(y) { y, part in
                let update = matmul(matmul(x, part.a.T), part.b.T)
                return y + (part.scale == 1 ? update : update * part.scale)
            }.asType(y.dtype)
        }
    }

    /// Builds the group for `targets` from the active adapters, or `nil` when none of them touches the group.
    static func make(targets: [String], baseShapes: [[Int]], active: [LoRAStack.Entry]) throws -> LoRAGroup? {
        var parts: [[Part]] = targets.map { _ in [] }
        for entry in active {
            for (index, target) in targets.enumerated() {
                guard let f = entry.adapter.targets[target] else { continue }
                let base = baseShapes[index]
                guard f.a.dim(1) == base[1], f.b.dim(0) == base[0], f.a.dim(0) == f.b.dim(1) else {
                    throw LoRAError.shapeMismatch(
                        file: entry.adapter.name, key: target, base: base, adapter: [f.b.dim(0), f.a.dim(1)])
                }
                parts[index].append(Part(a: f.a, b: f.b, scale: entry.scale * f.alphaOverRank))
            }
        }
        return parts.allSatisfy(\.isEmpty) ? nil : LoRAGroup(parts: parts)
    }
}
