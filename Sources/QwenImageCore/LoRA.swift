import Foundation
import MLX

/// A LoRA adapter file and the strength it is applied with.
public struct LoRASpec: Codable, Sendable, Hashable {
    public var path: String
    public var scale: Float

    public init(path: String, scale: Float = 1.0) {
        self.path = path
        self.scale = scale
    }

    public var name: String { URL(fileURLWithPath: path).lastPathComponent }
}

public enum LoRAError: Error, CustomStringConvertible {
    case unreadable(String)
    case unmatched(file: String, keys: [String])
    case shapeMismatch(file: String, key: String, base: [Int], delta: [Int])

    public var description: String {
        switch self {
        case .unreadable(let file): "LoRA file unreadable: \(file)"
        case .unmatched(let file, let keys):
            "LoRA \(file) targets \(keys.count) layer(s) this model does not have, e.g. \(keys.prefix(3).joined(separator: ", "))"
        case .shapeMismatch(let file, let key, let base, let delta):
            "LoRA \(file) does not fit \(key): base \(base), delta \(delta)"
        }
    }
}

extension WeightStore {
    /// Folds PEFT LoRA adapters into the stored base weights: `W' = W + scale * (alpha / r) * B A`.
    ///
    /// The product and the sum are computed in fp32 and rounded once to the weight dtype, so the merged model costs
    /// exactly as much per step as the base model. Every adapter key must map onto a base weight; an adapter that
    /// does not fit is an error rather than being skipped silently.
    mutating func bake(_ specs: [LoRASpec], prefix: String = "transformer.") throws {
        for spec in specs where spec.scale != 0 {
            let url = URL(fileURLWithPath: spec.path)
            guard let (arrays, metadata) = try? loadArraysAndMetadata(url: url) else {
                throw LoRAError.unreadable(spec.path)
            }
            let peftAlpha = Self.peftAlpha(metadata)
            var unmatched: [String] = []
            for key in arrays.keys.sorted() where key.hasSuffix(".lora_A.weight") {
                let module = String(key.dropLast(".lora_A.weight".count))
                let baseKey = (module.hasPrefix(prefix) ? String(module.dropFirst(prefix.count)) : module) + ".weight"
                guard let a = arrays[key], let b = arrays[module + ".lora_B.weight"], let base = tensors[baseKey]
                else {
                    unmatched.append(module)
                    continue
                }
                let rank = a.dim(0)
                let alpha = arrays[module + ".alpha"]?.item(Float.self) ?? peftAlpha ?? Float(rank)
                let delta = matmul(b.asType(.float32), a.asType(.float32)) * (spec.scale * alpha / Float(rank))
                guard delta.shape == base.shape else {
                    throw LoRAError.shapeMismatch(file: spec.name, key: baseKey, base: base.shape, delta: delta.shape)
                }
                let merged = (base.asType(.float32) + delta).asType(base.dtype)
                // Evaluated per layer so the fp32 temporaries never pile up.
                eval(merged)
                tensors[baseKey] = merged
            }
            if !unmatched.isEmpty { throw LoRAError.unmatched(file: spec.name, keys: unmatched) }
        }
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
