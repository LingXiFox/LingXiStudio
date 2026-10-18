import Foundation
import MLX

public enum WeightError: Error, CustomStringConvertible {
    case invalidBatch(Int)
    case missingDirectory(String)
    case missingTensor(String, component: String)
    case shapeMismatch(String, expected: [Int], actual: [Int])

    public var description: String {
        switch self {
        case .invalidBatch(let batch): "Loading batch must be 4, 8 or 16 (got \(batch))"
        case .missingDirectory(let path): "Model component directory not found: \(path)"
        case .missingTensor(let key, let component): "Tensor \(key) missing in \(component)"
        case .shapeMismatch(let key, let expected, let actual):
            "Tensor \(key) has shape \(actual), expected \(expected)"
        }
    }
}

/// Lazily-loaded view over every `.safetensors` shard of one model component.
///
/// MLX loads safetensors lazily: a tensor's bytes are read only when it is first evaluated, so tensors
/// that are never requested (the vision tower and `lm_head` of the text encoder) never occupy memory.
struct WeightStore {
    let component: String
    private var tensors: [String: MLXArray]

    init(directory: URL, component: String) throws {
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw WeightError.missingDirectory(directory.path)
        }
        self.component = component
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var tensors: [String: MLXArray] = [:]
        for file in files {
            tensors.merge(try loadArrays(url: file)) { first, _ in first }
        }
        self.tensors = tensors
    }

    var keys: Dictionary<String, MLXArray>.Keys { tensors.keys }

    func contains(_ key: String) -> Bool { tensors[key] != nil }

    /// Returns the tensor cast to `dtype` (when given), checking its shape when `shape` is given.
    func get(_ key: String, dtype: DType? = nil, shape: [Int]? = nil) throws -> MLXArray {
        guard var value = tensors[key] else { throw WeightError.missingTensor(key, component: component) }
        if let shape, value.shape != shape {
            throw WeightError.shapeMismatch(key, expected: shape, actual: value.shape)
        }
        if let dtype, value.dtype != dtype { value = value.asType(dtype) }
        return value
    }

    /// Transfer ownership so a streamed layer is freed after use rather than retained by the source dictionary.
    mutating func take(_ key: String, shape: [Int]? = nil) throws -> MLXArray {
        let value = try get(key, shape: shape)
        tensors[key] = nil
        return value
    }
}

/// Bias-free linear layer `y = x W^T`.
struct Linear {
    let weight: MLXArray

    init(_ weight: MLXArray) { self.weight = weight }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T)
    }
}

/// Collects every parameter of a component so it can be materialized in one pass.
protocol ParameterContainer {
    var parameters: [MLXArray] { get }
}

extension ParameterContainer {
    /// Reads all weights from disk now instead of on first use, so loading progress and memory are explicit.
    func materialize(batch: Int = 16, checkCancelled: () throws -> Void = {}, progress: ((Double) -> Void)? = nil) rethrows {
        let all = parameters
        var index = 0
        while index < all.count {
            try checkCancelled()
            let end = min(index + batch, all.count)
            eval(Array(all[index..<end]))
            index = end
            progress?(Double(index) / Double(all.count))
        }
    }
}
