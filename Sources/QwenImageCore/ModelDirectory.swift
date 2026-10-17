import Foundation

/// A Qwen-Image-2.1 diffusers-layout model folder and its integrity report.
public struct ModelDirectory: Sendable {
    public struct Component: Sendable {
        public var name: String
        public var url: URL
        public var files: Int
        public var bytes: Int64
        public var problems: [String]
    }

    public let root: URL
    public var processor: URL { root.appending(path: "processor") }
    public var scheduler: URL { root.appending(path: "scheduler") }
    public var textEncoder: URL { root.appending(path: "text_encoder") }
    public var transformer: URL { root.appending(path: "transformer") }
    public var vae: URL { root.appending(path: "vae") }

    public enum ValidationError: Error, CustomStringConvertible {
        case incomplete([String])
        public var description: String {
            switch self {
            case .incomplete(let problems): "Model folder is incomplete:\n- " + problems.joined(separator: "\n- ")
            }
        }
    }

    public init(root: URL) {
        self.root = root
    }

    /// Checks the pipeline class, required files, and that every shard referenced by an index exists with data.
    public func validate() throws -> [Component] {
        var problems: [String] = []
        let fm = FileManager.default

        let indexURL = root.appending(path: "model_index.json")
        if let data = try? Data(contentsOf: indexURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            if let name = json["_class_name"] as? String, name != "QwenImage21Pipeline" {
                problems.append("model_index.json declares \(name), expected QwenImage21Pipeline")
            }
        } else {
            problems.append("model_index.json missing or unreadable")
        }

        let required: [(String, [String])] = [
            ("processor", ["tokenizer.json"]),
            ("scheduler", ["scheduler_config.json"]),
            ("text_encoder", ["config.json", "model.safetensors.index.json"]),
            ("transformer", ["config.json", "diffusion_pytorch_model.safetensors.index.json"]),
            ("vae", ["config.json", "diffusion_pytorch_model.safetensors"]),
        ]
        var components: [Component] = []
        for (name, files) in required {
            let dir = root.appending(path: name)
            var component = Component(name: name, url: dir, files: 0, bytes: 0, problems: [])
            for file in files where !fm.fileExists(atPath: dir.appending(path: file).path) {
                component.problems.append("\(name)/\(file) missing")
            }
            // Every shard named by a safetensors index must exist and be non-empty.
            for file in files where file.hasSuffix(".index.json") {
                guard let data = try? Data(contentsOf: dir.appending(path: file)),
                    let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                    let map = json["weight_map"] as? [String: String]
                else { continue }
                for shard in Set(map.values).sorted() {
                    let size = (try? fm.attributesOfItem(atPath: dir.appending(path: shard).path)[.size] as? Int64) ?? 0
                    if size <= 0 { component.problems.append("\(name)/\(shard) missing or empty") }
                }
            }
            let contents = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
            for url in contents where url.pathExtension == "safetensors" {
                component.files += 1
                component.bytes += Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
            problems.append(contentsOf: component.problems)
            components.append(component)
        }
        if !problems.isEmpty { throw ValidationError.incomplete(problems) }
        return components
    }
}
