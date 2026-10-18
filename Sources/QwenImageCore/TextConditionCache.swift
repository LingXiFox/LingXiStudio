import CryptoKit
import Darwin
import Foundation
import MLX

/// One lossless CPU copy of the latest text/vision conditioning; never retains a model or an MLX graph.
struct TextConditionCache {
    struct Encoded {
        let embedding: MLXArray
        let imageRanges: [Range<Int>]
    }

    private struct Stored {
        let array: MLXArray.MLXArrayData
        let ranges: [Range<Int>]
    }

    private var entry: (key: Data, values: [Stored])?
    let byteLimit: Int

    init(byteLimit: Int = 128 << 20) { self.byteLimit = byteLimit }

    mutating func clear() { entry = nil }

    func lookup(_ key: Data) -> [Encoded]? {
        guard let entry, entry.key == key else { return nil }
        return entry.values.map { Encoded(embedding: MLXArray(data: $0.array), imageRanges: $0.ranges) }
    }

    mutating func store(_ values: [Encoded], key: Data) {
        guard !values.isEmpty, values.reduce(0, { $0 + $1.embedding.nbytes }) <= byteLimit else {
            entry = nil
            return
        }
        entry = (key, values.map { Stored(array: $0.embedding.asData(access: .copy), ranges: $0.imageRanges) })
    }

    static func key(revision: Data, request: GenerationRequest, conditions: [RGBAImage]) -> Data {
        var hash = SHA256()
        func bytes(_ value: Data) {
            var length = UInt64(value.count).littleEndian
            withUnsafeBytes(of: &length) { hash.update(bufferPointer: $0) }
            hash.update(data: value)
        }
        bytes(Data("conditioning-v1".utf8))
        bytes(revision)
        bytes(Data(request.prompt.utf8))
        bytes(Data((request.negativePrompt ?? "").utf8))
        bytes(Data(request.usesTrueCFG ? [1] : [0]))
        // Only these process settings affect text/vision arithmetic. Never collect the rest of the environment.
        for name in ["LINGXI_TE_RESIDUAL", "MLX_ENABLE_TF32"] {
            bytes(Data((ProcessInfo.processInfo.environment[name] ?? "default").utf8))
        }
        for image in conditions {
            bytes(Data("\(image.width)x\(image.height)".utf8))
            bytes(Data(image.pixels))
        }
        return Data(hash.finalize())
    }

    /// Device/inode/ctime invalidate replacements and in-place edits even when size and mtime are restored.
    /// Small configuration/tokenizer files are hashed; weight bytes are not reread on a cache hit.
    static func modelRevision(_ model: ModelDirectory) throws -> Data {
        var hash = SHA256()
        hash.update(data: Data(model.root.resolvingSymlinksInPath().path.utf8))
        for directory in [model.processor, model.textEncoder, model.scheduler] {
            let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter { ["json", "txt", "safetensors"].contains($0.pathExtension) }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            for file in files {
                var info = stat()
                guard stat(file.path, &info) == 0 else { throw CocoaError(.fileReadUnknown) }
                let stamp = "\(file.path)|\(info.st_dev)|\(info.st_ino)|\(info.st_size)|"
                    + "\(info.st_mtimespec.tv_sec):\(info.st_mtimespec.tv_nsec)|"
                    + "\(info.st_ctimespec.tv_sec):\(info.st_ctimespec.tv_nsec)\n"
                hash.update(data: Data(stamp.utf8))
                if file.pathExtension != "safetensors" { hash.update(data: try Data(contentsOf: file)) }
            }
        }
        return Data(hash.finalize())
    }
}
