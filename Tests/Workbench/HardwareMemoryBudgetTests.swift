import CryptoKit
import Foundation
import MLX
import XCTest
@testable import QwenImageCore

final class HardwareMemoryBudgetTests: XCTestCase {
    func testSoftBudgetPreservesFullFourLoRAReferencePixels() throws {
        guard ProcessInfo.processInfo.environment["LINGXI_RUN_MEMORY_BUDGET_TESTS"] == "1" else {
            throw XCTSkip("Opt-in full model memory scheduling experiment")
        }
        MLXRuntime.configure()
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let work = root.appending(path: ".tmp/runtime-v1-20261018/soft-budget").appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        try Data("\(getpid())".utf8).write(to: work.appending(path: "pid"))
        print("Soft budget evidence: \(work.path)")
        XCTAssertEqual(ProcessInfo.processInfo.thermalState, .nominal)
        let baseline = root.appending(path: ".tmp/runtime-v1-20261018/gui-profile/outputs/20261018-230045-42.png")
        let metadata = try GenerationMetadata.read(sidecarOf: baseline)
        let expected = try RGBAImage(contentsOf: baseline)
        var inputs = GenerationInputs()
        for input in metadata.request.inputs { inputs.conditions.append(try RGBAImage(contentsOf: URL(fileURLWithPath: input.path))) }
        let model = ModelDirectory(root: URL(fileURLWithPath: try XCTUnwrap(metadata.modelPath)))
        let previous = Memory.memoryLimit
        Memory.memoryLimit = min(previous, 20 << 30)
        defer { Memory.memoryLimit = previous }
        let eventURL = work.appending(path: "events.jsonl")
        FileManager.default.createFile(atPath: eventURL.path, contents: nil)
        let file = try FileHandle(forWritingTo: eventURL)
        defer { try? file.close() }
        let pipeline = try QwenImagePipeline(model: model)
        let result = try pipeline.generate(metadata.request, inputs: inputs, useEmbeddingCache: false,
            progress: { p in
                let row: [String: Any] = ["stage": p.stage.rawValue, "step": p.step, "fraction": p.fraction,
                    "uptime": ProcessInfo.processInfo.systemUptime, "active": p.memory.activeBytes,
                    "peak": p.memory.peakBytes, "footprint": ProcessMemory.footprint(), "memoryLimit": Memory.memoryLimit]
                do { try file.write(contentsOf: JSONSerialization.data(withJSONObject: row) + Data([10])) }
                catch { XCTFail("Could not preserve memory experiment: \(error)") }
            }, isCancelled: {
                FileManager.default.fileExists(atPath: work.appending(path: "resource-stop.json").path)
            })
        XCTAssertTrue(result.rgba == expected.pixels, "Every output pixel must match the frozen four-LoRA baseline")
        XCTAssertLessThan(Memory.activeMemory, 1 << 20)
        XCTAssertFalse(FileManager.default.fileExists(atPath: work.appending(path: "resource-stop.json").path))
        try ImageWriter.write(result, model: metadata.model, modelPath: metadata.modelPath, to: work.appending(path: "result.png"))
        let summary: [String: Any] = ["pixelIdentical": result.rgba == expected.pixels,
            "sha256": SHA256.hash(data: Data(result.rgba)).map { String(format: "%02x", $0) }.joined(),
            "postActive": Memory.activeMemory, "previousMemoryLimit": previous, "testMemoryLimit": Memory.memoryLimit]
        try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]).write(to: work.appending(path: "result.json"))
    }
}
