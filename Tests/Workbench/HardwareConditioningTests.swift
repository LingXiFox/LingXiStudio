import Foundation
import XCTest
@testable import QwenImageCore

/// Opt-in real-model checks; never substitute a synthetic model for the acceptance run.
final class HardwareConditioningTests: XCTestCase {
    func testSeedChangeUsesCacheAndMatchesUncachedPixels() throws {
        try verify(reference: false)
    }

    func testReferenceConditioningCacheMatchesUncachedPixels() throws {
        try verify(reference: true)
    }

    private func verify(reference: Bool) throws {
        guard ProcessInfo.processInfo.environment["LINGXI_RUN_MODEL_TESTS"] == "1" else {
            throw XCTSkip("Set LINGXI_RUN_MODEL_TESTS=1 for gated real-model acceptance")
        }
        MLXRuntime.configure()
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let work = root.appending(path: ".tmp/conditioning-performance-20261018").appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        let model = ModelDirectory(root: FileManager.default.homeDirectoryForCurrentUser.appending(path: "Models/Qwen-Image-2.1-BF16"))
        _ = try model.validate()
        let pipeline = try QwenImagePipeline(model: model)
        let lora = FileManager.default.homeDirectoryForCurrentUser.appending(path: "Models/Qwen-Image-2.1-LoRA/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors")
        var request = GenerationRequest(
            prompt: "A small blue ceramic fox on a wooden table", width: 256, height: 256, steps: 6, seed: 42,
            scheduler: .viggleTurbo, loras: [LoRASpec(path: lora.path, scale: 1)], mode: reference ? .reference : .textToImage)
        var inputs = GenerationInputs()
        if reference {
            let pixels: [UInt8] = (0..<256).flatMap { (index: Int) -> [UInt8] in
                [UInt8(index % 256), UInt8((index * 3) % 256), 127, 255]
            }
            inputs.conditions = [RGBAImage(width: 16, height: 16, pixels: pixels)]
        }
        var outputs: [GeneratedImage] = []
        for index in 0..<3 {
            let gate = Process()
            gate.executableURL = URL(fileURLWithPath: "/bin/zsh")
            gate.arguments = [root.appending(path: "Scripts/thermal-gate.sh").path]
            try gate.run()
            gate.waitUntilExit()
            XCTAssertEqual(gate.terminationStatus, 0)
            guard gate.terminationStatus == 0 else { return }
            request.seed = index == 0 ? 42 : 43
            var stages: [GenerationStage] = []
            let output = try autoreleasepool {
                try pipeline.generate(request, inputs: inputs, useEmbeddingCache: index != 2) {
                    stages.append($0.stage)
                    if $0.stage == .loadingVAE && $0.fraction == 0 {
                        XCTAssertLessThan($0.memory.activeBytes, 1 << 30, "DiT weights must be released before VAE loading")
                    }
                }
            }
            XCTAssertEqual(output.conditioningCacheHit, index == 1)
            XCTAssertEqual(stages.contains(.loadingTextEncoder), index != 1)
            try ImageWriter.write(output, model: model.root.lastPathComponent,
                to: work.appending(path: "cache-\(reference ? "reference" : "text")-\(index).png"))
            outputs.append(output)
        }
        XCTAssertEqual(outputs[1].rgba, outputs[2].rgba, "Cached and uncached full six-step outputs must match byte for byte")
    }
}
