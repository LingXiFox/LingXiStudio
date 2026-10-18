import Foundation
import MLX
import XCTest
@testable import QwenImageCore

final class HardwareLoRAStackTests: XCTestCase {
    func testFourNativeAdaptersKeepPixelsAndReleaseBetweenGenerations() throws {
        guard ProcessInfo.processInfo.environment["LINGXI_RUN_LORA_TESTS"] == "1" else {
            throw XCTSkip("Set LINGXI_RUN_LORA_TESTS=1 for real four-adapter residency acceptance")
        }
        MLXRuntime.configure()
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let work = root.appending(path: ".tmp/lora-reference-20261018/lifecycle").appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        let home = FileManager.default.homeDirectoryForCurrentUser
        let model = ModelDirectory(root: home.appending(path: "Models/Qwen-Image-2.1-BF16"))
        let pipeline = try QwenImagePipeline(model: model)
        let files = ["Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors",
                     "qwen2.1-detail-fix-2.0.safetensors", "canon_qwen21.safetensors", "samsung_qwen21.safetensors"]
        let originalOverride = ProcessInfo.processInfo.environment["LINGXI_DIT_CACHE"]
        defer {
            if let originalOverride { setenv("LINGXI_DIT_CACHE", originalOverride, 1) }
            else { unsetenv("LINGXI_DIT_CACHE") }
        }
        var previousPixels: [UInt8]?
        for count in 1...4 {
            let loras = files.prefix(count).map { LoRASpec(path: home.appending(path: "Models/Qwen-Image-2.1-LoRA/\($0)").path) }
            for lora in loras { XCTAssertTrue(try LoRAAdapter.inspect(path: lora.path).isCompatible) }
            let request = GenerationRequest(prompt: "A small blue ceramic fox on a wooden table", width: 256, height: 256,
                steps: 6, seed: 42, scheduler: .viggleTurbo, loras: loras)
            var controlPixels: [UInt8]?
            for cacheMB in ["0", "512"] {
                setenv("LINGXI_DIT_CACHE", cacheMB == "0" ? "0" : "1", 1)
                let gate = Process()
                gate.executableURL = URL(fileURLWithPath: "/bin/zsh")
                gate.arguments = [root.appending(path: "Scripts/thermal-gate.sh").path]
                try gate.run()
                gate.waitUntilExit()
                guard gate.terminationStatus == 0 else { XCTFail("Thermal gate failed"); return }
                let oldLimit = Memory.cacheLimit
                let image = try autoreleasepool {
                    try pipeline.generate(request) { progress in
                        if progress.stage == .loadingVAE && progress.fraction == 0 {
                            XCTAssertLessThan(progress.memory.activeBytes, 1 << 30, "All native adapter and DiT arrays must be released")
                        }
                    }
                }
                XCTAssertEqual(Memory.cacheLimit, oldLimit)
                if cacheMB == "0" {
                    if let previousPixels { XCTAssertNotEqual(image.rgba, previousPixels, "Each added real adapter must affect the output") }
                    previousPixels = image.rgba
                    controlPixels = image.rgba
                } else {
                    XCTAssertEqual(image.rgba, controlPixels, "Buffer cache policy must not change any pixel")
                }
                XCTAssertEqual(image.conditioningCacheHit, count > 1 || cacheMB == "512", "LoRA changes do not alter text conditioning")
                XCTAssertEqual(image.stepSeconds.count, 6)
                try ImageWriter.write(image, model: model.root.lastPathComponent,
                    to: work.appending(path: "stack-\(count)-cache\(cacheMB).png"))
                Stream.gpu.synchronize()
                Memory.clearCache()
                XCTAssertLessThan(Memory.activeMemory, 1 << 20, "Only independent CPU conditioning may remain between requests")
            }
        }
    }
}
