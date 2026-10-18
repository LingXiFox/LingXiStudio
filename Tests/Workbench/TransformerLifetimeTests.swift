import Foundation
import MLX
import XCTest
@testable import QwenImageCore

final class TransformerLifetimeTests: XCTestCase {
    func testCompiledWeightsAreReleasedAcrossPromptShapes() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["LINGXI_TEST_DEVICE"] == "cpu", "Metal-only compiled graph lifetime regression; run locally on Apple Silicon")
        MLXRuntime.configure()
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = QwenImageTransformer.Config(dim: 512, heads: 8, headDim: 64, layers: 2,
            contextDim: 512, axesDims: [16, 24, 24])
        try makeWeights(directory, config: config)
        Stream.gpu.synchronize()
        Memory.clearCache()
        let baseline = Memory.activeMemory
        for length in [5, 2, 3] {
            try autoreleasepool { try run(directory, config: config, textLength: length) }
            Stream.gpu.synchronize()
            Memory.clearCache()
            XCTAssertLessThan(Memory.activeMemory - baseline, 1 << 20, "Compiled constants must not retain the model")
        }
    }

    @inline(never) private func run(_ directory: URL, config: QwenImageTransformer.Config, textLength: Int) throws {
        let model = try QwenImageTransformer(store: WeightStore(directory: directory, component: "test"), config: config)
        try model.materialize()
        let rope = model.rope(textLength: textLength, latentHeight: 2, latentWidth: 2)
        let cache = model.prefill(text: MLXArray.ones([1, textLength, config.contextDim], dtype: .bfloat16), rope: rope.text)
        let output = model(latents: MLXArray.ones([1, 4, 64], dtype: .bfloat16), sigma: 0.5, cache: cache, rope: rope.image)
        eval(output)
        model.discardCompiledStep()
    }

    @inline(never) private func makeWeights(_ directory: URL, config: QwenImageTransformer.Config) throws {
        let d = config.dim
        var arrays: [String: MLXArray] = [:]
        func add(_ name: String, _ shape: [Int]) {
            arrays[name + ".weight"] = (MLXArray.ones(shape, dtype: .bfloat16) * 0.01).asType(.bfloat16)
        }
        add("img_in", [d, 64])
        add("txt_in.text_norm", [d])
        add("txt_in.in_layer", [d, d])
        add("txt_in.out_layer", [d, d])
        add("time_text_embed.timestep_embedder.linear_1", [d, 256])
        add("time_text_embed.timestep_embedder.linear_2", [d, d])
        add("modulation.1", [4 * d, d])
        add("norm_out.linear", [d, d])
        add("proj_out", [64, d])
        for index in 0..<config.layers {
            let p = "transformer_blocks.\(index)"
            for name in ["to_q", "to_k", "to_v", "to_out.0"] { add("\(p).attn.\(name)", [d, d]) }
            for name in ["norm_q", "norm_k"] { add("\(p).attn.\(name)", [config.headDim]) }
            for name in ["gate_layer", "proj"] { add("\(p).img_mlp.\(name)", [d * 3, d]) }
            add("\(p).img_mlp.out", [d, d * 3])
        }
        try save(arrays: arrays, url: directory.appending(path: "weights.safetensors"))
    }
}
