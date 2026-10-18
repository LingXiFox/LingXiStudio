import Foundation
import MLX
import XCTest
@testable import QwenImageCore

final class LoRAFormatTests: MLXTestCase {
    private func file(_ arrays: [String: MLXArray]) throws -> URL {
        let url = URL(fileURLWithPath: "/tmp", isDirectory: true).appending(path: "lora-\(UUID().uuidString).safetensors")
        try save(arrays: arrays, url: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func testPackedGateUpPreservesFactorBytesAndOrder() throws {
        MLXRuntime.configure()
        let a = MLXArray((0..<8).map { Float($0) / 16 }).reshaped(2, 4)
        let b = MLXArray((0..<24).map { Float($0) / 32 }).reshaped(12, 2)
        let path = try file(["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight": a,
                             "diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight": b,
                             "diffusion_model.transformer_blocks.0.img_mlp.gate_up.alpha": MLXArray(Float(4))])
        let adapter = try LoRAAdapter(path: path.path)
        let gate = try XCTUnwrap(adapter.targets["transformer_blocks.0.img_mlp.gate_layer"])
        let proj = try XCTUnwrap(adapter.targets["transformer_blocks.0.img_mlp.proj"])
        XCTAssertEqual(adapter.targetCount, 2)
        XCTAssertEqual(gate.alphaOverRank, 2)
        XCTAssertEqual(proj.alphaOverRank, 2)
        XCTAssertEqual(gate.a.asData(access: .copy).data, a.asData(access: .copy).data)
        XCTAssertEqual(gate.b.asData(access: .copy).data, b[..<6].asData(access: .copy).data)
        XCTAssertEqual(proj.b.asData(access: .copy).data, b[6...].asData(access: .copy).data)
        let x = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 4)
        let full = matmul(matmul(x, a.T), b.T)
        XCTAssertEqual(matmul(matmul(x, gate.a.T), gate.b.T).asArray(Float.self), full[0..., ..<6].asArray(Float.self))
        XCTAssertEqual(matmul(matmul(x, proj.a.T), proj.b.T).asArray(Float.self), full[0..., 6...].asArray(Float.self))
    }

    func testMalformedPairsAndDuplicateTargetsAreRejected() throws {
        let a = MLXArray.ones([2, 4], dtype: .bfloat16)
        let b = MLXArray.ones([4, 2], dtype: .bfloat16)
        for arrays in [
            ["x.lora_A.weight": a], ["x.lora_B.weight": b],
            ["x.lora_A.weight": a, "x.lora_B.weight": b, "x.weight": b],
            ["x.lora_A.weight": a, "x.lora_B.weight": MLXArray.ones([4, 3])],
            ["x.lora_A.weight": a, "x.lora_B.weight": b, "x.alpha": MLXArray([Float(1), 2])],
            ["transformer.x.lora_A.weight": a, "transformer.x.lora_B.weight": b,
             "diffusion_model.x.lora_A.weight": a, "diffusion_model.x.lora_B.weight": b],
            ["diffusion_model.x.img_mlp.gate_up.lora_A.weight": a,
             "diffusion_model.x.img_mlp.gate_up.lora_B.weight": MLXArray.ones([5, 2], dtype: .bfloat16)],
        ] {
            let path = try file(arrays)
            XCTAssertThrowsError(try LoRAAdapter(path: path.path))
        }
    }

    func testFourAdaptersKeepNativePrecisionOrderAndBaseBytes() throws {
        let target = "transformer_blocks.0.attn.to_q"
        let x = MLXArray([Float(1), -2, 3, -4]).reshaped(1, 4).asType(.bfloat16)
        let base = MLXArray.ones([1, 4], dtype: .bfloat16)
        let original = base.asData(access: .copy).data
        let stack = LoRAStack()
        var factors: [(MLXArray, MLXArray, Float)] = []
        for (i, dtype) in [DType.bfloat16, .float32, .float16, .float16].enumerated() {
            let a = (MLXArray.ones([2, 4]) * Float(i + 1) / 16).asType(dtype)
            let b = (MLXArray.ones([4, 2]) / Float(i + 2)).asType(dtype)
            let path = try file(["diffusion_model.\(target).lora_A.weight": a, "diffusion_model.\(target).lora_B.weight": b])
            let scale: Float = i == 0 ? 1 : 0.7
            try stack.load(LoRASpec(path: path.path, scale: scale))
            factors.append((a, b, scale))
            XCTAssertEqual(stack.entries.last?.adapter.targets[target]?.a.dtype, dtype)
        }
        var expected = base
        for (a, b, scale) in factors {
            let delta = matmul(matmul(x, a.T), b.T)
            expected = expected + (scale == 1 ? delta : delta * scale)
        }
        expected = expected.asType(base.dtype)
        let group = try XCTUnwrap(LoRAGroup.make(targets: [target], baseShapes: [[4, 4]], active: stack.active))
        XCTAssertEqual(group.apply(x, to: [base])[0].dtype, .bfloat16)
        XCTAssertEqual(group.apply(x, to: [base])[0].asData(access: .copy).data, expected.asData(access: .copy).data)
        XCTAssertEqual(base.asData(access: .copy).data, original)
        for entry in stack.entries { try stack.setEnabled(entry.id, false) }
        XCTAssertNil(try LoRAGroup.make(targets: [target], baseShapes: [[4, 4]], active: stack.active))
        for entry in stack.entries { try stack.setEnabled(entry.id, true) }
        let restored = try XCTUnwrap(LoRAGroup.make(targets: [target], baseShapes: [[4, 4]], active: stack.active))
        XCTAssertEqual(restored.apply(x, to: [base])[0].asData(access: .copy).data, expected.asData(access: .copy).data)
        for entry in stack.entries { try stack.unload(entry.id) }
        XCTAssertTrue(stack.active.isEmpty)
    }

    func testInspectionChecksCanonicalNamesAndShapes() throws {
        for prefix in ["transformer.", "diffusion_model.", ""] {
            let path = try file([prefix + "transformer_blocks.0.attn.to_q.lora_A.weight": MLXArray.ones([1, 4096], dtype: .float16),
                                 prefix + "transformer_blocks.0.attn.to_q.lora_B.weight": MLXArray.ones([4096, 1], dtype: .float16)])
            let inspection = try LoRAAdapter.inspect(path: path.path)
            XCTAssertTrue(inspection.isCompatible)
            XCTAssertEqual(inspection.targetCount, 1)
            XCTAssertEqual(inspection.dtype, "float16")
        }
        let bad = try file(["diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": MLXArray.ones([1, 2]),
                            "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": MLXArray.ones([2, 1])])
        XCTAssertFalse(try LoRAAdapter.inspect(path: bad.path).isCompatible)
    }
}
