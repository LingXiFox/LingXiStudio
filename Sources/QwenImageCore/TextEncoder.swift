import Foundation
import MLX

/// Text stack of the Qwen3-VL encoder used by Qwen-Image-2.1 (text-only path, no vision tower).
///
/// Text-only inputs place the same position on all three interleaved M-RoPE axes, which makes M-RoPE identical
/// to standard 1D RoPE, so the fast RoPE kernel is used directly.
final class QwenTextEncoder: ParameterContainer {
    struct Config {
        var hiddenSize = 4096
        var layers = 36
        var heads = 32
        var kvHeads = 8
        var headDim = 128
        var intermediateSize = 12288
        var ropeTheta: Float = 5_000_000
        var eps: Float = 1e-6
        /// Keep the residual stream in fp32 (default). `false` reproduces the reference's pure bf16 stream and
        /// exists for parity tests (`LINGXI_TE_RESIDUAL=bf16`).
        var residualFloat32 = ProcessInfo.processInfo.environment["LINGXI_TE_RESIDUAL"] != "bf16"
    }

    struct Layer {
        let inputNorm: MLXArray
        let postNorm: MLXArray
        let q: Linear
        let k: Linear
        let v: Linear
        let o: Linear
        let qNorm: MLXArray
        let kNorm: MLXArray
        let gate: Linear
        let up: Linear
        let down: Linear
    }

    let config: Config
    let embedTokens: MLXArray
    let layers: [Layer]
    let finalNorm: MLXArray

    var parameters: [MLXArray] {
        [embedTokens, finalNorm]
            + layers.flatMap {
                [
                    $0.inputNorm, $0.postNorm, $0.q.weight, $0.k.weight, $0.v.weight, $0.o.weight, $0.qNorm, $0.kNorm,
                    $0.gate.weight, $0.up.weight, $0.down.weight,
                ]
            }
    }

    init(store: WeightStore, config: Config = Config()) throws {
        self.config = config
        let prefix = "model.language_model"
        let d = config.hiddenSize
        embedTokens = try store.get("\(prefix).embed_tokens.weight")
        finalNorm = try store.get("\(prefix).norm.weight", shape: [d])
        layers = try (0..<config.layers).map { index in
            let p = "\(prefix).layers.\(index)"
            return Layer(
                inputNorm: try store.get("\(p).input_layernorm.weight", shape: [d]),
                postNorm: try store.get("\(p).post_attention_layernorm.weight", shape: [d]),
                q: Linear(try store.get("\(p).self_attn.q_proj.weight", shape: [config.heads * config.headDim, d])),
                k: Linear(try store.get("\(p).self_attn.k_proj.weight", shape: [config.kvHeads * config.headDim, d])),
                v: Linear(try store.get("\(p).self_attn.v_proj.weight", shape: [config.kvHeads * config.headDim, d])),
                o: Linear(try store.get("\(p).self_attn.o_proj.weight", shape: [d, config.heads * config.headDim])),
                qNorm: try store.get("\(p).self_attn.q_norm.weight", shape: [config.headDim]),
                kNorm: try store.get("\(p).self_attn.k_norm.weight", shape: [config.headDim]),
                gate: Linear(try store.get("\(p).mlp.gate_proj.weight", shape: [config.intermediateSize, d])),
                up: Linear(try store.get("\(p).mlp.up_proj.weight", shape: [config.intermediateSize, d])),
                down: Linear(try store.get("\(p).mlp.down_proj.weight", shape: [d, config.intermediateSize]))
            )
        }
    }

    /// Qwen3 RMSNorm: normalize in fp32, round to the activation dtype, then scale by the learned weight.
    private func rmsNorm(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let normalized = xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + config.eps)
        return normalized.asType(x.dtype) * weight
    }

    /// Returns the output of the last decoder layer, `[1, L, hidden]`.
    ///
    /// Qwen-Image-2.1 conditions on the last decoder layer output *before* the final RMSNorm
    /// (see the reference pipeline); `applyFinalNorm` exists only for A/B diagnostics.
    func callAsFunction(_ ids: [Int], applyFinalNorm: Bool = false, evalEachLayer: Bool = true) -> MLXArray {
        let length = ids.count
        // The residual stream is kept in fp32: activations reach the thousands, where bf16 accumulation
        // drifts measurably from an fp32 reference. Matmuls and attention still run in bf16.
        var hidden = embedTokens[MLXArray(ids.map { Int32($0) })].expandedDimensions(axis: 0)
        let dtype = embedTokens.dtype
        if config.residualFloat32 { hidden = hidden.asType(.float32) }
        let c = config
        let scale = 1 / Float(c.headDim).squareRoot()
        for layer in layers {
            let residual = hidden
            let x = rmsNorm(hidden, layer.inputNorm).asType(dtype)
            var q = layer.q(x).reshaped(1, length, c.heads, c.headDim)
            var k = layer.k(x).reshaped(1, length, c.kvHeads, c.headDim)
            let v = layer.v(x).reshaped(1, length, c.kvHeads, c.headDim).transposed(0, 2, 1, 3)
            q = rmsNorm(q, layer.qNorm).transposed(0, 2, 1, 3)
            k = rmsNorm(k, layer.kNorm).transposed(0, 2, 1, 3)
            q = MLXFast.RoPE(q, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: 0)
            k = MLXFast.RoPE(k, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: 0)
            let attention = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .causal
            ).transposed(0, 2, 1, 3).reshaped(1, length, c.heads * c.headDim)
            hidden = residual + layer.o(attention)

            let y = rmsNorm(hidden, layer.postNorm).asType(dtype)
            hidden = hidden + layer.down(silu(layer.gate(y)) * layer.up(y))
            if evalEachLayer { eval(hidden) }
        }
        return (applyFinalNorm ? rmsNorm(hidden, finalNorm) : hidden).asType(dtype)
    }
}
