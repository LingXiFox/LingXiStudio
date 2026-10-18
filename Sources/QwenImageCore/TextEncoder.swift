import Foundation
import MLX

/// Text stack of the Qwen3-VL encoder used by Qwen-Image-2.1. Condition images enter through `VisionInput`.
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
            try Self.loadLayer(index, config: config) { key, shape in try store.get(key, shape: shape) }
        }
    }

    private static func loadLayer(_ index: Int, config: Config, weight: (String, [Int]) throws -> MLXArray) throws -> Layer {
        let d = config.hiddenSize
        let p = "model.language_model.layers.\(index)"
        return Layer(
            inputNorm: try weight("\(p).input_layernorm.weight", [d]),
            postNorm: try weight("\(p).post_attention_layernorm.weight", [d]),
            q: Linear(try weight("\(p).self_attn.q_proj.weight", [config.heads * config.headDim, d])),
            k: Linear(try weight("\(p).self_attn.k_proj.weight", [config.kvHeads * config.headDim, d])),
            v: Linear(try weight("\(p).self_attn.v_proj.weight", [config.kvHeads * config.headDim, d])),
            o: Linear(try weight("\(p).self_attn.o_proj.weight", [d, config.heads * config.headDim])),
            qNorm: try weight("\(p).self_attn.q_norm.weight", [config.headDim]),
            kNorm: try weight("\(p).self_attn.k_norm.weight", [config.headDim]),
            gate: Linear(try weight("\(p).mlp.gate_proj.weight", [config.intermediateSize, d])),
            up: Linear(try weight("\(p).mlp.up_proj.weight", [config.intermediateSize, d])),
            down: Linear(try weight("\(p).mlp.down_proj.weight", [d, config.intermediateSize])))
    }

    private init(config: Config, embedding: MLXArray, finalNorm: MLXArray) {
        self.config = config
        self.embedTokens = embedding
        self.finalNorm = finalNorm
        self.layers = []
    }

    /// Qwen3 RMSNorm: normalize in fp32, round to the activation dtype, then scale by the learned weight.
    private func rmsNorm(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let normalized = xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + config.eps)
        return normalized.asType(x.dtype) * weight
    }

    /// Image context for a prompt with condition images: the vision tower's outputs and where they go.
    struct VisionInput {
        /// Merged vision embeddings of all images, `[visual tokens, hidden]`, in placeholder order.
        let embeds: MLXArray
        /// Deepstack features added to the hidden states after the first `deepstack.count` layers.
        let deepstack: [MLXArray]
        /// Contiguous runs of `<|image_pad|>` positions in the token sequence, one per image.
        let ranges: [Range<Int>]
        /// M-RoPE positions `[3][L]` (temporal, height, width), as `Qwen3VLModel.get_rope_index` builds them.
        let positions: [[Int32]]
    }

    /// Interleaved M-RoPE (`mrope_section = [24, 20, 20]`): frequency `i` reads the height position when
    /// `i % 3 == 1`, the width position when `i % 3 == 2` (both below 60), and the temporal position otherwise.
    private func mropeTables(_ positions: [[Int32]]) -> (MLXArray, MLXArray) {
        let half = config.headDim / 2
        let length = positions[0].count
        var angles = [Float](repeating: 0, count: length * config.headDim)
        for i in 0..<half {
            let inv = 1 / powf(config.ropeTheta, Float(2 * i) / Float(config.headDim))
            let axis = i < 60 && i % 3 == 1 ? 1 : i < 60 && i % 3 == 2 ? 2 : 0
            for t in 0..<length {
                let a = Float(positions[axis][t]) * inv
                angles[t * config.headDim + i] = a
                angles[t * config.headDim + half + i] = a
            }
        }
        let a = MLXArray(angles).reshaped(1, 1, length, config.headDim)
        return (cos(a), sin(a))
    }

    private func applyRope(_ x: MLXArray, _ tables: (MLXArray, MLXArray)) -> MLXArray {
        let xf = x.asType(.float32)
        let half = config.headDim / 2
        let rotated = concatenated([-xf[.ellipsis, half...], xf[.ellipsis, ..<half]], axis: -1)
        return (xf * tables.0 + rotated * tables.1).asType(x.dtype)
    }

    /// Replaces the rows of `ranges` in `hidden` ([1, L, d]) by consecutive rows of `values`, or adds them.
    private func scatter(_ hidden: MLXArray, _ values: MLXArray, ranges: [Range<Int>], add: Bool) -> MLXArray {
        var pieces: [MLXArray] = []
        var cursor = 0
        var offset = 0
        for range in ranges {
            if range.lowerBound > cursor { pieces.append(hidden[0..., cursor..<range.lowerBound]) }
            var rows = values[offset..<(offset + range.count)].expandedDimensions(axis: 0).asType(hidden.dtype)
            if add { rows = hidden[0..., range] + rows }
            pieces.append(rows)
            offset += range.count
            cursor = range.upperBound
        }
        if cursor < hidden.dim(1) { pieces.append(hidden[0..., cursor...]) }
        return concatenated(pieces, axis: 1)
    }

    /// Returns the output of the last decoder layer, `[1, L, hidden]`.
    ///
    /// Qwen-Image-2.1 conditions on the last decoder layer output *before* the final RMSNorm
    /// (see the reference pipeline); `applyFinalNorm` exists only for A/B diagnostics.
    func callAsFunction(
        _ ids: [Int], vision: VisionInput? = nil, applyFinalNorm: Bool = false, evalEachLayer: Bool = true,
        checkCancelled: () throws -> Void = {}
    ) rethrows -> MLXArray {
        var state = initialState(ids, vision: vision)
        for (index, layer) in layers.enumerated() {
            try checkCancelled()
            state.hidden = apply(layer, index: index, state: state)
            if evalEachLayer { eval(state.hidden) }
        }
        return (applyFinalNorm ? rmsNorm(state.hidden, finalNorm) : state.hidden).asType(embedTokens.dtype)
    }

    private struct EncodingState {
        var hidden: MLXArray
        let vision: VisionInput?
        let tables: (MLXArray, MLXArray)?
    }

    private func initialState(_ ids: [Int], vision: VisionInput?) -> EncodingState {
        // The residual stream is kept in fp32: activations reach the thousands, where bf16 accumulation
        // drifts measurably from an fp32 reference. Matmuls and attention still run in bf16.
        var hidden = embedTokens[MLXArray(ids.map { Int32($0) })].expandedDimensions(axis: 0)
        if config.residualFloat32 { hidden = hidden.asType(.float32) }
        // After the cast, so fp32 vision embeddings enter the fp32 residual stream unrounded.
        if let vision { hidden = scatter(hidden, vision.embeds, ranges: vision.ranges, add: false) }
        let tables = vision.map { mropeTables($0.positions) }
        return EncodingState(hidden: hidden, vision: vision, tables: tables)
    }

    private func apply(_ layer: Layer, index: Int, state: EncodingState) -> MLXArray {
        var hidden = state.hidden
        let length = hidden.dim(1)
        let dtype = embedTokens.dtype
        let c = config
        let scale = 1 / Float(c.headDim).squareRoot()
        let vision = state.vision
        let tables = state.tables
        let residual = hidden
        let x = rmsNorm(hidden, layer.inputNorm).asType(dtype)
        var q = layer.q(x).reshaped(1, length, c.heads, c.headDim)
        var k = layer.k(x).reshaped(1, length, c.kvHeads, c.headDim)
        let v = layer.v(x).reshaped(1, length, c.kvHeads, c.headDim).transposed(0, 2, 1, 3)
        q = rmsNorm(q, layer.qNorm).transposed(0, 2, 1, 3)
        k = rmsNorm(k, layer.kNorm).transposed(0, 2, 1, 3)
        if let tables {
            q = applyRope(q, tables)
            k = applyRope(k, tables)
        } else {
            // Text-only: all three M-RoPE axes carry the same position, which is plain 1D RoPE.
            q = MLXFast.RoPE(q, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: 0)
            k = MLXFast.RoPE(k, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: 0)
        }
        let attention = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal
        ).transposed(0, 2, 1, 3).reshaped(1, length, c.heads * c.headDim)
        hidden = residual + layer.o(attention)

        let y = rmsNorm(hidden, layer.postNorm).asType(dtype)
        hidden = hidden + layer.down(silu(layer.gate(y)) * layer.up(y))
        if let vision, index < vision.deepstack.count {
            hidden = scatter(hidden, vision.deepstack[index], ranges: vision.ranges, add: true)
        }
        return hidden
    }

    /// The existing layer math and evaluation boundaries, with only one language layer's weights resident.
    static func encodeStreaming(
        directory: URL, prompts: [(ids: [Int], vision: VisionInput?)], config: Config = Config(),
        checkCancelled: () throws -> Void = {}, progress: ((Double) -> Void)? = nil
    ) throws -> [MLXArray] {
        var store = try WeightStore(directory: directory, component: "text_encoder")
        let prefix = "model.language_model"
        let kernel = QwenTextEncoder(config: config,
            embedding: try store.take("\(prefix).embed_tokens.weight"),
            finalNorm: try store.take("\(prefix).norm.weight", shape: [config.hiddenSize]))
        var states = prompts.map { kernel.initialState($0.ids, vision: $0.vision) }
        let cacheLimit = Memory.cacheLimit
        Memory.cacheLimit = min(cacheLimit, 256 << 20)
        defer { Memory.cacheLimit = cacheLimit }
        for index in 0..<config.layers {
            try checkCancelled()
            try autoreleasepool {
                let layer = try loadLayer(index, config: config) { key, shape in try store.take(key, shape: shape) }
                for i in states.indices { states[i].hidden = kernel.apply(layer, index: index, state: states[i]) }
                eval(states.map(\.hidden))
            }
            Stream.gpu.synchronize()
            Memory.clearCache()
            progress?(Double(index + 1) / Double(config.layers))
        }
        return states.map { $0.hidden.asType(kernel.embedTokens.dtype) }
    }
}
