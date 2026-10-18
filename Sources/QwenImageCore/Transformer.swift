import Foundation
import MLX

/// Single-stream block-causal DiT of Qwen-Image-2.1 (`QwenImage21Transformer2DModel`).
///
/// The joint sequence is `[text | target image]`. Text is causal and, under `causal_condition`, modulated from
/// `t = 0`, so its activations never change across denoising steps: the text prefix is run once (`prefill`) and
/// each step only recomputes the image tokens against the cached per-block text K/V.
final class QwenImageTransformer: ParameterContainer {
    struct Config {
        var dim = 4096
        var heads = 32
        var headDim = 128
        var layers = 32
        var mlpRatio = 3
        var inChannels = 64
        var contextDim = 4096
        var axesDims = [16, 56, 56]
        var ropeTheta: Double = 10000
        var eps: Float = 1e-6
    }

    struct Block {
        let q: Linear
        let k: Linear
        let v: Linear
        let out: Linear
        let normQ: MLXArray
        let normK: MLXArray
        let mlpGate: Linear
        let mlpProj: Linear
        let mlpOut: Linear
    }

    /// Post-RoPE text keys/values for every block, valid for one prompt at one resolution.
    struct TextCache {
        let keys: [MLXArray]
        let values: [MLXArray]
        let textLength: Int
        var arrays: [MLXArray] { keys + values }
    }

    struct Rope {
        let cos: MLXArray
        let sin: MLXArray
    }

    /// Execution switches; every combination computes the same function (used for A/B measurements).
    struct Options {
        /// Single Metal kernel for Q/K RMSNorm + RoPE instead of composed ops.
        var fusedQKNormRope = true
        /// Trace each denoising step once with `compile` so element-wise ops are fused.
        var compiledStep = true

        static func fromEnvironment() -> Options {
            let env = ProcessInfo.processInfo.environment
            var options = Options()
            if env["LINGXI_FUSED_QK"] == "0" { options.fusedQKNormRope = false }
            if env["LINGXI_COMPILE"] == "0" { options.compiledStep = false }
            return options
        }
    }

    let config: Config
    let options: Options
    let imgIn: Linear
    let txtNorm: MLXArray
    let txtIn: Linear
    let txtOut: Linear
    let timeLinear1: Linear
    let timeLinear2: Linear
    let modulation: Linear
    let normOut: Linear
    let projOut: Linear
    let blocks: [Block]

    /// Active LoRA bindings, rebuilt from a `LoRAStack` by `bind(_:)`. Base weights are never modified.
    private var loraGroups: [String: LoRAGroup] = [:]
    private var boundRevision: Int?
    private var boundStack: ObjectIdentifier?

    var parameters: [MLXArray] {
        [imgIn.weight, txtNorm, txtIn.weight, txtOut.weight, timeLinear1.weight, timeLinear2.weight,
         modulation.weight, normOut.weight, projOut.weight]
            + blocks.flatMap {
                [$0.q.weight, $0.k.weight, $0.v.weight, $0.out.weight, $0.normQ, $0.normK,
                 $0.mlpGate.weight, $0.mlpProj.weight, $0.mlpOut.weight]
            }
    }

    init(store: WeightStore, config: Config = Config(), options: Options = .fromEnvironment()) throws {
        self.config = config
        self.options = options
        let d = config.dim
        let mlp = d * config.mlpRatio
        imgIn = Linear(try store.get("img_in.weight", shape: [d, config.inChannels]))
        txtNorm = try store.get("txt_in.text_norm.weight", shape: [config.contextDim])
        txtIn = Linear(try store.get("txt_in.in_layer.weight", shape: [d, config.contextDim]))
        txtOut = Linear(try store.get("txt_in.out_layer.weight", shape: [d, d]))
        timeLinear1 = Linear(try store.get("time_text_embed.timestep_embedder.linear_1.weight", shape: [d, 256]))
        timeLinear2 = Linear(try store.get("time_text_embed.timestep_embedder.linear_2.weight", shape: [d, d]))
        modulation = Linear(try store.get("modulation.1.weight", shape: [4 * d, d]))
        normOut = Linear(try store.get("norm_out.linear.weight", shape: [d, d]))
        projOut = Linear(try store.get("proj_out.weight", shape: [config.inChannels, d]))
        blocks = try (0..<config.layers).map { index in
            let p = "transformer_blocks.\(index)"
            return Block(
                q: Linear(try store.get("\(p).attn.to_q.weight", shape: [d, d])),
                k: Linear(try store.get("\(p).attn.to_k.weight", shape: [d, d])),
                v: Linear(try store.get("\(p).attn.to_v.weight", shape: [d, d])),
                out: Linear(try store.get("\(p).attn.to_out.0.weight", shape: [d, d])),
                normQ: try store.get("\(p).attn.norm_q.weight", shape: [config.headDim]),
                normK: try store.get("\(p).attn.norm_k.weight", shape: [config.headDim]),
                mlpGate: Linear(try store.get("\(p).img_mlp.gate_layer.weight", shape: [mlp, d])),
                mlpProj: Linear(try store.get("\(p).img_mlp.proj.weight", shape: [mlp, d])),
                mlpOut: Linear(try store.get("\(p).img_mlp.out.weight", shape: [d, mlp]))
            )
        }
    }

    // MARK: - LoRA

    /// Every base linear layer a LoRA may target, grouped by shared input. Groups with several targets run
    /// their LoRA down-projections as one matmul.
    private func loraLayout() -> [(group: String, targets: [String], shapes: [[Int]])] {
        let d = config.dim
        let mlp = d * config.mlpRatio
        var layout: [(String, [String], [[Int]])] = [
            ("img_in", ["img_in"], [[d, config.inChannels]]),
            ("txt_in.in_layer", ["txt_in.in_layer"], [[d, config.contextDim]]),
            ("txt_in.out_layer", ["txt_in.out_layer"], [[d, d]]),
            ("time.linear_1", ["time_text_embed.timestep_embedder.linear_1"], [[d, 256]]),
            ("time.linear_2", ["time_text_embed.timestep_embedder.linear_2"], [[d, d]]),
            ("modulation", ["modulation.1"], [[4 * d, d]]),
            ("norm_out", ["norm_out.linear"], [[d, d]]),
            ("proj_out", ["proj_out"], [[config.inChannels, d]]),
        ]
        for i in 0..<config.layers {
            let p = "transformer_blocks.\(i)"
            layout.append(("\(i).qkv", ["\(p).attn.to_q", "\(p).attn.to_k", "\(p).attn.to_v"], [[d, d], [d, d], [d, d]]))
            layout.append(("\(i).out", ["\(p).attn.to_out.0"], [[d, d]]))
            layout.append(("\(i).mlp_in", ["\(p).img_mlp.gate_layer", "\(p).img_mlp.proj"], [[mlp, d], [mlp, d]]))
            layout.append(("\(i).mlp_out", ["\(p).img_mlp.out"], [[d, mlp]]))
        }
        return layout
    }

    /// Applies the stack's active composition. Cheap when nothing changed since the last call.
    ///
    /// The previous bindings are simply dropped, so disabling or unloading every adapter returns the model to its
    /// exact base behaviour. The compiled step is reset because it captures the bound factors.
    func bind(_ stack: LoRAStack) throws {
        if boundStack == ObjectIdentifier(stack), boundRevision == stack.revision { return }
        let active = stack.active
        let layout = loraLayout()
        let known = Set(layout.flatMap(\.targets))
        for entry in active {
            let unmatched = entry.adapter.targets.keys.filter { !known.contains($0) }.sorted()
            if !unmatched.isEmpty { throw LoRAError.unmatched(file: entry.adapter.name, keys: unmatched) }
        }
        var groups: [String: LoRAGroup] = [:]
        // Profiling aid: restrict LoRA to group names containing one of these substrings.
        let only = ProcessInfo.processInfo.environment["LINGXI_LORA_ONLY"]?.split(separator: ",").map(String.init)
        for item in layout where only == nil || only!.contains(where: { item.group.contains($0) }) {
            if let group = try LoRAGroup.make(targets: item.targets, baseShapes: item.shapes, active: active) {
                groups[item.group] = group
            }
        }
        eval(groups.values.flatMap(\.arrays))
        loraGroups = groups
        boundStack = ObjectIdentifier(stack)
        boundRevision = stack.revision
        compiledImageStep = nil
    }

    var activeLoRAGroups: Int { loraGroups.count }

    private func linear(_ group: String, _ base: Linear, _ x: MLXArray) -> MLXArray {
        let y = base(x)
        guard let lora = loraGroups[group] else { return y }
        return lora.apply(x, to: [y])[0]
    }

    // MARK: - Geometry

    /// 3-axis RoPE angles. Text advances one shared position on every axis; the target image freezes the frame
    /// axis at the text length and lays tokens on a height/width grid centred on zero.
    ///
    /// Angles follow the reference's fp32 arithmetic: `1 / theta^(2k/d)` and `position * omega` in fp32.
    func rope(textLength: Int, latentHeight: Int, latentWidth: Int) -> (text: Rope, image: Rope) {
        func frequencies(_ dim: Int) -> [Float] {
            stride(from: 0, to: dim, by: 2).map { 1 / powf(Float(config.ropeTheta), Float($0) / Float(dim)) }
        }
        let axes = config.axesDims.map(frequencies)
        let half = axes.reduce(0) { $0 + $1.count }

        func table(_ positions: [(Int, Int, Int)]) -> Rope {
            var angles = [Float]()
            angles.reserveCapacity(positions.count * half)
            for (f, h, w) in positions {
                for (axis, value) in [(0, f), (1, h), (2, w)] {
                    for omega in axes[axis] { angles.append(Float(value) * omega) }
                }
            }
            let array = MLXArray(angles, [positions.count, half])
            return Rope(cos: MLX.cos(array), sin: MLX.sin(array))
        }

        let text = table((0..<textLength).map { ($0, $0, $0) })
        var image: [(Int, Int, Int)] = []
        image.reserveCapacity(latentHeight * latentWidth)
        for h in -(latentHeight - latentHeight / 2)..<(latentHeight / 2) {
            for w in -(latentWidth - latentWidth / 2)..<(latentWidth / 2) {
                image.append((textLength, h, w))
            }
        }
        return (text, table(image))
    }

    // MARK: - Conditioning

    /// Rows `[t, 0]` of the timestep embedding (`causal_condition` adds the `t = 0` row for the text prefix).
    ///
    /// Reproduces the reference rounding: the pipeline casts `sigma * 1000` to bf16, divides by 1000 in bf16,
    /// and the embedding rescales by 1000 in fp32.
    func timeEmbedding(sigma: Float) -> MLXArray {
        let t = (MLXArray([sigma * 1000]).asType(.bfloat16) / MLXArray(Float(1000)).asType(.bfloat16))
        let rows = concatenated([t, MLXArray.zeros([1], dtype: .bfloat16)]).asType(.float32) * 1000
        let half = 128
        let freqs = exp(-log(Float(10000)) * MLXArray((0..<half).map { Float($0) }) / Float(half))
        let args = rows.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        let projection = concatenated([MLX.cos(args), MLX.sin(args)], axis: -1).asType(.bfloat16)
        return linear("time.linear_2", timeLinear2, silu(linear("time.linear_1", timeLinear1, projection)))
    }

    /// Shared modulation `[t-row, 0-row]` x `[scale1 | gate1 | scale2 | gate2]`.
    func modulationParameters(_ temb: MLXArray) -> MLXArray {
        linear("modulation", modulation, silu(temb))
    }

    func textProjection(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let normed = xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + config.eps) * (txtNorm.asType(.float32) + 1)
        let hidden = geluTanh(linear("txt_in.in_layer", txtIn, normed.asType(x.dtype)))
        return linear("txt_in.out_layer", txtOut, hidden)
    }

    // MARK: - Attention

    /// diffusers `RMSNorm` with a bf16 weight: normalize in fp32, round, then multiply by the weight (rounded).
    private func referenceRMSNorm(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        return (xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + config.eps)).asType(x.dtype) * weight
    }

    private func applyRope(_ x: MLXArray, _ rope: Rope) -> MLXArray {
        let shape = x.shape
        let pairs = x.asType(.float32).reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
        let real = pairs[.ellipsis, 0]
        let imag = pairs[.ellipsis, 1]
        let cos = rope.cos[.newAxis, 0..., .newAxis, 0...]
        let sin = rope.sin[.newAxis, 0..., .newAxis, 0...]
        let rotated = stacked([real * cos - imag * sin, real * sin + imag * cos], axis: -1)
        return rotated.reshaped(shape).asType(x.dtype)
    }

    /// Projects to `[1, heads, L, headDim]` query/key/value with QK-norm and RoPE applied.
    private func project(_ index: Int, _ x: MLXArray, _ rope: Rope) -> (MLXArray, MLXArray, MLXArray) {
        let block = blocks[index]
        let length = x.dim(1)
        let c = config
        var qkv = [block.q(x), block.k(x), block.v(x)]
        if let lora = loraGroups["\(index).qkv"] { qkv = lora.apply(x, to: qkv) }
        var q = qkv[0].reshaped(1, length, c.heads, c.headDim)
        var k = qkv[1].reshaped(1, length, c.heads, c.headDim)
        let v = qkv[2].reshaped(1, length, c.heads, c.headDim).transposed(0, 2, 1, 3)
        if options.fusedQKNormRope {
            (q, k) = FusedKernels.qkNormRope(
                q: q, k: k, weightQ: block.normQ, weightK: block.normK, cos: rope.cos, sin: rope.sin, eps: c.eps)
            return (q.transposed(0, 2, 1, 3), k.transposed(0, 2, 1, 3), v)
        }
        q = applyRope(referenceRMSNorm(q, block.normQ), rope).transposed(0, 2, 1, 3)
        k = applyRope(referenceRMSNorm(k, block.normK), rope).transposed(0, 2, 1, 3)
        return (q, k, v)
    }

    private func merge(_ attention: MLXArray, length: Int) -> MLXArray {
        attention.transposed(0, 2, 1, 3).reshaped(1, length, config.dim)
    }

    // MARK: - Blocks

    private struct Modulation {
        let scale1, gate1, scale2, gate2: MLXArray
    }

    private func modulationRow(_ modulation: MLXArray, row: Int) -> Modulation {
        let d = config.dim
        let r = modulation[row]
        return Modulation(
            scale1: r[0..<d], gate1: tanh(r[d..<(2 * d)]), scale2: r[(2 * d)..<(3 * d)], gate2: tanh(r[(3 * d)...]))
    }

    private func attentionOut(_ index: Int, _ x: MLXArray) -> MLXArray {
        linear("\(index).out", blocks[index].out, x)
    }

    private func feedForward(_ index: Int, _ x: MLXArray) -> MLXArray {
        let block = blocks[index]
        var gateProj = [block.mlpGate(x), block.mlpProj(x)]
        if let lora = loraGroups["\(index).mlp_in"] { gateProj = lora.apply(x, to: gateProj) }
        return linear("\(index).mlp_out", block.mlpOut, silu(gateProj[0]) * gateProj[1])
    }

    private func layerNorm(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, eps: config.eps)
    }

    /// Runs the text prefix through every block once and caches its post-RoPE keys/values.
    func prefill(text: MLXArray, rope: Rope) -> TextCache {
        let length = text.dim(1)
        let mod = modulationRow(modulationParameters(timeEmbedding(sigma: 0)), row: 1)
        let scale = 1 / Float(config.headDim).squareRoot()
        var hidden = textProjection(text)
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for index in blocks.indices {
            let (q, k, v) = project(index, layerNorm(hidden) * (1 + mod.scale1), rope)
            keys.append(k)
            values.append(v)
            // The last block's text output feeds nothing downstream, only its K/V matter.
            if index == blocks.count - 1 { break }
            let attention = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .causal)
            hidden = hidden + mod.gate1 * attentionOut(index, merge(attention, length: length))
            hidden = hidden + mod.gate2 * feedForward(index, layerNorm(hidden) * (1 + mod.scale2))
            eval([hidden, k, v])
        }
        return TextCache(keys: keys, values: values, textLength: length)
    }

    /// Predicts the flow velocity for the packed target latents `[1, N, 64]`.
    func callAsFunction(latents: MLXArray, sigma: Float, cache: TextCache, rope: Rope) -> MLXArray {
        let inputs = [latents, timeEmbedding(sigma: sigma), rope.cos, rope.sin] + cache.keys + cache.values
        guard options.compiledStep else { return imageStep(inputs)[0] }
        if compiledImageStep == nil {
            // `compile` retraces by itself when input shapes change (resolution or prompt length).
            compiledImageStep = compile { [unowned self] in self.imageStep($0) }
        }
        return compiledImageStep!(inputs)[0]
    }

    private var compiledImageStep: (([MLXArray]) -> [MLXArray])?

    /// Same function as `callAsFunction`, evaluated eagerly (honours `blockTrace`).
    func uncompiled(latents: MLXArray, sigma: Float, cache: TextCache, rope: Rope) -> MLXArray {
        imageStep([latents, timeEmbedding(sigma: sigma), rope.cos, rope.sin] + cache.keys + cache.values)[0]
    }

    /// Debug hook: receives the image-token hidden state after every block (uncompiled path only).
    var blockTrace: ((Int, MLXArray) -> Void)?

    /// One denoising evaluation of the image tokens against the cached text prefix.
    /// Inputs: `[latents, temb, ropeCos, ropeSin, keys..., values...]`.
    private func imageStep(_ inputs: [MLXArray]) -> [MLXArray] {
        let latents = inputs[0]
        let temb = inputs[1]
        let rope = Rope(cos: inputs[2], sin: inputs[3])
        let keys = inputs[4..<(4 + blocks.count)]
        let values = inputs[(4 + blocks.count)...]
        let length = latents.dim(1)
        let mod = modulationRow(modulationParameters(temb), row: 0)
        let scale = 1 / Float(config.headDim).squareRoot()
        var hidden = linear("img_in", imgIn, latents)
        for index in blocks.indices {
            let (q, k, v) = project(index, layerNorm(hidden) * (1 + mod.scale1), rope)
            let attention = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: concatenated([keys[keys.startIndex + index], k], axis: 2),
                values: concatenated([values[values.startIndex + index], v], axis: 2),
                scale: scale, mask: .none)
            hidden = hidden + mod.gate1 * attentionOut(index, merge(attention, length: length))
            hidden = hidden + mod.gate2 * feedForward(index, layerNorm(hidden) * (1 + mod.scale2))
            blockTrace?(index, hidden)
        }
        let outScale = linear("norm_out", normOut, silu(temb))[0]
        return [linear("proj_out", projOut, layerNorm(hidden) * (1 + outScale))]
    }
}
