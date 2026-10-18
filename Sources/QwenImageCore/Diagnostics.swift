import Foundation
import MLX

/// Component-level entry points used to check numerics against a reference implementation.
/// Reference dumps use fp32; loading benchmarks preserve the native dtype.
public enum Diagnostics {
    /// Controlled loading experiment; outputs use identical full-precision weights for all batch sizes.
    public static func benchTextEncoder(model: ModelDirectory, prompt: String, batch: Int, streamed: Bool = false, to url: URL) throws -> (
        loading: Double, encoding: Double, footprint: Int, mlxPeak: Int
    ) {
        guard [4, 8, 16].contains(batch) else { throw WeightError.invalidBatch(batch) }
        let sampler = FootprintSampler()
        sampler.start()
        defer { sampler.stop() }
        Memory.peakMemory = 0
        let pipeline = try QwenImagePipeline(model: model)
        let tokens = try pipeline.promptTokens(prompt)
        let start = Date()
        if streamed {
            let hidden = try QwenTextEncoder.encodeStreaming(directory: model.textEncoder, prompts: [(tokens.ids, nil)])[0]
            eval(hidden)
            let duration = Date().timeIntervalSince(start)
            let peak = sampler.takePeak()
            let mlxPeak = Memory.peakMemory
            try save(arrays: ["hidden": hidden, "ids": MLXArray(tokens.ids.map { Int32($0) })], url: url)
            return (0, duration, peak, mlxPeak)
        }
        let encoder = try QwenTextEncoder(store: WeightStore(directory: model.textEncoder, component: "text_encoder"))
        try encoder.materialize(batch: batch)
        let loading = Date().timeIntervalSince(start)
        let loadPeak = sampler.takePeak()
        let encodeStart = Date()
        let hidden = try encoder(tokens.ids)
        eval(hidden)
        let encoding = Date().timeIntervalSince(encodeStart)
        let peak = max(loadPeak, sampler.takePeak())
        let mlxPeak = Memory.peakMemory
        try save(arrays: ["hidden": hidden, "ids": MLXArray(tokens.ids.map { Int32($0) })], url: url)
        return (loading, encoding, peak, mlxPeak)
    }

    public static func tokenize(model: ModelDirectory, text: String) throws -> [Int] {
        try QwenTokenizer(directory: model.processor).encode(text)
    }

    /// Text-encoder hidden states (pre final norm) for the full templated prompt.
    public static func dumpTextEncoder(model: ModelDirectory, prompt: String, to url: URL) throws {
        let pipeline = try QwenImagePipeline(model: model)
        let tokens = try pipeline.promptTokens(prompt)
        let encoder = try QwenTextEncoder(store: WeightStore(directory: model.textEncoder, component: "text_encoder"))
        let hidden = try encoder(tokens.ids)
        try save(
            arrays: [
                "hidden": hidden.asType(.float32),
                "ids": MLXArray(tokens.ids.map { Int32($0) }),
                "drop": MLXArray([Int32(tokens.drop)]),
            ], url: url)
    }

    /// Multimodal prompt encoding: vision tower outputs and the text encoder's pre-norm hidden states.
    public static func dumpTextImage(model: ModelDirectory, prompt: String, image url: URL, to out: URL) throws {
        let tokenizer = try QwenTokenizer(directory: model.processor)
        let image = try RGBAImage(contentsOf: url)
        let mp = try MultimodalPrompt(prompt: prompt, imageSizes: [(image.width, image.height)], tokenizer: tokenizer)
        let store = try WeightStore(directory: model.textEncoder, component: "text_encoder")
        var traced: [String: MLXArray] = [:]
        let vision = try QwenVisionEncoder(store: store).encode(image) { traced["v_\($0)"] = $1.asType(.float32) }
        let encoder = try QwenTextEncoder(store: store)
        let hidden = try encoder(
            mp.ids,
            vision: .init(embeds: vision.embeds, deepstack: vision.deepstack, ranges: mp.imageRanges, positions: mp.positions))
        var arrays: [String: MLXArray] = [
            "hidden": hidden.asType(.float32),
            "ids": MLXArray(mp.ids.map { Int32($0) }),
            "drop": MLXArray([Int32(mp.drop)]),
            "positions": MLXArray(mp.positions.flatMap { $0 }).reshaped(3, -1),
            "vision": vision.embeds.asType(.float32),
            "rgb": MLXArray(image.rgbOverWhite()).reshaped(image.height, image.width, 3),
        ]
        for (i, d) in vision.deepstack.enumerated() { arrays["deepstack\(i)"] = d.asType(.float32) }
        arrays.merge(traced) { a, _ in a }
        try save(arrays: arrays, url: out)
    }

    private static func loadTransformer(_ model: ModelDirectory, loras: [LoRASpec]) throws -> (
        QwenImageTransformer, LoRAStack
    ) {
        let transformer = try QwenImageTransformer(
            store: WeightStore(directory: model.transformer, component: "transformer"))
        try transformer.materialize()
        let stack = LoRAStack()
        for spec in loras { try stack.load(spec) }
        try transformer.bind(stack)
        return (transformer, stack)
    }

    private static func inputs(_ url: URL, _ names: [String]) throws -> [MLXArray] {
        let arrays = try loadArrays(url: url)
        return try names.map { name in
            guard let value = arrays[name] else { throw WeightError.missingTensor(name, component: url.path) }
            return value
        }
    }

    /// Conditioning stage: timestep embedding rows `[t, 0]`, shared modulation, and the projected text stream.
    public static func dumpConditioning(
        model: ModelDirectory, input: URL, sigma: Float, loras: [LoRASpec], to url: URL
    ) throws {
        let embeds = try inputs(input, ["embeds"])[0].asType(.bfloat16)
        let (transformer, _) = try loadTransformer(model, loras: loras)
        let temb = transformer.timeEmbedding(sigma: sigma)
        try save(
            arrays: [
                "temb": temb.asType(.float32),
                "modulation": transformer.modulationParameters(temb).asType(.float32),
                "txt": transformer.textProjection(embeds).asType(.float32),
            ], url: url)
    }

    /// One transformer evaluation. Reads `embeds` [1,T,4096] and `latents` [1,N,64] from `input`; with `trace`
    /// it runs uncompiled and also dumps the image-token hidden state after every block.
    public static func dumpTransformer(
        model: ModelDirectory, input: URL, sigmas: [Float], latentHeight: Int, latentWidth: Int,
        loras: [LoRASpec] = [], trace: Bool = false, to url: URL
    ) throws {
        let values = try inputs(input, ["embeds", "latents"])
        let (embeds, latents) = (values[0].asType(.bfloat16), values[1].asType(.bfloat16))
        let (transformer, _) = try loadTransformer(model, loras: loras)
        let rope = transformer.rope(textLength: embeds.dim(1), latentHeight: latentHeight, latentWidth: latentWidth)
        let cache = transformer.prefill(text: embeds, rope: rope.text)
        var dumped: [String: MLXArray] = [:]
        for (i, sigma) in sigmas.enumerated() {
            if trace {
                transformer.blockTrace = { index, hidden in
                    eval(hidden)
                    dumped["block\(index).s\(i)"] = hidden.asType(.float32)
                }
            }
            let out = trace
                ? transformer.uncompiled(latents: latents, sigma: sigma, cache: cache, rope: rope.image)
                : transformer(latents: latents, sigma: sigma, cache: cache, rope: rope.image)
            dumped["velocity.s\(i)"] = out.asType(.float32)
            eval(Array(dumped.values))
        }
        try save(arrays: dumped, url: url)
    }

    /// One condition image: `embeds` [1, L, 4096] (encoder output after the system drop, pad slots included),
    /// `pad_mask` [L], `cond` [1, ch*cw, 64] with `cond_hw` = [ch, cw], target `latents` [1, h*w, 64].
    public static func dumpTransformerCondition(
        model: ModelDirectory, input: URL, sigmas: [Float], latentHeight: Int, latentWidth: Int,
        loras: [LoRASpec] = [], to url: URL
    ) throws {
        let values = try inputs(input, ["embeds", "pad_mask", "cond", "cond_hw", "latents"])
        let embeds = values[0].asType(.bfloat16)
        let pad = values[1].asType(.int32).asArray(Int32.self)
        let hw = values[3].asType(.int32).asArray(Int32.self)
        let condition = QwenImageTransformer.ConditionImage(
            latents: values[2].asType(.bfloat16), height: Int(hw[0]), width: Int(hw[1]))
        let latents = values[4].asType(.bfloat16)
        guard let start = pad.firstIndex(of: 1), let end = pad.lastIndex(of: 1) else {
            throw WeightError.missingTensor("pad_mask (no image slots)", component: input.path)
        }
        let ranges = [start..<(end + 1)]
        let (transformer, _) = try loadTransformer(model, loras: loras)
        let layout = transformer.layout(
            textLength: embeds.dim(1), imageRanges: ranges, conditions: [(condition.height, condition.width)],
            latentHeight: latentHeight, latentWidth: latentWidth)
        let cache = transformer.prefill(text: embeds, conditions: [condition], imageRanges: ranges, layout: layout)
        var dumped: [String: MLXArray] = [:]
        for (i, sigma) in sigmas.enumerated() {
            dumped["velocity.s\(i)"] = transformer(latents: latents, sigma: sigma, cache: cache, rope: layout.target)
                .asType(.float32)
            eval(Array(dumped.values))
        }
        try save(arrays: dumped, url: url)
    }

    /// Several denoising steps from a given initial noise, recording the latents after every step.
    public static func dumpDenoise(
        model: ModelDirectory, input: URL, request: GenerationRequest, to url: URL
    ) throws {
        let values = try inputs(input, ["embeds", "noise"])
        let embeds = values[0].asType(.bfloat16)
        var latents = values[1].asType(.bfloat16)
        let (lh, lw) = (request.height / 16, request.width / 16)
        let pipeline = try QwenImagePipeline(model: model)
        let schedule = pipeline.schedule(for: request, imageTokens: lh * lw)
        let (transformer, _) = try loadTransformer(model, loras: request.loras)
        let rope = transformer.rope(textLength: embeds.dim(1), latentHeight: lh, latentWidth: lw)
        let cache = transformer.prefill(text: embeds, rope: rope.text)
        var dumped: [String: MLXArray] = ["sigmas": MLXArray(schedule.sigmas)]
        for step in 0..<schedule.steps {
            let velocity = transformer(latents: latents, sigma: schedule.sigmas[step], cache: cache, rope: rope.image)
            latents = schedule.step(latents: latents, velocity: velocity, index: step)
            eval(latents)
            dumped["latents.\(step)"] = latents.asType(.float32)
        }
        try save(arrays: dumped, url: url)
    }

    /// Euler update on given `latents`/`velocity` (bf16) at every schedule index, plus the schedule itself.
    public static func dumpEuler(model: ModelDirectory, input: URL, request: GenerationRequest, to url: URL) throws {
        let values = try inputs(input, ["latents", "velocity"])
        let pipeline = try QwenImagePipeline(model: model)
        let schedule = pipeline.schedule(for: request, imageTokens: values[0].dim(1))
        var dumped: [String: MLXArray] = ["sigmas": MLXArray(schedule.sigmas)]
        for index in 0..<schedule.steps {
            dumped["next.\(index)"] = schedule
                .step(latents: values[0].asType(.bfloat16), velocity: values[1].asType(.bfloat16), index: index)
                .asType(.float32)
        }
        try save(arrays: dumped, url: url)
    }

    /// Image postprocessing of a float image `[1,H,W,C]` in [-1, 1] to 8-bit.
    public static func dumpPostprocess(input: URL, to url: URL) throws {
        let image = try inputs(input, ["image"])[0]
        try save(arrays: ["pixels": QwenImagePipeline.toRGBA8(image).asType(.int32)], url: url)
    }

    /// LoRA lifecycle on one transformer: base -> load -> disable -> enable -> rescale -> duplicate -> unload.
    /// Returns a report of output differences that must be exactly zero or within float tolerance.
    public static func checkLoRALifecycle(
        model: ModelDirectory, input: URL, lora: String, sigma: Float, latentHeight: Int, latentWidth: Int
    ) throws -> [String] {
        let values = try inputs(input, ["embeds", "latents"])
        let (embeds, latents) = (values[0].asType(.bfloat16), values[1].asType(.bfloat16))
        let (transformer, stack) = try loadTransformer(model, loras: [])
        let checksum = { () -> [Float] in
            transformer.parameters.map { sum($0.asType(.float32)).item(Float.self) }
        }
        let weightsBefore = checksum()
        let rope = transformer.rope(textLength: embeds.dim(1), latentHeight: latentHeight, latentWidth: latentWidth)
        func run() throws -> MLXArray {
            try transformer.bind(stack)
            let cache = transformer.prefill(text: embeds, rope: rope.text)
            let out = transformer(latents: latents, sigma: sigma, cache: cache, rope: rope.image).asType(.float32)
            eval(out)
            return out
        }
        func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
            (sqrt(sum(square(a - b))) / sqrt(sum(square(b)))).item(Float.self)
        }
        var lines: [String] = []
        let base = try run()
        let id = try stack.load(LoRASpec(path: lora, scale: 1))
        let withLoRA = try run()
        lines.append(String(format: "loaded (scale 1) vs base: rel-L2 %.4f (must be > 0)", rel(withLoRA, base)))
        try stack.setEnabled(id, false)
        lines.append(String(format: "disabled vs base: rel-L2 %.3g (must be 0)", rel(try run(), base)))
        try stack.setEnabled(id, true)
        lines.append(String(format: "re-enabled vs loaded: rel-L2 %.3g (must be 0)", rel(try run(), withLoRA)))
        try stack.setScale(id, 0)
        lines.append(String(format: "scale 0 vs base: rel-L2 %.3g (must be 0)", rel(try run(), base)))
        try stack.setScale(id, 0.5)
        let second = try stack.load(LoRASpec(path: lora, scale: 0.5))
        lines.append(String(format: "two adapters at 0.5 vs one at 1.0: rel-L2 %.4f (float level)", rel(try run(), withLoRA)))
        try stack.unload(second)
        try stack.unload(id)
        lines.append(String(format: "unloaded vs base: rel-L2 %.3g (must be 0)", rel(try run(), base)))
        let weightsAfter = checksum()
        lines.append("base weight checksums unchanged: \(weightsBefore == weightsAfter) (\(weightsBefore.count) tensors)")
        return lines
    }

    /// VAE decode of `latents` [1,N,64] (normalized, packed) read from `input`.
    public static func dumpVAE(
        model: ModelDirectory, input: URL, latentHeight: Int, latentWidth: Int, float32: Bool = false, to url: URL
    ) throws {
        let arrays = try loadArrays(url: input)
        guard let latents = arrays["latents"] else {
            throw WeightError.missingTensor("latents", component: input.path)
        }
        let vae = try QwenImageVAEDecoder(
            store: WeightStore(directory: model.vae, component: "vae"), configURL: model.vae.appending(path: "config.json"),
            dtype: float32 ? .float32 : .bfloat16)
        var dumped: [String: MLXArray] = [:]
        let image = vae.decode(latents, latentHeight: latentHeight, latentWidth: latentWidth) { name, value in
            dumped[name] = value.asType(.float32)
        }
        dumped["image"] = image.asType(.float32)
        try save(arrays: dumped, url: url)
    }

    /// Encodes `image` ([1, H, W, 4] in [-1, 1]) with the VAE encoder; writes packed normalized `latents`.
    public static func dumpVAEEncode(model: ModelDirectory, input: URL, to url: URL) throws {
        guard let image = try loadArrays(url: input)["image"] else {
            throw WeightError.missingTensor("image", component: input.path)
        }
        let vae = try QwenImageVAEEncoder(
            store: WeightStore(directory: model.vae, component: "vae"), configURL: model.vae.appending(path: "config.json"))
        var dumped: [String: MLXArray] = [:]
        let latents = vae.encode(image) { name, value in dumped[name] = value }
        dumped["latents"] = latents
        try save(arrays: dumped, url: url)
    }

    public static func schedule(steps: Int, imageSeqLen: Int, model: ModelDirectory) throws -> [Float] {
        let config = try FlowMatchSchedule.Config(contentsOf: model.scheduler.appending(path: "scheduler_config.json"))
        return FlowMatchSchedule(steps: steps, imageSeqLen: imageSeqLen, config: config).sigmas
    }
}

extension Diagnostics {
    /// Checks the fused Q/K RMSNorm + RoPE kernel against the reference's composed ops (same rounding points).
    /// Returns the max absolute difference and the max difference measured in bf16 ulps of the reference.
    public static func checkFusedQKNormRope(length: Int = 300, heads: Int = 32, headDim: Int = 128) -> (
        Float, Float, Float
    ) {
        let key = MLXRandom.key(7)
        let keys = MLXRandom.split(key: key, into: 4)
        let q = (MLXRandom.normal([1, length, heads, headDim], key: keys[0]) * 3).asType(.bfloat16)
        let k = (MLXRandom.normal([1, length, heads, headDim], key: keys[1]) * 3).asType(.bfloat16)
        let wq = (1 + 0.1 * MLXRandom.normal([headDim], key: keys[2])).asType(.bfloat16)
        let wk = (1 + 0.1 * MLXRandom.normal([headDim], key: keys[3])).asType(.bfloat16)
        let angles = MLXArray((0..<(length * headDim / 2)).map { Float($0 % 977) * 0.37 }, [length, headDim / 2])
        let (cos, sin) = (MLX.cos(angles), MLX.sin(angles))

        func reference(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
            let xf = x.asType(.float32)
            let normed = (xf * rsqrt(mean(xf * xf, axis: -1, keepDims: true) + 1e-6)).asType(.bfloat16) * w
            let pairs = normed.asType(.float32).reshaped(1, length, heads, headDim / 2, 2)
            let (re, im) = (pairs[.ellipsis, 0], pairs[.ellipsis, 1])
            let c = cos[.newAxis, 0..., .newAxis, 0...]
            let s = sin[.newAxis, 0..., .newAxis, 0...]
            return stacked([re * c - im * s, re * s + im * c], axis: -1).reshaped(x.shape).asType(.bfloat16)
        }
        let (fq, fk) = FusedKernels.qkNormRope(q: q, k: k, weightQ: wq, weightK: wk, cos: cos, sin: sin, eps: 1e-6)
        var maxAbs: Float = 0
        var maxUlp: Float = 0
        var mismatches: Float = 0
        for (fused, ref) in [(fq, reference(q, wq)), (fk, reference(k, wk))] {
            let refF = ref.asType(.float32)
            let diff = abs(fused.asType(.float32) - refF)
            // bf16 has 8 significand bits: one ulp at |x| is 2^(floor(log2|x|) - 7).
            let ulp = MLX.pow(MLXArray(Float(2)), floor(log2(maximum(abs(refF), MLXArray(Float(1e-30))))) - 7)
            maxAbs = max(maxAbs, diff.max().item(Float.self))
            maxUlp = max(maxUlp, (diff / ulp).max().item(Float.self))
            mismatches += (diff .> 0).asType(.float32).mean().item(Float.self) / 2
        }
        return (maxAbs, maxUlp, mismatches)
    }
}

extension Diagnostics {
    public struct ThermalProbe: Sendable {
        public var thermalState: String
        /// Sustained BF16 matmul throughput of a short fixed workload; drops when the GPU is clock-limited.
        public var tflops: Double
    }

    public static func thermalStateName(_ state: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState)
        -> String
    {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    /// ~1 s of 4096³ BF16 matmuls, the same operation that dominates a denoising step.
    public static func probeThermal(iterations: Int = 24) -> ThermalProbe {
        let a = MLXRandom.normal([4096, 4096], key: MLXRandom.key(1)).asType(.bfloat16)
        let b = MLXRandom.normal([4096, 4096], key: MLXRandom.key(2)).asType(.bfloat16)
        eval(a, b)
        eval(matmul(a, b))  // warm-up
        let start = Date()
        var acc = a
        for _ in 0..<iterations {
            acc = matmul(acc, b) * 0.01
            eval(acc)
        }
        let seconds = Date().timeIntervalSince(start)
        let flops = 2.0 * 4096 * 4096 * 4096 * Double(iterations)
        return ThermalProbe(thermalState: thermalStateName(), tflops: flops / seconds / 1e12)
    }
}

extension Diagnostics {
    /// Short denoising benchmark of the transformer alone at a given size: one warm-up step, then `steps` timed
    /// steps. Text embeddings are random (their values do not affect cost); LoRA is bound dynamically.
    public static func benchTransformer(
        model: ModelDirectory, width: Int, height: Int, textLength: Int, steps: Int, loras: [LoRASpec]
    ) throws -> (steps: [Double], footprint: Int, mlxPeak: Int) {
        let cacheLimit = Memory.cacheLimit
        if ProcessInfo.processInfo.environment["LINGXI_DIT_CACHE"] != "0" {
            Memory.cacheLimit = min(cacheLimit, 512 << 20)
        }
        defer { Memory.cacheLimit = cacheLimit }
        print("DiT idle cache limit: \(Memory.cacheLimit) bytes")
        let (transformer, _) = try loadTransformer(model, loras: loras)
        let (lh, lw) = (height / 16, width / 16)
        let embeds = (MLXRandom.normal([1, textLength, 4096], key: MLXRandom.key(5)) * 4).asType(.bfloat16)
        var latents = MLXRandom.normal([1, lh * lw, 64], key: MLXRandom.key(6)).asType(.bfloat16)
        let rope = transformer.rope(textLength: textLength, latentHeight: lh, latentWidth: lw)
        let cache = transformer.prefill(text: embeds, rope: rope.text)
        eval(cache.arrays + [rope.image.cos, rope.image.sin])
        Memory.peakMemory = 0
        let sampler = FootprintSampler()
        sampler.start()
        defer { sampler.stop() }
        _ = sampler.takePeak()
        var times: [Double] = []
        for step in 0...steps {
            let start = Date()
            let sigma = Float(1 - Double(step) / Double(steps + 2))
            let velocity = transformer(latents: latents, sigma: sigma, cache: cache, rope: rope.image)
            latents = (latents.asType(.float32) - 0.05 * velocity.asType(.float32)).asType(.bfloat16)
            eval(latents)
            if step > 0 { times.append(Date().timeIntervalSince(start)) }
        }
        return (times, sampler.takePeak(), Memory.peakMemory)
    }
}
