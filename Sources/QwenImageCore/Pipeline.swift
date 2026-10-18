import Foundation
import MLX

/// Creation modes. Text-to-image, reference and edit follow the official Qwen-Image-2.1 pipeline; image-to-image
/// (SDEdit start from a noised image) and inpainting (edit conditioning plus per-step latent blending) are generic
/// diffusion methods applied on top of it, as Qwen-Image-2.1 has no official pipeline for them.
public enum GenerationMode: String, Codable, Sendable, CaseIterable {
    case textToImage = "text-to-image"
    case reference
    case imageToImage = "image-to-image"
    case edit
    case inpaint
}

/// Where an input image came from, recorded with every result so a run can be reproduced.
public struct InputImageRecord: Codable, Sendable, Hashable {
    public enum Role: String, Codable, Sendable { case condition, initial, mask }
    public var role: Role
    /// The stored copy the run used (inside the output folder's `inputs/`), or the original file.
    public var path: String
    /// File the user picked, when different from `path`.
    public var source: String?
    public var sha256: String
    public var width: Int
    public var height: Int

    public init(role: Role, path: String, source: String?, sha256: String, width: Int, height: Int) {
        self.role = role
        self.path = path
        self.source = source
        self.sha256 = sha256
        self.width = width
        self.height = height
    }
}

public struct GenerationRequest: Codable, Sendable {
    public var prompt: String
    public var negativePrompt: String?
    public var width: Int
    public var height: Int
    public var steps: Int
    /// True classifier-free guidance; only active when `> 1` and a negative prompt is given.
    public var trueCFGScale: Float
    public var seed: UInt64
    public var scheduler: SamplerSchedule
    public var loras: [LoRASpec]
    public var mode: GenerationMode
    /// Input images in the order the pipeline uses them (conditions first, then the initial image and mask).
    public var inputs: [InputImageRecord]
    /// Share of the schedule that runs from the (noised) initial image: 1 starts from pure noise.
    public var strength: Float
    /// Inpainting: blur radius (pixels) of the mask edge used when pasting the result over the original.
    public var maskFeather: Int
    public var maskInvert: Bool
    /// Inpainting: keep the original pixels outside the (feathered) mask exactly.
    public var keepUnmaskedPixels: Bool

    public init(
        prompt: String, negativePrompt: String? = nil, width: Int = 1024, height: Int = 1024, steps: Int = 40,
        trueCFGScale: Float = 1.0, seed: UInt64, scheduler: SamplerSchedule = .linear, loras: [LoRASpec] = [],
        mode: GenerationMode = .textToImage, inputs: [InputImageRecord] = [], strength: Float = 1,
        maskFeather: Int = 8, maskInvert: Bool = false, keepUnmaskedPixels: Bool = true
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.trueCFGScale = trueCFGScale
        self.seed = seed
        self.scheduler = scheduler
        self.loras = loras
        self.mode = mode
        self.inputs = inputs
        self.strength = strength
        self.maskFeather = maskFeather
        self.maskInvert = maskInvert
        self.keepUnmaskedPixels = keepUnmaskedPixels
    }

    // Records written before the image modes existed decode as text-to-image.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        prompt = try c.decode(String.self, forKey: .prompt)
        negativePrompt = try c.decodeIfPresent(String.self, forKey: .negativePrompt)
        width = try c.decode(Int.self, forKey: .width)
        height = try c.decode(Int.self, forKey: .height)
        steps = try c.decode(Int.self, forKey: .steps)
        trueCFGScale = try c.decode(Float.self, forKey: .trueCFGScale)
        seed = try c.decode(UInt64.self, forKey: .seed)
        scheduler = try c.decodeIfPresent(SamplerSchedule.self, forKey: .scheduler) ?? .linear
        loras = try c.decodeIfPresent([LoRASpec].self, forKey: .loras) ?? []
        mode = try c.decodeIfPresent(GenerationMode.self, forKey: .mode) ?? .textToImage
        inputs = try c.decodeIfPresent([InputImageRecord].self, forKey: .inputs) ?? []
        strength = try c.decodeIfPresent(Float.self, forKey: .strength) ?? 1
        maskFeather = try c.decodeIfPresent(Int.self, forKey: .maskFeather) ?? 8
        maskInvert = try c.decodeIfPresent(Bool.self, forKey: .maskInvert) ?? false
        keepUnmaskedPixels = try c.decodeIfPresent(Bool.self, forKey: .keepUnmaskedPixels) ?? true
    }

    public var usesTrueCFG: Bool {
        trueCFGScale > 1 && !(negativePrompt ?? "").isEmpty
    }
}

/// Image data for a request (the request itself only records where the images came from).
public struct GenerationInputs: Sendable {
    /// Condition images in prompt order (Picture 1, 2, ...). For edit and inpaint the first one is the main image.
    public var conditions: [RGBAImage]
    /// Starting image for image-to-image, partial-strength edit and inpaint.
    public var initial: RGBAImage?
    /// Inpainting mask: bright = repaint.
    public var mask: RGBAImage?

    public init(conditions: [RGBAImage] = [], initial: RGBAImage? = nil, mask: RGBAImage? = nil) {
        self.conditions = conditions
        self.initial = initial
        self.mask = mask
    }

    public static let none = GenerationInputs()
}

public enum SamplerSchedule: String, Codable, Sendable {
    /// Default 40-step grid with the terminal stretch.
    case linear
    /// Fixed 6-node schedule of the Viggle turbo LoRA, without the terminal stretch.
    case viggleTurbo = "viggle_turbo"
}

public enum GenerationStage: String, Codable, Sendable {
    case validating, loadingTextEncoder, encodingPrompt, encodingImages, loadingTransformer, prefill, denoising
    case loadingVAE, decoding, done
}

public struct MemoryStats: Codable, Sendable {
    public var activeBytes: Int
    public var cacheBytes: Int
    public var peakBytes: Int
    /// Process `phys_footprint`, including memory MLX does not account for.
    public var footprintBytes: Int

    static func current() -> MemoryStats {
        MemoryStats(
            activeBytes: Memory.activeMemory, cacheBytes: Memory.cacheMemory, peakBytes: Memory.peakMemory,
            footprintBytes: ProcessMemory.footprint())
    }
}

public struct GenerationProgress: Sendable {
    public var stage: GenerationStage
    /// Fraction of the current stage, 0...1.
    public var fraction: Double
    public var step: Int
    public var totalSteps: Int
    public var secondsPerStep: Double?
    public var elapsed: TimeInterval
    public var memory: MemoryStats
}

public struct StageTiming: Codable, Sendable {
    public var stage: GenerationStage
    public var seconds: Double
    /// Peak process footprint while the stage ran.
    public var peakFootprintBytes: Int
}

public struct GeneratedImage: Sendable {
    public var width: Int
    public var height: Int
    /// Interleaved 8-bit RGBA, row-major.
    public var rgba: [UInt8]
    public var request: GenerationRequest
    public var timings: [StageTiming]
    public var peakMemoryBytes: Int
    public var totalSeconds: Double
    /// Wall time of every denoising step (transformer + Euler update, synchronized).
    public var stepSeconds: [Double]
}

public enum PipelineError: Error, CustomStringConvertible {
    case invalidSize(Int, Int)
    case cancelled
    case emptyPrompt
    case scheduleSteps(SamplerSchedule, Int)
    case missingInput(String)

    public var description: String {
        switch self {
        case .missingInput(let what): "Missing input: \(what)"
        case .invalidSize(let w, let h): "Width and height must be positive multiples of 32 (got \(w)x\(h))"
        case .cancelled: "Generation cancelled"
        case .emptyPrompt: "Prompt is empty"
        case .scheduleSteps(let schedule, let steps): "\(schedule.rawValue) needs exactly \(steps) steps"
        }
    }
}

/// Qwen-Image-2.1 text-to-image pipeline with staged component residency.
///
/// Only one large component is resident at a time (text encoder ~16 GB, then transformer ~14 GB, then VAE),
/// which keeps a BF16 run within a 32 GB unified-memory budget.
public final class QwenImagePipeline: @unchecked Sendable {
    public let model: ModelDirectory
    let tokenizer: QwenTokenizer
    private let scheduleConfig: FlowMatchSchedule.Config

    static let systemPrompt = "Comprehend and analyze the provided prompt."
    static let systemPrefix = "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"

    public init(model: ModelDirectory) throws {
        self.model = model
        tokenizer = try QwenTokenizer(directory: model.processor)
        scheduleConfig = try FlowMatchSchedule.Config(contentsOf: model.scheduler.appending(path: "scheduler_config.json"))
    }

    /// `VaeImageProcessor.postprocess`: `(x / 2 + 0.5).clamp(0, 1)`, then `round(x * 255)` to uint8.
    static func toRGBA8(_ image: MLXArray) -> MLXArray {
        round(clip(image * 0.5 + 0.5, min: 0, max: 1) * 255).asType(.uint8)
    }

    func schedule(for request: GenerationRequest, imageTokens: Int) -> FlowMatchSchedule {
        switch request.scheduler {
        case .linear: FlowMatchSchedule(steps: request.steps, imageSeqLen: imageTokens, config: scheduleConfig)
        case .viggleTurbo:
            FlowMatchSchedule(
                nodes: FlowMatchSchedule.viggleTurboNodes, imageSeqLen: imageTokens, config: scheduleConfig,
                useTerminal: false)
        }
    }

    /// Token ids of the prompt as the reference pipeline builds them, and how many leading system tokens to drop.
    public func promptTokens(_ prompt: String) throws -> (ids: [Int], drop: Int) {
        let text = prompt.isEmpty ? " " : prompt
        let full = "\(Self.systemPrefix)<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        return (try tokenizer.encode(full), try tokenizer.encode(Self.systemPrefix).count)
    }

    public func generate(
        _ request: GenerationRequest,
        inputs: GenerationInputs = .none,
        progress: @escaping (GenerationProgress) -> Void = { _ in },
        isCancelled: @escaping () -> Bool = { false }
    ) throws -> GeneratedImage {
        guard request.width > 0, request.height > 0, request.width % 32 == 0, request.height % 32 == 0 else {
            throw PipelineError.invalidSize(request.width, request.height)
        }
        if request.scheduler == .viggleTurbo, request.steps != FlowMatchSchedule.viggleTurboNodes.count {
            throw PipelineError.scheduleSteps(.viggleTurbo, FlowMatchSchedule.viggleTurboNodes.count)
        }
        let plan = try ModePlan(request: request, inputs: inputs)
        let start = Date()
        var timings: [StageTiming] = []
        var stageStart = Date()
        Memory.peakMemory = 0
        let sampler = FootprintSampler()
        sampler.start()
        defer { sampler.stop() }
        // Cancellation and errors leave a stage early; its freed buffers must not stay in the MLX cache.
        defer { releaseStageMemory() }
        _ = sampler.takePeak()

        let latentHeight = request.height / 16
        let latentWidth = request.width / 16
        let imageTokens = latentHeight * latentWidth
        let schedule = schedule(for: request, imageTokens: imageTokens)
        let startIndex = plan.startIndex(steps: schedule.steps)
        let stepsToRun = schedule.steps - startIndex

        func report(_ stage: GenerationStage, _ fraction: Double, step: Int = 0, secondsPerStep: Double? = nil) {
            progress(
                GenerationProgress(
                    stage: stage, fraction: fraction, step: step, totalSteps: stepsToRun,
                    secondsPerStep: secondsPerStep, elapsed: Date().timeIntervalSince(start),
                    memory: .current()))
        }
        func finish(_ stage: GenerationStage) {
            timings.append(
                StageTiming(
                    stage: stage, seconds: Date().timeIntervalSince(stageStart), peakFootprintBytes: sampler.takePeak()))
            stageStart = Date()
        }
        func checkCancelled() throws {
            if isCancelled() { throw PipelineError.cancelled }
        }

        // 1. Prompt encoding (with the vision tower for condition images). Released before the transformer loads.
        report(.validating, 0)
        let conditionSizes = plan.conditions.map { ($0.width, $0.height) }
        struct Encoded {
            let embedding: MLXArray
            let imageRanges: [Range<Int>]
        }
        var prompts: [(ids: [Int], drop: Int, ranges: [Range<Int>], positions: [[Int32]]?)] = []
        for text in [request.prompt] + (request.usesTrueCFG ? [request.negativePrompt ?? ""] : []) {
            if plan.conditions.isEmpty {
                let tokens = try promptTokens(text)
                prompts.append((tokens.ids, tokens.drop, [], nil))
            } else {
                let mp = try MultimodalPrompt(prompt: text, imageSizes: conditionSizes, tokenizer: tokenizer)
                prompts.append((mp.ids, mp.drop, mp.imageRanges, mp.positions))
            }
        }
        finish(.validating)

        report(.loadingTextEncoder, 0)
        var encoded: [Encoded] = []
        // Each stage runs in its own autorelease pool: Metal command buffers are autoreleased and keep the buffers
        // they used alive, and a thread without a run loop would otherwise hold them until the whole run ends.
        try autoreleasepool {
            let store = try WeightStore(directory: model.textEncoder, component: "text_encoder")
            var vision: [QwenVisionEncoder.Output] = []
            if !plan.conditions.isEmpty {
                try autoreleasepool {
                    let tower = try QwenVisionEncoder(store: store)
                    try tower.materialize(checkCancelled: checkCancelled)
                    for image in plan.conditions {
                        try checkCancelled()
                        let out = tower.encode(image)
                        eval([out.embeds] + out.deepstack)
                        vision.append(out)
                    }
                }
                releaseStageMemory()
            }
            let encoder = try QwenTextEncoder(store: store)
            try encoder.materialize(checkCancelled: checkCancelled) { report(.loadingTextEncoder, $0) }
            finish(.loadingTextEncoder)
            try checkCancelled()

            report(.encodingPrompt, 0)
            let visionInput: (([[Int32]], [Range<Int>]) -> QwenTextEncoder.VisionInput)? =
                vision.isEmpty
                ? nil
                : { positions, ranges in
                    QwenTextEncoder.VisionInput(
                        embeds: concatenated(vision.map(\.embeds), axis: 0),
                        deepstack: (0..<vision[0].deepstack.count).map { i in
                            concatenated(vision.map { $0.deepstack[i] }, axis: 0)
                        },
                        ranges: ranges, positions: positions)
                }
            for prompt in prompts {
                try checkCancelled()
                let hidden = try encoder(
                    prompt.ids, vision: prompt.positions.flatMap { p in visionInput?(p, prompt.ranges) },
                    checkCancelled: checkCancelled)
                let embedding = hidden[0..., prompt.drop...]
                eval(embedding)
                encoded.append(
                    Encoded(
                        embedding: embedding,
                        imageRanges: prompt.ranges.map { ($0.lowerBound - prompt.drop)..<($0.upperBound - prompt.drop) }))
            }
        }
        releaseStageMemory()
        finish(.encodingPrompt)
        try checkCancelled()

        // 2. VAE encoding of condition images and of the initial image.
        var conditionLatents: [QwenImageTransformer.ConditionImage] = []
        var initialLatents: MLXArray?
        if !plan.conditions.isEmpty || plan.initial != nil {
            report(.encodingImages, 0)
            try autoreleasepool {
                let encoder = try QwenImageVAEEncoder(
                    store: WeightStore(directory: model.vae, component: "vae"),
                    configURL: model.vae.appending(path: "config.json"))
                try encoder.materialize(checkCancelled: checkCancelled)
                for image in plan.conditions {
                    try checkCancelled()
                    let latents = encoder.encode(image.vaeInput()).asType(.bfloat16)
                    eval(latents)
                    conditionLatents.append(.init(latents: latents, height: image.height / 16, width: image.width / 16))
                }
                if let initial = plan.initial {
                    try checkCancelled()
                    let latents = encoder.encode(initial.vaeInput()).asType(.bfloat16)
                    eval(latents)
                    initialLatents = latents
                }
            }
            releaseStageMemory()
            finish(.encodingImages)
            try checkCancelled()
        }

        // 3. Denoising.
        report(.loadingTransformer, 0)
        var stepTimes: [Double] = []
        let noise = MLXRandom.normal([1, imageTokens, 64], key: MLXRandom.key(request.seed)).asType(.bfloat16)
        let latentMask = plan.latentMask(height: latentHeight, width: latentWidth)
        var latents = noise
        if let initialLatents, startIndex > 0 {
            latents = Self.scaleNoise(initialLatents, sigma: schedule.sigmas[startIndex], noise: noise)
        }
        try autoreleasepool {
            let transformer = try QwenImageTransformer(
                store: WeightStore(directory: model.transformer, component: "transformer"))
            try transformer.materialize(checkCancelled: checkCancelled) { report(.loadingTransformer, $0) }
            // LoRA factors stay separate from the base weights and are applied on the fly.
            let loras = LoRAStack()
            for spec in request.loras {
                try checkCancelled()
                try loras.load(spec)
            }
            try transformer.bind(loras)
            finish(.loadingTransformer)
            try checkCancelled()

            report(.prefill, 0)
            var caches: [(QwenImageTransformer.TextCache, QwenImageTransformer.Rope)] = []
            for item in encoded {
                try checkCancelled()
                let layout = transformer.layout(
                    textLength: item.embedding.dim(1), imageRanges: item.imageRanges,
                    conditions: conditionLatents.map { ($0.height, $0.width) }, latentHeight: latentHeight,
                    latentWidth: latentWidth)
                let cache = transformer.prefill(
                    text: item.embedding, conditions: conditionLatents, imageRanges: item.imageRanges, layout: layout)
                eval(cache.arrays + [layout.target.cos, layout.target.sin])
                caches.append((cache, layout.target))
            }
            encoded.removeAll()
            conditionLatents.removeAll()
            finish(.prefill)

            for step in startIndex..<schedule.steps {
                try checkCancelled()
                let stepStart = Date()
                let sigma = schedule.sigmas[step]
                var velocity = transformer(latents: latents, sigma: sigma, cache: caches[0].0, rope: caches[0].1)
                if caches.count > 1 {
                    let unconditional = transformer(
                        latents: latents, sigma: sigma, cache: caches[1].0, rope: caches[1].1)
                    velocity = unconditional + request.trueCFGScale * (velocity - unconditional)
                }
                latents = schedule.step(latents: latents, velocity: velocity, index: step)
                if let latentMask, let initialLatents {
                    // Outside the mask the sample follows the original image, noised to the next sigma
                    // (`QwenImageEditInpaintPipeline`); after the last step that is the clean original.
                    let next = schedule.sigmas[step + 1]
                    let original = next > 0 ? Self.scaleNoise(initialLatents, sigma: next, noise: noise) : initialLatents
                    latents = ((1 - latentMask) * original + latentMask * latents).asType(latents.dtype)
                }
                eval(latents)
                stepTimes.append(Date().timeIntervalSince(stepStart))
                // The first step includes kernel compilation, so the rate excludes it once more are available.
                let recent = stepTimes.count > 1 ? Array(stepTimes.dropFirst().suffix(5)) : stepTimes
                report(
                    .denoising, Double(step - startIndex + 1) / Double(stepsToRun), step: step - startIndex + 1,
                    secondsPerStep: recent.reduce(0, +) / Double(recent.count))
            }
        }
        releaseStageMemory()
        finish(.denoising)
        try checkCancelled()

        // 4. Decoding.
        report(.loadingVAE, 0)
        let pixels = try autoreleasepool { () throws -> MLXArray in
            let vae = try QwenImageVAEDecoder(
                store: WeightStore(directory: model.vae, component: "vae"),
                configURL: model.vae.appending(path: "config.json"), dtype: .float32)
            try vae.materialize(checkCancelled: checkCancelled) { report(.loadingVAE, $0) }
            finish(.loadingVAE)
            try checkCancelled()

            report(.decoding, 0)
            let image = vae.decode(latents, latentHeight: latentHeight, latentWidth: latentWidth)
            var pixels = QwenImagePipeline.toRGBA8(image)
            if let composite = plan.pixelComposite(width: request.width, height: request.height) {
                pixels = composite(pixels)
            }
            eval(pixels)
            return pixels
        }
        try checkCancelled()
        let rgba = pixels.asArray(UInt8.self)
        releaseStageMemory()
        finish(.decoding)
        report(.done, 1)

        return GeneratedImage(
            width: request.width, height: request.height, rgba: rgba, request: request, timings: timings,
            peakMemoryBytes: Memory.peakMemory, totalSeconds: Date().timeIntervalSince(start), stepSeconds: stepTimes)
    }

    /// Flow-matching forward noising (`FlowMatchEulerDiscreteScheduler.scale_noise`): `sigma * noise + (1 - sigma) * x`.
    static func scaleNoise(_ x: MLXArray, sigma: Float, noise: MLXArray) -> MLXArray {
        (sigma * noise.asType(.float32) + (1 - sigma) * x.asType(.float32)).asType(x.dtype)
    }

    /// Returns a finished stage's buffers to the system. Command buffer completion handlers can still hold the stage's
    /// arrays after `eval` returns; waiting for the stream first makes sure they are freed before the cache is cleared,
    /// instead of landing in the cache while the next stage loads.
    private func releaseStageMemory() {
        Stream.gpu.synchronize()
        Memory.clearCache()
    }
}
