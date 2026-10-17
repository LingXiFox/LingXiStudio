import Foundation
import MLX

public struct GenerationRequest: Codable, Sendable {
    public var prompt: String
    public var negativePrompt: String?
    public var width: Int
    public var height: Int
    public var steps: Int
    /// True classifier-free guidance; only active when `> 1` and a negative prompt is given.
    public var trueCFGScale: Float
    public var seed: UInt64

    public init(
        prompt: String, negativePrompt: String? = nil, width: Int = 1024, height: Int = 1024, steps: Int = 40,
        trueCFGScale: Float = 1.0, seed: UInt64
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.trueCFGScale = trueCFGScale
        self.seed = seed
    }

    public var usesTrueCFG: Bool {
        trueCFGScale > 1 && !(negativePrompt ?? "").isEmpty
    }
}

public enum GenerationStage: String, Codable, Sendable {
    case validating, loadingTextEncoder, encodingPrompt, loadingTransformer, prefill, denoising
    case loadingVAE, decoding, done
}

public struct MemoryStats: Codable, Sendable {
    public var activeBytes: Int
    public var cacheBytes: Int
    public var peakBytes: Int

    static func current() -> MemoryStats {
        MemoryStats(activeBytes: Memory.activeMemory, cacheBytes: Memory.cacheMemory, peakBytes: Memory.peakMemory)
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
}

public enum PipelineError: Error, CustomStringConvertible {
    case invalidSize(Int, Int)
    case cancelled
    case emptyPrompt

    public var description: String {
        switch self {
        case .invalidSize(let w, let h): "Width and height must be positive multiples of 32 (got \(w)x\(h))"
        case .cancelled: "Generation cancelled"
        case .emptyPrompt: "Prompt is empty"
        }
    }
}

/// Qwen-Image-2.1 text-to-image pipeline with staged component residency.
///
/// Only one large component is resident at a time (text encoder ~16 GB, then transformer ~14 GB, then VAE),
/// which keeps a BF16 run within a 32 GB unified-memory budget.
public final class QwenImagePipeline: @unchecked Sendable {
    public let model: ModelDirectory
    private let tokenizer: QwenTokenizer
    private let scheduleConfig: FlowMatchSchedule.Config

    static let systemPrompt = "Comprehend and analyze the provided prompt."
    static let systemPrefix = "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"

    public init(model: ModelDirectory) throws {
        self.model = model
        tokenizer = try QwenTokenizer(directory: model.processor)
        scheduleConfig = try FlowMatchSchedule.Config(contentsOf: model.scheduler.appending(path: "scheduler_config.json"))
    }

    /// Token ids of the prompt as the reference pipeline builds them, and how many leading system tokens to drop.
    public func promptTokens(_ prompt: String) throws -> (ids: [Int], drop: Int) {
        let text = prompt.isEmpty ? " " : prompt
        let full = "\(Self.systemPrefix)<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        return (try tokenizer.encode(full), try tokenizer.encode(Self.systemPrefix).count)
    }

    public func generate(
        _ request: GenerationRequest,
        progress: @escaping (GenerationProgress) -> Void = { _ in },
        isCancelled: @escaping () -> Bool = { false }
    ) throws -> GeneratedImage {
        guard request.width > 0, request.height > 0, request.width % 32 == 0, request.height % 32 == 0 else {
            throw PipelineError.invalidSize(request.width, request.height)
        }
        let start = Date()
        var timings: [StageTiming] = []
        var stageStart = Date()
        Memory.peakMemory = 0

        func report(_ stage: GenerationStage, _ fraction: Double, step: Int = 0, secondsPerStep: Double? = nil) {
            progress(
                GenerationProgress(
                    stage: stage, fraction: fraction, step: step, totalSteps: request.steps,
                    secondsPerStep: secondsPerStep, elapsed: Date().timeIntervalSince(start),
                    memory: .current()))
        }
        func finish(_ stage: GenerationStage) {
            timings.append(StageTiming(stage: stage, seconds: Date().timeIntervalSince(stageStart)))
            stageStart = Date()
        }
        func checkCancelled() throws {
            if isCancelled() { throw PipelineError.cancelled }
        }

        // 1. Text encoding. The encoder is released before the transformer is loaded.
        report(.validating, 0)
        let positive = try promptTokens(request.prompt)
        let negative = request.usesTrueCFG ? try promptTokens(request.negativePrompt ?? "") : nil
        finish(.validating)

        report(.loadingTextEncoder, 0)
        var embeddings: [MLXArray] = []
        do {
            let encoder = try QwenTextEncoder(store: WeightStore(directory: model.textEncoder, component: "text_encoder"))
            encoder.materialize { report(.loadingTextEncoder, $0) }
            finish(.loadingTextEncoder)
            try checkCancelled()

            report(.encodingPrompt, 0)
            for tokens in [positive] + (negative.map { [$0] } ?? []) {
                let hidden = encoder(tokens.ids)
                let embedding = hidden[0..., tokens.drop...]
                eval(embedding)
                embeddings.append(embedding)
            }
        }
        Memory.clearCache()
        finish(.encodingPrompt)
        try checkCancelled()

        // 2. Denoising.
        report(.loadingTransformer, 0)
        let latentHeight = request.height / 16
        let latentWidth = request.width / 16
        let imageTokens = latentHeight * latentWidth
        let schedule = FlowMatchSchedule(steps: request.steps, imageSeqLen: imageTokens, config: scheduleConfig)
        var latents = MLXRandom.normal([1, imageTokens, 64], key: MLXRandom.key(request.seed)).asType(.bfloat16)
        do {
            let transformer = try QwenImageTransformer(
                store: WeightStore(directory: model.transformer, component: "transformer"))
            transformer.materialize { report(.loadingTransformer, $0) }
            finish(.loadingTransformer)
            try checkCancelled()

            report(.prefill, 0)
            var caches: [(QwenImageTransformer.TextCache, QwenImageTransformer.Rope)] = []
            for embedding in embeddings {
                let rope = transformer.rope(
                    textLength: embedding.dim(1), latentHeight: latentHeight, latentWidth: latentWidth)
                let cache = transformer.prefill(text: embedding, rope: rope.text)
                eval(cache.arrays + [rope.image.cos, rope.image.sin])
                caches.append((cache, rope.image))
            }
            embeddings.removeAll()
            finish(.prefill)

            var stepTimes: [Double] = []
            for step in 0..<schedule.steps {
                try checkCancelled()
                let stepStart = Date()
                let sigma = schedule.sigmas[step]
                var velocity = transformer(latents: latents, sigma: sigma, cache: caches[0].0, rope: caches[0].1)
                if caches.count > 1 {
                    let unconditional = transformer(
                        latents: latents, sigma: sigma, cache: caches[1].0, rope: caches[1].1)
                    velocity = unconditional + request.trueCFGScale * (velocity - unconditional)
                }
                // Euler step in fp32, stored back in the model dtype like the reference scheduler.
                let dt = schedule.sigmas[step + 1] - sigma
                latents = (latents.asType(.float32) + dt * velocity.asType(.float32)).asType(.bfloat16)
                eval(latents)
                stepTimes.append(Date().timeIntervalSince(stepStart))
                // The first step includes kernel compilation, so the rate excludes it once more are available.
                let recent = stepTimes.count > 1 ? Array(stepTimes.dropFirst().suffix(5)) : stepTimes
                report(
                    .denoising, Double(step + 1) / Double(schedule.steps), step: step + 1,
                    secondsPerStep: recent.reduce(0, +) / Double(recent.count))
            }
        }
        Memory.clearCache()
        finish(.denoising)
        try checkCancelled()

        // 3. Decoding.
        report(.loadingVAE, 0)
        var pixels: MLXArray
        do {
            let vae = try QwenImageVAEDecoder(
                store: WeightStore(directory: model.vae, component: "vae"),
                configURL: model.vae.appending(path: "config.json"), dtype: .float32)
            vae.materialize { report(.loadingVAE, $0) }
            finish(.loadingVAE)

            report(.decoding, 0)
            let image = vae.decode(latents, latentHeight: latentHeight, latentWidth: latentWidth)
            // (x / 2 + 0.5) * 255, rounded, as the reference image processor does.
            pixels = round(clip(image * 0.5 + 0.5, min: 0, max: 1) * 255).asType(.uint8)
            eval(pixels)
        }
        let rgba = pixels.asArray(UInt8.self)
        Memory.clearCache()
        finish(.decoding)
        report(.done, 1)

        return GeneratedImage(
            width: request.width, height: request.height, rgba: rgba, request: request, timings: timings,
            peakMemoryBytes: Memory.peakMemory, totalSeconds: Date().timeIntervalSince(start))
    }
}
