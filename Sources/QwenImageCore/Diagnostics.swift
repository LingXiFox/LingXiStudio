import Foundation
import MLX

/// Component-level entry points used to check numerics against a reference implementation.
/// Each dumps its tensors to a safetensors file in fp32.
public enum Diagnostics {
    public static func tokenize(model: ModelDirectory, text: String) throws -> [Int] {
        try QwenTokenizer(directory: model.processor).encode(text)
    }

    /// Text-encoder hidden states (pre final norm) for the full templated prompt.
    public static func dumpTextEncoder(model: ModelDirectory, prompt: String, to url: URL) throws {
        let pipeline = try QwenImagePipeline(model: model)
        let tokens = try pipeline.promptTokens(prompt)
        let encoder = try QwenTextEncoder(store: WeightStore(directory: model.textEncoder, component: "text_encoder"))
        let hidden = encoder(tokens.ids)
        try save(
            arrays: [
                "hidden": hidden.asType(.float32),
                "ids": MLXArray(tokens.ids.map { Int32($0) }),
                "drop": MLXArray([Int32(tokens.drop)]),
            ], url: url)
    }

    /// One transformer evaluation on given inputs. Reads `embeds` [1,T,4096] and `latents` [1,N,64] from `input`.
    public static func dumpTransformer(
        model: ModelDirectory, input: URL, sigma: Float, latentHeight: Int, latentWidth: Int, to url: URL
    ) throws {
        let arrays = try loadArrays(url: input)
        guard let embeds = arrays["embeds"], let latents = arrays["latents"] else {
            throw WeightError.missingTensor("embeds/latents", component: input.path)
        }
        let transformer = try QwenImageTransformer(
            store: WeightStore(directory: model.transformer, component: "transformer"))
        let rope = transformer.rope(textLength: embeds.dim(1), latentHeight: latentHeight, latentWidth: latentWidth)
        let cache = transformer.prefill(text: embeds.asType(.bfloat16), rope: rope.text)
        let out = transformer(latents: latents.asType(.bfloat16), sigma: sigma, cache: cache, rope: rope.image)
        try save(
            arrays: ["velocity": out.asType(.float32), "temb": transformer.timeEmbedding(sigma: sigma).asType(.float32)],
            url: url)
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

    public static func schedule(steps: Int, imageSeqLen: Int, model: ModelDirectory) throws -> [Float] {
        let config = try FlowMatchSchedule.Config(contentsOf: model.scheduler.appending(path: "scheduler_config.json"))
        return FlowMatchSchedule(steps: steps, imageSeqLen: imageSeqLen, config: config).sigmas
    }
}
