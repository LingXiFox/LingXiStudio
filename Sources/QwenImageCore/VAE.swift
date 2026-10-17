import Foundation
import MLX

/// Image decoder of `AutoencoderKLQwenImage21` for single-frame (image) latents.
///
/// For one frame every causal 3D convolution reduces to a 2D convolution, the temporal `time_conv` of the
/// first chunk is skipped, and the parameter-free `DupUp` shortcut keeps only its last temporal slot.
/// Tensors are NHWC throughout.
final class QwenImageVAEDecoder: ParameterContainer {
    struct Conv {
        let weight: MLXArray  // [out, kh, kw, in]
        let bias: MLXArray

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let padding = weight.dim(1) / 2
            return conv2d(x, weight, padding: .init(padding)) + bias
        }
    }

    struct ResBlock {
        let norm1: MLXArray
        let conv1: Conv
        let norm2: MLXArray
        let conv2: Conv
        let shortcut: Conv?
    }

    struct UpBlock {
        let resnets: [ResBlock]
        let upsampler: Conv?
        let inDim: Int
        let outDim: Int
        let temporalFactor: Int
    }

    struct Attention {
        let norm: MLXArray
        let qkv: Conv
        let proj: Conv
    }

    let dtype: DType
    let latentsMean: MLXArray
    let latentsStd: MLXArray
    let postQuantConv: Conv
    let convIn: Conv
    let midResnets: [ResBlock]
    let midAttention: Attention
    let upBlocks: [UpBlock]
    let normOut: MLXArray
    let convOut: Conv

    var parameters: [MLXArray] {
        func conv(_ c: Conv) -> [MLXArray] { [c.weight, c.bias] }
        func res(_ r: ResBlock) -> [MLXArray] {
            [r.norm1, r.norm2] + conv(r.conv1) + conv(r.conv2) + (r.shortcut.map(conv) ?? [])
        }
        return [latentsMean, latentsStd, normOut] + conv(postQuantConv) + conv(convIn) + conv(convOut)
            + midResnets.flatMap(res) + [midAttention.norm] + conv(midAttention.qkv) + conv(midAttention.proj)
            + upBlocks.flatMap { $0.resnets.flatMap(res) + ($0.upsampler.map(conv) ?? []) }
    }

    init(store: WeightStore, configURL: URL, dtype: DType = .bfloat16) throws {
        self.dtype = dtype
        let data = try Data(contentsOf: configURL)
        let config = (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        let mean = (config["latents_mean"] as? [Double]) ?? []
        let std = (config["latents_std"] as? [Double]) ?? []
        let zDim = (config["z_dim"] as? Int) ?? 64
        let baseDim = (config["decoder_base_dim"] as? Int) ?? 144
        let dimMult = (config["dim_mult"] as? [Int]) ?? [1, 2, 4, 8, 8]
        let numRes = (config["num_res_blocks"] as? Int) ?? 2
        let temporal = (config["temperal_downsample"] as? [Bool]) ?? [false, true, true, true]
        guard mean.count == zDim, std.count == zDim else {
            throw WeightError.shapeMismatch("latents_mean/std", expected: [zDim], actual: [mean.count])
        }
        latentsMean = MLXArray(mean.map { Float($0) })
        latentsStd = MLXArray(std.map { Float($0) })

        func conv(_ key: String) throws -> Conv {
            let weight = try store.get("\(key).weight")
            // PyTorch [out, in, kh, kw] -> MLX [out, kh, kw, in]
            return Conv(weight: weight.transposed(0, 2, 3, 1).asType(dtype), bias: try store.get("\(key).bias", dtype: dtype))
        }
        func gamma(_ key: String) throws -> MLXArray {
            let value = try store.get(key)
            return value.reshaped(value.dim(0)).asType(.float32)
        }
        func res(_ key: String) throws -> ResBlock {
            ResBlock(
                norm1: try gamma("\(key).norm1.gamma"), conv1: try conv("\(key).conv1"),
                norm2: try gamma("\(key).norm2.gamma"), conv2: try conv("\(key).conv2"),
                shortcut: store.contains("\(key).conv_shortcut.weight") ? try conv("\(key).conv_shortcut") : nil)
        }

        postQuantConv = try conv("post_quant_conv")
        convIn = try conv("decoder.conv_in")
        midResnets = try [res("decoder.mid_block.resnets.0"), res("decoder.mid_block.resnets.1")]
        midAttention = Attention(
            norm: try gamma("decoder.mid_block.attentions.0.norm.gamma"),
            qkv: try conv("decoder.mid_block.attentions.0.to_qkv"),
            proj: try conv("decoder.mid_block.attentions.0.proj"))

        let dims = [dimMult.last!] + dimMult.reversed()
        let channels = dims.map { $0 * baseDim }
        let temporalUp = Array(temporal.reversed())
        upBlocks = try (0..<(channels.count - 1)).map { i in
            let key = "decoder.up_blocks.\(i)"
            let up = i != dimMult.count - 1
            return UpBlock(
                resnets: try (0...numRes).map { try res("\(key).resnets.\($0)") },
                upsampler: up ? try conv("\(key).upsampler.resample.1") : nil,
                inDim: channels[i], outDim: channels[i + 1],
                temporalFactor: up && i < temporalUp.count && temporalUp[i] ? 2 : 1)
        }
        normOut = try gamma("decoder.norm_out.gamma")
        convOut = try conv("decoder.conv_out")
    }

    /// Wan-style RMS norm over channels: `F.normalize(x) * sqrt(C) * gamma`, normalized in fp32.
    private func channelNorm(_ x: MLXArray, _ gamma: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let norm = maximum(sqrt(sum(xf * xf, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        let normalized = (xf / norm).asType(x.dtype)
        return normalized * (Float(gamma.dim(0)).squareRoot() * gamma).asType(x.dtype)
    }

    private func resBlock(_ block: ResBlock, _ x: MLXArray) -> MLXArray {
        let h = block.shortcut.map { $0(x) } ?? x
        var y = block.conv1(silu(channelNorm(x, block.norm1)))
        y = block.conv2(silu(channelNorm(y, block.norm2)))
        return y + h
    }

    private func attention(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let qkv = midAttention.qkv(channelNorm(x, midAttention.norm)).reshaped(b, 1, h * w, 3 * c)
        let parts = split(qkv, parts: 3, axis: -1)
        let out = MLXFast.scaledDotProductAttention(
            queries: parts[0], keys: parts[1], values: parts[2], scale: 1 / Float(c).squareRoot(), mask: .none)
        return x + midAttention.proj(out.reshaped(b, h, w, c))
    }

    private func nearestUpsample(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let expanded = broadcast(x.reshaped(b, h, 1, w, 1, c), to: [b, h, 2, w, 2, c])
        return expanded.reshaped(b, 2 * h, 2 * w, c)
    }

    /// `QwenImage21DupUp3D` on the first chunk: channel repeat_interleave, rearranged into 2x spatial
    /// upsampling, keeping only the last of the `temporalFactor` time slots.
    private func dupUp(_ x: MLXArray, inDim: Int, outDim: Int, temporalFactor: Int) -> MLXArray {
        let (b, h, w) = (x.dim(0), x.dim(1), x.dim(2))
        let repeats = outDim * temporalFactor * 4 / inDim
        let y = repeated(x, count: repeats, axis: -1)
            .reshaped(b, h, w, outDim, temporalFactor, 2, 2)[0..., 0..., 0..., 0..., temporalFactor - 1]
        // [b, h, w, out, i, j] -> [b, h, i, w, j, out]
        return y.transposed(0, 1, 4, 2, 5, 3).reshaped(b, 2 * h, 2 * w, outDim)
    }

    /// Decodes normalized packed latents `[1, H/16 * W/16, 64]` to an RGBA image in `[-1, 1]`, `[1, H, W, 4]`.
    func decode(
        _ packed: MLXArray, latentHeight: Int, latentWidth: Int, evalEachStage: Bool = true,
        trace: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        var x = packed.asType(.float32).reshaped(1, latentHeight, latentWidth, -1)
        x = (x * latentsStd + latentsMean).asType(dtype)
        x = convIn(postQuantConv(x))
        trace?("conv_in", x)
        x = resBlock(midResnets[0], x)
        trace?("mid.res0", x)
        x = attention(x)
        trace?("mid.attn", x)
        x = resBlock(midResnets[1], x)
        trace?("mid", x)
        if evalEachStage { eval(x) }
        for (index, block) in upBlocks.enumerated() {
            let input = x
            for resnet in block.resnets { x = resBlock(resnet, x) }
            if let upsampler = block.upsampler {
                x = upsampler(nearestUpsample(x))
                x = x + dupUp(input, inDim: block.inDim, outDim: block.outDim, temporalFactor: block.temporalFactor)
            }
            trace?("up.\(index)", x)
            if evalEachStage { eval(x) }
        }
        x = convOut(silu(channelNorm(x, normOut)))
        return clip(x.asType(.float32), min: -1, max: 1)
    }
}
