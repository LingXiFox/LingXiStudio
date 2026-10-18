import Foundation
import MLX

/// Qwen3-VL vision tower (`model.visual` in the text encoder checkpoint), used for condition images.
///
/// Matches `Qwen3VLVisionModel` for one image:
/// - patches of 16x16 are embedded by a Conv3d over 2 duplicated frames;
/// - the 48x48 learned position table is bilinearly resampled (align_corners) to the patch grid;
/// - a 2D rotary embedding is applied in every block;
/// - after 27 blocks the 2x2 merger projects to the 4096-dim text space.
/// Patches run in spatial-merge-block order (each 2x2 group is contiguous), as the processor emits them, so the
/// merger only needs to fold consecutive groups of four tokens.
final class QwenVisionEncoder: ParameterContainer {
    struct Config {
        var hidden = 1152
        var depth = 27
        var heads = 16
        var intermediate = 4304
        var patch = 16
        var temporalPatch = 2
        var merge = 2
        var outHidden = 4096
        var positionGrid = 48
        var deepstackLayers = [8, 16, 24]
        var ropeTheta: Float = 10000
        var eps: Float = 1e-6
    }

    struct Dense {
        let weight: MLXArray
        let bias: MLXArray

        func callAsFunction(_ x: MLXArray) -> MLXArray { addMM(bias, x, weight.T) }
    }

    struct Block {
        let norm1: (MLXArray, MLXArray)
        let norm2: (MLXArray, MLXArray)
        let qkv: Dense
        let proj: Dense
        let fc1: Dense
        let fc2: Dense
    }

    struct Merger {
        let norm: (MLXArray, MLXArray)
        let fc1: Dense
        let fc2: Dense
        /// Deepstack mergers fold the 2x2 group first and normalize the 4608-dim vector; the final merger
        /// normalizes each 1152-dim token before folding.
        let normAfterFold: Bool
    }

    /// Output for one image.
    struct Output {
        /// `[tokens, 4096]`, one per 2x2 patch group, in raster order of the merged grid.
        let embeds: MLXArray
        /// Deepstack features from the configured layers, same shape as `embeds`.
        let deepstack: [MLXArray]
        /// Merged grid size (patch grid / 2).
        let gridHeight: Int
        let gridWidth: Int
    }

    let config: Config
    let patchWeight: MLXArray  // [hidden, 3 * 2 * 16 * 16]
    let patchBias: MLXArray
    let positionTable: MLXArray  // [48 * 48, hidden]
    let blocks: [Block]
    let merger: Merger
    let deepstackMergers: [Merger]

    var parameters: [MLXArray] {
        func dense(_ d: Dense) -> [MLXArray] { [d.weight, d.bias] }
        func merger(_ m: Merger) -> [MLXArray] { [m.norm.0, m.norm.1] + dense(m.fc1) + dense(m.fc2) }
        return [patchWeight, patchBias, positionTable] + merger(self.merger) + deepstackMergers.flatMap(merger)
            + blocks.flatMap {
                [$0.norm1.0, $0.norm1.1, $0.norm2.0, $0.norm2.1] + dense($0.qkv) + dense($0.proj) + dense($0.fc1)
                    + dense($0.fc2)
            }
    }

    /// The tower runs in fp32 by default: its activations are badly conditioned in bf16 (the reference's own
    /// bf16 run is ~10 % rel-L2 from fp32), the error carries into every text token after the image, and the
    /// tower is small enough (0.4 B parameters) that fp32 costs well under a second per image.
    init(store: WeightStore, config: Config = Config(), dtype: DType = .float32) throws {
        self.config = config
        let p = "model.visual"
        func dense(_ key: String) throws -> Dense {
            Dense(weight: try store.get("\(key).weight", dtype: dtype), bias: try store.get("\(key).bias", dtype: dtype))
        }
        func norm(_ key: String) throws -> (MLXArray, MLXArray) {
            (try store.get("\(key).weight", dtype: dtype), try store.get("\(key).bias", dtype: dtype))
        }
        func makeMerger(_ key: String, after: Bool) throws -> Merger {
            Merger(
                norm: try norm("\(key).norm"), fc1: try dense("\(key).linear_fc1"), fc2: try dense("\(key).linear_fc2"),
                normAfterFold: after)
        }
        let c = config
        let patchDim = 3 * c.temporalPatch * c.patch * c.patch
        patchWeight = try store.get("\(p).patch_embed.proj.weight", dtype: dtype, shape: [c.hidden, 3, c.temporalPatch, c.patch, c.patch])
            .reshaped(c.hidden, patchDim)
        patchBias = try store.get("\(p).patch_embed.proj.bias", dtype: dtype, shape: [c.hidden])
        positionTable = try store.get("\(p).pos_embed.weight", dtype: dtype, shape: [c.positionGrid * c.positionGrid, c.hidden])
        blocks = try (0..<c.depth).map { i in
            let b = "\(p).blocks.\(i)"
            return Block(
                norm1: try norm("\(b).norm1"), norm2: try norm("\(b).norm2"), qkv: try dense("\(b).attn.qkv"),
                proj: try dense("\(b).attn.proj"), fc1: try dense("\(b).mlp.linear_fc1"),
                fc2: try dense("\(b).mlp.linear_fc2"))
        }
        merger = try makeMerger("\(p).merger", after: false)
        deepstackMergers = try c.deepstackLayers.indices.map { try makeMerger("\(p).deepstack_merger_list.\($0)", after: true) }
    }

    /// Patch grid `(row, col)` of every token in spatial-merge-block order.
    private func patchCoordinates(gridH: Int, gridW: Int) -> [(Int, Int)] {
        let m = config.merge
        var coords: [(Int, Int)] = []
        coords.reserveCapacity(gridH * gridW)
        for br in 0..<(gridH / m) {
            for bc in 0..<(gridW / m) {
                for ir in 0..<m {
                    for ic in 0..<m { coords.append((br * m + ir, bc * m + ic)) }
                }
            }
        }
        return coords
    }

    /// Bilinear (align_corners) resampling of the square position table to the patch grid, gathered per token.
    private func positionEmbeddings(_ coords: [(Int, Int)], gridH: Int, gridW: Int) -> MLXArray {
        let side = config.positionGrid
        func taps(_ index: Int, _ size: Int) -> [(Int, Float)] {
            let src = Float(index) * Float(side - 1) / Float(max(size - 1, 1))
            let lo = Float(Int(src.rounded(.down)))
            return (0..<2).map { o in
                let tap = min(max(Int(lo) + o, 0), side - 1)
                let weight = max(1 - abs(src - lo - Float(o)), 0)
                return (tap, weight)
            }
        }
        var indices: [Int32] = []
        var weights: [Float] = []
        for (row, col) in coords {
            for (hTap, hWeight) in taps(row, gridH) {
                for (wTap, wWeight) in taps(col, gridW) {
                    indices.append(Int32(hTap * side + wTap))
                    weights.append(hWeight * wWeight)
                }
            }
        }
        let n = coords.count
        let gathered = positionTable[MLXArray(indices)].reshaped(n, 4, config.hidden).asType(.float32)
        return (gathered * MLXArray(weights).reshaped(n, 4, 1)).sum(axis: 1)
    }

    /// 2D rotary angles: 18 frequencies on the row, 18 on the column, concatenated and repeated (rotate-half).
    private func rotary(_ coords: [(Int, Int)]) -> (MLXArray, MLXArray) {
        let headDim = config.hidden / config.heads
        let spatial = headDim / 2
        let inv = stride(from: 0, to: spatial, by: 2).map { 1 / powf(config.ropeTheta, Float($0) / Float(spatial)) }
        var angles: [Float] = []
        angles.reserveCapacity(coords.count * headDim)
        for (row, col) in coords {
            let half = inv.map { Float(row) * $0 } + inv.map { Float(col) * $0 }
            angles += half + half
        }
        let a = MLXArray(angles).reshaped(coords.count, 1, headDim)
        return (cos(a), sin(a))
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    private func layerNorm(_ x: MLXArray, _ p: (MLXArray, MLXArray)) -> MLXArray {
        MLXFast.layerNorm(x, weight: p.0, bias: p.1, eps: config.eps)
    }

    /// Exact (erf) GELU in fp32, as `nn.GELU()` computes it for bf16 inputs.
    private func gelu(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        return (0.5 * xf * (1 + erf(xf / Float(2).squareRoot()))).asType(x.dtype)
    }

    private func applyMerger(_ m: Merger, _ x: MLXArray) -> MLXArray {
        let fold = config.merge * config.merge * config.hidden
        let folded = m.normAfterFold ? layerNorm(x.reshaped(-1, fold), m.norm) : layerNorm(x, m.norm).reshaped(-1, fold)
        return m.fc2(gelu(m.fc1(folded)))
    }

    /// `pixels`: `[H, W, 3]` RGB normalized to `[-1, 1]` (`(x / 255 - 0.5) / 0.5`), H and W multiples of 32.
    func callAsFunction(_ pixels: MLXArray, trace: ((String, MLXArray) -> Void)? = nil) -> Output {
        let c = config
        let (height, width) = (pixels.dim(0), pixels.dim(1))
        let (gridH, gridW) = (height / c.patch, width / c.patch)
        let m = c.merge
        let dtype = patchWeight.dtype
        // [gh/m, m, p, gw/m, m, p, C] -> [gh/m, gw/m, m, m, C, p, p]: merge-block order, channel-major patch vectors.
        var patches = pixels.reshaped(gridH / m, m, c.patch, gridW / m, m, c.patch, 3).transposed(0, 3, 1, 4, 6, 2, 5)
        patches = patches.reshaped(gridH * gridW, 3, 1, c.patch, c.patch)
        // Both temporal slots of the Conv3d see the same frame.
        patches = broadcast(patches, to: [gridH * gridW, 3, c.temporalPatch, c.patch, c.patch])
            .reshaped(gridH * gridW, -1).asType(dtype)
        var hidden = addMM(patchBias, patches, patchWeight.T)

        let coords = patchCoordinates(gridH: gridH, gridW: gridW)
        hidden = hidden + positionEmbeddings(coords, gridH: gridH, gridW: gridW).asType(dtype)
        let (cosA, sinA) = rotary(coords)
        trace?("embed", hidden)

        let n = gridH * gridW
        let headDim = c.hidden / c.heads
        let scale = 1 / Float(headDim).squareRoot()
        var deepstack: [MLXArray] = []
        for (index, block) in blocks.enumerated() {
            let qkv = block.qkv(layerNorm(hidden, block.norm1)).reshaped(n, 3, c.heads, headDim)
            var q = qkv[0..., 0].asType(.float32)
            var k = qkv[0..., 1].asType(.float32)
            let v = qkv[0..., 2]
            q = (q * cosA + rotateHalf(q) * sinA).asType(dtype)
            k = (k * cosA + rotateHalf(k) * sinA).asType(dtype)
            let attention = MLXFast.scaledDotProductAttention(
                queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
                keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
                values: v.transposed(1, 0, 2).expandedDimensions(axis: 0), scale: scale, mask: .none
            )[0].transposed(1, 0, 2).reshaped(n, c.hidden)
            hidden = hidden + block.proj(attention)
            hidden = hidden + block.fc2(geluTanh(block.fc1(layerNorm(hidden, block.norm2))))
            if index == 0 || index == c.depth - 1 { trace?("block\(index)", hidden) }
            if let slot = c.deepstackLayers.firstIndex(of: index) {
                deepstack.append(applyMerger(deepstackMergers[slot], hidden))
            }
            eval(hidden)
        }
        return Output(embeds: applyMerger(merger, hidden), deepstack: deepstack, gridHeight: gridH / m, gridWidth: gridW / m)
    }
}
