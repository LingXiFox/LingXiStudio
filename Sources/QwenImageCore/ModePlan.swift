import Foundation
import MLX

/// What a request needs from its input images, prepared at the sizes the pipeline uses.
struct ModePlan {
    /// Condition images at their condition sizes (`calculate_dimensions(1024², ratio)`), in prompt order.
    let conditions: [RGBAImage]
    /// Starting image at the output size, when the mode starts from an image.
    let initial: RGBAImage?
    /// Repaint weight per output pixel in [0, 1] (`[H, W]`), for inpainting.
    let mask: MLXArray?
    let strength: Float
    let feather: Int
    let keepUnmasked: Bool

    init(request: GenerationRequest, inputs: GenerationInputs) throws {
        let (w, h) = (request.width, request.height)
        func conditionSized(_ image: RGBAImage) -> RGBAImage {
            let size = conditionDimensions(ratio: Double(image.width) / Double(image.height))
            return image.resized(width: size.width, height: size.height)
        }
        if inputs.conditions.count > 10 { throw ConditioningError.tooManyImages(inputs.conditions.count) }
        strength = min(max(request.strength, 0), 1)
        feather = max(request.maskFeather, 0)
        keepUnmasked = request.keepUnmaskedPixels
        switch request.mode {
        case .textToImage:
            conditions = []
            initial = nil
            mask = nil
        case .reference:
            guard !inputs.conditions.isEmpty else { throw PipelineError.missingInput("at least one reference image") }
            conditions = inputs.conditions.map(conditionSized)
            initial = nil
            mask = nil
        case .edit:
            guard let main = inputs.conditions.first else { throw PipelineError.missingInput("the image to edit") }
            conditions = inputs.conditions.map(conditionSized)
            // Below full strength the edit starts from the noised main image (generic SDEdit start).
            initial = strength < 1 ? main.resized(width: w, height: h) : nil
            mask = nil
        case .imageToImage:
            guard let start = inputs.initial ?? inputs.conditions.first else {
                throw PipelineError.missingInput("the starting image")
            }
            conditions = []
            initial = start.resized(width: w, height: h)
            mask = nil
        case .inpaint:
            guard let main = inputs.conditions.first else { throw PipelineError.missingInput("the image to repaint") }
            guard let maskImage = inputs.mask else { throw PipelineError.missingInput("a mask") }
            conditions = inputs.conditions.map(conditionSized)
            initial = main.resized(width: w, height: h)
            mask = Self.maskWeights(maskImage.resized(width: w, height: h), invert: request.maskInvert)
        }
    }

    /// First schedule index to run (`get_timesteps`): `strength` 1 runs every step from pure noise.
    func startIndex(steps: Int) -> Int {
        guard initial != nil else { return 0 }
        let initSteps = min(Float(steps) * strength, Float(steps))
        return min(Int(max(Float(steps) - initSteps, 0)), steps - 1)
    }

    /// Luma of the mask (ITU-R 601, like PIL "L"), times alpha, as repaint weights.
    static func maskWeights(_ image: RGBAImage, invert: Bool) -> MLXArray {
        let rgba = MLXArray(image.pixels).reshaped(image.height, image.width, 4).asType(.float32) / 255
        let luma = 0.299 * rgba[0..., 0..., 0] + 0.587 * rgba[0..., 0..., 1] + 0.114 * rgba[0..., 0..., 2]
        let weights = luma * rgba[0..., 0..., 3]
        return invert ? 1 - weights : weights
    }

    /// Binary latent mask `[1, h * w, 1]`: a latent cell is repainted when any pixel of its 16x16 tile is marked,
    /// so thin strokes are not lost by point sampling.
    func latentMask(height: Int, width: Int) -> MLXArray? {
        guard let mask else { return nil }
        let binary = (mask .>= 0.5).asType(.float32)
        let tiles = binary.reshaped(height, 16, width, 16).max(axes: [1, 3])
        return tiles.reshaped(1, height * width, 1)
    }

    /// Pastes the generated image over the original outside the feathered mask, so unselected pixels stay exact.
    func pixelComposite(width: Int, height: Int) -> ((MLXArray) -> MLXArray)? {
        guard let mask, keepUnmasked, let initial else { return nil }
        var alpha = (mask .>= 0.5).asType(.float32)
        if feather > 0 { alpha = Self.blur(alpha, radius: feather) }
        let original = MLXArray(initial.pixels).reshaped(1, height, width, 4).asType(.float32)
        let a = alpha.reshaped(1, height, width, 1)
        return { generated in
            round(a * generated.asType(.float32) + (1 - a) * original).asType(.uint8)
        }
    }

    /// Separable Gaussian blur (sigma = radius / 2) with edge clamping, on a `[H, W]` map.
    static func blur(_ x: MLXArray, radius: Int) -> MLXArray {
        let sigma = Float(radius) / 2
        let taps = (-radius...radius).map { exp(-Float($0 * $0) / (2 * sigma * sigma)) }
        let total = taps.reduce(0, +)
        let kernel = MLXArray(taps.map { $0 / total })
        func pass(_ v: MLXArray, axis: Int) -> MLXArray {
            let n = v.dim(axis)
            let padded = concatenated(
                [repeated(axis == 0 ? v[0..<1] : v[0..., 0..<1], count: radius, axis: axis), v,
                 repeated(axis == 0 ? v[(n - 1)...] : v[0..., (n - 1)...], count: radius, axis: axis)], axis: axis)
            var out = MLXArray.zeros(v.shape, dtype: .float32)
            for (i, weight) in zip(0..., taps.indices.map { kernel[$0] }) {
                out = out + weight * (axis == 0 ? padded[i..<(i + n)] : padded[0..., i..<(i + n)])
            }
            return out
        }
        return pass(pass(x, axis: 0), axis: 1)
    }
}
