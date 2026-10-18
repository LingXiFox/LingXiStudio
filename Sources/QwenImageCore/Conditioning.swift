import CoreGraphics
import Foundation
import ImageIO
import MLX

/// An 8-bit RGBA image in memory (row-major, straight alpha).
public struct RGBAImage: Sendable {
    public var width: Int
    public var height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// Decodes any ImageIO-readable file into straight-alpha RGBA8 (sRGB).
    public init(contentsOf url: URL) throws {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldAllowFloat: false] as CFDictionary)
        else { throw ConditioningError.unreadableImage(url.path) }
        self = try RGBAImage(cgImage: image)
    }

    public init(cgImage image: CGImage) throws {
        if let raw = Self.rawPixels(image) {
            self = raw
            return
        }
        // Other layouts (16-bit, float, CMYK, indexed...) are converted by drawing into an sRGB bitmap.
        let (w, h) = (image.width, image.height)
        // Premultiplied is the only RGBA layout CGBitmapContext draws into; alpha is un-premultiplied afterwards.
        var data = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = data.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw ConditioningError.unreadableImage("bitmap") }
        Self.unpremultiply(&data)
        width = w
        height = h
        pixels = data
    }

    /// Reads 8-bit RGB(A) / gray(+alpha) pixels exactly as stored. Drawing through a CGContext would color-match
    /// even sRGB to sRGB (measured up to 7 levels off), which changes the pixels the model sees.
    private static func rawPixels(_ image: CGImage) -> RGBAImage? {
        guard image.bitsPerComponent == 8, let model = image.colorSpace?.model,
            model == .rgb || model == .monochrome,
            let provider = image.dataProvider, let cfData = provider.data
        else { return nil }
        let info = image.bitmapInfo
        let byteOrder = info.intersection(.byteOrderMask)
        guard byteOrder == [] || byteOrder == .byteOrder32Big || byteOrder == .byteOrderDefault else { return nil }
        let alpha = image.alphaInfo
        let channels = model == .rgb ? 3 : 1
        let hasAlpha = alpha != .none && alpha != .noneSkipLast && alpha != .noneSkipFirst
        let stride = image.bitsPerPixel / 8
        guard stride == channels + (alpha == .none ? 0 : 1) else { return nil }
        let alphaFirst = alpha == .first || alpha == .premultipliedFirst || alpha == .noneSkipFirst
        let premultiplied = alpha == .premultipliedFirst || alpha == .premultipliedLast
        let (w, h, rowBytes) = (image.width, image.height, image.bytesPerRow)
        let bytes = CFDataGetBytePtr(cfData)!
        var out = [UInt8](repeating: 255, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let src = y * rowBytes + x * stride
                let colorBase = src + (alphaFirst ? 1 : 0)
                let o = (y * w + x) * 4
                if channels == 3 {
                    out[o] = bytes[colorBase]
                    out[o + 1] = bytes[colorBase + 1]
                    out[o + 2] = bytes[colorBase + 2]
                } else {
                    out[o] = bytes[colorBase]
                    out[o + 1] = bytes[colorBase]
                    out[o + 2] = bytes[colorBase]
                }
                if hasAlpha { out[o + 3] = bytes[alphaFirst ? src : src + channels] }
            }
        }
        if premultiplied { unpremultiply(&out) }
        return RGBAImage(width: w, height: h, pixels: out)
    }

    private static func unpremultiply(_ data: inout [UInt8]) {
        for i in stride(from: 0, to: data.count, by: 4) {
            let a = Int(data[i + 3])
            guard a > 0, a < 255 else { continue }
            for c in 0..<3 { data[i + c] = UInt8(min(255, (Int(data[i + c]) * 255 + a / 2) / a)) }
        }
    }

    public var cgImage: CGImage? {
        let provider = CGDataProvider(data: Data(pixels) as CFData)
        return provider.flatMap {
            CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: $0, decode: nil,
                shouldInterpolate: true, intent: .defaultIntent)
        }
    }

    /// High-quality resample to exactly `width` x `height` (Core Graphics, high interpolation quality).
    public func resized(width w: Int, height h: Int) -> RGBAImage {
        guard w != width || h != height, let source = cgImage else { return self }
        var data = [UInt8](repeating: 0, count: w * h * 4)
        data.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return }
            context.interpolationQuality = .high
            context.draw(source, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        Self.unpremultiply(&data)
        return RGBAImage(width: w, height: h, pixels: data)
    }

    /// RGB with alpha composited over white, as the reference pipeline feeds the vision encoder.
    func rgbOverWhite() -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for p in 0..<(width * height) {
            let a = Int(pixels[p * 4 + 3])
            for c in 0..<3 {
                let v = Int(pixels[p * 4 + c])
                rgb[p * 3 + c] = a == 255 ? UInt8(v) : UInt8((v * a + 255 * (255 - a) + 127) / 255)
            }
        }
        return rgb
    }

    /// `[1, H, W, 4]` float32 in `[-1, 1]`, the VAE encoder input (all four channels, like the reference).
    func vaeInput() -> MLXArray {
        MLXArray(pixels.map { Float($0) / 127.5 - 1 }).reshaped(1, height, width, 4)
    }
}

public enum ConditioningError: Error, CustomStringConvertible {
    case unreadableImage(String)
    case tooManyImages(Int)

    public var description: String {
        switch self {
        case .unreadableImage(let path): "Could not read image \(path)"
        case .tooManyImages(let n): "At most 10 condition images are supported (got \(n))"
        }
    }
}

/// `calculate_dimensions` of the reference: the size with area `target` and aspect `ratio`, rounded to 32.
public func conditionDimensions(target: Int = 1024 * 1024, ratio: Double) -> (width: Int, height: Int) {
    let width = (Double(target) * ratio).squareRoot()
    let height = width / ratio
    return (Int((width / 32).rounded(.toNearestOrEven)) * 32, Int((height / 32).rounded(.toNearestOrEven)) * 32)
}

/// Tokens, positions and vision context of a prompt with condition images (`prompt_template_ti2i`).
struct MultimodalPrompt {
    let ids: [Int]
    /// Leading system tokens dropped from the hidden states.
    let drop: Int
    /// Per image: placeholder token range in `ids`, and the merged grid (latent grid / 2).
    let imageRanges: [Range<Int>]
    let grids: [(height: Int, width: Int)]
    let positions: [[Int32]]

    /// `images` are already at their condition sizes (multiples of 32).
    init(prompt: String, imageSizes: [(width: Int, height: Int)], tokenizer: QwenTokenizer) throws {
        let systemPrefix = QwenImagePipeline.systemPrefix
        var vision = "<image1><|vision_start|><|image_pad|><|vision_end|>"
        if imageSizes.count > 1 {
            for i in 2...imageSizes.count { vision += " <image\(i)><|vision_start|><|image_pad|><|vision_end|>" }
        }
        let text = prompt.isEmpty ? " " : prompt
        let template = "\(systemPrefix)<|im_start|>user\n\(vision)\(text)<|im_end|>\n<|im_start|>assistant\n"
        let raw = try tokenizer.encode(template)
        let pad = try tokenizer.encode("<|image_pad|>")
        precondition(pad.count == 1, "<|image_pad|> must be a single added token")
        // The processor expands each placeholder to one token per 2x2 patch group (= one per merged cell).
        var ids: [Int] = []
        var ranges: [Range<Int>] = []
        var grids: [(Int, Int)] = []
        var next = 0
        for token in raw {
            if token == pad[0] {
                let size = imageSizes[next]
                let grid = (size.height / 32, size.width / 32)
                let start = ids.count
                ids += Array(repeating: token, count: grid.0 * grid.1)
                ranges.append(start..<ids.count)
                grids.append(grid)
                next += 1
            } else {
                ids.append(token)
            }
        }
        self.ids = ids
        drop = try tokenizer.encode(systemPrefix).count
        imageRanges = ranges
        self.grids = grids

        // `get_rope_index`: text advances all axes together; an image puts t at the current position and lays
        // h/w on its merged grid offset by it, then the position jumps by max(h, w).
        var t: [Int32] = []
        var h: [Int32] = []
        var w: [Int32] = []
        var position: Int32 = 0
        var cursor = 0
        for (range, grid) in zip(ranges, grids) {
            for _ in cursor..<range.lowerBound {
                t.append(position)
                h.append(position)
                w.append(position)
                position += 1
            }
            for row in 0..<grid.0 {
                for col in 0..<grid.1 {
                    t.append(position)
                    h.append(position + Int32(row))
                    w.append(position + Int32(col))
                }
            }
            position += Int32(max(grid.0, grid.1))
            cursor = range.upperBound
        }
        for _ in cursor..<ids.count {
            t.append(position)
            h.append(position)
            w.append(position)
            position += 1
        }
        positions = [t, h, w]
    }
}

extension QwenVisionEncoder {
    /// Vision tower on an RGBA condition image (alpha over white, normalized to [-1, 1]).
    func encode(_ image: RGBAImage, trace: ((String, MLXArray) -> Void)? = nil) -> Output {
        // Processor order: rescale by 1/255, then normalize with mean 0.5 and std 0.5.
        let rescale: Float = 0.00392156862745098
        let rgb = MLXArray(image.rgbOverWhite().map { (Float($0) * rescale - 0.5) / 0.5 })
            .reshaped(image.height, image.width, 3)
        return self(rgb, trace: trace)
    }
}

import CryptoKit

extension InputImageRecord {
    /// Record for an image file used by a run (hash of the file bytes, decoded size).
    public static func make(role: Role, url: URL, source: URL? = nil) throws -> (InputImageRecord, RGBAImage) {
        let data = try Data(contentsOf: url)
        let image = try RGBAImage(contentsOf: url)
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return (
            InputImageRecord(
                role: role, path: url.path, source: source.map(\.path), sha256: hash, width: image.width,
                height: image.height),
            image
        )
    }
}

extension GenerationMode {
    /// Default output size for a mode: the official pipeline derives it from the condition image's aspect ratio at
    /// 1024² area (the main image for edit and inpaint, the last one for reference, the start image for img2img).
    public static func defaultSize(mode: GenerationMode, conditions: [RGBAImage], initial: RGBAImage?)
        -> (width: Int, height: Int)?
    {
        let basis: RGBAImage? =
            switch mode {
            case .textToImage: nil
            case .reference: conditions.last
            case .edit, .inpaint: conditions.first
            case .imageToImage: initial ?? conditions.first
            }
        return basis.map { conditionDimensions(ratio: Double($0.width) / Double($0.height)) }
    }
}
