import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public enum ImageWriterError: Error, CustomStringConvertible {
    case encodeFailed(String)
    public var description: String {
        switch self {
        case .encodeFailed(let path): "Could not write PNG to \(path)"
        }
    }
}

/// Generation record stored next to every image (sidecar JSON) and inside the PNG description.
public struct GenerationMetadata: Codable, Sendable {
    public var app = "LingXiStudio"
    public var model: String
    public var precision = "bf16"
    public var mode = "text-to-image"
    public var request: GenerationRequest
    public var timings: [StageTiming]
    public var stepSeconds: [Double]
    public var totalSeconds: Double
    public var peakMemoryBytes: Int
    public var createdAt: Date
}

extension GenerationMetadata {
    /// Reads the sidecar `<name>.json` written next to a PNG.
    public static func read(sidecarOf png: URL) throws -> GenerationMetadata {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let data = try Data(contentsOf: png.deletingPathExtension().appendingPathExtension("json"))
        return try decoder.decode(GenerationMetadata.self, from: data)
    }
}

public enum ImageWriter {
    /// Writes a lossless RGBA PNG plus `<name>.json` with the full generation record.
    public static func write(_ image: GeneratedImage, model: String, to url: URL) throws {
        let metadata = GenerationMetadata(
            model: model, mode: image.request.mode.rawValue, request: image.request, timings: image.timings, stepSeconds: image.stepSeconds, totalSeconds: image.totalSeconds,
            peakMemoryBytes: image.peakMemoryBytes, createdAt: Date())
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let json = try encoder.encode(metadata)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)

        let data = Data(image.rgba) as CFData
        guard let provider = CGDataProvider(data: data),
            let cgImage = CGImage(
                width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
            let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw ImageWriterError.encodeFailed(url.path) }

        let properties: [CFString: Any] = [
            kCGImagePropertyPNGDictionary: [
                kCGImagePropertyPNGDescription: String(decoding: json, as: UTF8.self),
                kCGImagePropertyPNGSoftware: "LingXiStudio",
            ]
        ]
        CGImageDestinationAddImage(destination, cgImage, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw ImageWriterError.encodeFailed(url.path) }
        try json.write(to: url.deletingPathExtension().appendingPathExtension("json"))
    }
}
