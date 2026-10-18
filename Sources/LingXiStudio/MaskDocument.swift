import AppKit
import CoreGraphics
import Foundation
import Observation
import QwenImageCore

/// The inpainting mask being painted: an 8-bit gray bitmap at the main image's resolution (255 = repaint).
///
/// Strokes are drawn with Core Graphics into the bitmap; each finished stroke is one undo step. The canvas shows the
/// mask as a tinted overlay on the main image; the pipeline receives it as a PNG input.
@MainActor
@Observable
final class MaskDocument {
    enum Tool: String, CaseIterable { case brush, eraser }

    private(set) var width = 0
    private(set) var height = 0
    private var bitmap: [UInt8] = []
    private var imageID: String?
    /// Bumped on every pixel change so views can redraw the overlay.
    private(set) var revision = 0
    private var undoStack: [[UInt8]] = []
    private var redoStack: [[UInt8]] = []

    var tool: Tool = .brush
    /// Brush diameter in image pixels.
    var brushSize: Double = 80
    /// 0 = soft edge, 1 = hard edge.
    var hardness: Double = 0.7
    var opacity: Double = 1
    var showOverlay = true

    var isEmpty: Bool { !bitmap.contains { $0 > 0 } }
    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }
    var hasCanvas: Bool { width > 0 }

    /// Keeps strokes only while the same source image and dimensions remain selected.
    func prepare(width: Int, height: Int, imageID: String?) {
        guard width != self.width || height != self.height || imageID != self.imageID else { return }
        self.imageID = imageID
        self.width = width
        self.height = height
        bitmap = [UInt8](repeating: 0, count: width * height)
        undoStack.removeAll()
        redoStack.removeAll()
        revision += 1
    }

    private func snapshot() {
        undoStack.append(bitmap)
        if undoStack.count > 40 { undoStack.removeFirst() }
        redoStack.removeAll()
    }

    func undo() {
        guard let last = undoStack.popLast() else { return }
        redoStack.append(bitmap)
        bitmap = last
        revision += 1
    }

    func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(bitmap)
        bitmap = next
        revision += 1
    }

    func clear() {
        guard hasCanvas else { return }
        snapshot()
        bitmap = [UInt8](repeating: 0, count: width * height)
        revision += 1
    }

    func invert() {
        guard hasCanvas else { return }
        snapshot()
        bitmap = bitmap.map { 255 - $0 }
        revision += 1
    }

    /// Starts a stroke (records the undo point).
    func beginStroke() { snapshot() }

    /// Paints a dab at image coordinates (origin top-left).
    func dab(at point: CGPoint) {
        guard hasCanvas else { return }
        let radius = max(brushSize / 2, 1)
        bitmap.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
            else { return }
            // Core Graphics has a bottom-left origin.
            let center = CGPoint(x: point.x, y: CGFloat(height) - point.y)
            let value: CGFloat = tool == .brush ? 1 : 0
            let colors = [
                CGColor(gray: value, alpha: opacity), CGColor(gray: value, alpha: opacity),
                CGColor(gray: value, alpha: 0),
            ]
            let inner = CGFloat(min(max(hardness, 0), 0.99))
            if let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceGray(), colors: colors as CFArray, locations: [0, inner, 1])
            {
                context.drawRadialGradient(
                    gradient, startCenter: center, startRadius: 0, endCenter: center, endRadius: radius, options: [])
            }
        }
        revision += 1
    }

    /// Paints dabs along a segment so fast drags leave a continuous line.
    func stroke(from a: CGPoint, to b: CGPoint) {
        let distance = hypot(b.x - a.x, b.y - a.y)
        let spacing = max(brushSize * 0.15, 1)
        let steps = max(Int(distance / spacing), 1)
        for i in 1...steps {
            let t = CGFloat(i) / CGFloat(steps)
            dab(at: CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t))
        }
    }

    /// Red overlay (alpha = mask) for display.
    func overlayImage(color: NSColor = .systemRed) -> CGImage? {
        guard hasCanvas else { return nil }
        let c = color.usingColorSpace(.sRGB) ?? .red
        let (r, g, b) = (UInt8(c.redComponent * 255), UInt8(c.greenComponent * 255), UInt8(c.blueComponent * 255))
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) where bitmap[i] > 0 {
            let a = UInt16(bitmap[i]) * 140 / 255
            rgba[i * 4] = UInt8(UInt16(r) * a / 255)
            rgba[i * 4 + 1] = UInt8(UInt16(g) * a / 255)
            rgba[i * 4 + 2] = UInt8(UInt16(b) * a / 255)
            rgba[i * 4 + 3] = UInt8(a)
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)
        return provider.flatMap {
            CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: $0,
                decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        }
    }

    /// The mask as an opaque RGBA image (white = repaint), for the pipeline and for export.
    func rgbaImage() -> RGBAImage? {
        guard hasCanvas else { return nil }
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            rgba[i * 4] = bitmap[i]
            rgba[i * 4 + 1] = bitmap[i]
            rgba[i * 4 + 2] = bitmap[i]
        }
        return RGBAImage(width: width, height: height, pixels: rgba)
    }

    /// Loads a mask image (any size; resampled to the canvas, luma used as repaint weight).
    func load(_ image: RGBAImage) {
        guard hasCanvas else { return }
        snapshot()
        let resized = image.resized(width: width, height: height)
        bitmap = (0..<(width * height)).map { i in
            let p = resized.pixels
            let luma = 0.299 * Double(p[i * 4]) + 0.587 * Double(p[i * 4 + 1]) + 0.114 * Double(p[i * 4 + 2])
            return UInt8(min(255, luma * Double(p[i * 4 + 3]) / 255))
        }
        revision += 1
    }

    /// Share of the image marked for repainting.
    var coverage: Double {
        guard hasCanvas else { return 0 }
        return Double(bitmap.lazy.filter { $0 >= 128 }.count) / Double(bitmap.count)
    }
}
