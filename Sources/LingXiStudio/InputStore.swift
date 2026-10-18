import AppKit
import CryptoKit
import Foundation
import ImageIO
import QwenImageCore
import UniformTypeIdentifiers

/// Content-addressed copies of input images under Application Support, so queued tasks and recorded results keep
/// pointing at exactly the pixels they used even if the original file changes or moves.
enum InputStore {
    static var directory: URL {
        let dir = AppPaths.support.appending(path: "inputs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Imports an image file (any ImageIO format). Never modifies the original.
    static func importFile(_ url: URL) throws -> InputImage {
        let data = try Data(contentsOf: url)
        return try store(data: data, source: url.path)
    }

    /// Imports image data from the pasteboard or a drop (PNG/TIFF/JPEG...).
    static func importData(_ data: Data, source: String? = nil) throws -> InputImage {
        try store(data: data, source: source)
    }

    static func importImage(_ image: NSImage, source: String? = nil) throws -> InputImage {
        guard let tiff = image.tiffRepresentation else { throw ConditioningError.unreadableImage("pasteboard") }
        return try store(data: tiff, source: source)
    }

    /// Writes an 8-bit RGBA bitmap (e.g. a mask) as a stored PNG input.
    static func importRGBA(_ image: RGBAImage, name: String) throws -> InputImage {
        guard let cg = image.cgImage else { throw ConditioningError.unreadableImage(name) }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw ConditioningError.unreadableImage(name)
        }
        CGImageDestinationAddImage(dest, cg, nil)
        CGImageDestinationFinalize(dest)
        return try store(data: data as Data, source: nil)
    }

    private static func store(data: Data, source: String?) throws -> InputImage {
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw ConditioningError.unreadableImage(source ?? "data") }
        let target = directory.appending(path: "\(hash).png")
        if !FileManager.default.fileExists(atPath: target.path) {
            // Stored as PNG regardless of the source format so every later read is lossless and format-independent.
            guard let dest = CGImageDestinationCreateWithURL(target as CFURL, UTType.png.identifier as CFString, 1, nil)
            else { throw ConditioningError.unreadableImage(target.path) }
            CGImageDestinationAddImage(dest, cg, nil)
            guard CGImageDestinationFinalize(dest) else { throw ConditioningError.unreadableImage(target.path) }
        }
        // The stored PNG's own hash identifies the pixels the pipeline reads.
        let stored = try Data(contentsOf: target)
        let storedHash = SHA256.hash(data: stored).map { String(format: "%02x", $0) }.joined()
        return InputImage(storedPath: target.path, sourcePath: source, sha256: storedHash, width: cg.width, height: cg.height)
    }

    /// Resolve a moved output folder without accepting changed input bytes.
    static func restore(_ record: InputImageRecord, nextTo artwork: URL) throws -> InputImage {
        let original = URL(fileURLWithPath: record.path)
        let adjacent = artwork.deletingLastPathComponent().appending(path: "inputs/\(record.sha256).png")
        for url in [original, adjacent] {
            guard let data = try? Data(contentsOf: url),
                SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == record.sha256
            else { continue }
            return InputImage(storedPath: url.path, sourcePath: record.source, sha256: record.sha256,
                width: record.width, height: record.height)
        }
        throw ConditioningError.unreadableImage(record.path)
    }

    /// Reads image files or image data from a pasteboard.
    @MainActor
    static func fromPasteboard(_ pasteboard: NSPasteboard = .general) -> [InputImage] {
        if let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL],
            !urls.isEmpty
        {
            return urls.compactMap { try? importFile($0) }
        }
        if let images = pasteboard.readObjects(forClasses: [NSImage.self]) as? [NSImage] {
            return images.compactMap { try? importImage($0, source: "剪贴板") }
        }
        return []
    }
}

enum AppPaths {
    /// An opt-in profile keeps development acceptance away from the user's library and preferences.
    static var profile: URL? {
        guard let path = ProcessInfo.processInfo.environment["LINGXI_PROFILE_DIRECTORY"] else { return nil }
        precondition(path.hasPrefix("/"), "LINGXI_PROFILE_DIRECTORY must be an absolute path")
        return URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
    }

    static var preferencesDomain: String? {
        profile.map { "com.lingxifox.LingXiStudio.profile." + SHA256.hash(data: Data($0.path.utf8)).map { String(format: "%02x", $0) }.joined() }
    }

    static var defaults: UserDefaults {
        guard let domain = preferencesDomain else { return .standard }
        guard let defaults = UserDefaults(suiteName: domain) else { preconditionFailure("Could not open isolated preferences") }
        return defaults
    }

    static var support: URL {
        let dir = profile ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "LingXiStudio")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static var logs: URL {
        let dir = support.appending(path: "logs")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

/// Plain-text diagnostic log (one file per day) kept for the task/diagnostics views.
enum AppLog {
    private static let queue = DispatchQueue(label: "LingXiStudio.log")

    static var todayFile: URL {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        return AppPaths.logs.appending(path: "\(f.string(from: Date())).log")
    }

    static func write(_ message: String) {
        let line = "\(ISO8601DateFormatter().string(from: Date())) \(message)\n"
        let url = todayFile
        queue.async {
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(Data(line.utf8))
                try? handle.close()
            } else {
                try? Data(line.utf8).write(to: url)
            }
        }
    }
}
