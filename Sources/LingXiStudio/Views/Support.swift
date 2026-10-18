import AppKit
import ImageIO
import QwenImageCore
import SwiftUI
import UniformTypeIdentifiers

/// Two panes with a draggable divider. `fraction` is the share of the first pane; double-click restores `defaultFraction`.
struct ProportionalSplit<First: View, Second: View>: View {
    enum Axis { case horizontal, vertical }

    let axis: Axis
    @Binding var fraction: Double
    let defaultFraction: Double
    let minFirst: CGFloat
    let minSecond: CGFloat
    @ViewBuilder let first: First
    @ViewBuilder let second: Second

    @State private var dragStart: Double?

    var body: some View {
        GeometryReader { geo in
            let total = axis == .horizontal ? geo.size.width : geo.size.height
            let available = max(total - 1, 1)
            let firstSize = min(max(available * fraction, minFirst), max(available - minSecond, minFirst))
            let layout = axis == .horizontal ? AnyLayout(HStackLayout(spacing: 0)) : AnyLayout(VStackLayout(spacing: 0))
            layout {
                first.frame(
                    width: axis == .horizontal ? firstSize : nil, height: axis == .vertical ? firstSize : nil)
                handle(available: available)
                second.frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private func handle(available: CGFloat) -> some View {
        Rectangle()
            .fill(Color(nsColor: .separatorColor))
            .frame(width: axis == .horizontal ? 1 : nil, height: axis == .vertical ? 1 : nil)
            .overlay {
                Color.clear
                    .frame(width: axis == .horizontal ? 8 : nil, height: axis == .vertical ? 8 : nil)
                    .contentShape(Rectangle())
                    .onHover { inside in
                        if inside {
                            (axis == .horizontal ? NSCursor.resizeLeftRight : NSCursor.resizeUpDown).push()
                        } else {
                            NSCursor.pop()
                        }
                    }
                    .gesture(
                        DragGesture(minimumDistance: 1)
                            .onChanged { value in
                                let start = dragStart ?? fraction
                                dragStart = start
                                let delta = axis == .horizontal ? value.translation.width : value.translation.height
                                fraction = min(max(start + delta / available, 0.15), 0.85)
                            }
                            .onEnded { _ in dragStart = nil }
                    )
                    .onTapGesture(count: 2) { fraction = defaultFraction }
            }
    }
}

/// Downscaled thumbnails decoded off the main thread and cached by file URL.
@MainActor
final class ThumbnailCache {
    static let shared = ThumbnailCache()
    private let cache = NSCache<NSURL, NSImage>()

    func image(for url: URL, maxPixels: Int = 320) async -> NSImage? {
        if let hit = cache.object(forKey: url as NSURL) { return hit }
        let cg = await Task.detached { () -> CGImage? in
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPixels,
                kCGImageSourceCreateThumbnailWithTransform: true,
            ]
            return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        }.value
        guard let cg else { return nil }
        let image = NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
        cache.setObject(image, forKey: url as NSURL)
        return image
    }
}

struct Thumbnail: View {
    let url: URL
    var size: CGFloat = 96
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            Color(nsColor: .quaternaryLabelColor).opacity(0.3)
            if let image {
                Image(nsImage: image).resizable().scaledToFill()
            } else {
                ProgressView().controlSize(.small)
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .task(id: url) { image = await ThumbnailCache.shared.image(for: url) }
    }
}

/// Actions available on a finished image, used by the preview toolbar, history grid and context menus.
struct ImageActionsMenu: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    let item: HistoryItem

    var body: some View {
        Button("在独立窗口中查看", systemImage: "arrow.up.left.and.arrow.down.right") { openWindow(id: "viewer", value: item.url) }
        Divider()
        Button("复用全部参数", systemImage: "arrow.uturn.backward") { model.reuse(item) }
            .disabled(item.metadata == nil)
        Button("使用相同种子再生成", systemImage: "arrow.clockwise") {
            model.reuse(item)
            model.enqueue()
        }
        .disabled(item.metadata == nil)
        Divider()
        Button("复制图像内容", systemImage: "photo") { ImageFileActions.copyImage(item.url) }
        Button("复制文件", systemImage: "doc.on.doc") { ImageFileActions.copyFile(item.url) }
        Button("复制文件路径", systemImage: "link") { ImageFileActions.copyText(item.url.path) }
        Button("复制提示词", systemImage: "text.quote") { ImageFileActions.copyText(item.metadata?.request.prompt ?? "") }
            .disabled(item.metadata == nil)
        Divider()
        Button("另存为…", systemImage: "square.and.arrow.down") { ImageFileActions.saveAs(item.url) }
        Button("用默认应用打开", systemImage: "arrow.up.forward.app") { NSWorkspace.shared.open(item.url) }
        Button("在 Finder 中显示", systemImage: "folder") {
            NSWorkspace.shared.activateFileViewerSelecting([item.url])
        }
        Divider()
        Button("移到废纸篓…", systemImage: "trash", role: .destructive) { ImageFileActions.trash(item.url, model: model) }
    }
}

@MainActor
enum ImageFileActions {
    static func copyImage(_ url: URL) {
        guard let image = NSImage(contentsOf: url) else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.writeObjects([image])
    }

    static func copyFile(_ url: URL) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.writeObjects([url as NSURL])
    }

    static func copyText(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    /// Copies the PNG and its JSON record to a user-chosen place. Overwriting needs the save panel's own confirmation.
    static func saveAs(_ url: URL) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = url.lastPathComponent
        guard panel.runModal() == .OK, let target = panel.url, target != url else { return }
        let fm = FileManager.default
        let sidecar = url.deletingPathExtension().appendingPathExtension("json")
        let targetSidecar = target.deletingPathExtension().appendingPathExtension("json")
        do {
            for (from, to) in [(url, target), (sidecar, targetSidecar)] where fm.fileExists(atPath: from.path) {
                if fm.fileExists(atPath: to.path) { try fm.trashItem(at: to, resultingItemURL: nil) }
                try fm.copyItem(at: from, to: to)
            }
        } catch {
            NSAlert(error: error).runModal()
        }
    }

    static func trash(_ url: URL, model: AppModel) {
        let alert = NSAlert()
        alert.messageText = "把「\(url.lastPathComponent)」移到废纸篓？"
        alert.informativeText = "图片和它的参数记录（JSON）会一起移到废纸篓，可从废纸篓恢复。"
        alert.addButton(withTitle: "移到废纸篓")
        alert.addButton(withTitle: "取消")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let sidecar = url.deletingPathExtension().appendingPathExtension("json")
        for file in [url, sidecar] where FileManager.default.fileExists(atPath: file.path) {
            try? FileManager.default.trashItem(at: file, resultingItemURL: nil)
        }
        if model.selection == url { model.selection = nil }
        model.reloadHistory()
    }
}

/// A labelled value row for the metrics and info grids.
struct InfoRow: View {
    let label: String
    let value: String
    var tint: Color? = nil

    var body: some View {
        GridRow {
            Text(label).foregroundStyle(.secondary).gridColumnAlignment(.leading)
            Text(value).monospacedDigit().foregroundStyle(tint ?? .primary).gridColumnAlignment(.leading)
                .textSelection(.enabled)
        }
    }
}

/// Section header with an optional trailing accessory, used for the left-panel groups.
struct SectionHeader<Accessory: View>: View {
    let title: String
    let symbol: String
    @ViewBuilder var accessory: Accessory

    var body: some View {
        HStack {
            Label(title, systemImage: symbol).font(.headline)
            Spacer()
            accessory
        }
    }
}

extension SectionHeader where Accessory == EmptyView {
    init(title: String, symbol: String) {
        self.init(title: title, symbol: symbol) { EmptyView() }
    }
}
