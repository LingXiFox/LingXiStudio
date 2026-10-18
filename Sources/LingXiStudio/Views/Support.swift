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
                first.frame(width: axis == .horizontal ? firstSize : nil, height: axis == .vertical ? firstSize : nil)
                handle(available: available)
                second.frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private func handle(available: CGFloat) -> some View {
        Rectangle()
            .fill(Design.hairline)
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
                    .help("拖动调整比例，双击恢复默认")
            }
    }
}

/// Downscaled thumbnails decoded off the main thread and cached by file URL.
@MainActor
final class ThumbnailCache {
    static let shared = ThumbnailCache()
    private let cache = NSCache<NSURL, NSImage>()

    func image(for url: URL, maxPixels: Int = 320) async -> NSImage? {
        let key = "\(url.path)#\(maxPixels)" as NSString
        if let hit = cache.object(forKey: NSURL(fileURLWithPath: key as String)) { return hit }
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
        cache.setObject(image, forKey: NSURL(fileURLWithPath: key as String))
        return image
    }
}

struct Thumbnail: View {
    let url: URL
    var size: CGFloat = 96
    var corner: CGFloat = Design.innerCorner
    var fit = false
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            Color.primary.opacity(0.05)
            if let image {
                if fit {
                    Image(nsImage: image).resizable().scaledToFit()
                } else {
                    Image(nsImage: image).resizable().scaledToFill()
                }
            } else {
                ProgressView().controlSize(.small)
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: corner, style: .continuous))
        .task(id: url) { image = await ThumbnailCache.shared.image(for: url, maxPixels: Int(size * 2.5)) }
    }
}

/// Actions available on a finished image, used by the preview toolbar, history grid, gallery and context menus.
struct ImageActionsMenu: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    let item: Artwork

    var body: some View {
        Button("放大查看", systemImage: "arrow.up.left.and.arrow.down.right") { openWindow(id: "viewer", value: item.url) }
        Button(model.favorites.contains(item.url.path) ? "取消收藏" : "收藏",
               systemImage: model.favorites.contains(item.url.path) ? "star.slash" : "star") { model.toggleFavorite(item) }
        Divider()
        Button("复用全部参数", systemImage: "arrow.uturn.backward") { model.reuse(item) }
            .disabled(!item.isReusable)
        Button("使用相同种子再生成", systemImage: "arrow.clockwise") {
            model.reuse(item)
            model.enqueue()
        }
        .disabled(!item.isReusable)
        Menu("送去创作", systemImage: "paperplane") {
            Button("作为参考图", systemImage: "photo.stack") { model.send(item, to: .reference) }
            Button("作为初始图（图生图）", systemImage: "photo.on.rectangle.angled") { model.send(item, to: .initial) }
            Button("图像编辑", systemImage: "wand.and.stars") { model.send(item, to: .edit) }
            Button("局部重绘", systemImage: "paintbrush.pointed") { model.send(item, to: .inpaint) }
        }
        Divider()
        Button("复制图像内容", systemImage: "photo") { ImageFileActions.copyImage(item.url) }
        Button("复制文件", systemImage: "doc.on.doc") { ImageFileActions.copyFile(item.url) }
        Button("复制文件路径", systemImage: "link") { ImageFileActions.copyText(item.url.path) }
        Button("复制提示词", systemImage: "text.quote") { ImageFileActions.copyText(item.prompt ?? "") }
            .disabled(item.prompt == nil)
        Divider()
        Button("另存为…", systemImage: "square.and.arrow.down") { ImageFileActions.saveAs(item, settings: model.exportSettings) }
        Button("用默认应用打开", systemImage: "arrow.up.forward.app") { NSWorkspace.shared.open(item.url) }
        Button("在 Finder 中显示", systemImage: "folder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
        Divider()
        Button("移到废纸篓…", systemImage: "trash", role: .destructive) {
            if ImageFileActions.confirmTrash(count: 1, name: item.url.lastPathComponent) { model.trashArtworks([item]) }
        }
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

    /// Exports one image in the chosen format with its record; the save panel asks before replacing a file.
    static func saveAs(_ item: Artwork, settings: ExportSettings) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [settings.format.type]
        panel.nameFieldStringValue = settings.fileName(for: item, index: 1)
        guard panel.runModal() == .OK, let target = panel.url, target != item.url else { return }
        do {
            if FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.trashItem(at: target, resultingItemURL: nil)
            }
            var single = settings
            single.template = target.deletingPathExtension().lastPathComponent
            _ = try single.export(item, index: 1, to: target.deletingLastPathComponent())
        } catch {
            NSAlert(error: error).runModal()
        }
    }

    static func confirmTrash(count: Int, name: String) -> Bool {
        let alert = NSAlert()
        alert.messageText = count == 1 ? "把「\(name)」移到废纸篓？" : "把选中的 \(count) 张图移到废纸篓？"
        alert.informativeText = "图片和它的参数记录会一起移到废纸篓，可从废纸篓恢复。"
        alert.addButton(withTitle: "移到废纸篓")
        alert.addButton(withTitle: "取消")
        return alert.runModal() == .alertFirstButtonReturn
    }

    static func chooseFolder(prompt: String = "选择") -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = prompt
        return panel.runModal() == .OK ? panel.url : nil
    }

    static func chooseImages(multiple: Bool = true) -> [URL] {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = multiple
        return panel.runModal() == .OK ? panel.urls : []
    }

    static func ask(_ title: String, placeholder: String = "", initial: String = "") -> String? {
        let alert = NSAlert()
        alert.messageText = title
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 280, height: 24))
        field.placeholderString = placeholder
        field.stringValue = initial
        alert.accessoryView = field
        alert.addButton(withTitle: "确定")
        alert.addButton(withTitle: "取消")
        alert.window.initialFirstResponder = field
        guard alert.runModal() == .alertFirstButtonReturn else { return nil }
        let text = field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }
}

/// A labelled value row for metric grids.
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

/// Multi-line text editor with a placeholder and native field styling.
struct PromptEditor: View {
    @Binding var text: String
    var placeholder: String
    var minHeight: CGFloat = 90

    var body: some View {
        TextEditor(text: $text)
            .font(.body)
            .scrollContentBackground(.hidden)
            .padding(.horizontal, 6).padding(.vertical, 7)
            .frame(minHeight: minHeight, maxHeight: minHeight * 2.6)
            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: Design.innerCorner))
            .overlay(RoundedRectangle(cornerRadius: Design.innerCorner).strokeBorder(Design.hairline))
            .overlay(alignment: .topLeading) {
                if text.isEmpty {
                    Text(placeholder).foregroundStyle(.tertiary).padding(.horizontal, 11).padding(.vertical, 7)
                        .allowsHitTesting(false)
                }
            }
    }
}
