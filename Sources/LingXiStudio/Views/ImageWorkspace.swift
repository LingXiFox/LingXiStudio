import AppKit
import SwiftUI

/// Right top: one image area that switches between preview, reference images and mask.
struct ImageWorkspace: View {
    enum Tab: String, CaseIterable { case preview = "预览", references = "参考图", mask = "蒙版" }

    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @State private var tab: Tab = .preview
    @State private var zoom = ZoomState()

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Picker("", selection: $tab) {
                    ForEach(Tab.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
                Spacer()
                if tab == .preview, let item = model.selectedItem {
                    ZoomControls(zoom: $zoom)
                    Button("放大查看", systemImage: "arrow.up.left.and.arrow.down.right") {
                        openWindow(id: "viewer", value: item.url)
                    }
                    .labelStyle(.iconOnly)
                    .help("在独立窗口中查看")
                    Menu("操作", systemImage: "ellipsis.circle") { ImageActionsMenu(item: item) }
                        .menuStyle(.borderlessButton).fixedSize()
                }
            }
            .padding(8)
            Divider()
            switch tab {
            case .preview: preview
            case .references: unavailable(CreationMode.reference)
            case .mask: unavailable(CreationMode.inpaint)
            }
        }
    }

    @ViewBuilder
    private var preview: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .bottom) {
                if let item = model.selectedItem {
                    ZoomableImage(url: item.url, zoom: $zoom)
                        .contextMenu { ImageActionsMenu(item: item) }
                } else {
                    ContentUnavailableView("还没有图像", systemImage: "photo", description: Text("在左侧填写提示词并按 ⌘↩ 生成"))
                }
                if let live = model.live {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(live.stage == .denoising ? "去噪 \(live.step)/\(live.totalSteps)" : live.stageLabel)
                        if live.imageCount > 1 { Text("· 第 \(live.imageIndex + 1)/\(live.imageCount) 张") }
                    }
                    .font(.callout)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(.regularMaterial, in: Capsule())
                    .padding(10)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            if let item = model.selectedItem { details(item) }
            if !model.history.isEmpty { strip }
        }
    }

    private func details(_ item: HistoryItem) -> some View {
        HStack(spacing: 10) {
            if let meta = item.metadata {
                let r = meta.request
                Text("\(r.width)×\(r.height)")
                Text("\(r.steps) 步")
                Text("种子 \(String(r.seed))").textSelection(.enabled)
                if !r.loras.isEmpty { Text("\(r.loras.count) 个 LoRA") }
                Text(Format.duration(meta.totalSeconds))
            } else {
                Text("无生成记录（未记录）")
            }
            Spacer()
            Text(item.url.lastPathComponent).lineLimit(1).truncationMode(.middle)
        }
        .font(.caption).foregroundStyle(.secondary)
        .padding(.horizontal, 8).padding(.vertical, 4)
    }

    private var strip: some View {
        ScrollView(.horizontal) {
            LazyHStack(spacing: 6) {
                ForEach(model.history.prefix(30)) { item in
                    Thumbnail(url: item.url, size: 52)
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(item.url == model.selection ? Color.accentColor : .clear, lineWidth: 2))
                        .onTapGesture { model.selection = item.url }
                        .contextMenu { ImageActionsMenu(item: item) }
                }
            }
            .padding(6)
        }
        .frame(height: 64)
        .background(Color(nsColor: .underPageBackgroundColor))
    }

    private func unavailable(_ mode: CreationMode) -> some View {
        ContentUnavailableView(
            mode == .inpaint ? "蒙版画布未就绪" : "参考图未就绪", systemImage: mode.symbol,
            description: Text(mode.unavailableReason ?? ""))
    }
}

struct ZoomState: Equatable {
    /// `nil` = fit to the view; otherwise image pixels per screen pixel.
    var scale: Double?
}

struct ZoomControls: View {
    @Binding var zoom: ZoomState

    var body: some View {
        HStack(spacing: 2) {
            Button("适配窗口", systemImage: "arrow.down.right.and.arrow.up.left") { zoom.scale = nil }
                .help("适配窗口")
            Button("100%", systemImage: "1.magnifyingglass") { zoom.scale = 1 }
                .help("100% 像素")
            Button("缩小", systemImage: "minus.magnifyingglass") { zoom.scale = max((zoom.scale ?? 0.5) / 1.5, 0.05) }
            Button("放大", systemImage: "plus.magnifyingglass") { zoom.scale = min((zoom.scale ?? 0.5) * 1.5, 8) }
            Text(zoom.scale.map { "\(Int($0 * 100))%" } ?? "适配").font(.caption).monospacedDigit().frame(width: 40)
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.borderless)
    }
}

/// An image that fits the view by default, can show 100 % device pixels, zoom with pinch and pan by scrolling.
struct ZoomableImage: View {
    let url: URL
    @Binding var zoom: ZoomState
    @State private var image: NSImage?
    @State private var pinchBase: Double?
    @Environment(\.displayScale) private var displayScale

    var body: some View {
        GeometryReader { geo in
            if let image, let rep = image.representations.first {
                let pixels = CGSize(width: rep.pixelsWide, height: rep.pixelsHigh)
                let fit = min(geo.size.width / pixels.width, geo.size.height / pixels.height) * displayScale
                let scale = zoom.scale ?? fit
                let size = CGSize(width: pixels.width * scale / displayScale, height: pixels.height * scale / displayScale)
                ScrollView([.horizontal, .vertical]) {
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(scale >= 2 ? .none : .high)
                        .frame(width: size.width, height: size.height)
                        .frame(minWidth: geo.size.width, minHeight: geo.size.height)
                }
                .scrollIndicators(zoom.scale == nil ? .hidden : .automatic)
                .gesture(
                    MagnifyGesture()
                        .onChanged { value in
                            let base = pinchBase ?? scale
                            pinchBase = base
                            zoom.scale = min(max(base * value.magnification, 0.05), 8)
                        }
                        .onEnded { _ in pinchBase = nil }
                )
                .onTapGesture(count: 2) { zoom.scale = zoom.scale == nil ? 1 : nil }
            } else {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task(id: url) { image = NSImage(contentsOf: url) }
    }
}

/// Standalone viewer window for large inspection; the workbench layout stays as it was.
struct ImageViewerWindow: View {
    @Environment(AppModel.self) private var model
    let url: URL
    @State private var zoom = ZoomState()

    var body: some View {
        let item = model.history.first { $0.url == url } ?? HistoryItem(url: url, metadata: nil, created: Date())
        VStack(spacing: 0) {
            ZoomableImage(url: url, zoom: $zoom)
                .contextMenu { ImageActionsMenu(item: item) }
            Divider()
            HStack {
                ZoomControls(zoom: $zoom)
                Spacer()
                if let r = item.metadata?.request {
                    Text("\(r.width)×\(r.height) · \(r.steps) 步 · 种子 \(String(r.seed))").font(.caption)
                        .foregroundStyle(.secondary)
                }
                Menu("操作", systemImage: "ellipsis.circle") { ImageActionsMenu(item: item) }
                    .menuStyle(.borderlessButton).fixedSize()
            }
            .padding(8)
        }
        .navigationTitle(url.lastPathComponent)
    }
}
