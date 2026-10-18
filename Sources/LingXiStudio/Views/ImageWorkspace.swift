import AppKit
import QwenImageCore
import SwiftUI
import UniformTypeIdentifiers

/// Right top: the artwork canvas, with the input images and mask canvas one tap away in the image modes.
struct ImageWorkspace: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 0) {
            if model.mode != .textToImage {
                HStack(spacing: 8) {
                    BrandSegmented(
                        selection: $model.workspaceTab,
                        options: [
                            .init(value: .preview, title: "作品", symbol: "photo"),
                            .init(
                                value: .inputs,
                                title: model.mode.needsInitial ? "初始图" : "输入图\(model.conditions.isEmpty ? "" : " \(model.conditions.count)")",
                                symbol: "photo.stack"),
                        ] + (model.mode.needsMask ? [.init(value: .mask, title: "蒙版", symbol: "paintbrush.pointed")] : []))
                    Spacer()
                    if model.workspaceTab == .mask && model.mainImage != nil {
                        Button("大画布", systemImage: "arrow.up.left.and.arrow.down.right") { openWindow(id: "mask") }
                            .buttonStyle(.ghost)
                            .help("在独立窗口中精细涂抹，返回后蒙版保留（⇧⌘K）")
                    }
                }
                .padding(.horizontal, Metrics.m).padding(.vertical, Metrics.s)
            }
            Group {
                switch model.mode == .textToImage ? .preview : model.workspaceTab {
                case .preview: PreviewPane()
                case .inputs: InputsPane()
                case .mask: MaskPane()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Brand.canvas)
    }
}

// MARK: - Preview

struct PreviewPane: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @State private var zoom = ZoomState()

    var body: some View {
        VStack(spacing: 0) {
            if let item = model.selectedArtwork, !model.isRunning {
                CanvasToolbar(item: item, zoom: $zoom)
            }
            ZStack {
                if let item = model.selectedArtwork, !model.isRunning {
                    ZoomableImage(url: item.url, zoom: $zoom)
                        .contextMenu { ImageActionsMenu(item: item) }
                        .onTapGesture(count: 2) { openWindow(id: "viewer", value: item.url) }
                } else if !model.isRunning {
                    EmptyState(symbol: "sparkles", title: "作品会出现在这里", message: "写下提示词，按 ⌘↩ 生成")
                }
                if let live = model.live {
                    ZStack {
                        Brand.canvas
                        ProgressOverlay(live: live).padding(Metrics.xl)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .contentShape(Rectangle())
                } else if model.isRunning {
                    VStack(spacing: Metrics.m) {
                        Text(model.cancellationRequested ? "正在停止…" : "准备生成…")
                        Button(model.cancellationRequested ? "正在停止…" : "取消生成", systemImage: "stop.fill") { model.stopCurrent() }
                            .buttonStyle(SecondaryButtonStyle())
                            .disabled(model.cancellationRequested)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Brand.canvas)
                } else if let notice = model.canvasNotice {
                    VStack {
                        Spacer()
                        NoticeOverlay(notice: notice).padding(Metrics.m)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

/// Preview-only scrim, fed by runtime callbacks; the rest of the workbench stays interactive.
struct ProgressOverlay: View {
    @Environment(AppModel.self) private var model
    let live: LiveProgress

    enum Phase: Int, CaseIterable {
        case conditioning, denoising, decoding

        var title: String {
            switch self {
            case .conditioning: "条件编码"
            case .denoising: "DiT 去噪"
            case .decoding: "VAE 解码"
            }
        }

        init(_ stage: GenerationStage) {
            switch stage {
            case .validating, .loadingTextEncoder, .encodingPrompt, .encodingImages, .loadingTransformer, .prefill:
                self = .conditioning
            case .denoising: self = .denoising
            case .loadingVAE, .decoding, .done: self = .decoding
            }
        }
    }

    var body: some View {
        let phase = Phase(live.stage)
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                ForEach(Phase.allCases, id: \.self) { p in
                    if p != .conditioning {
                        Image(systemName: "chevron.right").font(.system(size: 8, weight: .bold)).foregroundStyle(Brand.textTertiary)
                    }
                    HStack(spacing: 4) {
                        if p.rawValue < phase.rawValue {
                            Image(systemName: "checkmark").font(.system(size: 9, weight: .bold))
                        }
                        Text(p.title)
                    }
                    .font(.caption.weight(p == phase ? .semibold : .regular))
                    .foregroundStyle(p == phase ? Brand.foxText : p.rawValue < phase.rawValue ? Brand.textSecondary : Brand.textTertiary)
                }
            }
            Text(model.cancellationRequested ? "正在停止…" : live.stageLabel).font(.title3.weight(.semibold))
            if model.cancellationRequested {
                Text("停止请求已收到，等待当前 GPU 计算完成")
                    .font(.caption).foregroundStyle(Brand.textSecondary)
            }
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                if phase == .denoising {
                    Text("\(live.step) / \(live.totalSteps)").font(.title3.weight(.semibold)).monospacedDigit()
                    Text("Steps").font(.caption).foregroundStyle(Brand.textSecondary)
                }
                if live.imageCount > 1 {
                    Text("第 \(live.imageIndex + 1)/\(live.imageCount) 张").font(.caption).foregroundStyle(Brand.textSecondary)
                }
                Spacer(minLength: 8)
                if let rate = live.secondsPerStep {
                    Text(String(format: "%.2f s/step", rate)).font(.body.weight(.semibold)).monospacedDigit()
                        .help("已完成去噪步的实测平均耗时")
                } else {
                    Text("等待首步测速").font(.caption).foregroundStyle(Brand.textSecondary)
                }
            }
            ProgressBar(fraction: live.overallFraction)
            HStack {
                if phase == .denoising, let remaining = live.remaining {
                    Text("去噪剩余约 \(Format.duration(remaining))").font(.caption).monospacedDigit().foregroundStyle(Brand.textSecondary)
                        .help(live.secondsPerStep.map { String(format: "按实测 %.2f s/step 估计", $0) } ?? "")
                }
                Spacer(minLength: 8)
                Button(model.cancellationRequested ? "正在停止…" : "取消生成", systemImage: "stop.fill") { model.stopCurrent() }
                    .buttonStyle(SecondaryButtonStyle())
                    .disabled(model.cancellationRequested)
                    .help("取消生成（⌘.）")
            }
        }
        .frame(maxWidth: 380)
    }
}

/// Failure or cancellation of the last task, dismissed automatically after a brief display.
struct NoticeOverlay: View {
    @Environment(AppModel.self) private var model
    let notice: AppModel.CanvasNotice

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: notice.failed ? "xmark.octagon.fill" : "minus.circle.fill")
                .foregroundStyle(notice.failed ? Brand.danger : Brand.textSecondary)
            VStack(alignment: .leading, spacing: 3) {
                Text(notice.title).font(.callout.weight(.semibold))
                if let detail = notice.detail {
                    Text(detail).font(.caption).foregroundStyle(Brand.textSecondary).lineLimit(3).textSelection(.enabled)
                }
            }
            Spacer(minLength: 8)
            if notice.failed {
                Button("任务") { model.managementTab = .tasks }.buttonStyle(.ghost).font(.caption)
            }
            Button {
                model.canvasNotice = nil
            } label: {
                Image(systemName: "xmark")
            }
            .buttonStyle(.ghost).help("关闭")
        }
        .padding(12)
        .frame(maxWidth: 380)
        .background(Brand.paper, in: RoundedRectangle(cornerRadius: Metrics.radiusL, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: Metrics.radiusL, style: .continuous)
            .strokeBorder(notice.failed ? Brand.danger.opacity(0.4) : Brand.hairline))
        .shadow(color: .black.opacity(0.12), radius: 10, y: 3)
        .task(id: notice.id) {
            do { try await Task.sleep(for: .seconds(notice.failed ? 6 : 3)) }
            catch { return }
            if model.canvasNotice?.id == notice.id { model.canvasNotice = nil }
        }
    }
}

/// Compact canvas tools stay outside the artwork so keyboard users can reach them without hovering.
struct CanvasToolbar: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    let item: Artwork
    @Binding var zoom: ZoomState

    var body: some View {
        HStack(spacing: 2) {
            Text("作品").font(TypeScale.section).foregroundStyle(Brand.textSecondary)
            Spacer(minLength: 4)
            Menu {
                Button("适配窗口") { zoom.scale = nil }
                Button("100% 像素") { zoom.scale = 1 }
                Divider()
                Button("放大") { zoom.scale = min((zoom.scale ?? 0.5) * 1.5, 8) }
                Button("缩小") { zoom.scale = max((zoom.scale ?? 0.5) / 1.5, 0.05) }
            } label: {
                Text(zoom.scale.map { "\(Int($0 * 100))%" } ?? "适配").font(.caption).monospacedDigit()
            }
            .menuStyle(.borderlessButton).fixedSize()
            .padding(.horizontal, 7)
            .accessibilityLabel("画布缩放")
            Button("复制图像", systemImage: "doc.on.doc") { ImageFileActions.copyImage(item.url) }
                .help("复制图像")
            Button("另存为", systemImage: "square.and.arrow.down") { ImageFileActions.saveAs(item, settings: model.exportSettings) }
                .help("另存为…")
            Menu {
                ImageActionsMenu(item: item)
            } label: {
                Image(systemName: "ellipsis")
            }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
            .padding(.horizontal, 5)
            .accessibilityLabel("作品操作")
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.ghost)
        .padding(.horizontal, Metrics.m).padding(.vertical, 4)
    }
}

struct ZoomState: Equatable {
    /// `nil` = fit to the view; otherwise image pixels per device pixel.
    var scale: Double?
}

struct ZoomControls: View {
    @Binding var zoom: ZoomState

    var body: some View {
        HStack(spacing: 2) {
            Button("适配窗口", systemImage: "arrow.down.right.and.arrow.up.left") { zoom.scale = nil }
            Button("100% 像素", systemImage: "1.magnifyingglass") { zoom.scale = 1 }
            Button("缩小", systemImage: "minus.magnifyingglass") { zoom.scale = max((zoom.scale ?? 0.5) / 1.5, 0.05) }
            Button("放大", systemImage: "plus.magnifyingglass") { zoom.scale = min((zoom.scale ?? 0.5) * 1.5, 8) }
            Text(zoom.scale.map { "\(Int($0 * 100))%" } ?? "适配").font(.caption).monospacedDigit().frame(width: 38)
                .foregroundStyle(Brand.textSecondary)
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.ghost)
    }
}

/// An image that fits the view by default, shows 100 % device pixels on demand, zooms with pinch, pans by scrolling.
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
                let fit = min(max(geo.size.width - 40, 40) / pixels.width, max(geo.size.height - 40, 40) / pixels.height) * displayScale
                let scale = zoom.scale ?? fit
                let size = CGSize(width: pixels.width * scale / displayScale, height: pixels.height * scale / displayScale)
                ScrollView([.horizontal, .vertical]) {
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(scale >= 2 ? .none : .high)
                        .frame(width: size.width, height: size.height)
                        .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
                        .shadow(color: .black.opacity(0.22), radius: 14, y: 5)
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
            } else {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task(id: url) { image = NSImage(contentsOf: url) }
    }
}

// MARK: - Input images

struct InputsPane: View {
    @Environment(AppModel.self) private var model
    @State private var targeted = false

    var body: some View {
        Group {
            if model.mode == .textToImage {
                EmptyState(symbol: "text.below.photo", title: "文生图不使用输入图", message: "在左侧切换到参考图创作、图生图、图像编辑或局部重绘")
            } else if model.mode.needsInitial {
                initialPane
            } else {
                conditionsPane
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .overlay {
            if targeted {
                RoundedRectangle(cornerRadius: Design.corner).strokeBorder(Brand.fox, style: StrokeStyle(lineWidth: 2, dash: [6]))
                    .padding(8)
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            if model.mode.needsInitial, let first = urls.first { model.setInitial(url: first) } else { model.importConditions(urls: urls) }
            return true
        } isTargeted: { targeted = $0 }
        .onPasteCommand(of: [.fileURL, .image, .png, .tiff]) { _ in
            if model.mode.needsInitial {
                if let image = InputStore.fromPasteboard().first { model.initialImage = image }
            } else {
                model.pasteConditions()
            }
        }
    }

    private var conditionsPane: some View {
        VStack(spacing: 0) {
            HStack {
                Text(model.mode == .reference ? "参考图（按顺序作为 图 1、图 2…）" : "第 1 张为主图，其余为附加参考")
                    .font(.caption).foregroundStyle(Brand.textSecondary)
                Spacer()
                Text("\(model.conditions.count)/10").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Button("添加…", systemImage: "plus") { model.importConditions(urls: ImageFileActions.chooseImages()) }
                Button("粘贴", systemImage: "doc.on.clipboard") { model.pasteConditions() }
                Button("清空", systemImage: "trash") { model.conditions.removeAll() }.disabled(model.conditions.isEmpty)
            }
            .labelStyle(.iconOnly).buttonStyle(.ghost)
            .padding(.horizontal, 10).padding(.vertical, 6)
            if model.conditions.isEmpty {
                VStack(spacing: 0) {
                    EmptyState(symbol: "square.and.arrow.down.on.square", title: "拖入或粘贴图片", message: "支持 Finder 拖放、⌘V 粘贴、从图库「送去创作」，最多 10 张")
                        .frame(maxHeight: 220)
                    Button("选择图片…") { model.importConditions(urls: ImageFileActions.chooseImages()) }
                        .buttonStyle(SecondaryButtonStyle())
                    Spacer()
                }
            } else {
                List {
                    ForEach(Array(model.conditions.enumerated()), id: \.element.id) { index, image in
                        InputRow(image: image, index: index)
                    }
                    .onMove { model.conditions.move(fromOffsets: $0, toOffset: $1) }
                }
                .listStyle(.inset)
                .scrollContentBackground(.hidden)
            }
        }
    }

    private var initialPane: some View {
        VStack(spacing: 10) {
            if let image = model.initialImage {
                Thumbnail(url: image.url, size: 220, corner: 10, fit: true)
                Text("\(image.name) · \(image.width)×\(image.height)").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("替换…") { if let url = ImageFileActions.chooseImages(multiple: false).first { model.setInitial(url: url) } }
                    Button("移除", role: .destructive) { model.initialImage = nil }
                }
                .controlSize(.small)
            } else {
                EmptyState(symbol: "photo.on.rectangle.angled", title: "添加初始图", message: "图像作为初始状态，按强度加噪后重绘。拖入、粘贴或选择一张图片。")
                    .frame(maxHeight: 220)
                Button("选择图片…") { if let url = ImageFileActions.chooseImages(multiple: false).first { model.setInitial(url: url) } }
                    .buttonStyle(SecondaryButtonStyle())
            }
        }
        .padding()
    }
}

struct InputRow: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    let image: InputImage
    let index: Int

    var body: some View {
        HStack(spacing: 10) {
            Thumbnail(url: image.url, size: 52, corner: 6)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    if index == 0 && model.mode != .reference {
                        Text("主图").font(.caption2.weight(.bold)).padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Brand.fox, in: Capsule()).foregroundStyle(.white)
                    } else {
                        Text("图 \(index + 1)").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                    }
                    Text(image.name).lineLimit(1).truncationMode(.middle)
                }
                let target = conditionDimensions(ratio: Double(image.width) / Double(image.height))
                Text("\(image.width)×\(image.height) → 条件尺寸 \(target.width)×\(target.height)")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Menu {
                if index > 0 && model.mode != .reference { Button("设为主图") { model.makeMain(image.id) } }
                Button("上移") { move(-1) }.disabled(index == 0)
                Button("下移") { move(1) }.disabled(index == model.conditions.count - 1)
                Button("置顶") { model.makeMain(image.id) }.disabled(index == 0)
                Divider()
                Button("查看原图") { openWindow(id: "viewer", value: image.url) }
                Button("替换…") {
                    if let url = ImageFileActions.chooseImages(multiple: false).first { model.replaceCondition(image.id, with: url) }
                }
                if let source = image.sourcePath, FileManager.default.fileExists(atPath: source) {
                    Button("在 Finder 中显示原文件") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: source)]) }
                }
                Divider()
                Button("移除", role: .destructive) { model.removeCondition(image.id) }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
        }
        .padding(.vertical, 2)
    }

    private func move(_ delta: Int) {
        let target = index + delta
        guard model.conditions.indices.contains(target) else { return }
        model.conditions.swapAt(index, target)
    }
}

// MARK: - Mask

struct MaskPane: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let main = model.mainImage {
            VStack(spacing: 0) {
                MaskToolbar(compact: true)
                MaskCanvas(image: main)
            }
        } else {
            EmptyState(symbol: "paintbrush.pointed", title: "蒙版需要一张主图", message: "在「输入图」中添加要重绘的图片，或从图库「送去创作 → 局部重绘」")
        }
    }
}

/// Brush controls shared by the inline mask pane and the large mask window.
struct MaskToolbar: View {
    @Environment(AppModel.self) private var model
    var compact = false
    @State private var importing = false

    var body: some View {
        @Bindable var mask = model.mask
        HStack(spacing: 10) {
            Picker("", selection: $mask.tool) {
                Label("画笔", systemImage: "paintbrush.pointed.fill").tag(MaskDocument.Tool.brush)
                Label("橡皮擦", systemImage: "eraser.fill").tag(MaskDocument.Tool.eraser)
            }
            .pickerStyle(.segmented).labelsHidden().fixedSize()
            .labelStyle(.iconOnly)
            HStack(spacing: 4) {
                Image(systemName: "circle.fill").font(.system(size: 6)).foregroundStyle(.secondary)
                Slider(value: $mask.brushSize, in: 4...400).frame(width: compact ? 80 : 140)
                Image(systemName: "circle.fill").font(.system(size: 13)).foregroundStyle(.secondary)
                Text("\(Int(mask.brushSize))").font(.caption.monospacedDigit()).frame(width: 28)
            }
            .help("笔刷大小（图像像素）")
            if !compact {
                HStack(spacing: 4) {
                    Text("硬度").font(.caption).foregroundStyle(.secondary)
                    Slider(value: $mask.hardness, in: 0...1).frame(width: 90)
                }
                HStack(spacing: 4) {
                    Text("不透明度").font(.caption).foregroundStyle(.secondary)
                    Slider(value: $mask.opacity, in: 0.05...1).frame(width: 90)
                }
            }
            Spacer()
            Group {
                Button("撤销", systemImage: "arrow.uturn.backward") { mask.undo() }.disabled(!mask.canUndo)
                    .keyboardShortcut("z", modifiers: .command)
                Button("重做", systemImage: "arrow.uturn.forward") { mask.redo() }.disabled(!mask.canRedo)
                    .keyboardShortcut("z", modifiers: [.command, .shift])
                Button(mask.showOverlay ? "隐藏蒙版" : "显示蒙版", systemImage: mask.showOverlay ? "eye" : "eye.slash") {
                    mask.showOverlay.toggle()
                }
                Button("反转", systemImage: "circle.lefthalf.filled") { mask.invert() }
                Button("清空", systemImage: "trash") { mask.clear() }
                Menu {
                    Button("导入蒙版图片…") { importing = true }
                    Button("导出蒙版 PNG…") { exportMask() }.disabled(mask.isEmpty)
                } label: {
                    Image(systemName: "square.and.arrow.up.on.square")
                }
                .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
            }
            .labelStyle(.iconOnly).buttonStyle(.ghost)
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(Brand.side)
        .fileImporter(isPresented: $importing, allowedContentTypes: [.image]) { result in
            if case .success(let url) = result, let image = try? RGBAImage(contentsOf: url) { model.mask.load(image) }
        }
    }

    private func exportMask() {
        guard let rgba = model.mask.rgbaImage(), let cg = rgba.cgImage else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "mask.png"
        guard panel.runModal() == .OK, let url = panel.url,
            let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(dest, cg, nil)
        CGImageDestinationFinalize(dest)
    }
}

/// Paintable canvas: the main image with the mask overlay; zoom with pinch, brush cursor follows the pointer.
struct MaskCanvas: View {
    @Environment(AppModel.self) private var model
    let image: InputImage
    @State private var base: NSImage?
    @State private var overlay: CGImage?
    @State private var lastPoint: CGPoint?
    @State private var hover: CGPoint?
    @State private var zoom: CGFloat = 1
    @State private var pinchBase: CGFloat?

    var body: some View {
        GeometryReader { geo in
            let fit = min(geo.size.width / CGFloat(image.width), geo.size.height / CGFloat(image.height)) * 0.96
            let scale = fit * zoom
            let size = CGSize(width: CGFloat(image.width) * scale, height: CGFloat(image.height) * scale)
            ScrollView([.horizontal, .vertical]) {
                ZStack {
                    Checkerboard()
                    if let base { Image(nsImage: base).resizable().interpolation(.high) }
                    if model.mask.showOverlay, let overlay {
                        Image(decorative: overlay, scale: 1).resizable().interpolation(.medium)
                    }
                    if let hover {
                        Circle()
                            .strokeBorder(Color.white, lineWidth: 1.5)
                            .background(Circle().strokeBorder(Color.black.opacity(0.5), lineWidth: 3))
                            .frame(width: model.mask.brushSize * scale, height: model.mask.brushSize * scale)
                            .position(x: hover.x * scale, y: hover.y * scale)
                            .allowsHitTesting(false)
                    }
                }
                .frame(width: size.width, height: size.height)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0, coordinateSpace: .local)
                        .onChanged { value in
                            let p = CGPoint(x: value.location.x / scale, y: value.location.y / scale)
                            if let last = lastPoint {
                                model.mask.stroke(from: last, to: p)
                            } else {
                                model.mask.beginStroke()
                                model.mask.dab(at: p)
                            }
                            lastPoint = p
                            hover = p
                        }
                        .onEnded { _ in lastPoint = nil }
                )
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let location): hover = CGPoint(x: location.x / scale, y: location.y / scale)
                    case .ended: hover = nil
                    }
                }
                .frame(minWidth: geo.size.width, minHeight: geo.size.height)
            }
            .simultaneousGesture(
                MagnifyGesture()
                    .onChanged { value in
                        let b = pinchBase ?? zoom
                        pinchBase = b
                        zoom = min(max(b * value.magnification, 0.5), 8)
                    }
                    .onEnded { _ in pinchBase = nil }
            )
            .overlay(alignment: .bottomTrailing) {
                HStack(spacing: 6) {
                    Button("缩小", systemImage: "minus.magnifyingglass") { zoom = max(zoom / 1.4, 0.5) }
                    Text("\(Int(zoom * 100))%").font(.caption.monospacedDigit()).frame(width: 40)
                    Button("放大", systemImage: "plus.magnifyingglass") { zoom = min(zoom * 1.4, 8) }
                    Button("适配", systemImage: "arrow.down.right.and.arrow.up.left") { zoom = 1 }
                }
                .labelStyle(.iconOnly).buttonStyle(.ghost)
                .padding(.horizontal, 10).padding(.vertical, 5)
                .glassEffect(.regular, in: Capsule())
                .padding(10)
            }
        }
        .task(id: image.storedPath) { base = NSImage(contentsOf: image.url) }
        .onChange(of: model.mask.revision, initial: true) { _, _ in overlay = model.mask.overlayImage() }
    }
}
