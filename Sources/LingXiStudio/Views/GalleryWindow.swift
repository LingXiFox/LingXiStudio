import QwenImageCore
import SwiftUI

/// Gallery detail view: filters, multi-selection, batch export / move / delete, compare and full parameters.
struct GalleryWindow: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @State private var selected: Set<URL> = []
    @State private var thumbSize: CGFloat = 150
    @State private var message: String?

    var body: some View {
        @Bindable var model = model
        NavigationSplitView {
            List {
                Section("范围") {
                    Picker("时间", selection: $model.galleryFilter.dateRange) {
                        Text("全部").tag(GalleryFilter.DateRange.all)
                        Text("今天").tag(GalleryFilter.DateRange.today)
                        Text("近 7 天").tag(GalleryFilter.DateRange.week)
                        Text("近 30 天").tag(GalleryFilter.DateRange.month)
                    }
                    Picker("模式", selection: $model.galleryFilter.mode) {
                        Text("全部模式").tag(GenerationMode?.none)
                        ForEach(CreationMode.allCases) { Text($0.title).tag(GenerationMode?.some($0)) }
                    }
                    Picker("尺寸", selection: $model.galleryFilter.size) {
                        Text("全部尺寸").tag(String?.none)
                        ForEach(sizes, id: \.self) { Text($0.replacingOccurrences(of: "x", with: "×")).tag(String?.some($0)) }
                    }
                    Picker("模型", selection: $model.galleryFilter.model) {
                        Text("全部模型").tag(String?.none)
                        ForEach(modelNames, id: \.self) { Text($0).tag(String?.some($0)) }
                    }
                    Toggle("只看收藏", isOn: $model.galleryFilter.favoritesOnly)
                    Toggle("只看含参考图", isOn: $model.galleryFilter.withInputsOnly)
                }
                Section("文件夹") {
                    Label(model.outputDirectory.lastPathComponent, systemImage: "folder.fill")
                        .help(model.outputDirectory.path)
                    ForEach(model.galleryFolders, id: \.self) { path in
                        Label(URL(fileURLWithPath: path).lastPathComponent, systemImage: "archivebox")
                            .help(path)
                            .contextMenu {
                                Button("从图库移除（不删除文件）") { model.galleryFolders.removeAll { $0 == path } }
                            }
                    }
                    Button("添加文件夹…", systemImage: "plus") {
                        if let url = ImageFileActions.chooseFolder(prompt: "添加") { model.galleryFolders.append(url.path) }
                    }
                    if let legacy = model.legacyGalleryCandidate {
                        Button("导入旧版 WebUI 图库", systemImage: "square.and.arrow.down") {
                            model.galleryFolders.append(legacy.path)
                        }
                        .help(legacy.path)
                    }
                }
            }
            .navigationSplitViewColumnWidth(min: 200, ideal: 220)
        } detail: {
            VStack(spacing: 0) {
                HStack(spacing: 10) {
                    TextField("搜索提示词、种子或文件名", text: $model.galleryFilter.text)
                        .textFieldStyle(.roundedBorder).frame(maxWidth: 320)
                    Text("\(items.count) 张").foregroundStyle(.secondary).monospacedDigit()
                    Spacer()
                    Slider(value: $thumbSize, in: 90...280).frame(width: 120).help("缩略图大小")
                }
                .padding(10)
                Divider()
                if items.isEmpty {
                    ContentUnavailableView("没有符合条件的作品", systemImage: "photo.on.rectangle")
                } else {
                    ScrollView {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: thumbSize), spacing: 10)], spacing: 10) {
                            ForEach(items) { item in cell(item) }
                        }
                        .padding(12)
                    }
                }
                if let message {
                    Text(message).font(.caption).padding(6).frame(maxWidth: .infinity).background(.bar)
                }
            }
            .toolbar {
                ToolbarItemGroup {
                    Button("刷新", systemImage: "arrow.clockwise") { model.reloadGallery() }
                        .help("重新扫描图库文件夹")
                    Button("全选") { selected = Set(items.map(\.url)) }
                    Button("取消选择") { selected.removeAll() }.disabled(selected.isEmpty)
                    Button("比较", systemImage: "square.split.2x1") {
                        let urls = Array(selected.prefix(2))
                        openWindow(id: "compare", value: CompareRequest(left: urls[0], right: urls[1]))
                    }
                    .disabled(selected.count != 2)
                    .help("选中两张图进行前后对比")
                    Button("导出…", systemImage: "square.and.arrow.up") { export() }.disabled(selected.isEmpty)
                    Button("移动…", systemImage: "folder") { move() }.disabled(selected.isEmpty)
                    Button("删除…", systemImage: "trash") {
                        let chosen = items.filter { selected.contains($0.url) }
                        if ImageFileActions.confirmTrash(count: chosen.count, name: chosen.first?.url.lastPathComponent ?? "") {
                            model.trashArtworks(chosen)
                            selected.removeAll()
                        }
                    }
                    .disabled(selected.isEmpty)
                }
            }
        }
        .navigationTitle("图库")
        .inspector(isPresented: .constant(selectedItem != nil)) {
            if let item = selectedItem { ArtworkInspector(item: item) }
        }
    }

    private var items: [Artwork] { model.filteredArtworks }
    private var selectedItem: Artwork? { selected.count == 1 ? items.first { selected.contains($0.url) } : nil }
    private var sizes: [String] { Array(Set(model.artworks.map { "\($0.pixelWidth)x\($0.pixelHeight)" })).sorted() }
    private var modelNames: [String] { Array(Set(model.artworks.compactMap(\.modelName))).sorted() }

    private func cell(_ item: Artwork) -> some View {
        let isSelected = selected.contains(item.url)
        return VStack(alignment: .leading, spacing: 4) {
            Thumbnail(url: item.url, size: thumbSize)
                .overlay(
                    RoundedRectangle(cornerRadius: Design.innerCorner)
                        .strokeBorder(isSelected ? Color.accentColor : .clear, lineWidth: 3))
                .overlay(alignment: .topTrailing) {
                    HStack(spacing: 3) {
                        if item.origin == .legacy { Image(systemName: "archivebox.fill").help("旧版 WebUI 作品") }
                        if model.favorites.contains(item.url.path) { Image(systemName: "star.fill").foregroundStyle(.yellow) }
                    }
                    .font(.caption).padding(5)
                }
            Text(item.prompt ?? "未记录参数").font(.caption).lineLimit(1).foregroundStyle(item.prompt == nil ? .tertiary : .primary)
            Text([item.mode?.title, item.seed.map { "种子 \($0)" }].compactMap { $0 }.joined(separator: " · "))
                .font(.caption2).foregroundStyle(.secondary).lineLimit(1)
        }
        .frame(width: thumbSize)
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { openWindow(id: "viewer", value: item.url) }
        .onTapGesture {
            if NSEvent.modifierFlags.contains(.command) {
                if isSelected { selected.remove(item.url) } else { selected.insert(item.url) }
            } else {
                selected = [item.url]
                model.selection = item.url
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
        .accessibilityValue(isSelected ? "已选择" : "未选择")
        .accessibilityAction { selected = [item.url]; model.selection = item.url }
        .contextMenu { ImageActionsMenu(item: item) }
    }

    private func export() {
        guard let folder = ImageFileActions.chooseFolder(prompt: "导出到此处") else { return }
        let chosen = items.filter { selected.contains($0.url) }
        let (done, errors) = model.exportArtworks(chosen, to: folder)
        message = "已导出 \(done) 张（\(model.exportSettings.format.label)）" + (errors.isEmpty ? "" : "，失败 \(errors.count) 张：\(errors.first!)")
    }

    private func move() {
        guard let folder = ImageFileActions.chooseFolder(prompt: "移动到此处") else { return }
        model.moveArtworks(items.filter { selected.contains($0.url) }, to: folder)
        selected.removeAll()
        message = "已移动（含参数记录），同名文件不会被覆盖"
    }
}

/// Full generation record of one image, with reuse actions.
struct ArtworkInspector: View {
    @Environment(AppModel.self) private var model
    let item: Artwork

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Thumbnail(url: item.url, size: 240, fit: true)
                HStack {
                    Button("复用全部参数") { model.reuse(item) }.disabled(!item.isReusable)
                    Menu("送去创作") {
                        Button("作为参考图") { model.send(item, to: .reference) }
                        Button("作为初始图") { model.send(item, to: .initial) }
                        Button("图像编辑") { model.send(item, to: .edit) }
                        Button("局部重绘") { model.send(item, to: .inpaint) }
                    }
                    .fixedSize()
                }
                .controlSize(.small)
                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 10, verticalSpacing: 5) {
                    row("文件", item.url.lastPathComponent)
                    row("尺寸", "\(item.pixelWidth)×\(item.pixelHeight)")
                    row("来源", item.origin == .legacy ? "旧版 WebUI" : "LingXiStudio")
                    row("模式", item.mode?.title)
                    row("模型", item.modelName)
                    row("种子", item.seed.map(String.init))
                    row("步数", item.steps.map(String.init))
                    row("Guidance", item.guidance.map { String(format: "%.1f", $0) })
                    if let r = item.metadata?.request {
                        row("调度", r.scheduler.rawValue)
                        if r.mode != .textToImage && r.mode != .reference { row("强度", String(format: "%.2f", r.strength)) }
                        row("输入图", r.inputs.isEmpty ? "无" : r.inputs.map { "\($0.role.rawValue) \($0.width)×\($0.height)" }.joined(separator: "，"))
                        row("精度", item.metadata?.precision)
                        row("耗时", item.metadata.map { Format.duration($0.totalSeconds) })
                        row("创建", item.metadata?.createdAt.formatted(date: .numeric, time: .standard))
                    }
                    row("LoRA", item.loras.isEmpty ? "无" : item.loras.map {
                        "\(URL(fileURLWithPath: $0.path).deletingPathExtension().lastPathComponent) ×\(String(format: "%.2f", $0.scale))"
                    }.joined(separator: "\n"))
                }
                .font(.callout)
                if let prompt = item.prompt {
                    GroupBox("提示词") { Text(prompt).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
                }
                if let negative = item.metadata?.request.negativePrompt ?? item.legacy?.negativePrompt, !negative.isEmpty {
                    GroupBox("负面提示词") { Text(negative).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
                }
            }
            .padding(14)
        }
        .inspectorColumnWidth(min: 260, ideal: 300)
    }

    private func row(_ label: String, _ value: String?) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value ?? "未记录").foregroundStyle(value == nil ? .tertiary : .primary).textSelection(.enabled)
        }
    }
}

struct CompareRequest: Codable, Hashable {
    let left: URL
    let right: URL
}

/// Before/after comparison: slider over two images, or side by side.
struct CompareWindow: View {
    let request: CompareRequest
    @State private var split: CGFloat = 0.5
    @State private var sideBySide = false
    @State private var left: NSImage?
    @State private var right: NSImage?

    var body: some View {
        VStack(spacing: 0) {
            if sideBySide {
                HStack(spacing: 2) {
                    image(left)
                    image(right)
                }
            } else {
                GeometryReader { geo in
                    ZStack {
                        image(right)
                        image(left).mask(alignment: .leading) {
                            Rectangle().frame(width: geo.size.width * split)
                        }
                        Rectangle().fill(.white).frame(width: 2).shadow(radius: 2)
                            .position(x: geo.size.width * split, y: geo.size.height / 2)
                    }
                    .contentShape(Rectangle())
                    .gesture(DragGesture(minimumDistance: 0).onChanged { split = min(max($0.location.x / geo.size.width, 0), 1) })
                }
            }
            HStack {
                Text(request.left.lastPathComponent).lineLimit(1)
                Spacer()
                Picker("", selection: $sideBySide) {
                    Text("滑杆").tag(false)
                    Text("并排").tag(true)
                }
                .pickerStyle(.segmented).fixedSize()
                Spacer()
                Text(request.right.lastPathComponent).lineLimit(1)
            }
            .font(.caption).padding(8).background(.bar)
        }
        .background(Design.canvasBackground)
        .task {
            left = NSImage(contentsOf: request.left)
            right = NSImage(contentsOf: request.right)
        }
        .navigationTitle("对比")
    }

    private func image(_ image: NSImage?) -> some View {
        Group {
            if let image { Image(nsImage: image).resizable().scaledToFit() } else { ProgressView() }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Standalone viewer window for large inspection; the workbench layout stays as it was.
struct ImageViewerWindow: View {
    @Environment(AppModel.self) private var model
    let url: URL
    @State private var zoom = ZoomState()

    var body: some View {
        let item = model.artworks.first { $0.url == url }
        ZStack(alignment: .bottom) {
            ZoomableImage(url: url, zoom: $zoom)
                .contextMenu { if let item { ImageActionsMenu(item: item) } }
            HStack(spacing: 10) {
                ZoomControls(zoom: $zoom)
                if let item {
                    Divider().frame(height: 16)
                    Text([item.mode?.title, "\(item.pixelWidth)×\(item.pixelHeight)", item.seed.map { "种子 \($0)" }]
                        .compactMap { $0 }.joined(separator: " · "))
                        .font(.caption).foregroundStyle(.secondary)
                    Menu {
                        ImageActionsMenu(item: item)
                    } label: {
                        Image(systemName: "ellipsis")
                    }
                    .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 7)
            .glassEffect(.regular, in: Capsule())
            .padding(14)
        }
        .background(Design.canvasBackground)
        .navigationTitle(url.lastPathComponent)
    }
}

/// Large mask canvas window; shares the mask document with the workbench, so nothing is lost when it closes.
struct MaskEditorWindow: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 0) {
            if let main = model.mainImage {
                MaskToolbar()
                MaskCanvas(image: main)
                HStack {
                    Text(String(format: "已选 %.1f%% · 白色 / 涂抹区域 = 待重绘区域", model.mask.coverage * 100))
                    Spacer()
                    Text("\(main.width)×\(main.height)")
                }
                .font(.caption).foregroundStyle(.secondary)
                .padding(8).background(.bar)
            } else {
                ContentUnavailableView("没有主图", systemImage: "paintbrush.pointed", description: Text("在工作台中先添加要重绘的图片"))
            }
        }
        .background(Design.canvasBackground)
        .navigationTitle("蒙版画布")
    }
}
