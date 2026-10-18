import AppKit
import QwenImageCore
import SwiftUI

@main
struct LingXiStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var model: AppModel

    init() {
        // MLX reads its runtime settings once, when the first array touches the GPU.
        MLXRuntime.configure()
        _model = State(initialValue: AppModel())
    }

    var body: some Scene {
        Window("LingXiStudio", id: "main") {
            WorkbenchView()
                .environment(model)
                .frame(minWidth: 1180, minHeight: 760)
        }
        .defaultSize(width: 1560, height: 980)
        .commands { WorkbenchCommands(model: model) }

        WindowGroup("图像查看", id: "viewer", for: URL.self) { $url in
            if let url { ImageViewerWindow(url: url).environment(model) }
        }
        .defaultSize(width: 1100, height: 1000)

        WindowGroup("对比", id: "compare", for: CompareRequest.self) { $request in
            if let request { CompareWindow(request: request) }
        }
        .defaultSize(width: 1300, height: 900)

        Window("图库", id: "gallery") {
            GalleryWindow().environment(model)
        }
        .defaultSize(width: 1280, height: 860)

        Window("蒙版画布", id: "mask") {
            MaskEditorWindow().environment(model)
        }
        .defaultSize(width: 1200, height: 1000)

        Window("模型库", id: "models") {
            ModelLibraryWindow().environment(model)
        }
        .defaultSize(width: 860, height: 620)

        Window("LoRA 库", id: "loras") {
            LoRALibraryWindow().environment(model)
        }
        .defaultSize(width: 900, height: 680)

        Settings {
            SettingsView().environment(model)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Needed when launched as a bare SwiftPM executable rather than from the .app bundle.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
    }
}

struct WorkbenchCommands: Commands {
    let model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandMenu("生成") {
            Button("加入队列并生成") { model.enqueue() }
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!model.blockingIssues.isEmpty)
            Button("停止当前任务") { model.stopCurrent() }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(!model.isRunning)
            Divider()
            Button("清空等待中的任务") { model.clearWaiting() }
                .disabled(model.waitingCount == 0)
            Divider()
            ForEach(Array(CreationMode.allCases.enumerated()), id: \.element.id) { index, mode in
                Button(mode.title) { model.mode = mode }
                    .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
            }
        }
        CommandMenu("素材") {
            Button("添加输入图…") {
                let urls = ImageFileActions.chooseImages(multiple: !model.mode.needsInitial)
                if model.mode.needsInitial, let first = urls.first { model.setInitial(url: first) } else { model.importConditions(urls: urls) }
            }
            .keyboardShortcut("i", modifiers: [.command, .shift])
            Button("从剪贴板粘贴为参考图") { model.pasteConditions() }
                .keyboardShortcut("v", modifiers: [.command, .shift])
        }
        CommandMenu("窗口工具") {
            Button("图库") { openWindow(id: "gallery") }.keyboardShortcut("g", modifiers: [.command, .shift])
            Button("模型库") { openWindow(id: "models") }.keyboardShortcut("m", modifiers: [.command, .shift])
            Button("LoRA 库") { openWindow(id: "loras") }.keyboardShortcut("l", modifiers: [.command, .shift])
            Button("蒙版画布") { openWindow(id: "mask") }.keyboardShortcut("k", modifiers: [.command, .shift])
            Divider()
            Button("重新校验模型") { model.validateModel() }
            Button("释放模型会话状态") { model.unloadModel() }.disabled(model.isRunning)
        }
    }
}
