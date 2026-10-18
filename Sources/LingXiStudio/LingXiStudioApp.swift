import AppKit
import QwenImageCore
import SwiftUI

@main
struct LingXiStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var model: AppModel

    init() {
        // MLX reads its runtime settings once, when the first array touches the GPU.
        RuntimeOverrides.apply()
        MLXRuntime.configure()
        _model = State(initialValue: AppModel())
    }

    var body: some Scene {
        Window("LingXiStudio", id: "main") {
            WorkbenchView()
                .environment(model)
                .frame(minWidth: 980, minHeight: 640)
                .onAppear { model.applyAppearance() }
        }
        .defaultSize(width: 1560, height: 980)
        .commands { WorkbenchCommands(model: model) }

        WindowGroup("图像查看", id: "viewer", for: URL.self) { $url in
            if let url { ImageViewerWindow(url: url).environment(model).brandWindow(Brand.canvas) }
        }
        .defaultSize(width: 1100, height: 1000)

        WindowGroup("对比", id: "compare", for: CompareRequest.self) { $request in
            if let request { CompareWindow(request: request).brandWindow(Brand.canvas) }
        }
        .defaultSize(width: 1300, height: 900)

        Window("图库", id: "gallery") {
            GalleryWindow().environment(model).brandWindow()
        }
        .defaultSize(width: 1280, height: 860)

        Window("蒙版画布", id: "mask") {
            MaskEditorWindow().environment(model).brandWindow(Brand.canvas)
        }
        .defaultSize(width: 1200, height: 1000)

        Window("模型库", id: "models") {
            ModelLibraryWindow().environment(model).brandWindow()
        }
        .defaultSize(width: 860, height: 620)

        Window("LoRA 库", id: "loras") {
            LoRALibraryWindow().environment(model).brandWindow()
        }
        .defaultSize(width: 900, height: 680)

        Window("提示词预设", id: "prompt-presets") {
            PromptLibraryWindow(source: .presets).environment(model)
        }
        .defaultSize(width: 760, height: 500)

        Window("提示词历史", id: "prompt-history") {
            PromptLibraryWindow(source: .history).environment(model)
        }
        .defaultSize(width: 760, height: 500)

        Window("Runtime Inspector", id: "inspector") {
            RuntimeInspector().environment(model).brandWindow()
        }
        .defaultSize(width: 520, height: 760)

        Settings {
            SettingsView().environment(model)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var focusMonitor: Any?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Needed when launched as a bare SwiftPM executable rather than from the .app bundle.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
        focusMonitor = NSEvent.addLocalMonitorForEvents(matching: .leftMouseDown) { event in
            MainActor.assumeIsolated {
                if let window = event.window { Self.dismissTextFocus(in: window, at: event.locationInWindow) }
            }
            return event
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let focusMonitor { NSEvent.removeMonitor(focusMonitor) }
        focusMonitor = nil
    }

    /// Resigning before dispatch commits numeric fields while letting the clicked control receive its normal event.
    @MainActor
    static func dismissTextFocus(in window: NSWindow, at point: NSPoint) {
        guard let editor = window.firstResponder as? NSTextView else { return }
        let input: NSView = editor.isFieldEditor
            ? (editor.delegate as? NSView ?? editor)
            : (editor.enclosingScrollView ?? editor)
        if !input.convert(input.bounds, to: nil).contains(point) { window.makeFirstResponder(nil) }
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
            Button(model.cancellationRequested ? "正在停止…" : "停止当前任务") { model.stopCurrent() }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(!model.isRunning || model.cancellationRequested)
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
            Button("Runtime Inspector") { openWindow(id: "inspector") }.keyboardShortcut("i", modifiers: [.command, .option])
            Divider()
            Button("重新校验模型") { model.validateModel() }
            Button("释放模型会话状态") { model.unloadModel() }.disabled(model.isRunning)
        }
    }
}
