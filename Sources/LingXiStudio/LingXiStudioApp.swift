import AppKit
import QwenImageCore
import SwiftUI

@main
struct LingXiStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var model: AppModel

    init() {
        // MLX reads its command buffer settings once, when the first array touches the GPU.
        MLXRuntime.configure()
        _model = State(initialValue: AppModel())
    }

    var body: some Scene {
        Window("LingXiStudio", id: "main") {
            WorkbenchView()
                .environment(model)
                .frame(minWidth: 1100, minHeight: 720)
        }
        .defaultSize(width: 1500, height: 940)
        .commands { WorkbenchCommands(model: model) }

        WindowGroup("图像查看", id: "viewer", for: URL.self) { $url in
            if let url {
                ImageViewerWindow(url: url).environment(model)
            }
        }
        .defaultSize(width: 1100, height: 1000)

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
        }
        CommandMenu("模型") {
            Button("重新校验模型") { model.validateModel() }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            Button("释放模型状态") { model.unloadModel() }
                .disabled(model.isRunning)
        }
    }
}
