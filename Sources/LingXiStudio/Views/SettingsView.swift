import SwiftUI

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @State private var choosingModel = false
    @State private var choosingOutput = false

    var body: some View {
        Form {
            Section("路径") {
                LabeledContent("模型文件夹") {
                    HStack {
                        Text(model.modelRoot?.path ?? "未选择").lineLimit(1).truncationMode(.middle)
                        Button("选择…") { choosingModel = true }
                    }
                }
                LabeledContent("输出目录") {
                    HStack {
                        Text(model.outputDirectory.path).lineLimit(1).truncationMode(.middle)
                        Button("选择…") { choosingOutput = true }
                    }
                }
            }
            Section("运行环境") {
                LabeledContent("推理", value: "原生 Swift + MLX，完全离线")
                LabeledContent("Metal 命令缓冲", value: "100 op / 200 MB（MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER）")
                LabeledContent("统一内存", value: Format.bytes(model.metrics.physicalMemory))
            }
        }
        .formStyle(.grouped)
        .frame(width: 560)
        .fileImporter(isPresented: $choosingModel, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { model.modelRoot = url }
        }
        .fileImporter(isPresented: $choosingOutput, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { model.outputDirectory = url }
        }
    }
}
