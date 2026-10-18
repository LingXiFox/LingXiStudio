import QwenImageCore
import SwiftUI

/// Model library and loader: current model, integrity, components and their live residency.
struct ModelSection: View {
    @Environment(AppModel.self) private var model
    @State private var choosing = false
    @State private var expanded = true

    var body: some View {
        GroupBox {
            DisclosureGroup(isExpanded: $expanded) {
                VStack(alignment: .leading, spacing: 10) {
                    if let root = model.modelRoot {
                        Text(root.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                            .lineLimit(1).truncationMode(.middle)
                    }
                    if case .error(let message) = model.modelStatus {
                        Text(message).font(.callout).foregroundStyle(.red).textSelection(.enabled)
                    }
                    if !model.components.isEmpty { componentGrid }
                    HStack {
                        Button("选择模型文件夹…", systemImage: "folder") { choosing = true }
                        Button("重新校验", systemImage: "checkmark.shield") { model.validateModel() }
                            .disabled(model.modelRoot == nil || model.modelStatus == .validating)
                        Button("释放", systemImage: "eject") { model.unloadModel() }
                            .disabled(model.isRunning || model.modelRoot == nil)
                            .help("清除分词器等会话状态；权重本就不在任务之间驻留")
                        Spacer()
                    }
                    .controlSize(.small)
                    Text("32 GB 平衡档：每个任务按 文本编码器 → Transformer → VAE 顺序加载，同一时刻只驻留一个大组件，用完即释放。")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, 6)
            } label: {
                header
            }
        }
        .fileImporter(isPresented: $choosing, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { model.modelRoot = url }
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Label("模型", systemImage: "cube.box").font(.headline)
            Text(model.modelName).lineLimit(1)
            statusBadge
            Spacer()
            if model.modelBytes > 0 {
                Text("BF16 · Safetensors · \(Format.bytes(model.modelBytes))").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var statusBadge: some View {
        let color: Color =
            switch model.modelStatus {
            case .ready: .green
            case .inUse: .blue
            case .validating: .orange
            case .error: .red
            case .unselected: .secondary
            }
        return HStack(spacing: 4) {
            if model.modelStatus == .validating { ProgressView().controlSize(.mini) }
            Circle().fill(color).frame(width: 7, height: 7)
            Text(model.modelStatus.label).font(.caption)
        }
        .padding(.horizontal, 6).padding(.vertical, 2)
        .background(color.opacity(0.12), in: Capsule())
    }

    private var componentGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
            GridRow {
                Text("组件").foregroundStyle(.secondary)
                Text("文件").foregroundStyle(.secondary)
                Text("大小").foregroundStyle(.secondary)
                Text("校验").foregroundStyle(.secondary)
                Text("驻留").foregroundStyle(.secondary)
            }
            .font(.caption)
            ForEach(model.components, id: \.name) { component in
                GridRow {
                    Text(Self.title(component.name))
                    Text(component.files > 0 ? "\(component.files)" : "–").monospacedDigit()
                    Text(component.bytes > 0 ? Format.bytes(component.bytes) : "–").monospacedDigit()
                    Image(systemName: component.problems.isEmpty ? "checkmark.circle.fill" : "xmark.octagon.fill")
                        .foregroundStyle(component.problems.isEmpty ? .green : .red)
                    residencyLabel(component.name)
                }
            }
        }
        .font(.callout)
    }

    @ViewBuilder
    private func residencyLabel(_ name: String) -> some View {
        if component(name, carriesWeights: true) {
            let state = model.residency[name] ?? .notLoaded
            HStack(spacing: 4) {
                if state == .loading { ProgressView().controlSize(.mini) }
                Text(state.label).foregroundStyle(state == .resident ? .blue : .secondary)
            }
        } else {
            Text("配置").foregroundStyle(.secondary)
        }
    }

    private func component(_ name: String, carriesWeights: Bool) -> Bool {
        ["text_encoder", "transformer", "vae"].contains(name)
    }

    static func title(_ name: String) -> String {
        switch name {
        case "text_encoder": "文本编码器"
        case "transformer": "Transformer"
        case "vae": "VAE"
        case "processor": "分词器"
        case "scheduler": "调度配置"
        default: name
        }
    }
}
