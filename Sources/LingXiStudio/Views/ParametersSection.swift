import QwenImageCore
import SwiftUI

/// Generation parameters. Only parameters the executor really honours are editable; constraints are explained inline.
struct ParametersSection: View {
    @Environment(AppModel.self) private var model
    @State private var advanced = false
    @State private var choosingOutput = false

    var body: some View {
        @Bindable var model = model
        let p = model.parameters
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                SectionHeader(title: "生成参数", symbol: "slider.horizontal.3")

                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 12) {
                    GridRow {
                        label("尺寸")
                        sizeControls
                    }
                    GridRow {
                        label("调度")
                        VStack(alignment: .leading, spacing: 4) {
                            Picker("调度", selection: $model.parameters.preset) {
                                Text(SchedulePreset.standard.title).tag(SchedulePreset.standard)
                                Text(SchedulePreset.turbo.title).tag(SchedulePreset.turbo)
                                    .disabled(!model.turboAvailable)
                            }
                            .labelsHidden()
                            .pickerStyle(.radioGroup)
                            .horizontalRadioGroupLayout()
                            if !model.turboAvailable {
                                note("Turbo 需要在右下 LoRA 列表中启用一个「加速」用途的 LoRA")
                            } else if p.preset == .turbo {
                                note("Turbo 使用固定 6 个时间节点，步数锁定为 6")
                            }
                        }
                    }
                    GridRow {
                        label("步数")
                        HStack {
                            if p.preset == .turbo {
                                Text("6").monospacedDigit().frame(width: 60, alignment: .leading)
                                Image(systemName: "lock.fill").foregroundStyle(.secondary)
                            } else {
                                TextField("步数", value: $model.parameters.steps, format: .number)
                                    .frame(width: 60)
                                Stepper("", value: $model.parameters.steps, in: 1...100).labelsHidden()
                            }
                            estimate
                        }
                    }
                    GridRow {
                        label("Guidance")
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Slider(value: $model.parameters.guidance, in: 1...10, step: 0.5).frame(maxWidth: 260)
                                Text(String(format: "%.1f", p.guidance)).monospacedDigit().frame(width: 36)
                            }
                            note(
                                p.guidance <= 1
                                    ? "1.0 = 不做引导（模型默认）"
                                    : p.usesTrueCFG ? "真 CFG \(String(format: "%.1f", p.guidance))：每步正负两次前向" : "需要填写负面提示词才会生效")
                        }
                    }
                    GridRow {
                        label("种子")
                        seedControls
                    }
                    GridRow {
                        label("张数")
                        HStack {
                            Stepper(value: $model.parameters.count, in: 1...16) {
                                Text("\(p.count)").monospacedDigit().frame(width: 28, alignment: .leading)
                            }
                            if p.count > 1 {
                                Picker("种子规则", selection: $model.parameters.batchSeeds) {
                                    Text("逐张 +1").tag(GenerationParameters.BatchSeeds.increment)
                                    Text("逐张随机").tag(GenerationParameters.BatchSeeds.random)
                                }
                                .labelsHidden().fixedSize()
                            }
                        }
                    }
                }

                DisclosureGroup("高级与输出", isExpanded: $advanced) {
                    Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 8) {
                        GridRow {
                            label("输出目录")
                            HStack {
                                Text(model.outputDirectory.path).lineLimit(1).truncationMode(.middle)
                                    .textSelection(.enabled)
                                Button("更改…") { choosingOutput = true }.controlSize(.small)
                            }
                        }
                        GridRow {
                            label("文件")
                            note("无损 PNG（参数写入 PNG 描述字段）+ 同名 JSON 记录；命名：日期-时间-种子，不覆盖已有文件")
                        }
                        GridRow {
                            label("计算精度")
                            note("Transformer / 文本编码器 BF16 原始权重；VAE 解码 FP32。量化档位尚未验证，不提供")
                        }
                        GridRow {
                            label("步骤缓存")
                            note("不可用：会改变计算结果，执行器未实现，默认也不应开启")
                        }
                        GridRow {
                            label("VAE 分块")
                            note("不需要：1024² 解码峰值约 12 GB，由解码缓存上限控制")
                        }
                    }
                    .padding(.top, 6)
                }
            }
        }
        .fileImporter(isPresented: $choosingOutput, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { model.outputDirectory = url }
        }
    }

    private var sizeControls: some View {
        @Bindable var model = model
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                ForEach(SizePreset.all) { preset in
                    let selected = model.parameters.width == preset.width && model.parameters.height == preset.height
                    Button(preset.label) {
                        model.parameters.width = preset.width
                        model.parameters.height = preset.height
                    }
                    .buttonStyle(.bordered)
                    .tint(selected ? .accentColor : nil)
                    .controlSize(.small)
                    .help("\(preset.width)×\(preset.height)")
                }
            }
            HStack {
                TextField("宽", value: $model.parameters.width, format: .number.grouping(.never)).frame(width: 64)
                Stepper("", value: $model.parameters.width, in: 256...2048, step: 32).labelsHidden()
                Text("×")
                TextField("高", value: $model.parameters.height, format: .number.grouping(.never)).frame(width: 64)
                Stepper("", value: $model.parameters.height, in: 256...2048, step: 32).labelsHidden()
                Button("横竖互换", systemImage: "arrow.left.arrow.right") {
                    let w = model.parameters.width
                    model.parameters.width = model.parameters.height
                    model.parameters.height = w
                }
                .labelStyle(.iconOnly)
                Text("\(model.parameters.imageTokens) 图像 token").font(.caption).foregroundStyle(.secondary)
            }
            if model.parameters.width % 32 != 0 || model.parameters.height % 32 != 0 {
                Text("宽高必须是 32 的倍数").font(.caption).foregroundStyle(.red)
            }
        }
    }

    private var seedControls: some View {
        @Bindable var model = model
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Picker("种子", selection: $model.parameters.seedMode) {
                    Text("随机").tag(GenerationParameters.SeedMode.random)
                    Text("固定").tag(GenerationParameters.SeedMode.fixed)
                }
                .labelsHidden().pickerStyle(.segmented).fixedSize()
                if model.parameters.seedMode == .fixed {
                    TextField("种子", value: $model.parameters.seed, format: .number.grouping(.never)).frame(width: 130)
                    Button("随机取一个", systemImage: "dice") {
                        model.parameters.seed = UInt64.random(in: 0..<UInt64(UInt32.max))
                    }
                    .labelStyle(.iconOnly)
                }
            }
            if let last = model.lastSeed {
                HStack(spacing: 6) {
                    Text("上一张实际种子：\(String(last))").font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    Button("固定使用") {
                        model.parameters.seedMode = .fixed
                        model.parameters.seed = last
                    }
                    .controlSize(.mini)
                    Button("复制") { ImageFileActions.copyText(String(last)) }.controlSize(.mini)
                }
            }
        }
    }

    @ViewBuilder
    private var estimate: some View {
        let p = model.parameters
        let steps = p.preset == .turbo ? FlowMatchSchedule.viggleTurboNodes.count : p.steps
        if let rate = model.estimatedSecondsPerStep {
            Text("预计去噪 \(Format.duration(rate * Double(steps) * Double(p.count)))（\(String(format: "%.2f", rate)) s/step，按本机实测）")
                .font(.caption).foregroundStyle(.secondary)
        } else {
            Text("预计耗时：此尺寸与配置首次运行后可估算").font(.caption).foregroundStyle(.secondary)
        }
    }

    private func label(_ text: String) -> some View {
        Text(text).foregroundStyle(.secondary).frame(width: 64, alignment: .leading)
    }

    private func note(_ text: String) -> some View {
        Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }
}
