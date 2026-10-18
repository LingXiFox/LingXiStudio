import SwiftUI

/// Memory profile and live system metrics. Values the system does not expose are shown as unavailable.
struct PerformanceSection: View {
    @Environment(AppModel.self) private var model
    @State private var expanded = true

    var body: some View {
        GroupBox {
            DisclosureGroup(isExpanded: $expanded) {
                VStack(alignment: .leading, spacing: 10) {
                    profiles
                    Divider()
                    metrics
                }
                .padding(.top, 6)
            } label: {
                SectionHeader(title: "性能与内存", symbol: "memorychip") {
                    if let pressure = model.metrics.pressure {
                        Text("内存压力 \(pressure.label)").font(.caption)
                            .foregroundStyle(pressure == .normal ? Color.secondary : pressure == .warning ? .orange : .red)
                    }
                }
            }
        }
    }

    private var profiles: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                profile("32 GB 平衡", available: true)
                profile("高画质优先", available: false)
                profile("省内存", available: false)
                profile("高速实验", available: false)
            }
            Text("平衡档：BF16 原始权重，组件分阶段驻留，峰值约 18 GB。其余档位需要量化或缓存方案，经验证后才开放。")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func profile(_ title: String, available: Bool) -> some View {
        Text(title)
            .font(.callout)
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(available ? Color.accentColor.opacity(0.18) : Color.clear, in: Capsule())
            .overlay(Capsule().stroke(available ? Color.accentColor : Color(nsColor: .separatorColor)))
            .foregroundStyle(available ? .primary : .tertiary)
            .help(available ? "当前档位" : "尚未验证，暂不可用")
    }

    private var metrics: some View {
        let m = model.metrics
        let live = model.live
        return HStack(alignment: .top, spacing: 32) {
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                InfoRow(label: "统一内存", value: Format.bytes(m.physicalMemory))
                InfoRow(label: "应用占用", value: Format.bytes(m.appFootprint))
                InfoRow(
                    label: "MLX 分配",
                    value: live.map { "\(Format.bytes($0.mlxActive))（峰值 \(Format.bytes($0.mlxPeak))）" } ?? "空闲")
                InfoRow(
                    label: "Swap",
                    value: m.swapUsed.map { used in "\(Format.bytes(used)) / \(Format.bytes(m.swapTotal ?? 0))" } ?? "不可得")
                InfoRow(
                    label: "内存压力",
                    value: m.pressure.map { p in "\(p.label)\(m.freePercent.map { "（可用 \($0)%）" } ?? "")" } ?? "不可得",
                    tint: m.pressure == .critical ? .red : m.pressure == .warning ? .orange : nil)
                InfoRow(label: "热状态", value: m.thermal.label, tint: m.thermal.rawValue >= 2 ? .orange : nil)
            }
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                InfoRow(label: "阶段", value: live?.stageLabel ?? "空闲")
                InfoRow(
                    label: "进度",
                    value: live.map { l in
                        l.stage == .denoising
                            ? "第 \(l.imageIndex + 1)/\(l.imageCount) 张 · 步 \(l.step)/\(l.totalSteps)"
                            : "第 \(l.imageIndex + 1)/\(l.imageCount) 张"
                    } ?? "–")
                InfoRow(label: "速度", value: live?.secondsPerStep.map { String(format: "%.2f s/step", $0) } ?? "–")
                InfoRow(label: "本张耗时", value: live.map { Format.duration($0.elapsed) } ?? "–")
                InfoRow(label: "预计剩余", value: live?.remaining.map { Format.duration($0) } ?? "–")
                InfoRow(label: "队列", value: "等待 \(model.waitingCount) · 失败 \(model.failedCount)")
            }
        }
        .font(.callout)
    }
}

/// Generate / stop controls pinned under the parameter panel, with the live stage and blocking problems.
struct RunBar: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let live = model.live {
                HStack {
                    Text(live.stageLabel).font(.callout.weight(.medium))
                    if live.stage == .denoising {
                        Text("步 \(live.step)/\(live.totalSteps)").monospacedDigit()
                    }
                    if live.imageCount > 1 { Text("· 第 \(live.imageIndex + 1)/\(live.imageCount) 张") }
                    Spacer()
                    if let remaining = live.remaining { Text("剩余约 \(Format.duration(remaining))").foregroundStyle(.secondary) }
                }
                .font(.callout)
                ProgressView(value: overallFraction(live))
            }
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(model.blockingIssues, id: \.self) { issue in
                        Label(issue, systemImage: "exclamationmark.circle").foregroundStyle(.red)
                    }
                    ForEach(model.warnings, id: \.self) { warning in
                        Label(warning, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                    }
                    if model.blockingIssues.isEmpty && model.warnings.isEmpty {
                        Text(summary).foregroundStyle(.secondary)
                    }
                }
                .font(.caption)
                Spacer()
                Button(role: .cancel) {
                    model.stopCurrent()
                } label: {
                    Label("停止", systemImage: "stop.fill")
                }
                .disabled(!model.isRunning)
                .keyboardShortcut(".", modifiers: .command)
                Button {
                    model.enqueue()
                } label: {
                    Label(model.isRunning ? "加入队列" : "生成", systemImage: "sparkles")
                        .frame(minWidth: 90)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!model.blockingIssues.isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private var summary: String {
        let p = model.parameters
        let steps = p.preset == .turbo ? 6 : p.steps
        let loras = model.activeLoRAs.count
        return "\(p.width)×\(p.height) · \(steps) 步 · Guidance \(String(format: "%.1f", p.guidance)) · "
            + (p.seedMode == .random ? "随机种子" : "种子 \(p.seed)") + " · \(loras) 个 LoRA · ⌘↩ 生成"
    }

    /// Denoising dominates the run time, so the bar follows the step count and reserves a small share for the rest.
    private func overallFraction(_ live: LiveProgress) -> Double {
        let perImage: Double =
            switch live.stage {
            case .validating, .loadingTextEncoder, .encodingPrompt: 0.02
            case .loadingTransformer, .prefill: 0.05
            case .denoising: 0.05 + 0.85 * Double(live.step) / Double(max(live.totalSteps, 1))
            case .loadingVAE, .decoding: 0.92
            case .done: 1
            }
        return (Double(live.imageIndex) + perImage) / Double(max(live.imageCount, 1))
    }
}
