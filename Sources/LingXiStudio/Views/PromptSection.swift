import SwiftUI

/// Positive and negative prompts. The negative prompt only takes part when true CFG is on (guidance > 1).
struct PromptSection: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                SectionHeader(title: "提示词", symbol: "text.cursor") {
                    Menu("历史", systemImage: "clock.arrow.circlepath") {
                        ForEach(recentPrompts, id: \.self) { prompt in
                            Button(String(prompt.prefix(80))) { model.parameters.prompt = prompt }
                        }
                        if recentPrompts.isEmpty { Text("暂无") }
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                }
                editor($model.parameters.prompt, placeholder: "描述想要生成的画面（中文或英文均可）", minHeight: 96)
                Text("\(model.parameters.prompt.count) 字").font(.caption).foregroundStyle(.secondary)

                HStack {
                    Text("负面提示词").font(.subheadline)
                    Spacer()
                    if model.parameters.guidance <= 1 {
                        Text("Guidance ≤ 1 时不计算负面分支，此处内容不会生效")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("真 CFG：每步额外计算一次负面分支，耗时约翻倍").font(.caption).foregroundStyle(.orange)
                    }
                }
                editor($model.parameters.negativePrompt, placeholder: "可选", minHeight: 44)
                    .disabled(model.parameters.guidance <= 1)
                    .opacity(model.parameters.guidance <= 1 ? 0.5 : 1)
            }
        }
    }

    private var recentPrompts: [String] {
        var seen = Set<String>()
        return model.tasks.map(\.parameters.prompt).filter { !$0.isEmpty && seen.insert($0).inserted }.prefix(20).map { $0 }
    }

    private func editor(_ text: Binding<String>, placeholder: String, minHeight: CGFloat) -> some View {
        TextEditor(text: text)
            .font(.body)
            .scrollContentBackground(.hidden)
            .padding(6)
            .frame(minHeight: minHeight, maxHeight: minHeight * 2.5)
            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor)))
            .overlay(alignment: .topLeading) {
                if text.wrappedValue.isEmpty {
                    Text(placeholder).foregroundStyle(.tertiary).padding(.horizontal, 11).padding(.vertical, 6)
                        .allowsHitTesting(false)
                }
            }
    }
}
