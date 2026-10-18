import SwiftUI

/// Main window: parameters on the left (2/3), image workspace top right and management bottom right (1:1).
struct WorkbenchView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        ProportionalSplit(
            axis: .horizontal, fraction: $model.leftFraction, defaultFraction: 2.0 / 3.0, minFirst: 560,
            minSecond: 340
        ) {
            ParameterPanel()
        } second: {
            ProportionalSplit(
                axis: .vertical, fraction: $model.rightTopFraction, defaultFraction: 0.5, minFirst: 220,
                minSecond: 240
            ) {
                ImageWorkspace()
            } second: {
                ManagementPanel()
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Label(model.modelName, systemImage: "cube")
                    .labelStyle(.titleAndIcon)
                    .foregroundStyle(.secondary)
            }
            ToolbarItem(placement: .primaryAction) {
                Button("恢复默认布局", systemImage: "rectangle.split.2x1") {
                    model.leftFraction = 2.0 / 3.0
                    model.rightTopFraction = 0.5
                }
                .help("左 2/3 · 右 1/3，右栏上下 1:1")
            }
        }
    }
}

/// Left 2/3: model, prompt, parameters and performance, with the run controls pinned at the bottom.
struct ParameterPanel: View {
    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    ModelSection()
                    PromptSection()
                    ParametersSection()
                    PerformanceSection()
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider()
            RunBar()
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}
