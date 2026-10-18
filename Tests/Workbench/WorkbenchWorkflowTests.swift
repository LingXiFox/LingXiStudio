import Foundation
import XCTest
@testable import QwenImageCore
@testable import LingXiStudio

final class WorkbenchWorkflowTests: XCTestCase {
    @MainActor
    func testIsolatedProfileStartsWithoutHomeModelAndRestoresItsOwnPreferences() throws {
        let root = URL(fileURLWithPath: "/tmp").appending(path: "lingxi-profile-\(UUID().uuidString)")
        let old = ProcessInfo.processInfo.environment["LINGXI_PROFILE_DIRECTORY"]
        setenv("LINGXI_PROFILE_DIRECTORY", root.path, 1)
        let domain = try XCTUnwrap(AppPaths.preferencesDomain)
        defer {
            AppPaths.defaults.removePersistentDomain(forName: domain)
            if let old { setenv("LINGXI_PROFILE_DIRECTORY", old, 1) } else { unsetenv("LINGXI_PROFILE_DIRECTORY") }
            try? FileManager.default.removeItem(at: root)
        }
        XCTAssertEqual(AppPaths.support.path, root.path)
        let first = AppModel()
        XCTAssertNil(first.modelRoot)
        XCTAssertTrue(first.loras.isEmpty)
        XCTAssertTrue(first.tasks.isEmpty)
        XCTAssertEqual(first.outputDirectory, root.appending(path: "outputs"))
        first.parameters.prompt = "An isolated profile persistence check"
        first.promptPresets = [PromptPreset(name: "Profile only", text: "A small fox")]
        let second = AppModel()
        XCTAssertEqual(second.parameters.prompt, first.parameters.prompt)
        XCTAssertEqual(second.promptPresets, first.promptPresets)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appending(path: "prompt-presets.json").path))
    }

    @MainActor
    private func withProfile(_ body: (AppModel, URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "lingxi-workflow-\(UUID())")
        let old = ProcessInfo.processInfo.environment["LINGXI_PROFILE_DIRECTORY"]
        setenv("LINGXI_PROFILE_DIRECTORY", root.path, 1)
        let domain = try XCTUnwrap(AppPaths.preferencesDomain)
        defer {
            AppPaths.defaults.removePersistentDomain(forName: domain)
            if let old { setenv("LINGXI_PROFILE_DIRECTORY", old, 1) } else { unsetenv("LINGXI_PROFILE_DIRECTORY") }
            try? FileManager.default.removeItem(at: root)
        }
        try body(AppModel(), root)
    }

    @MainActor
    func testHistoryRestoresModelClearsStaleInputsAndReadsEmbeddedRecord() throws {
        try withProfile { model, root in
            let modelFolder = root.appending(path: "recorded-model")
            try FileManager.default.createDirectory(at: modelFolder, withIntermediateDirectories: true)
            let request = GenerationRequest(prompt: "A fox", negativePrompt: "blur", width: 32, height: 32, seed: 42)
            let image = GeneratedImage(width: 32, height: 32, rgba: .init(repeating: 255, count: 4096), request: request,
                timings: [], peakMemoryBytes: 0, totalSeconds: 1, stepSeconds: [0.2])
            let url = root.appending(path: "result.png")
            try ImageWriter.write(image, model: "recorded-model", modelPath: modelFolder.path, to: url)
            try FileManager.default.removeItem(at: url.deletingPathExtension().appendingPathExtension("json"))
            let artwork = try XCTUnwrap(GalleryScanner.scan(root, origin: .native).first)
            XCTAssertEqual(artwork.metadata?.modelPath, modelFolder.path)
            model.initialImage = try InputStore.importFile(url)
            model.conditions = [try InputStore.importFile(url)]
            model.mask.prepare(width: 32, height: 32, imageID: "old")
            model.reuse(artwork)
            XCTAssertEqual(model.modelRoot?.path, modelFolder.path)
            XCTAssertEqual(model.parameters.negativePrompt, "blur")
            XCTAssertEqual(model.parameters.seed, 42)
            XCTAssertNil(model.initialImage)
            XCTAssertTrue(model.conditions.isEmpty)
            XCTAssertTrue(model.mask.isEmpty)
            XCTAssertTrue(model.loras.isEmpty)
        }
    }

    @MainActor
    func testMissingInputIsReportedAndMovedInputRequiresExactHash() throws {
        try withProfile { model, root in
            let source = try InputStore.importRGBA(RGBAImage(width: 2, height: 2, pixels: .init(repeating: 255, count: 16)), name: "test")
            let record = source.record(role: .condition)
            let moved = root.appending(path: "moved")
            let inputFolder = moved.appending(path: "inputs")
            try FileManager.default.createDirectory(at: inputFolder, withIntermediateDirectories: true)
            let copy = inputFolder.appending(path: "\(record.sha256).png")
            try FileManager.default.copyItem(at: source.url, to: copy)
            try FileManager.default.removeItem(at: source.url)
            let imageURL = moved.appending(path: "result.png")
            XCTAssertEqual(try InputStore.restore(record, nextTo: imageURL).url, copy)
            try Data("changed".utf8).write(to: copy)
            XCTAssertThrowsError(try InputStore.restore(record, nextTo: imageURL))
            let request = GenerationRequest(prompt: "A fox", width: 32, height: 32, seed: 42, mode: .reference, inputs: [record])
            let image = GeneratedImage(width: 32, height: 32, rgba: .init(repeating: 255, count: 4096), request: request,
                timings: [], peakMemoryBytes: 0, totalSeconds: 1, stepSeconds: [])
            try ImageWriter.write(image, model: "missing", to: imageURL)
            let artwork = try XCTUnwrap(GalleryScanner.scan(moved, origin: .native).first)
            model.reuse(artwork)
            XCTAssertTrue(model.conditions.isEmpty)
            XCTAssertTrue(model.actionError?.contains("内容已改变") == true)
            let before = model.initialImage
            model.setInitial(url: copy)
            XCTAssertEqual(model.initialImage, before)
            XCTAssertTrue(model.actionError?.contains("无法导入") == true)
        }
    }

    func testLoRAMigrationAndInvalidEntryCanBeDisabledWithoutStaleMetadata() throws {
        var item = try JSONDecoder().decode(LoRAItem.self, from: Data(#"{"path":"/tmp/missing-turbo.safetensors","scale":0.65}"#.utf8))
        XCTAssertEqual(item.purpose, .acceleration)
        XCTAssertEqual(item.scale, 0.65)
        item.rank = 256
        item.targetCount = 227
        item.precision = "bfloat16"
        item.refresh()
        XCTAssertEqual(item.rank, 0)
        XCTAssertEqual(item.precision, "—")
        XCTAssertTrue(item.canToggle)
        item.enabled = false
        XCTAssertFalse(item.canToggle)
    }
    func testQueuedOutputPathAndLegacyTaskDecoding() throws {
        let task = TaskRecord(mode: .reference, modelName: "test", modelPath: "/model", parameters: GenerationParameters(),
            loras: [LoRASpec(path: "/b", scale: 0.65), LoRASpec(path: "/a", scale: -0.2, enabled: false)],
            conditions: [], initial: nil, mask: nil, outputPath: "/queued-output")
        let data = try JSONEncoder().encode(task)
        let decoded = try JSONDecoder().decode(TaskRecord.self, from: data)
        XCTAssertEqual(decoded.outputPath, "/queued-output")
        XCTAssertEqual(decoded.loras, task.loras)
        var old = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        old.removeValue(forKey: "outputPath")
        let legacy = try JSONDecoder().decode(TaskRecord.self, from: JSONSerialization.data(withJSONObject: old))
        XCTAssertNil(legacy.outputPath)
    }

    @MainActor
    func testLoRAPresetKeepsExactStrengthOrderAndDisabledState() throws {
        try withProfile { model, _ in
            model.loras = [LoRAItem(path: "/old")]
            let specs = [LoRASpec(path: "/second", scale: 0.65), LoRASpec(path: "/first", scale: -0.2, enabled: false)]
            model.applyLoRAPreset(LoRAPreset(name: "Exact order", entries: specs))
            XCTAssertEqual(Array(model.loras.prefix(2)).map(\.spec), specs)
            XCTAssertEqual(model.loras.last?.path, "/old")
            XCTAssertEqual(model.loras.last?.enabled, false)
            model.parameters.width = 0
            XCTAssertTrue(model.blockingIssues.contains("宽高必须大于零"))
            model.parameters.width = Int.max
            XCTAssertEqual(model.estimatedPeakBytes, Int64.max)
            model.loras[0].scale = .infinity
            XCTAssertTrue(model.blockingIssues.contains("Guidance 和 LoRA 强度必须是有限数值"))
        }
    }

    func testSchedulingWatermarkLeavesHeadroomAndNeverRaisesAnExistingLimit() {
        XCTAssertEqual(GenerationEngine.schedulingMemoryLimit(previous: 31 << 30, physical: 32 << 30), 20 << 30)
        XCTAssertEqual(GenerationEngine.schedulingMemoryLimit(previous: 18 << 30, physical: 32 << 30), 18 << 30)
        XCTAssertEqual(GenerationEngine.schedulingMemoryLimit(previous: 31 << 30, physical: 64 << 30), 31 << 30)
    }

}
