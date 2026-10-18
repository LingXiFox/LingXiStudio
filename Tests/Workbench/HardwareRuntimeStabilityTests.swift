import Foundation
import MLX
import XCTest
@testable import LingXiStudio
@testable import QwenImageCore

final class HardwareRuntimeStabilityTests: XCTestCase {
    func testFourLoRAsAndReferenceCompleteConsecutiveJobs() throws {
        guard ProcessInfo.processInfo.environment["LINGXI_RUN_STABILITY_TESTS"] == "1" else {
            throw XCTSkip("Set LINGXI_RUN_STABILITY_TESTS=1 for full-size consecutive runtime acceptance")
        }
        XCTAssertTrue(Thread.isMainThread)
        MLXRuntime.configure()
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let work = root.appending(path: ".tmp/runtime-stability-20261018").appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        print("Runtime stability evidence: \(work.path)")
        let recorder = try Recorder(work.appending(path: "events.jsonl"))
        try Data("\(getpid())".utf8).write(to: work.appending(path: "pid"))
        let home = FileManager.default.homeDirectoryForCurrentUser
        let model = home.appending(path: "Models/Qwen-Image-2.1-BF16")
        let baseline = root.appending(path: ".tmp/lora-reference-20261018/reference-cache512.png")
        var request = try GenerationMetadata.read(sidecarOf: baseline).request
        let files = ["Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors",
                     "qwen2.1-detail-fix-2.0.safetensors", "canon_qwen21.safetensors", "samsung_qwen21.safetensors"]
        request.loras = files.map { LoRASpec(path: home.appending(path: "Models/Qwen-Image-2.1-LoRA/\($0)").path) }
        for lora in request.loras { XCTAssertTrue(try LoRAAdapter.inspect(path: lora.path).isCompatible) }
        let gate = Process()
        gate.executableURL = URL(fileURLWithPath: "/bin/zsh")
        gate.arguments = [root.appending(path: "Scripts/thermal-gate.sh").path]
        try gate.run()
        gate.waitUntilExit()
        guard gate.terminationStatus == 0 else { XCTFail("Initial benchmark calibration failed"); return }
        let done = expectation(description: "Three complete full-size reference generations")
        let engine = GenerationEngine()
        let timer = DispatchSource.makeTimerSource(queue: .main)
        var previous = ProcessInfo.processInfo.systemUptime
        timer.schedule(deadline: .now(), repeating: .milliseconds(50), leeway: .milliseconds(5))
        timer.setEventHandler {
            let now = ProcessInfo.processInfo.systemUptime
            recorder.record(["kind": "mainRunLoop", "delayMs": max(0, (now - previous - 0.05) * 1000)])
            previous = now
            // Only the independent resource monitor may request a safety stop; throughput never does.
            if FileManager.default.fileExists(atPath: work.appending(path: "resource-stop.json").path) { engine.cancel() }
        }
        timer.resume()
        defer { timer.cancel() }
        recorder.record(["kind": "baselineStart"])
        RunLoop.current.run(until: Date().addingTimeInterval(3))
        recorder.record(["kind": "generationStart"])
        engine.submit(.init(taskID: UUID(), modelRoot: model, request: request,
                            seeds: [42, 43, 42], outputDirectory: work)) { event in
            switch event {
            case .started(let index, let seed):
                recorder.record(["kind": "started", "index": index, "seed": seed])
            case .progress(let index, let p):
                recorder.record(["kind": "progress", "index": index, "stage": p.stage.rawValue,
                                 "fraction": p.fraction, "step": p.step, "elapsed": p.elapsed,
                                 "activeBytes": p.memory.activeBytes, "cacheBytes": p.memory.cacheBytes,
                                 "footprintBytes": ProcessMemory.footprint(), "cacheHit": p.conditioningCacheHit])
                if (p.stage == .prefill && p.fraction == 0) || (p.stage == .denoising && [2, 4, 6].contains(p.step)) {
                    let probe = autoreleasepool { Diagnostics.probeThermal() }
                    recorder.record(["kind": "throughput", "index": index, "step": p.step,
                                     "thermal": probe.thermalState, "tflops": probe.tflops])
                }
            case .saved(let index, let url, let seconds, let steps):
                recorder.record(["kind": "saved", "index": index, "path": url.path,
                                 "seconds": seconds, "stepSeconds": steps, "activeBytes": Memory.activeMemory])
            case .finished(let result):
                let error: String
                switch result { case .success: error = ""; case .failure(let failure): error = String(describing: failure) }
                recorder.record(["kind": "finished", "error": error,
                                 "activeBytes": Memory.activeMemory, "cacheBytes": Memory.cacheMemory])
                done.fulfill()
            }
        }
        wait(for: [done], timeout: 1800)
        recorder.record(["kind": "recoveryStart"])
        RunLoop.current.run(until: Date().addingTimeInterval(5))
        let rows = recorder.rows
        XCTAssertFalse(FileManager.default.fileExists(atPath: work.appending(path: "resource-stop.json").path))
        XCTAssertEqual(rows.filter { $0["kind"] as? String == "finished" }.last?["error"] as? String, "")
        let saved = rows.filter { $0["kind"] as? String == "saved" }
        XCTAssertEqual(saved.count, 3)
        for row in saved {
            XCTAssertEqual((row["stepSeconds"] as? [Double])?.count, 6)
            XCTAssertLessThan(row["activeBytes"] as? Int ?? Int.max, 1 << 20)
        }
        for row in rows where row["stage"] as? String == "loadingVAE" && row["fraction"] as? Double == 0 {
            XCTAssertLessThan(row["activeBytes"] as? Int ?? Int.max, 1 << 30)
        }
        XCTAssertGreaterThan(rows.filter { $0["kind"] as? String == "mainRunLoop" }.count, 100)
        if saved.count == 3, let first = saved[0]["path"] as? String, let last = saved[2]["path"] as? String {
            XCTAssertEqual(try RGBAImage(contentsOf: URL(fileURLWithPath: first)).pixels,
                           try RGBAImage(contentsOf: URL(fileURLWithPath: last)).pixels)
            for (index, row) in saved.enumerated() {
                let metadata = try GenerationMetadata.read(sidecarOf: URL(fileURLWithPath: row["path"] as! String))
                XCTAssertEqual(metadata.conditioningCacheHit, index != 0)
            }
        }
    }

    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private let file: FileHandle
        private var events: [[String: Any]] = []
        init(_ url: URL) throws {
            FileManager.default.createFile(atPath: url.path, contents: nil)
            file = try FileHandle(forWritingTo: url)
        }
        func record(_ row: [String: Any]) {
            lock.lock()
            defer { lock.unlock() }
            let row = row.merging(["uptime": ProcessInfo.processInfo.systemUptime]) { first, _ in first }
            events.append(row)
            do { try file.write(contentsOf: JSONSerialization.data(withJSONObject: row) + Data([10])) }
            catch { XCTFail("Could not preserve stability event: \(error)") }
        }
        var rows: [[String: Any]] { lock.lock(); defer { lock.unlock() }; return events }
        deinit { try? file.close() }
    }
}
