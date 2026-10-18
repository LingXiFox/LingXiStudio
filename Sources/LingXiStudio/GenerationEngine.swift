import Foundation
import QwenImageCore

/// Runs generation jobs one at a time on a dedicated thread.
///
/// All MLX work happens on this one thread, so the GPU stream and compiled graphs are never shared across threads.
/// The thread gets a large stack because tracing the compiled 32-block step recurses deeply.
final class GenerationEngine: @unchecked Sendable {
    struct Job: Sendable {
        var taskID: UUID
        var modelRoot: URL
        var request: GenerationRequest
        var seeds: [UInt64]
        var outputDirectory: URL
    }

    enum Event: Sendable {
        case started(imageIndex: Int, seed: UInt64)
        case progress(imageIndex: Int, GenerationProgress)
        case saved(imageIndex: Int, url: URL, seconds: Double, stepSeconds: [Double])
        case finished(Result<Void, Error>)
    }

    private let condition = NSCondition()
    private var pending: [(Job, @Sendable (Event) -> Void)] = []
    private var cancelled = false
    private var pipelines: [URL: QwenImagePipeline] = [:]
    private var thread: Thread?

    init() {
        let thread = Thread { [unowned self] in self.loop() }
        thread.stackSize = 64 << 20
        thread.name = "LingXiStudio.generation"
        thread.qualityOfService = .userInitiated
        self.thread = thread
        thread.start()
    }

    func submit(_ job: Job, events: @escaping @Sendable (Event) -> Void) {
        condition.lock()
        cancelled = false
        pending.append((job, events))
        condition.signal()
        condition.unlock()
    }

    /// Stops the running job at the next stage or step boundary.
    func cancel() {
        condition.lock()
        cancelled = true
        condition.unlock()
    }

    /// Drops the cached tokenizer/scheduler state of a model (weights are never kept between tasks).
    func forget(model root: URL) {
        condition.lock()
        pipelines[root] = nil
        condition.unlock()
    }

    private var isCancelled: Bool {
        condition.lock()
        defer { condition.unlock() }
        return cancelled
    }

    private func loop() {
        while true {
            condition.lock()
            while pending.isEmpty { condition.wait() }
            let (job, events) = pending.removeFirst()
            condition.unlock()
            events(.finished(Result { try run(job, events) }))
        }
    }

    private func run(_ job: Job, _ events: @escaping @Sendable (Event) -> Void) throws {
        let model = ModelDirectory(root: job.modelRoot)
        _ = try model.validate()
        condition.lock()
        let cached = pipelines[job.modelRoot]
        condition.unlock()
        let pipeline = try cached ?? QwenImagePipeline(model: model)
        condition.lock()
        pipelines[job.modelRoot] = pipeline
        condition.unlock()
        try FileManager.default.createDirectory(at: job.outputDirectory, withIntermediateDirectories: true)
        for (index, seed) in job.seeds.enumerated() {
            if isCancelled { throw PipelineError.cancelled }
            var request = job.request
            request.seed = seed
            events(.started(imageIndex: index, seed: seed))
            // This thread has no run loop to drain autoreleased Metal objects (command buffers and the buffers they
            // reference), so each image gets its own pool; otherwise a finished run keeps its memory.
            try autoreleasepool {
                let image = try pipeline.generate(
                    request, progress: { events(.progress(imageIndex: index, $0)) }, isCancelled: { self.isCancelled })
                let url = Self.outputURL(in: job.outputDirectory, seed: seed)
                try ImageWriter.write(image, model: job.modelRoot.lastPathComponent, to: url)
                events(.saved(imageIndex: index, url: url, seconds: image.totalSeconds, stepSeconds: image.stepSeconds))
            }
        }
    }

    private static func outputURL(in directory: URL, seed: UInt64) -> URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let base = "\(formatter.string(from: Date()))-\(seed)"
        var url = directory.appending(path: base + ".png")
        var n = 2
        // Never overwrite an existing result.
        while FileManager.default.fileExists(atPath: url.path) {
            url = directory.appending(path: "\(base)-\(n).png")
            n += 1
        }
        return url
    }
}
