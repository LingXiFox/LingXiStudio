import Darwin
import Foundation

/// Process-level memory as the OS accounts it (`phys_footprint`, what Activity Monitor shows as Memory).
///
/// MLX's own counters only cover its active arrays and buffer cache; the footprint also includes Metal
/// allocations held elsewhere, file pages mapped dirty, and everything the app itself allocates.
public enum ProcessMemory {
    public static func footprint() -> Int {
        var info = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        return result == 0 ? Int(info.ri_phys_footprint) : 0
    }
}

/// Samples the process footprint on a background queue and keeps the maximum seen since the last `reset`.
final class FootprintSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak = 0
    private var timer: DispatchSourceTimer?

    func start(interval: DispatchTimeInterval = .milliseconds(20)) {
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now(), repeating: interval)
        timer.setEventHandler { [weak self] in self?.record(ProcessMemory.footprint()) }
        timer.resume()
        self.timer = timer
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }

    private func record(_ value: Int) {
        lock.lock()
        peak = max(peak, value)
        lock.unlock()
    }

    /// Returns the peak since the previous call and starts a new interval.
    func takePeak() -> Int {
        record(ProcessMemory.footprint())
        lock.lock()
        defer { lock.unlock() }
        let value = peak
        peak = 0
        return value
    }
}
