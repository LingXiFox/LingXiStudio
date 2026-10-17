import Foundation

/// `FlowMatchEulerDiscreteScheduler` as configured for Qwen-Image-2.1: exponential resolution-dependent time
/// shift followed by the terminal stretch, sampled with first-order Euler steps.
public struct FlowMatchSchedule: Sendable {
    public struct Config: Sendable {
        public var baseImageSeqLen = 256
        public var maxImageSeqLen = 8192
        public var baseShift = 0.5
        public var maxShift = 0.9
        public var shiftTerminal: Double? = 0.02
        public var useDynamicShifting = true
        public var shift = 1.0

        public init() {}

        public init(contentsOf url: URL) throws {
            let data = try Data(contentsOf: url)
            let json = (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            if let v = json["base_image_seq_len"] as? Int { baseImageSeqLen = v }
            if let v = json["max_image_seq_len"] as? Int { maxImageSeqLen = v }
            if let v = json["base_shift"] as? Double { baseShift = v }
            if let v = json["max_shift"] as? Double { maxShift = v }
            if let v = json["use_dynamic_shifting"] as? Bool { useDynamicShifting = v }
            if let v = json["shift"] as? Double { shift = v }
            shiftTerminal = json["shift_terminal"] as? Double
            if let type = json["time_shift_type"] as? String, type != "exponential" {
                throw ScheduleError.unsupported("time_shift_type \(type)")
            }
        }
    }

    public enum ScheduleError: Error, CustomStringConvertible {
        case unsupported(String)
        public var description: String {
            switch self {
            case .unsupported(let what): "Unsupported scheduler setting: \(what)"
            }
        }
    }

    /// `steps + 1` values from ~1 down to the terminal 0.
    public let sigmas: [Float]
    public let mu: Double

    public var steps: Int { sigmas.count - 1 }

    public init(steps: Int, imageSeqLen: Int, config: Config = Config()) {
        precondition(steps > 0)
        let m = (config.maxShift - config.baseShift) / Double(config.maxImageSeqLen - config.baseImageSeqLen)
        let b = config.baseShift - m * Double(config.baseImageSeqLen)
        let mu = Double(imageSeqLen) * m + b
        self.mu = mu

        // np.linspace(1.0, 1 / steps, steps)
        var values: [Double] = (0..<steps).map { i in
            steps == 1 ? 1.0 : 1.0 + (1.0 / Double(steps) - 1.0) * Double(i) / Double(steps - 1)
        }
        if config.useDynamicShifting {
            values = values.map { t in exp(mu) / (exp(mu) + (1 / t - 1)) }
        } else {
            values = values.map { t in config.shift * t / (1 + (config.shift - 1) * t) }
        }
        if let terminal = config.shiftTerminal, terminal > 0 {
            let scale = (1 - values[values.count - 1]) / (1 - terminal)
            values = values.map { 1 - (1 - $0) / scale }
        }
        sigmas = values.map { Float($0) } + [0]
    }
}
