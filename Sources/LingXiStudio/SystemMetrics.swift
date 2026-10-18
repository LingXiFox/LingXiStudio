import Darwin
import Foundation
import QwenImageCore

/// One sample of the memory and thermal figures the workbench shows. Values the system does not expose stay `nil`
/// and are shown as unavailable rather than estimated.
struct SystemMetrics: Sendable {
    enum Pressure: String, Sendable {
        case normal, warning, critical

        var label: String {
            switch self {
            case .normal: "正常"
            case .warning: "偏高"
            case .critical: "严重"
            }
        }
    }

    var physicalMemory: Int64
    var appFootprint: Int
    var swapUsed: Int64?
    var swapTotal: Int64?
    var pressure: Pressure?
    /// `kern.memorystatus_level`: percentage of memory the kernel considers available.
    var freePercent: Int?
    var thermal: ProcessInfo.ThermalState

    static func sample() -> SystemMetrics {
        var swap = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        let hasSwap = sysctlbyname("vm.swapusage", &swap, &size, nil, 0) == 0
        return SystemMetrics(
            physicalMemory: Int64(ProcessInfo.processInfo.physicalMemory),
            appFootprint: ProcessMemory.footprint(),
            swapUsed: hasSwap ? Int64(swap.xsu_used) : nil,
            swapTotal: hasSwap ? Int64(swap.xsu_total) : nil,
            pressure: sysctlInt("kern.memorystatus_vm_pressure_level").flatMap {
                switch $0 {
                case 1: .normal
                case 2: .warning
                case 4: .critical
                default: nil
                }
            },
            freePercent: sysctlInt("kern.memorystatus_level"),
            thermal: ProcessInfo.processInfo.thermalState)
    }

    private static func sysctlInt(_ name: String) -> Int? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        return sysctlbyname(name, &value, &size, nil, 0) == 0 ? Int(value) : nil
    }
}

extension ProcessInfo.ThermalState {
    var label: String {
        switch self {
        case .nominal: "正常"
        case .fair: "轻微发热"
        case .serious: "较热（降频）"
        case .critical: "过热"
        @unknown default: "未知"
        }
    }
}

enum Format {
    static func bytes<T: BinaryInteger>(_ value: T) -> String {
        let gb = Double(value) / 1_073_741_824
        return gb >= 1 ? String(format: "%.1f GB", gb) : String(format: "%.0f MB", Double(value) / 1_048_576)
    }

    static func duration(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "–" }
        if seconds < 60 { return String(format: "%.1f 秒", seconds) }
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return m < 60 ? "\(m) 分 \(s) 秒" : "\(m / 60) 时 \(m % 60) 分"
    }
}
