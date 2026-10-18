import Foundation
import MLX
import XCTest

/// Standard CI VMs have no Metal device; production inference remains GPU-only.
class MLXTestCase: XCTestCase {
    override func invokeTest() {
        if ProcessInfo.processInfo.environment["LINGXI_TEST_DEVICE"] == "cpu" {
            Device.withDefaultDevice(.cpu) { super.invokeTest() }
        } else { super.invokeTest() }
    }
}
