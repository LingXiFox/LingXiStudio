import Foundation
import MLX
import XCTest

/// Common CI math runs on CPU; Metal resource lookup still happens during MLX initialization.
class MLXTestCase: XCTestCase {
    override func invokeTest() {
        // Swift Build nests this bundle inside the xctest, outside MLX's automatic search paths.
        guard let resources = Bundle(for: MLXTestCase.self).resourceURL else {
            XCTFail("The test bundle has no resource directory")
            return
        }
        let library = resources.appending(path: "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
        guard FileManager.default.fileExists(atPath: library.path) else {
            XCTFail("Missing test Metal resource: \(library.path)")
            return
        }
        GPU.metallib = library
        if ProcessInfo.processInfo.environment["LINGXI_TEST_DEVICE"] == "cpu" {
            Device.withDefaultDevice(.cpu) { super.invokeTest() }
        } else { super.invokeTest() }
    }
}
