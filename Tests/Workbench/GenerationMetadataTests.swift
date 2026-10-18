import Foundation
import ImageIO
import XCTest
@testable import QwenImageCore

final class GenerationMetadataTests: XCTestCase {
    func testEveryModeIsRecordedInSidecarAndPNG() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        for mode in GenerationMode.allCases {
            let request = GenerationRequest(prompt: "test", width: 32, height: 32, seed: 42, mode: mode)
            let image = GeneratedImage(
                width: 32, height: 32, rgba: .init(repeating: 255, count: 32 * 32 * 4), request: request,
                timings: [], peakMemoryBytes: 0, totalSeconds: 0, stepSeconds: [])
            let url = directory.appending(path: "\(mode.rawValue).png")
            try ImageWriter.write(image, model: "test", to: url)
            let sidecar = try GenerationMetadata.read(sidecarOf: url)
            XCTAssertEqual(sidecar.mode, mode.rawValue)
            XCTAssertEqual(sidecar.request.mode, mode)
            let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
            let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
            let png = try XCTUnwrap(properties[kCGImagePropertyPNGDictionary] as? [CFString: Any])
            let description = try XCTUnwrap(png[kCGImagePropertyPNGDescription] as? String)
            let record = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(description.utf8)) as? [String: Any])
            XCTAssertEqual(record["mode"] as? String, mode.rawValue)
        }
    }
}
