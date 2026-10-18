import Foundation
import MLX
import XCTest
@testable import QwenImageCore

final class TextConditionCacheTests: XCTestCase {
    func testBF16NoncontiguousRoundTripPreservesBytesDtypeAndRanges() throws {
        MLXRuntime.configure()
        let array = MLXArray((0..<24).map { Float($0) / 7 }).reshaped(4, 6).asType(.bfloat16).T
        let bytes = array.asData(access: .copy).data
        var cache = TextConditionCache()
        cache.store([.init(embedding: array, imageRanges: [1..<3])], key: Data([1]))
        let restored = try XCTUnwrap(cache.lookup(Data([1]))?.first)
        XCTAssertEqual(restored.embedding.dtype, .bfloat16)
        XCTAssertEqual(restored.embedding.shape, [6, 4])
        XCTAssertEqual(restored.embedding.asData(access: .copy).data, bytes)
        XCTAssertEqual(restored.imageRanges, [1..<3])
        XCTAssertNil(cache.lookup(Data([2])))
        cache.clear()
        XCTAssertNil(cache.lookup(Data([1])))
    }

    func testCapacityAndReplacementNeverReturnAnOlderEntry() {
        MLXRuntime.configure()
        var cache = TextConditionCache(byteLimit: 8)
        cache.store([.init(embedding: MLXArray([Float(1)]), imageRanges: [])], key: Data([1]))
        cache.store([.init(embedding: MLXArray([Float(2)]), imageRanges: [])], key: Data([2]))
        XCTAssertNil(cache.lookup(Data([1])))
        XCTAssertNotNil(cache.lookup(Data([2])))
        cache.store([.init(embedding: MLXArray([Float(1), 2, 3]), imageRanges: [])], key: Data([3]))
        XCTAssertNil(cache.lookup(Data([2])))
        XCTAssertNil(cache.lookup(Data([3])))
    }

    func testSeedIsExcludedButPromptCFGModelAndActualImageContentAreValidated() {
        var request = GenerationRequest(prompt: "portrait", seed: 42)
        let a = RGBAImage(width: 1, height: 1, pixels: [1, 2, 3, 255])
        let b = RGBAImage(width: 1, height: 1, pixels: [3, 2, 1, 255])
        let revision = Data([1])
        let original = TextConditionCache.key(revision: revision, request: request, conditions: [a, b])
        request.seed = 43
        XCTAssertEqual(original, TextConditionCache.key(revision: revision, request: request, conditions: [a, b]))
        XCTAssertNotEqual(original, TextConditionCache.key(revision: revision, request: request, conditions: [b, a]))
        XCTAssertNotEqual(original, TextConditionCache.key(revision: Data([2]), request: request, conditions: [a, b]))
        request.prompt = "landscape"
        XCTAssertNotEqual(original, TextConditionCache.key(revision: revision, request: request, conditions: [a, b]))
        request.prompt = "portrait"
        request.negativePrompt = "blur"
        request.trueCFGScale = 2
        XCTAssertNotEqual(original, TextConditionCache.key(revision: revision, request: request, conditions: [a, b]))
    }

    func testModelRevisionDetectsSameSizeEditsWithRestoredModificationTime() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = ModelDirectory(root: root)
        for directory in [model.processor, model.textEncoder, model.scheduler] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        let weight = model.textEncoder.appending(path: "test.safetensors")
        try Data([1, 2, 3]).write(to: weight)
        let time = try FileManager.default.attributesOfItem(atPath: weight.path)[.modificationDate]
        let before = try TextConditionCache.modelRevision(model)
        try Data([3, 2, 1]).write(to: weight)
        try FileManager.default.setAttributes([.modificationDate: try XCTUnwrap(time)], ofItemAtPath: weight.path)
        XCTAssertNotEqual(before, try TextConditionCache.modelRevision(model))
        let after = try TextConditionCache.modelRevision(model)
        try Data("changed".utf8).write(to: model.processor.appending(path: "tokenizer.json"))
        XCTAssertNotEqual(after, try TextConditionCache.modelRevision(model))
    }
}
