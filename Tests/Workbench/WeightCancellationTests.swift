import MLX
import XCTest
@testable import QwenImageCore

final class WeightCancellationTests: MLXTestCase {
    private struct Weights: ParameterContainer {
        let parameters = [MLXArray([Float(1)]), MLXArray([Float(2)]), MLXArray([Float(3)])]
    }

    func testCancellationBeforeFirstBatchSkipsLoadingProgress() {
        MLXRuntime.configure()
        var progress: [Double] = []
        XCTAssertThrowsError(try Weights().materialize(checkCancelled: { throw PipelineError.cancelled }) { progress.append($0) }) {
            guard case PipelineError.cancelled = $0 else { return XCTFail("Unexpected error: \($0)") }
        }
        XCTAssertTrue(progress.isEmpty)
    }

    func testCancellationAtExistingBatchBoundaryStopsRemainingBatches() {
        MLXRuntime.configure()
        var checks = 0
        var progress: [Double] = []
        XCTAssertThrowsError(try Weights().materialize(batch: 1, checkCancelled: {
            checks += 1
            if checks == 2 { throw PipelineError.cancelled }
        }) { progress.append($0) })
        XCTAssertEqual(checks, 2)
        XCTAssertEqual(progress, [1.0 / 3.0])
    }

    func testUncancelledLoadingRetainsProgressAndValues() throws {
        MLXRuntime.configure()
        let weights = Weights()
        var progress: [Double] = []
        try weights.materialize(batch: 1, progress: { progress.append($0) })
        XCTAssertEqual(progress, [1.0 / 3.0, 2.0 / 3.0, 1])
        XCTAssertEqual(weights.parameters.map { $0.item(Float.self) }, [1, 2, 3])
    }

    func testTextEncodingChecksCancellationBetweenExistingLayerEvaluations() throws {
        MLXRuntime.configure()
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let prefix = "model.language_model"
        var arrays: [String: MLXArray] = [
            "\(prefix).embed_tokens.weight": MLXArray.ones([2, 4]),
            "\(prefix).norm.weight": MLXArray.ones([4]),
        ]
        for index in 0..<2 {
            let layer = "\(prefix).layers.\(index)"
            for name in ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"] {
                arrays["\(layer).\(name).weight"] = MLXArray.ones([4])
            }
            for name in ["q", "k", "v", "o"] {
                arrays["\(layer).self_attn.\(name)_proj.weight"] = MLXArray.zeros([4, 4])
            }
            for name in ["gate", "up"] { arrays["\(layer).mlp.\(name)_proj.weight"] = MLXArray.zeros([8, 4]) }
            arrays["\(layer).mlp.down_proj.weight"] = MLXArray.zeros([4, 8])
        }
        try save(arrays: arrays, url: directory.appending(path: "weights.safetensors"))
        let config = QwenTextEncoder.Config(hiddenSize: 4, layers: 2, heads: 1, kvHeads: 1, headDim: 4, intermediateSize: 8)
        let encoder = try QwenTextEncoder(store: WeightStore(directory: directory, component: "test"), config: config)
        var checks = 0
        XCTAssertThrowsError(try encoder([0], checkCancelled: {
            checks += 1
            if checks == 2 { throw PipelineError.cancelled }
        })) {
            guard case PipelineError.cancelled = $0 else { return XCTFail("Unexpected error: \($0)") }
        }
        XCTAssertEqual(checks, 2)
        XCTAssertEqual(try encoder([0]).asArray(Float.self), [1, 1, 1, 1])
        let streamed = try QwenTextEncoder.encodeStreaming(directory: directory,
            prompts: [([0], nil), ([1, 0], nil)], config: config)
        XCTAssertEqual(streamed[0].asData(access: .copy).data, try encoder([0]).asData(access: .copy).data)
        XCTAssertEqual(streamed[1].asData(access: .copy).data, try encoder([1, 0]).asData(access: .copy).data)
        let previousLimit = Memory.cacheLimit
        var streamedChecks = 0
        var completedLayers: [Double] = []
        XCTAssertThrowsError(try QwenTextEncoder.encodeStreaming(directory: directory,
            prompts: [([0], nil)], config: config, checkCancelled: {
                streamedChecks += 1
                if streamedChecks == 2 { throw PipelineError.cancelled }
            }, progress: { completedLayers.append($0) }))
        XCTAssertEqual(streamedChecks, 2)
        XCTAssertEqual(completedLayers, [0.5])
        XCTAssertEqual(Memory.cacheLimit, previousLimit)
    }
}
