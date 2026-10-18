import Foundation
import QwenImageCore

MLXRuntime.configure()

let usage = """
    qwen-image — native Qwen-Image-2.1 runner (Swift + MLX)

    USAGE:
      qwen-image generate --prompt <text> [options]
      qwen-image validate [--model <dir>]
      qwen-image tokenize --text <text> [--model <dir>]
      qwen-image schedule --steps <n> [--width W --height H]
      qwen-image debug-text --prompt <text> --out <file.safetensors>
      qwen-image debug-transformer --in <file> --sigma <s> --width W --height H --out <file>
      qwen-image debug-vae --in <file> --width W --height H --out <file>

    GENERATE OPTIONS:
      --model <dir>            Model folder (default ~/Models/Qwen-Image-2.1-BF16)
      --negative-prompt <text> Negative prompt (needs --cfg > 1)
      --cfg <float>            True CFG scale (default 1.0 = off)
      --width / --height <n>   Output size, multiples of 32 (default 1024)
      --steps <n>              Denoising steps (default 40)
      --seed <n>               Seed (default random)
      --output <file.png>      Output path (default ./outputs/<timestamp>-<seed>.png)
      --lora <file[:scale]>    LoRA applied at runtime, base weights untouched (comma-separate several)
      --scheduler <name>       linear (default) or viggle_turbo (6 steps)
      --mode <name>            text-to-image (default), reference, edit, img2img, inpaint
      --image <a.png,b.png>    Condition images (reference / edit / inpaint: the first is the main image)
      --init <file>            Start image for img2img
      --mask <file>            Inpaint mask, white = repaint (--invert 1, --feather <px>, --keep-unmasked 0)
      --strength <0..1>        Share of the schedule run from the noised start image (default 1)
    """

struct Arguments {
    var command: String
    var options: [String: String] = [:]

    init(_ argv: [String]) {
        command = argv.first ?? "help"
        var index = 1
        while index < argv.count {
            let key = argv[index]
            if key.hasPrefix("--"), index + 1 < argv.count {
                options[String(key.dropFirst(2))] = argv[index + 1]
                index += 2
            } else {
                index += 1
            }
        }
    }

    func string(_ key: String) -> String? { options[key] }
    func int(_ key: String, _ fallback: Int) -> Int { options[key].flatMap(Int.init) ?? fallback }
    func float(_ key: String, _ fallback: Float) -> Float { options[key].flatMap(Float.init) ?? fallback }
    func url(_ key: String) -> URL? {
        options[key].map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
    }
    func require(_ key: String) -> String {
        guard let value = options[key] else {
            FileHandle.standardError.write(Data("missing --\(key)\n\n\(usage)\n".utf8))
            exit(2)
        }
        return value
    }
}

/// `--lora a.safetensors:0.8,b.safetensors` -> specs (scale defaults to 1).
func loraSpecs() -> [LoRASpec] {
    (args.string("lora") ?? "").split(separator: ",").map { item in
        let parts = item.split(separator: ":", maxSplits: 1)
        let path = (String(parts[0]) as NSString).expandingTildeInPath
        return LoRASpec(path: path, scale: parts.count > 1 ? Float(parts[1]) ?? 1 : 1)
    }
}

/// Request plus decoded input images for `--mode` (reference / edit / img2img / inpaint).
func makeRequest(prompt: String) throws -> (GenerationRequest, GenerationInputs) {
    let mode: GenerationMode =
        switch args.string("mode") ?? "text-to-image" {
        case "reference": .reference
        case "edit": .edit
        case "img2img", "image-to-image": .imageToImage
        case "inpaint": .inpaint
        default: .textToImage
        }
    func paths(_ key: String) -> [URL] {
        (args.string(key) ?? "").split(separator: ",").map {
            URL(fileURLWithPath: (String($0) as NSString).expandingTildeInPath)
        }
    }
    var records: [InputImageRecord] = []
    var inputs = GenerationInputs()
    for url in paths("image") {
        let (record, image) = try InputImageRecord.make(role: .condition, url: url)
        records.append(record)
        inputs.conditions.append(image)
    }
    if let url = paths("init").first {
        let (record, image) = try InputImageRecord.make(role: .initial, url: url)
        records.append(record)
        inputs.initial = image
    }
    if let url = paths("mask").first {
        let (record, image) = try InputImageRecord.make(role: .mask, url: url)
        records.append(record)
        inputs.mask = image
    }
    let size = GenerationMode.defaultSize(mode: mode, conditions: inputs.conditions, initial: inputs.initial)
    let request = GenerationRequest(
        prompt: prompt, negativePrompt: args.string("negative-prompt"),
        width: args.int("width", size?.width ?? 1024), height: args.int("height", size?.height ?? 1024),
        steps: args.int("steps", 40), trueCFGScale: args.float("cfg", 1.0),
        seed: args.string("seed").flatMap(UInt64.init) ?? UInt64.random(in: 0..<UInt64(UInt32.max)),
        scheduler: SamplerSchedule(rawValue: args.string("scheduler") ?? "linear") ?? .linear,
        loras: loraSpecs(), mode: mode, inputs: records, strength: args.float("strength", 1),
        maskFeather: args.int("feather", 8), maskInvert: args.string("invert") == "1",
        keepUnmaskedPixels: args.string("keep-unmasked") != "0")
    return (request, inputs)
}

func formatBytes(_ bytes: Int) -> String {
    String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
}

// Line-buffered progress must interleave correctly with regular prints when piped to a log.
setvbuf(stdout, nil, _IOLBF, 0)
let args = Arguments(Array(CommandLine.arguments.dropFirst()))
let model = ModelDirectory(
    root: args.url("model") ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: "Models/Qwen-Image-2.1-BF16"))

do {
    switch args.command {
    case "generate":
        let components = try model.validate()
        for c in components where c.bytes > 0 {
            print("✓ \(c.name): \(c.files) file(s), \(formatBytes(Int(c.bytes)))")
        }
        let (request, inputs) = try makeRequest(prompt: args.require("prompt"))
        print("seed \(request.seed), \(request.width)x\(request.height), \(request.steps) steps"
            + (request.usesTrueCFG ? ", true CFG \(request.trueCFGScale)" : ""))

        let pipeline = try QwenImagePipeline(model: model)
        var lastStage: GenerationStage?
        let image = try pipeline.generate(request, inputs: inputs) { p in
            if p.stage != lastStage {
                if lastStage != nil { print("") }
                lastStage = p.stage
            }
            var line = "\r[\(String(format: "%6.1f", p.elapsed))s] \(p.stage.rawValue)"
            if p.stage == .denoising {
                line += " \(p.step)/\(p.totalSteps)"
                if let rate = p.secondsPerStep {
                    line += String(format: "  %.2f s/step  ETA %.0fs", rate, rate * Double(p.totalSteps - p.step))
                }
            } else {
                line += String(format: " %3.0f%%", p.fraction * 100)
            }
            line += "  mem \(formatBytes(p.memory.activeBytes)) (peak \(formatBytes(p.memory.peakBytes)))   "
            print(line, terminator: "")
            fflush(stdout)
        }
        print("")

        let output: URL
        if let explicit = args.url("output") {
            output = explicit
        } else {
            let dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appending(path: "outputs")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
            output = dir.appending(path: "\(stamp)-\(request.seed).png")
        }
        try ImageWriter.write(image, model: model.root.lastPathComponent, to: output)
        for t in image.timings {
            print(String(format: "  %-20@ %7.1fs  footprint peak %@", t.stage.rawValue as NSString, t.seconds,
                formatBytes(t.peakFootprintBytes)))
        }
        print("step seconds: " + image.stepSeconds.map { String(format: "%.2f", $0) }.joined(separator: " "))
        if image.stepSeconds.count > 1 {
            let steady = image.stepSeconds.dropFirst()
            print(String(format: "steady-state %.3f s/step (steps 2-%d)", steady.reduce(0, +) / Double(steady.count), image.stepSeconds.count))
        }
        print(String(format: "total %.1fs, peak MLX memory %@", image.totalSeconds, formatBytes(image.peakMemoryBytes)))
        print("saved \(output.path)")

    case "validate":
        for c in try model.validate() {
            print("✓ \(c.name): \(c.files) safetensors, \(formatBytes(Int(c.bytes)))")
        }

    case "tokenize":
        print(try Diagnostics.tokenize(model: model, text: args.require("text")))

    case "schedule":
        let tokens = (args.int("width", 1024) / 16) * (args.int("height", 1024) / 16)
        print(try Diagnostics.schedule(steps: args.int("steps", 40), imageSeqLen: tokens, model: model))

    case "debug-text":
        try Diagnostics.dumpTextEncoder(model: model, prompt: args.require("prompt"), to: args.url("out")!)

    case "debug-text-image":
        try Diagnostics.dumpTextImage(
            model: model, prompt: args.require("prompt"), image: args.url("image")!, to: args.url("out")!)

    case "debug-transformer":
        try Diagnostics.dumpTransformer(
            model: model, input: args.url("in")!,
            sigmas: (args.string("sigmas") ?? args.string("sigma") ?? "1.0").split(separator: ",").compactMap { Float($0) },
            latentHeight: args.int("height", 1024) / 16, latentWidth: args.int("width", 1024) / 16,
            loras: loraSpecs(), trace: args.string("trace") == "1", to: args.url("out")!)

    case "debug-transformer-cond":
        try Diagnostics.dumpTransformerCondition(
            model: model, input: args.url("in")!,
            sigmas: (args.string("sigmas") ?? "0.9").split(separator: ",").compactMap { Float($0) },
            latentHeight: args.int("height", 1024) / 16, latentWidth: args.int("width", 1024) / 16,
            loras: loraSpecs(), to: args.url("out")!)

    case "debug-conditioning":
        try Diagnostics.dumpConditioning(
            model: model, input: args.url("in")!, sigma: args.float("sigma", 1.0), loras: loraSpecs(),
            to: args.url("out")!)

    case "debug-denoise":
        try Diagnostics.dumpDenoise(model: model, input: args.url("in")!, request: makeRequest(prompt: "").0, to: args.url("out")!)

    case "debug-euler":
        try Diagnostics.dumpEuler(model: model, input: args.url("in")!, request: makeRequest(prompt: "").0, to: args.url("out")!)

    case "debug-postprocess":
        try Diagnostics.dumpPostprocess(input: args.url("in")!, to: args.url("out")!)

    case "debug-lora":
        let lines = try Diagnostics.checkLoRALifecycle(
            model: model, input: args.url("in")!, lora: (args.require("lora") as NSString).expandingTildeInPath,
            sigma: args.float("sigma", 0.75), latentHeight: args.int("height", 1024) / 16,
            latentWidth: args.int("width", 1024) / 16)
        lines.forEach { print($0) }

    case "debug-vae":
        try Diagnostics.dumpVAE(
            model: model, input: args.url("in")!, latentHeight: args.int("height", 1024) / 16,
            latentWidth: args.int("width", 1024) / 16, float32: args.string("dtype") == "f32", to: args.url("out")!)

    case "debug-vae-encode":
        try Diagnostics.dumpVAEEncode(model: model, input: args.url("in")!, to: args.url("out")!)

    case "bench-transformer":
        let probeBefore = Diagnostics.probeThermal()
        let r = try Diagnostics.benchTransformer(
            model: model, width: args.int("width", 1024), height: args.int("height", 1024),
            textLength: args.int("text-length", 31), steps: args.int("steps", 3), loras: loraSpecs())
        let probeAfter = Diagnostics.probeThermal()
        print("steps: " + r.steps.map { String(format: "%.3f", $0) }.joined(separator: " "))
        print(String(format: "mean %.3f s/step  footprint peak %@  MLX peak %@", r.steps.reduce(0, +) / Double(r.steps.count),
            formatBytes(r.footprint), formatBytes(r.mlxPeak)))
        print(String(format: "probe before %.2f TFLOPS (%@), after %.2f TFLOPS (%@)", probeBefore.tflops,
            probeBefore.thermalState, probeAfter.tflops, probeAfter.thermalState))

    case "probe":
        let probe = Diagnostics.probeThermal()
        print(String(format: "thermal %@  matmul %.2f TFLOPS", probe.thermalState, probe.tflops))

    case "debug-kernel":
        let (maxAbs, maxUlp, mismatched) = Diagnostics.checkFusedQKNormRope()
        print(String(format: "fused qk-norm-rope vs reference composition: max abs %.3g, max %.2f bf16 ulp, %.4f%% of elements differ",
            maxAbs, maxUlp, mismatched * 100))

    default:
        print(usage)
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
