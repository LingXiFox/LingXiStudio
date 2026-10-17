import Foundation
import QwenImageCore

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
        let request = GenerationRequest(
            prompt: args.require("prompt"), negativePrompt: args.string("negative-prompt"),
            width: args.int("width", 1024), height: args.int("height", 1024), steps: args.int("steps", 40),
            trueCFGScale: args.float("cfg", 1.0),
            seed: args.string("seed").flatMap(UInt64.init) ?? UInt64.random(in: 0..<UInt64(UInt32.max)))
        print("seed \(request.seed), \(request.width)x\(request.height), \(request.steps) steps"
            + (request.usesTrueCFG ? ", true CFG \(request.trueCFGScale)" : ""))

        let pipeline = try QwenImagePipeline(model: model)
        var lastStage: GenerationStage?
        let image = try pipeline.generate(request) { p in
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
        for t in image.timings { print(String(format: "  %-20@ %7.1fs", t.stage.rawValue as NSString, t.seconds)) }
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

    case "debug-transformer":
        try Diagnostics.dumpTransformer(
            model: model, input: args.url("in")!, sigma: args.float("sigma", 1.0),
            latentHeight: args.int("height", 1024) / 16, latentWidth: args.int("width", 1024) / 16,
            to: args.url("out")!)

    case "debug-vae":
        try Diagnostics.dumpVAE(
            model: model, input: args.url("in")!, latentHeight: args.int("height", 1024) / 16,
            latentWidth: args.int("width", 1024) / 16, float32: args.string("dtype") == "f32", to: args.url("out")!)

    default:
        print(usage)
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
