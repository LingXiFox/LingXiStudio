import Foundation

/// Byte-level BPE tokenizer compatible with the Qwen2/Qwen3 `tokenizer.json` shipped with Qwen-Image-2.1.
///
/// Pipeline: split on added tokens -> NFC normalize -> regex pre-tokenize -> byte-to-unicode -> BPE merges.
public final class QwenTokenizer: @unchecked Sendable {
    public enum TokenizerError: Error, CustomStringConvertible {
        case missingFile(String)
        case invalidFormat(String)
        case unknownToken(String)

        public var description: String {
            switch self {
            case .missingFile(let path): "Tokenizer file not found: \(path)"
            case .invalidFormat(let reason): "Invalid tokenizer file: \(reason)"
            case .unknownToken(let token): "Token not in vocabulary: \(token)"
            }
        }
    }

    private let vocab: [String: Int]
    private let mergeRanks: [String: Int]
    /// Added tokens sorted longest first so that greedy matching picks the longest special token.
    private let addedTokens: [(content: String, id: Int)]
    private let pretokenizer: NSRegularExpression
    private let byteEncoder: [Character]
    private var cache: [String: [Int]] = [:]
    private let lock = NSLock()

    public init(directory: URL) throws {
        let tokenizerURL = directory.appending(path: "tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw TokenizerError.missingFile(tokenizerURL.path)
        }
        let data = try Data(contentsOf: tokenizerURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let vocab = model["vocab"] as? [String: Int],
            let merges = model["merges"] as? [Any]
        else {
            throw TokenizerError.invalidFormat("missing model.vocab / model.merges")
        }
        guard (model["type"] as? String) == "BPE" else {
            throw TokenizerError.invalidFormat("only BPE models are supported")
        }

        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, entry) in merges.enumerated() {
            if let pair = entry as? [String], pair.count == 2 {
                ranks[pair[0] + " " + pair[1]] = rank
            } else if let line = entry as? String {
                ranks[line] = rank
            } else {
                throw TokenizerError.invalidFormat("unexpected merge entry at \(rank)")
            }
        }

        var added: [(String, Int)] = []
        for item in (root["added_tokens"] as? [[String: Any]]) ?? [] {
            if let content = item["content"] as? String, let id = item["id"] as? Int {
                added.append((content, id))
            }
        }
        added.sort { $0.0.count > $1.0.count }

        // The pattern is taken from tokenizer.json so it stays in sync with the checkpoint.
        var pattern =
            #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        if let pre = root["pre_tokenizer"] as? [String: Any],
            let steps = pre["pretokenizers"] as? [[String: Any]],
            let split = steps.first(where: { ($0["type"] as? String) == "Split" }),
            let regex = (split["pattern"] as? [String: Any])?["Regex"] as? String
        {
            pattern = regex
        }

        self.vocab = vocab
        self.mergeRanks = ranks
        self.addedTokens = added
        self.pretokenizer = try NSRegularExpression(pattern: pattern)
        self.byteEncoder = QwenTokenizer.makeByteEncoder()
    }

    public func tokenID(_ token: String) -> Int? {
        if let id = addedTokens.first(where: { $0.content == token })?.id { return id }
        return vocab[token]
    }

    public func encode(_ text: String) throws -> [Int] {
        var ids: [Int] = []
        for segment in splitOnAddedTokens(text) {
            switch segment {
            case .added(let id):
                ids.append(id)
            case .text(let raw):
                let normalized = raw.precomposedStringWithCanonicalMapping
                for piece in pretokenize(normalized) {
                    ids.append(contentsOf: try bpe(piece))
                }
            }
        }
        return ids
    }

    // MARK: - Internals

    private enum Segment {
        case text(String)
        case added(Int)
    }

    private func splitOnAddedTokens(_ text: String) -> [Segment] {
        var segments: [Segment] = []
        var pending = ""
        var index = text.startIndex
        outer: while index < text.endIndex {
            for token in addedTokens where text[index...].hasPrefix(token.content) {
                if !pending.isEmpty {
                    segments.append(.text(pending))
                    pending = ""
                }
                segments.append(.added(token.id))
                index = text.index(index, offsetBy: token.content.count)
                continue outer
            }
            pending.append(text[index])
            index = text.index(after: index)
        }
        if !pending.isEmpty { segments.append(.text(pending)) }
        return segments
    }

    private func pretokenize(_ text: String) -> [String] {
        let ns = text as NSString
        return pretokenizer.matches(in: text, range: NSRange(location: 0, length: ns.length)).map {
            ns.substring(with: $0.range)
        }
    }

    private func bpe(_ piece: String) throws -> [Int] {
        lock.lock()
        if let cached = cache[piece] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        var symbols = Array(piece.utf8).map { String(byteEncoder[Int($0)]) }
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(symbols.count - 1) {
                if let rank = mergeRanks[symbols[i] + " " + symbols[i + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if bestIndex < 0 { break }
            let first = symbols[bestIndex]
            let second = symbols[bestIndex + 1]
            // Merge every occurrence of the best pair, left to right, like the reference implementation.
            var merged: [String] = []
            merged.reserveCapacity(symbols.count)
            var i = 0
            while i < symbols.count {
                if i < symbols.count - 1 && symbols[i] == first && symbols[i + 1] == second {
                    merged.append(first + second)
                    i += 2
                } else {
                    merged.append(symbols[i])
                    i += 1
                }
            }
            symbols = merged
        }

        let ids = try symbols.map { symbol -> Int in
            guard let id = vocab[symbol] else { throw TokenizerError.unknownToken(symbol) }
            return id
        }
        lock.lock()
        cache[piece] = ids
        lock.unlock()
        return ids
    }

    /// GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest to code points from 256 upward.
    private static func makeByteEncoder() -> [Character] {
        var printable = Array(33...126) + Array(161...172) + Array(174...255)
        var codepoints = printable
        var next = 0
        for byte in 0..<256 where !printable.contains(byte) {
            printable.append(byte)
            codepoints.append(256 + next)
            next += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (byte, codepoint) in zip(printable, codepoints) {
            table[byte] = Character(Unicode.Scalar(UInt32(codepoint))!)
        }
        return table
    }
}
