import CryptoKit
import Foundation
import Observation

/// Downloads Hugging Face repositories (public, no login) into a local folder.
///
/// - The file list comes from the Hub tree API (path, size, LFS SHA-256).
/// - Files that already exist with the expected size are skipped.
/// - Each file downloads to `<name>.part` and resumes with an HTTP Range request after an interruption.
/// - Up to three files run in parallel; a failed file is retried three times with backoff.
/// - Finished files are verified by size and, for LFS files, by SHA-256 computed while downloading.
@MainActor
@Observable
final class DownloadManager {
    struct RemoteFile: Identifiable, Hashable {
        let path: String
        let size: Int64
        /// SHA-256 of LFS files (hex), nil for small Git files.
        let sha256: String?
        var id: String { path }
    }

    enum FileState: Equatable {
        case pending, skipped, downloading, verifying, done, failed(String), cancelled

        var label: String {
            switch self {
            case .pending: "等待"
            case .skipped: "已存在，跳过"
            case .downloading: "下载中"
            case .verifying: "校验中"
            case .done: "完成"
            case .failed(let m): "失败：\(m)"
            case .cancelled: "已取消"
            }
        }
    }

    struct FileProgress: Identifiable {
        let file: RemoteFile
        var state: FileState = .pending
        var received: Int64 = 0
        var attempts = 0
        var id: String { file.path }
    }

    struct Job: Identifiable {
        let id = UUID()
        let repo: String
        let revision: String
        let destination: URL
        var files: [FileProgress]
        var started = Date()
        var finished: Date?
        var error: String?

        var totalBytes: Int64 { files.reduce(0) { $0 + $1.file.size } }
        var receivedBytes: Int64 { files.reduce(0) { $0 + ($1.state == .skipped || $1.state == .done ? $1.file.size : $1.received) } }
        var isActive: Bool { finished == nil && error == nil }
    }

    enum DownloadError: Error, CustomStringConvertible {
        case http(Int, String)
        case sizeMismatch(String)
        case hashMismatch(String)
        case offline

        var description: String {
            switch self {
            case .http(let code, let url):
                code == 401 || code == 403 ? "需要授权（仓库受限或私有）：\(url)" : "HTTP \(code)：\(url)"
            case .sizeMismatch(let p): "大小不符：\(p)"
            case .hashMismatch(let p): "SHA-256 不符：\(p)"
            case .offline: "已开启离线模式"
            }
        }
    }

    private(set) var jobs: [Job] = []
    var offline = false
    private var tasks: [UUID: Task<Void, Never>] = [:]
    private nonisolated let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.httpMaximumConnectionsPerHost = 4
        return URLSession(configuration: config)
    }()

    /// `owner/name` from a repo id or a huggingface.co URL.
    static func repoID(from input: String) -> String? {
        var s = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if let range = s.range(of: "huggingface.co/") { s = String(s[range.upperBound...]) }
        let parts = s.split(separator: "/").map(String.init).filter { !$0.isEmpty && $0 != "tree" && $0 != "main" }
        guard parts.count >= 2 else { return nil }
        return "\(parts[0])/\(parts[1])"
    }

    /// Lists every file of the repository (follows the tree API's pagination).
    func listFiles(repo: String, revision: String = "main") async throws -> [RemoteFile] {
        if offline { throw DownloadError.offline }
        var url: URL? = URL(string: "https://huggingface.co/api/models/\(repo)/tree/\(revision)?recursive=true")
        var files: [RemoteFile] = []
        while let next = url {
            let (data, response) = try await session.data(from: next)
            let http = response as? HTTPURLResponse
            guard http?.statusCode == 200 else { throw DownloadError.http(http?.statusCode ?? 0, next.absoluteString) }
            let entries = (try JSONSerialization.jsonObject(with: data) as? [[String: Any]]) ?? []
            for e in entries where e["type"] as? String == "file" {
                let lfs = e["lfs"] as? [String: Any]
                files.append(
                    RemoteFile(
                        path: e["path"] as? String ?? "", size: (lfs?["size"] as? NSNumber ?? e["size"] as? NSNumber)?.int64Value ?? 0,
                        sha256: lfs?["oid"] as? String))
            }
            url = Self.nextPage(http?.value(forHTTPHeaderField: "Link"))
        }
        return files
    }

    private static func nextPage(_ link: String?) -> URL? {
        guard let link else { return nil }
        for part in link.split(separator: ",") where part.contains("rel=\"next\"") {
            if let start = part.firstIndex(of: "<"), let end = part.firstIndex(of: ">") {
                return URL(string: String(part[part.index(after: start)..<end]))
            }
        }
        return nil
    }

    @discardableResult
    func start(repo: String, revision: String = "main", files: [RemoteFile], destination: URL) -> UUID {
        let job = Job(repo: repo, revision: revision, destination: destination, files: files.map { FileProgress(file: $0) })
        jobs.insert(job, at: 0)
        let id = job.id
        tasks[id] = Task { [weak self] in await self?.run(id) }
        return id
    }

    func cancel(_ id: UUID) {
        tasks[id]?.cancel()
        update(id) { job in
            for i in job.files.indices where job.files[i].state == .pending || job.files[i].state == .downloading {
                job.files[i].state = .cancelled
            }
            job.finished = Date()
        }
    }

    /// Restarts failed or cancelled files; partial files resume.
    func retry(_ id: UUID) {
        update(id) { job in
            for i in job.files.indices {
                switch job.files[i].state {
                case .failed, .cancelled: job.files[i].state = .pending
                default: break
                }
            }
            job.finished = nil
            job.error = nil
        }
        tasks[id] = Task { [weak self] in await self?.run(id) }
    }

    private func update(_ id: UUID, _ change: (inout Job) -> Void) {
        guard let i = jobs.firstIndex(where: { $0.id == id }) else { return }
        change(&jobs[i])
    }

    private func setFile(_ id: UUID, _ path: String, _ change: (inout FileProgress) -> Void) {
        update(id) { job in
            if let i = job.files.firstIndex(where: { $0.file.path == path }) { change(&job.files[i]) }
        }
    }

    private func run(_ id: UUID) async {
        guard let job = jobs.first(where: { $0.id == id }) else { return }
        if offline {
            update(id) { $0.error = DownloadError.offline.description }
            return
        }
        let pending = job.files.filter { $0.state == .pending }.map(\.file)
        await withTaskGroup(of: Void.self) { group in
            var iterator = pending.makeIterator()
            func next() -> RemoteFile? { iterator.next() }
            for _ in 0..<3 {
                guard let file = next() else { break }
                group.addTask { await self.download(id, job, file) }
            }
            while await group.next() != nil {
                if let file = next() { group.addTask { await self.download(id, job, file) } }
            }
        }
        update(id) { $0.finished = Date() }
    }

    private func download(_ id: UUID, _ job: Job, _ file: RemoteFile) async {
        let target = job.destination.appending(path: file.path)
        let fm = FileManager.default
        // Already complete (same size): skip without re-downloading.
        if let size = (try? fm.attributesOfItem(atPath: target.path)[.size] as? NSNumber)?.int64Value, size == file.size {
            setFile(id, file.path) { $0.state = .skipped }
            return
        }
        for attempt in 1...3 {
            if Task.isCancelled { return }
            setFile(id, file.path) {
                $0.state = .downloading
                $0.attempts = attempt
            }
            do {
                let session = self.session
                try await Self.fetch(session: session, repo: job.repo, revision: job.revision, file: file, to: target) {
                    received, verifying in
                    await MainActor.run {
                        self.setFile(id, file.path) {
                            $0.received = received
                            if verifying { $0.state = .verifying }
                        }
                    }
                }
                setFile(id, file.path) { $0.state = .done }
                return
            } catch is CancellationError {
                setFile(id, file.path) { $0.state = .cancelled }
                return
            } catch {
                if attempt == 3 {
                    setFile(id, file.path) { $0.state = .failed("\(error)") }
                } else {
                    try? await Task.sleep(for: .seconds(Double(attempt * attempt) * 2))
                }
            }
        }
    }

    /// Runs off the main actor; `progress(received, verifying)` reports back.
    private nonisolated static func fetch(
        session: URLSession, repo: String, revision: String, file: RemoteFile, to target: URL,
        progress: @escaping @Sendable (Int64, Bool) async -> Void
    ) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        let part = target.appendingPathExtension("part")
        if !fm.fileExists(atPath: part.path) { fm.createFile(atPath: part.path, contents: nil) }
        var offset = ((try? fm.attributesOfItem(atPath: part.path)[.size] as? NSNumber)?.int64Value) ?? 0
        if offset > file.size { offset = 0; try Data().write(to: part) }

        // The hash covers the bytes already on disk, then the rest as it arrives.
        var hasher = SHA256()
        if offset > 0, file.sha256 != nil, let handle = try? FileHandle(forReadingFrom: part) {
            while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
            try handle.close()
        }
        let encodedPath = file.path.split(separator: "/").map {
            String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0)
        }.joined(separator: "/")
        var request = URLRequest(url: URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(encodedPath)")!)
        if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }

        let (bytes, response) = try await session.bytes(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 || status == 206 else { throw DownloadError.http(status, request.url!.absoluteString) }
        if status == 200 && offset > 0 {
            // The server ignored the Range: start over.
            offset = 0
            hasher = SHA256()
            try Data().write(to: part)
        }
        let handle = try FileHandle(forWritingTo: part)
        defer { try? handle.close() }
        try handle.seekToEnd()
        var received = offset
        var buffer = Data()
        buffer.reserveCapacity(4 << 20)
        var lastReport = Date()
        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= 4 << 20 {
                try handle.write(contentsOf: buffer)
                if file.sha256 != nil { hasher.update(data: buffer) }
                received += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                if Date().timeIntervalSince(lastReport) > 0.25 {
                    await progress(received, false)
                    lastReport = Date()
                }
                try Task.checkCancellation()
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            if file.sha256 != nil { hasher.update(data: buffer) }
            received += Int64(buffer.count)
        }
        await progress(received, true)
        guard received == file.size else { throw DownloadError.sizeMismatch(file.path) }
        if let expected = file.sha256 {
            let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            guard digest == expected.lowercased() else {
                try? FileManager.default.removeItem(at: part)
                throw DownloadError.hashMismatch(file.path)
            }
        }
        if fm.fileExists(atPath: target.path) { try fm.trashItem(at: target, resultingItemURL: nil) }
        try fm.moveItem(at: part, to: target)
    }
}
