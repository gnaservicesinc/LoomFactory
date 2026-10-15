import CryptoKit
import Foundation
import LoomNative

/// Uses the Hub's preupload, Git LFS and NDJSON commit interfaces directly.
/// Protocol references: huggingface_hub/_commit_api.py and huggingface_hub/lfs.py.
/// No credential is written into the source project, an upload file or a log.
public enum NativeHubPublisher {
    public static func publish(package: URL, repository: String,
                               progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> URL {
        let token = try authenticationToken(environment: ProcessInfo.processInfo.environment)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 24 * 60 * 60
        let session = URLSession(configuration: configuration, delegate: HubRedirectDelegate(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        return try await publish(package: package, repository: repository, token: token, session: session, progress: progress)
    }

    static func authenticationToken(environment: [String: String], home: URL = FileManager.default.homeDirectoryForCurrentUser) throws -> String {
        func clean(_ value: String?) -> String? {
            guard let value else { return nil }
            let result = value.replacingOccurrences(of: "\r", with: "").replacingOccurrences(of: "\n", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return result.isEmpty ? nil : result
        }
        if let token = clean(environment["HF_TOKEN"]) ?? clean(environment["HUGGING_FACE_HUB_TOKEN"]) { return token }
        let tokenPath: URL
        if let explicit = environment["HF_TOKEN_PATH"], !explicit.isEmpty {
            tokenPath = URL(fileURLWithPath: NSString(string: explicit).expandingTildeInPath)
        } else if let hfHome = environment["HF_HOME"], !hfHome.isEmpty {
            tokenPath = URL(fileURLWithPath: NSString(string: hfHome).expandingTildeInPath).appendingPathComponent("token")
        } else if let cache = environment["XDG_CACHE_HOME"], !cache.isEmpty {
            tokenPath = URL(fileURLWithPath: NSString(string: cache).expandingTildeInPath).appendingPathComponent("huggingface/token")
        } else { tokenPath = home.appendingPathComponent(".cache/huggingface/token") }
        if let contents = try? String(contentsOf: tokenPath, encoding: .utf8), let token = clean(contents) { return token }
        throw LoomError("Sign in to Hugging Face with hf auth login, or provide HF_TOKEN, before publishing.")
    }

    static func publish(package: URL, repository: String, token: String, session: URLSession,
                        progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> URL {
        let components = repository.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 2, components.allSatisfy({ component in
            !component.isEmpty && component.count <= 96 && !component.hasPrefix(".") && !component.hasPrefix("-") &&
            !component.hasSuffix(".") && !component.hasSuffix("-") && !component.contains("..") && !component.contains("--") &&
            component.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || "._-".contains($0)) }
        }) else { throw LoomError("Use a Hugging Face repository name such as your-account/Texture-Loom.") }
        let resolved = package.standardizedFileURL.resolvingSymlinksInPath()
        guard package.isFileURL, resolved.pathExtension == "textureloom",
              (try resolved.resourceValues(forKeys: [.isRegularFileKey])).isRegularFile == true else {
            throw LoomError("Export a Texture Loom package before publishing it.")
        }
        try Task.checkCancellation()
        progress(.init(type: "publish", message: "Checking your Hugging Face sign-in…"))
        let client = HubClient(session: session, token: token)
        _ = try await client.json(path: "/api/whoami-v2", method: "GET")
        let create = try await client.request(path: "/api/repos/create", method: "POST", body: ["name": String(components[1]),
            "organization": String(components[0]), "type": "model"])
        if create.response.statusCode == 409 {
            _ = try await client.json(path: "/api/models/" + repository, method: "GET")
        } else if [401, 402, 403].contains(create.response.statusCode) {
            // A fine-grained token may write an existing repository without create permission.
            _ = try await client.json(path: "/api/models/" + repository, method: "GET")
        } else { try client.requireSuccess(create.response, operation: "create the repository") }

        let card = Data("# Texture Loom\n\nA complete classifier and material-model package created with Loom Factory. Download TextureLoom.textureloom for Texture Studio build integration.\n".utf8)
        progress(.init(type: "publish", message: "Checking the exported package bytes…"))
        let task = Task.detached(priority: .userInitiated) { try HubFile(path: "TextureLoom.textureloom", source: .file(resolved)) }
        let archive = try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
        let files = [archive, try HubFile(path: "README.md", source: .bytes(card))]
        let preupload = try await client.json(path: "/api/models/" + repository + "/preupload/main", method: "POST",
            body: ["files": files.map { ["path": $0.path, "sample": $0.sample.base64EncodedString(), "size": $0.size] as [String: Any] }])
        guard let entries = preupload["files"] as? [[String: Any]], entries.count == files.count else {
            throw LoomError("Hugging Face returned an incomplete upload inventory.")
        }
        var modes: [String: String] = [:]
        for entry in entries {
            guard let path = entry["path"] as? String, files.contains(where: { $0.path == path }), modes[path] == nil,
                  let mode = entry["uploadMode"] as? String, ["regular", "lfs"].contains(mode),
                  entry["shouldIgnore"] as? Bool != true else {
                throw LoomError("Hugging Face cannot upload a required package or model-card file with this repository's current rules.")
            }
            modes[path] = mode
        }
        progress(.init(type: "publish", message: "Uploading the Texture Loom package…", progress: 0.05))
        for file in files where modes[file.path] == "lfs" {
            let batch = try await client.json(path: "/" + repository + ".git/info/lfs/objects/batch", method: "POST",
                body: ["operation": "upload", "transfers": ["basic", "multipart"], "hash_algo": "sha256",
                    "ref": ["name": "main"], "objects": [["oid": file.sha256, "size": file.size]]],
                contentType: "application/vnd.git-lfs+json")
            guard let objects = batch["objects"] as? [[String: Any]], objects.count == 1,
                  let object = objects.first, object["oid"] as? String == file.sha256,
                  (object["size"] as? NSNumber)?.uint64Value == file.size, object["error"] == nil else {
                throw LoomError("Hugging Face could not prepare the large-file upload.")
            }
            if let actions = object["actions"] as? [String: Any] {
                guard let upload = actions["upload"] as? [String: Any] else { throw LoomError("The large-file upload instructions are incomplete.") }
                try await client.upload(file: file, action: upload, progress: progress)
                if let verify = actions["verify"] as? [String: Any] {
                    let url = try client.actionURL(verify)
                    let headers = (verify["header"] as? [String: String]) ?? [:]
                    let result = try await client.request(url: url, method: "POST", body: ["oid": file.sha256, "size": file.size],
                        headers: headers, authenticated: url.host == "huggingface.co")
                    try client.requireSuccess(result.response, operation: "verify the large-file upload")
                }
            }
        }
        try Task.checkCancellation()
        progress(.init(type: "publish", message: "Checking package integrity before publishing…", progress: 0.85))
        try archive.verifyUnchanged()
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("loom-hub-commit-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary) }
        try writeCommit(files: files, modes: modes, to: temporary)
        try archive.verifyUnchanged()
        progress(.init(type: "publish", message: "Saving the package and model card on Hugging Face…", progress: 0.9))
        var request = client.urlRequest(url: URL(string: "https://huggingface.co/api/models/" + repository + "/commit/main")!, method: "POST", authenticated: true)
        request.setValue("application/x-ndjson", forHTTPHeaderField: "Content-Type")
        let committed = try await client.upload(request: request, from: temporary)
        try client.requireSuccess(committed.response, operation: "publish the package commit")
        guard let result = try? JSONSerialization.jsonObject(with: committed.data) as? [String: Any],
              let oid = result["commitOid"] as? String, !oid.isEmpty, result["commitUrl"] as? String != nil else {
            throw LoomError("Hugging Face accepted the request but did not return a confirmed commit. Check the repository before retrying.")
        }
        progress(.init(type: "completed", message: "Texture Loom published", progress: 1))
        return URL(string: "https://huggingface.co/" + repository)!
    }

    private static func writeCommit(files: [HubFile], modes: [String: String], to destination: URL) throws {
        guard FileManager.default.createFile(atPath: destination.path, contents: nil) else { throw LoomError("The upload payload could not be prepared.") }
        let output = try FileHandle(forWritingTo: destination); defer { try? output.close() }
        func line(_ value: [String: Any]) throws {
            var bytes = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]); bytes.append(10)
            try output.write(contentsOf: bytes)
        }
        try line(["key": "header", "value": ["summary": "Publish Texture Loom package", "description": "Package and model card exported by Loom Factory."]])
        for file in files {
            try Task.checkCancellation()
            if modes[file.path] == "lfs" {
                try line(["key": "lfsFile", "value": ["path": file.path, "algo": "sha256", "oid": file.sha256, "size": file.size]])
            } else {
                // Base64 is streamed in multiples of three bytes: even a server-selected
                // regular archive does not require loading its whole contents into RAM.
                let path = String(data: try JSONSerialization.data(withJSONObject: file.path, options: [.fragmentsAllowed]), encoding: .utf8)!
                try output.write(contentsOf: Data(("{\"key\":\"file\",\"value\":{\"path\":" + path + ",\"encoding\":\"base64\",\"content\":\"").utf8))
                switch file.source {
                case .file(let source):
                    let input = try FileHandle(forReadingFrom: source); defer { try? input.close() }
                    var remainder = Data()
                    while try autoreleasepool(invoking: { () throws -> Bool in
                        try Task.checkCancellation()
                        guard let bytes = try input.read(upToCount: 3 * 1_048_576), !bytes.isEmpty else { return false }
                        remainder.append(bytes)
                        let count = remainder.count - remainder.count % 3
                        if count > 0 {
                            try output.write(contentsOf: Data(remainder.prefix(count).base64EncodedString().utf8))
                            remainder.removeFirst(count)
                        }
                        return true
                    }) {}
                    if !remainder.isEmpty { try output.write(contentsOf: Data(remainder.base64EncodedString().utf8)) }
                case .bytes(let bytes): try output.write(contentsOf: Data(bytes.base64EncodedString().utf8))
                }
                try output.write(contentsOf: Data("\"}}\n".utf8))
            }
        }
        try output.synchronize()
    }
}

private struct HubFile: Sendable {
    enum Source: Sendable { case file(URL), bytes(Data) }
    let path: String
    let source: Source
    let size: UInt64
    let sha256: String
    let sample: Data
    private let modified: Date?
    init(path: String, source: Source) throws {
        self.path = path; self.source = source
        switch source {
        case .bytes(let bytes):
            size = UInt64(bytes.count); sample = bytes.prefix(512); modified = nil
            sha256 = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        case .file(let file):
            let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
            size = (attributes[.size] as! NSNumber).uint64Value; modified = attributes[.modificationDate] as? Date
            let input = try FileHandle(forReadingFrom: file); defer { try? input.close() }
            sample = try input.read(upToCount: 512) ?? Data(); try input.seek(toOffset: 0)
            sha256 = try NativeFileHash.sha256(file) { try Task.checkCancellation() }
        }
    }
    func verifyUnchanged() throws {
        if case .file(let file) = source {
            let current = try FileManager.default.attributesOfItem(atPath: file.path)
            guard (current[.size] as? NSNumber)?.uint64Value == size, current[.modificationDate] as? Date == modified,
                  try HubFile(path: path, source: source).sha256 == sha256 else {
                throw LoomError("The exported package changed during upload. Export it again before publishing.")
            }
        }
    }
}

private struct HubClient: Sendable {
    let session: URLSession
    let token: String
    struct Result: Sendable { let data: Data; let response: HTTPURLResponse }
    func urlRequest(url: URL, method: String, authenticated: Bool) -> URLRequest {
        var request = URLRequest(url: url); request.httpMethod = method
        request.setValue("LoomFactory/0.2.0 (native Swift)", forHTTPHeaderField: "User-Agent")
        if authenticated { request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization") }
        return request
    }
    func request(path: String, method: String, body: [String: Any]? = nil,
                 contentType: String = "application/json") async throws -> Result {
        try await request(url: URL(string: "https://huggingface.co" + path)!, method: method, body: body,
            headers: ["Accept": contentType, "Content-Type": contentType], authenticated: true)
    }
    func request(url: URL, method: String, body: [String: Any]?, headers: [String: String], authenticated: Bool) async throws -> Result {
        try Task.checkCancellation()
        var request = urlRequest(url: url, method: method, authenticated: authenticated)
        for (key, value) in headers { request.setValue(value, forHTTPHeaderField: key) }
        if let body {
            request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
            if request.value(forHTTPHeaderField: "Content-Type") == nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        }
        do {
            let (data, response) = try await session.data(for: request)
            try Task.checkCancellation()
            guard let response = response as? HTTPURLResponse else { throw LoomError("Hugging Face returned an invalid response.") }
            return Result(data: data, response: response)
        } catch {
            if Task.isCancelled || (error as? URLError)?.code == .cancelled { throw CancellationError() }
            if let error = error as? LoomError { throw error }
            throw LoomError("The Hugging Face request could not finish. Check your connection, then try again.")
        }
    }
    func json(path: String, method: String, body: [String: Any]? = nil,
              contentType: String = "application/json") async throws -> [String: Any] {
        let result = try await request(path: path, method: method, body: body, contentType: contentType)
        try requireSuccess(result.response, operation: "prepare the package upload")
        guard let json = try? JSONSerialization.jsonObject(with: result.data) as? [String: Any] else {
            throw LoomError("Hugging Face returned invalid upload information.")
        }
        return json
    }
    func requireSuccess(_ response: HTTPURLResponse, operation: String) throws {
        guard (200..<300).contains(response.statusCode) else {
            if [401, 403].contains(response.statusCode) { throw LoomError("Hugging Face could not \(operation). Check your sign-in and write permission for this repository.") }
            throw LoomError("Hugging Face could not \(operation) (HTTP \(response.statusCode)).")
        }
    }
    func actionURL(_ action: [String: Any]) throws -> URL {
        guard let address = action["href"] as? String, let url = URL(string: address), url.scheme == "https",
              url.host != nil, url.user == nil, url.password == nil else { throw LoomError("Hugging Face returned invalid upload instructions.") }
        return url
    }
    func upload(request: URLRequest, from file: URL, delegate: HubUploadProgress? = nil) async throws -> Result {
        do {
            let (data, response) = try await session.upload(for: request, fromFile: file, delegate: delegate)
            try Task.checkCancellation()
            guard let response = response as? HTTPURLResponse else { throw LoomError("Hugging Face returned an invalid upload response.") }
            return Result(data: data, response: response)
        } catch {
            if Task.isCancelled || (error as? URLError)?.code == .cancelled { throw CancellationError() }
            if let error = error as? LoomError { throw error }
            throw LoomError("The Hugging Face upload could not finish. Check your connection, then try again.")
        }
    }
    func upload(file: HubFile, action: [String: Any], progress: @escaping @Sendable (TrainingEvent) -> Void) async throws {
        let url = try actionURL(action)
        var headers: [String: String] = [:]
        for (key, value) in (action["header"] as? [String: Any]) ?? [:] {
            if let text = value as? String { headers[key] = text }
            else if key == "chunk_size", let number = value as? NSNumber { headers[key] = number.stringValue }
            else { throw LoomError("Hugging Face returned an invalid upload header.") }
        }
        let source: URL
        let temporaryBytes = FileManager.default.temporaryDirectory.appendingPathComponent("loom-hub-body-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporaryBytes) }
        switch file.source {
        case .file(let file): source = file
        case .bytes(let bytes): try bytes.write(to: temporaryBytes, options: .atomic); source = temporaryBytes
        }
        if let rawSize = headers["chunk_size"] {
            guard let chunkSize = UInt64(rawSize), chunkSize > 0, file.size > 0 else { throw LoomError("Hugging Face returned an invalid multipart chunk size.") }
            let count = Int(1 + (file.size - 1) / chunkSize)
            let parts = headers.compactMap { key, value -> (Int, URL)? in
                guard let part = Int(key), part > 0, let address = URL(string: value), address.scheme == "https", address.host != nil,
                      address.user == nil, address.password == nil else { return nil }
                return (part, address)
            }.sorted { $0.0 < $1.0 }
            guard parts.count == count, parts.map(\.0) == Array(1...count) else { throw LoomError("Hugging Face returned an incomplete multipart upload.") }
            var completed: [[String: Any]] = []
            for (part, partURL) in parts {
                try Task.checkCancellation()
                let offset = UInt64(part - 1) * chunkSize, length = min(chunkSize, file.size - offset)
                let slice = FileManager.default.temporaryDirectory.appendingPathComponent("loom-hub-part-" + UUID().uuidString)
                defer { try? FileManager.default.removeItem(at: slice) }
                try Self.copy(source: source, offset: offset, length: length, to: slice)
                let request = urlRequest(url: partURL, method: "PUT", authenticated: false)
                let result = try await upload(request: request, from: slice,
                    delegate: HubUploadProgress(offset: offset, total: file.size, progress: progress))
                try requireSuccess(result.response, operation: "upload a package part")
                guard let etag = result.response.value(forHTTPHeaderField: "ETag"), !etag.isEmpty else { throw LoomError("The package part upload did not return its receipt.") }
                completed.append(["partNumber": part, "etag": etag])
            }
            let result = try await request(url: url, method: "POST", body: ["oid": file.sha256, "parts": completed],
                headers: ["Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"], authenticated: false)
            try requireSuccess(result.response, operation: "complete the multipart upload")
        } else {
            var request = urlRequest(url: url, method: "PUT", authenticated: false)
            for (key, value) in headers { request.setValue(value, forHTTPHeaderField: key) }
            let result = try await upload(request: request, from: source, delegate: HubUploadProgress(offset: 0, total: file.size, progress: progress))
            try requireSuccess(result.response, operation: "upload the package")
        }
    }
    private static func copy(source: URL, offset: UInt64, length: UInt64, to destination: URL) throws {
        let input = try FileHandle(forReadingFrom: source); defer { try? input.close() }
        try input.seek(toOffset: offset)
        guard FileManager.default.createFile(atPath: destination.path, contents: nil) else { throw LoomError("A package upload part could not be prepared.") }
        let output = try FileHandle(forWritingTo: destination); defer { try? output.close() }
        var remaining = length
        while remaining > 0 {
            try autoreleasepool {
                try Task.checkCancellation()
                guard let bytes = try input.read(upToCount: Int(min(remaining, 1_048_576))), !bytes.isEmpty else {
                    throw LoomError("The package changed while its upload was being prepared.")
                }
                try output.write(contentsOf: bytes); remaining -= UInt64(bytes.count)
            }
        }
    }
}

private final class HubRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        guard request.url?.scheme == "https" else { completionHandler(nil); return }
        var redirected = request
        if request.url?.host != response.url?.host { redirected.setValue(nil, forHTTPHeaderField: "Authorization") }
        completionHandler(redirected)
    }
}
private final class HubUploadProgress: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    let offset: UInt64
    let total: UInt64
    let progress: @Sendable (TrainingEvent) -> Void
    private let lock = NSLock()
    private var lastEmission: ContinuousClock.Instant?
    init(offset: UInt64, total: UInt64, progress: @escaping @Sendable (TrainingEvent) -> Void) {
        self.offset = offset; self.total = total; self.progress = progress
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        let sent = min(total, offset + UInt64(max(0, totalBytesSent)))
        let now = ContinuousClock.now
        let emit = lock.withLock {
            guard lastEmission == nil || now - lastEmission! >= .milliseconds(100) || totalBytesSent == totalBytesExpectedToSend else { return false }
            lastEmission = now; return true
        }
        guard emit else { return }
        progress(.init(type: "publish", message: "Uploaded \(sent.formatted()) of \(total.formatted()) package bytes",
            progress: 0.05 + 0.8 * Double(sent) / Double(max(1, total))))
    }
}
