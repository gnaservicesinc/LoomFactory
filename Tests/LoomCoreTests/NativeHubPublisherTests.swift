import CryptoKit
import Foundation
import XCTest
@testable import LoomCore

final class NativeHubPublisherTests: XCTestCase, @unchecked Sendable {
    func testNativeRegularCommitStreamsArchiveAndModelCardTogether() async throws {
        let package = try fixture(Data((0..<3_145_735).map { UInt8($0 % 251) }))
        defer { try? FileManager.default.removeItem(at: package) }
        let original = try Data(contentsOf: package)
        HubMockProtocol.script.configure([
            .init("GET", "/api/whoami-v2", json: ["name": "native-test"]),
            .init("POST", "/api/repos/create", json: ["url": "https://huggingface.co/native-test/Texture-Loom"]),
            .init("POST", "/api/models/native-test/Texture-Loom/preupload/main", json: ["files": [
                ["path": "TextureLoom.textureloom", "uploadMode": "regular", "shouldIgnore": false],
                ["path": "README.md", "uploadMode": "regular", "shouldIgnore": false]]]),
            .init("POST", "/api/models/native-test/Texture-Loom/commit/main", json: ["commitOid": "123abc", "commitUrl": "https://huggingface.co/native-test/Texture-Loom/commit/123abc"], check: { request, body in
                XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer test-only-token")
                XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/x-ndjson")
                let lines = try body.split(separator: 10).map { try JSONSerialization.jsonObject(with: Data($0)) as! [String: Any] }
                XCTAssertEqual(lines.count, 3); XCTAssertEqual(lines[0]["key"] as? String, "header")
                let archive = lines[1]["value"] as! [String: String]
                XCTAssertEqual(archive["path"], "TextureLoom.textureloom")
                XCTAssertEqual(Data(base64Encoded: archive["content"]!), original)
                let readme = lines[2]["value"] as! [String: String]
                XCTAssertEqual(readme["path"], "README.md")
                XCTAssertTrue(String(data: Data(base64Encoded: readme["content"]!)!, encoding: .utf8)!.contains("created with Loom Factory"))
            })
        ])
        let session = session(); defer { session.invalidateAndCancel() }
        let result = try await NativeHubPublisher.publish(package: package, repository: "native-test/Texture-Loom", token: "test-only-token", session: session, progress: { _ in })
        XCTAssertEqual(result.absoluteString, "https://huggingface.co/native-test/Texture-Loom")
        XCTAssertEqual(HubMockProtocol.script.remaining, 0)
        XCTAssertEqual(try Data(contentsOf: package), original)
    }

    func testNativeLFSBasicUploadUsesServerInstructionsWithoutForwardingHubToken() async throws {
        let bytes = Data("real archive bytes in a mock transport".utf8), package = try fixture(bytes)
        defer { try? FileManager.default.removeItem(at: package) }
        let sha = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        HubMockProtocol.script.configure([
            .init("GET", "/api/whoami-v2", json: ["name": "native-test"]),
            .init("POST", "/api/repos/create", status: 409, json: ["error": "Already exists"]),
            .init("GET", "/api/models/native-test/Texture-Loom", json: ["id": "native-test/Texture-Loom"]),
            preupload(),
            .init("POST", "/native-test/Texture-Loom.git/info/lfs/objects/batch", json: ["objects": [["oid": sha, "size": bytes.count,
                "actions": ["upload": ["href": "https://uploads.unit.test/object", "header": ["x-upload-contract": "signed-by-server"]],
                    "verify": ["href": "https://huggingface.co/verify-object"]]]]], check: { _, body in
                let request = try JSONSerialization.jsonObject(with: body) as! [String: Any]
                XCTAssertEqual(request["transfers"] as? [String], ["basic", "multipart"])
            }),
            .init("PUT", "/object", json: [:], check: { request, body in
                XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
                XCTAssertEqual(request.value(forHTTPHeaderField: "x-upload-contract"), "signed-by-server")
                XCTAssertEqual(body, bytes)
            }),
            .init("POST", "/verify-object", json: [:], check: { request, body in
                XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer test-only-token")
                let value = try JSONSerialization.jsonObject(with: body) as! [String: Any]
                XCTAssertEqual(value["oid"] as? String, sha)
            }),
            commit(sha: sha, size: bytes.count)
        ])
        let session = session(); defer { session.invalidateAndCancel() }
        _ = try await NativeHubPublisher.publish(package: package, repository: "native-test/Texture-Loom", token: "test-only-token", session: session, progress: { _ in })
        XCTAssertEqual(HubMockProtocol.script.remaining, 0)
    }

    func testNativeLFSMultipartUploadsExactSlicesAndReceiptPayload() async throws {
        let bytes = Data("abcdefghijk".utf8), package = try fixture(bytes)
        defer { try? FileManager.default.removeItem(at: package) }
        let sha = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        var steps: [HubMockStep] = [
            .init("GET", "/api/whoami-v2", json: ["name": "native-test"]),
            .init("POST", "/api/repos/create", json: ["url": "https://huggingface.co/native-test/Texture-Loom"]),
            preupload(),
            .init("POST", "/native-test/Texture-Loom.git/info/lfs/objects/batch", json: ["objects": [["oid": sha, "size": bytes.count,
                "actions": ["upload": ["href": "https://uploads.unit.test/complete", "header": ["chunk_size": 4,
                    "1": "https://uploads.unit.test/part1", "2": "https://uploads.unit.test/part2", "3": "https://uploads.unit.test/part3"]]]]]])
        ]
        for index in 0..<3 {
            let expected = Data(bytes[(index * 4)..<min(bytes.count, index * 4 + 4)])
            steps.append(.init("PUT", "/part\(index + 1)", headers: ["ETag": "part-\(index + 1)"], json: [:], check: { request, body in
                XCTAssertNil(request.value(forHTTPHeaderField: "Authorization")); XCTAssertEqual(body, expected)
            }))
        }
        steps.append(.init("POST", "/complete", json: [:], check: { request, body in
            XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
            let completion = try JSONSerialization.jsonObject(with: body) as! [String: Any]
            XCTAssertEqual(completion["oid"] as? String, sha)
            let parts = completion["parts"] as! [[String: Any]]
            XCTAssertEqual(parts.map { $0["partNumber"] as! Int }, [1, 2, 3])
            XCTAssertEqual(parts.map { $0["etag"] as! String }, ["part-1", "part-2", "part-3"])
        }))
        steps.append(commit(sha: sha, size: bytes.count))
        HubMockProtocol.script.configure(steps)
        let session = session(); defer { session.invalidateAndCancel() }
        _ = try await NativeHubPublisher.publish(package: package, repository: "native-test/Texture-Loom", token: "test-only-token", session: session, progress: { _ in })
        XCTAssertEqual(HubMockProtocol.script.remaining, 0)
    }

    func testTokenStorePrecedenceAndMissingSignInStayLocal() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("hub-auth-test-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        for path in ["explicit", "hf/token", "xdg/huggingface/token", ".cache/huggingface/token"] {
            let file = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data((path + "\n").utf8).write(to: file)
        }
        XCTAssertEqual(try NativeHubPublisher.authenticationToken(environment: ["HF_TOKEN": "env-test\n", "HF_TOKEN_PATH": root.appendingPathComponent("explicit").path], home: root), "env-test")
        XCTAssertEqual(try NativeHubPublisher.authenticationToken(environment: ["HF_TOKEN_PATH": root.appendingPathComponent("explicit").path, "HF_HOME": root.appendingPathComponent("hf").path], home: root), "explicit")
        XCTAssertEqual(try NativeHubPublisher.authenticationToken(environment: ["HF_HOME": root.appendingPathComponent("hf").path, "XDG_CACHE_HOME": root.appendingPathComponent("xdg").path], home: root), "hf/token")
        XCTAssertEqual(try NativeHubPublisher.authenticationToken(environment: ["XDG_CACHE_HOME": root.appendingPathComponent("xdg").path], home: root), "xdg/huggingface/token")
        XCTAssertEqual(try NativeHubPublisher.authenticationToken(environment: [:], home: root), ".cache/huggingface/token")
        XCTAssertThrowsError(try NativeHubPublisher.authenticationToken(environment: [:], home: root.appendingPathComponent("absent")))
    }

    func testDeniedUploadDoesNotEchoResponseOrCredential() async throws {
        let package = try fixture(Data("small archive".utf8)); defer { try? FileManager.default.removeItem(at: package) }
        HubMockProtocol.script.configure([.init("GET", "/api/whoami-v2", status: 401, json: ["error": "test-only-token must never appear in an error"] )])
        let session = session(); defer { session.invalidateAndCancel() }
        do {
            _ = try await NativeHubPublisher.publish(package: package, repository: "native-test/Texture-Loom", token: "test-only-token", session: session, progress: { _ in })
            XCTFail("Unauthorized requests must stop before any upload.")
        } catch {
            XCTAssertFalse(error.localizedDescription.contains("test-only-token"))
            XCTAssertTrue(error.localizedDescription.contains("sign-in"))
        }
        XCTAssertEqual(HubMockProtocol.script.remaining, 0)
    }

    private func fixture(_ bytes: Data) throws -> URL {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("native-hub-" + UUID().uuidString + ".textureloom")
        try bytes.write(to: file); return file
    }
    private func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral; configuration.protocolClasses = [HubMockProtocol.self]
        return URLSession(configuration: configuration)
    }
    private func preupload() -> HubMockStep {
        .init("POST", "/api/models/native-test/Texture-Loom/preupload/main", json: ["files": [
            ["path": "TextureLoom.textureloom", "uploadMode": "lfs", "shouldIgnore": false],
            ["path": "README.md", "uploadMode": "regular", "shouldIgnore": false]]])
    }
    private func commit(sha: String, size: Int) -> HubMockStep {
        .init("POST", "/api/models/native-test/Texture-Loom/commit/main", json: ["commitOid": "123abc", "commitUrl": "https://huggingface.co/native-test/Texture-Loom/commit/123abc"], check: { _, body in
            let lines = try body.split(separator: 10).map { try JSONSerialization.jsonObject(with: Data($0)) as! [String: Any] }
            XCTAssertEqual(lines[1]["key"] as? String, "lfsFile")
            let value = lines[1]["value"] as! [String: Any]
            XCTAssertEqual(value["oid"] as? String, sha); XCTAssertEqual(value["size"] as? Int, size)
            XCTAssertEqual(lines[2]["key"] as? String, "file")
        })
    }
}

private struct HubMockStep: @unchecked Sendable {
    let method: String
    let path: String
    let status: Int
    let headers: [String: String]
    let data: Data
    let check: @Sendable (URLRequest, Data) throws -> Void
    init(_ method: String, _ path: String, status: Int = 200, headers: [String: String] = [:], json: [String: Any],
         check: @escaping @Sendable (URLRequest, Data) throws -> Void = { _, _ in }) {
        self.method = method; self.path = path; self.status = status; self.headers = headers
        data = try! JSONSerialization.data(withJSONObject: json); self.check = check
    }
}
private final class HubMockScript: @unchecked Sendable {
    private let lock = NSLock()
    private var steps: [HubMockStep] = []
    var remaining: Int { lock.withLock { steps.count } }
    func configure(_ steps: [HubMockStep]) { lock.withLock { self.steps = steps } }
    func next() throws -> HubMockStep {
        try lock.withLock {
            guard !steps.isEmpty else { throw LoomError("Unexpected mock upload request.") }
            return steps.removeFirst()
        }
    }
}
private final class HubMockProtocol: URLProtocol, @unchecked Sendable {
    static let script = HubMockScript()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let step = try Self.script.next()
            XCTAssertEqual(request.httpMethod, step.method); XCTAssertEqual(request.url?.path, step.path)
            var body = request.httpBody ?? Data()
            if let stream = request.httpBodyStream {
                stream.open(); defer { stream.close() }
                var buffer = [UInt8](repeating: 0, count: 65536)
                while stream.hasBytesAvailable {
                    let count = stream.read(&buffer, maxLength: buffer.count)
                    if count > 0 { body.append(contentsOf: buffer.prefix(count)) }
                    else { break }
                }
            }
            try step.check(request, body)
            let response = HTTPURLResponse(url: request.url!, statusCode: step.status, httpVersion: "HTTP/1.1", headerFields: step.headers)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: step.data); client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}
