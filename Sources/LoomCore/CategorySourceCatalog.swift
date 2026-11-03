import Foundation

/// Provider identities are only hints until the published bytes have been checked.
enum CategorySourceProvider: String, Codable, Sendable {
    case polyHaven, ambientCG
}

struct CategorySourceIdentity: Sendable {
    let provider: CategorySourceProvider
    let asset: String
    let tier: Int
    let package: String?
    var pageURL: URL {
        URL(string: provider == .polyHaven ? "https://polyhaven.com/a/\(asset)" : "https://ambientcg.com/view?id=\(asset)")!
    }

    static func recognize(_ url: URL) -> Self? {
        let name = url.lastPathComponent
        if let match = name.range(of: #"_(diff|coll?[0-9]+|disp|rough|nor_gl|nor_dx|bump|ao)_(1|2|4|8|16)k\.(png|jpg|exr)$"#, options: .regularExpression) {
            let asset = String(name[..<match.lowerBound])
            let suffix = name[match].split(separator: "_").last!.split(separator: "k").first!
            guard asset.range(of: #"^[a-z0-9_]+$"#, options: .regularExpression) != nil,
                  let tier = Int(suffix) else { return nil }
            return .init(provider: .polyHaven, asset: asset, tier: tier, package: nil)
        }
        if let match = name.range(of: #"_(1|2|4|8|16)K-(PNG|JPG)_[A-Za-z0-9]+\.(png|jpg)$"#, options: .regularExpression) {
            let asset = String(name[..<match.lowerBound])
            let parts = name[match].dropFirst().split(separator: "_")
            guard asset.range(of: #"^[A-Za-z][A-Za-z0-9]+$"#, options: .regularExpression) != nil,
                  let tier = Int(parts[0].split(separator: "K")[0]) else { return nil }
            return .init(provider: .ambientCG, asset: asset, tier: tier, package: "\(asset)_\(parts[0]).zip")
        }
        return nil
    }
}

enum CategorySourceCatalog {
    struct Published: Sendable {
        let url: URL
        let size: Int64
        let md5: String?
    }
    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 30 * 60
        config.httpAdditionalHeaders = ["User-Agent": "LoomFactory/1.0 (category source recovery)"]
        return URLSession(configuration: config)
    }
    static func catalog(_ identity: CategorySourceIdentity, session: URLSession, allowMissing: Bool = false) async throws -> Data {
        let url = URL(string: identity.provider == .polyHaven ? "https://api.polyhaven.com/files/\(identity.asset)"
            : "https://ambientcg.com/api/v3/assets?id=\(identity.asset)&include=downloads")!
        let (data, response) = try await session.data(from: url)
        if allowMissing, (response as? HTTPURLResponse)?.statusCode == 404 { return Data("{}".utf8) }
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw LoomError("Could not verify \(identity.asset) with its provider. No source files were deleted.")
        }
        return data
    }
    static func polyFile(_ data: Data, filename: String) throws -> Published? {
        func find(_ value: Any) -> Published? {
            if let object = value as? [String: Any] {
                if let raw = object["url"] as? String, let url = URL(string: raw),
                   url.scheme == "https", url.host == "dl.polyhaven.org", url.lastPathComponent == filename,
                   let size = object["size"] as? Int64, size > 0,
                   let md5 = object["md5"] as? String,
                   md5.range(of: #"^[a-fA-F0-9]{32}$"#, options: .regularExpression) != nil {
                    return .init(url: url, size: size, md5: md5.lowercased())
                }
                for child in object.values { if let result = find(child) { return result } }
            }
            return nil
        }
        return find(try JSONSerialization.jsonObject(with: data))
    }
    static func ambientPackage(_ data: Data, identity: CategorySourceIdentity) throws -> Published? {
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let assets = json?["assets"] as? [[String: Any]]
        let asset = assets?.first { ($0["id"] as? String) == identity.asset }
        let downloads = asset?["downloads"] as? [[String: Any]] ?? []
        for file in downloads {
            guard let raw = file["url"] as? String, let url = URL(string: raw),
                  url.scheme == "https", url.host?.lowercased() == "ambientcg.com", url.path == "/get",
                  URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "file" })?.value == identity.package,
                  let size = file["size"] as? Int64, size > 0 else { continue }
            return .init(url: url, size: size, md5: nil)
        }
        return nil
    }

    struct ZIPEntry: Sendable, Equatable {
        let name: String
        let size: Int64
        let crc32: UInt32
    }
    /// Check only the published ZIP directory, without downloading a duplicate
    /// texture set just to decide whether the local originals can be removed.
    static func zipEntries(_ file: Published, session: URLSession) async throws -> [ZIPEntry] {
        let tailStart = max(0, file.size - 65_557)
        let tail = try await range(file.url, start: tailStart, end: file.size - 1, total: file.size, session: session)
        guard tail.count >= 22,
              let end = stride(from: tail.count - 22, through: 0, by: -1).first(where: {
                  tail.u32($0) == 0x06054b50 && $0 + 22 + Int(tail.u16($0 + 20)) == tail.count
              }), tail.u16(end + 4) == 0, tail.u16(end + 6) == 0,
              tail.u16(end + 8) == tail.u16(end + 10) else { throw LoomError("Unsupported provider ZIP directory; its local files were kept.") }
        let size = Int64(tail.u32(end + 12)), offset = Int64(tail.u32(end + 16))
        guard size > 0, size <= 16_777_216, offset + size <= file.size,
              tail.u16(end + 10) != UInt16.max else { throw LoomError("Unsupported provider ZIP directory; its local files were kept.") }
        let data: Data
        if offset >= tailStart, offset + size <= file.size {
            data = tail.subdata(in: Int(offset - tailStart)..<Int(offset - tailStart + size))
        } else {
            data = try await range(file.url, start: offset, end: offset + size - 1, total: file.size, session: session)
        }
        return try parseZIPDirectory(data, count: Int(tail.u16(end + 10)))
    }
    static func parseZIPDirectory(_ data: Data, count: Int) throws -> [ZIPEntry] {
        var offset = 0, entries: [ZIPEntry] = []
        for _ in 0..<count {
            guard offset + 46 <= data.count, data.u32(offset) == 0x02014b50 else { throw LoomError("Invalid provider ZIP directory.") }
            let length = Int(data.u16(offset + 28)), extra = Int(data.u16(offset + 30)), comment = Int(data.u16(offset + 32))
            let next = offset + 46 + length + extra + comment
            guard next <= data.count, data.u16(offset + 8) & 1 == 0,
                  let name = String(data: data.subdata(in: offset + 46..<offset + 46 + length), encoding: .utf8),
                  !entries.contains(where: { $0.name == name }) else { throw LoomError("Unsupported provider ZIP entry.") }
            entries.append(.init(name: name, size: Int64(data.u32(offset + 24)), crc32: data.u32(offset + 16)))
            offset = next
        }
        guard offset == data.count else { throw LoomError("Invalid provider ZIP directory length.") }
        return entries
    }
    private static func range(_ url: URL, start: Int64, end: Int64, total: Int64, session: URLSession) async throws -> Data {
        var request = URLRequest(url: url)
        request.setValue("bytes=\(start)-\(end)", forHTTPHeaderField: "Range")
        let (temporary, response) = try await session.download(for: request)
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard let http = response as? HTTPURLResponse, http.statusCode == 206,
              http.value(forHTTPHeaderField: "Content-Range") == "bytes \(start)-\(end)/\(total)",
              try temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize == Int(end - start + 1) else {
            throw LoomError("Provider cannot verify this archive with a partial download; its local files were kept.")
        }
        return try Data(contentsOf: temporary)
    }
}

private extension Data {
    func u16(_ i: Int) -> UInt16 { UInt16(self[i]) | UInt16(self[i + 1]) << 8 }
    func u32(_ i: Int) -> UInt32 { UInt32(u16(i)) | UInt32(u16(i + 2)) << 16 }
}
