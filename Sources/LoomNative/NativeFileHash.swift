import CryptoKit
import Foundation

/// Hash original bytes with bounded temporary storage even when called by a
/// long-lived training worker whose outer autorelease pool has not drained.
public enum NativeFileHash {
    public static func sha256(_ url: URL, checkCancellation: () throws -> Void = {}) throws -> String {
        try checkCancellation()
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while try autoreleasepool(invoking: { () throws -> Bool in
            try checkCancellation()
            guard let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty else { return false }
            digest.update(data: chunk)
            return true
        }) {}
        try checkCancellation()
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
