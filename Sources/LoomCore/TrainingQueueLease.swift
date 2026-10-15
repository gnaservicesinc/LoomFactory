import Darwin
import Foundation

/// Held for the entire GPU operation. The OS releases it when a process exits.
/// A queue owns the outer lease and may run its own serial project operations.
public final class TrainingQueueLease: @unchecked Sendable {
    private let descriptor: Int32
    public init(projectFolder: URL, controlFolder: URL) throws {
        let fd = try Self.openLease(projectFolder)
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            close(fd)
            throw LoomError("Another model operation is already using this source project.")
        }
        try Self.writeReceipt(fd, controlFolder: controlFolder)
        descriptor = fd
    }
    private init(lockedDescriptor: Int32) { descriptor = lockedDescriptor }
    private static func writeReceipt(_ fd: Int32, controlFolder: URL) throws {
        do {
            let data = try JSONSerialization.data(withJSONObject: ["pid": getpid(), "control_folder": controlFolder.path])
            guard ftruncate(fd, 0) == 0 else { throw LoomError("Cannot update the model operation lease.") }
            let count = data.withUnsafeBytes { Darwin.write(fd, $0.baseAddress!, $0.count) }
            guard count == data.count, fsync(fd) == 0 else { throw LoomError("Cannot save the model operation lease.") }
        } catch { close(fd); throw error }
    }
    deinit { _ = flock(descriptor, LOCK_UN); close(descriptor) }

    private static func openLease(_ folder: URL) throws -> Int32 {
        let fd = open(folder.appendingPathComponent("training-queue.lock").path,
                      O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard fd >= 0 else { throw LoomError("Cannot open the model operation lease.") }
        return fd
    }

    static func requireAvailable(projectFolder: URL) throws -> TrainingQueueLease? {
        let fd = try openLease(projectFolder)
        if flock(fd, LOCK_EX | LOCK_NB) == 0 {
            try writeReceipt(fd, controlFolder: projectFolder)
            return TrainingQueueLease(lockedDescriptor: fd)
        }
        defer { close(fd) }
        let value = (try? Data(contentsOf: projectFolder.appendingPathComponent("training-queue.lock")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        if value?["pid"] as? Int32 == getpid() { return nil }
        let control = value?["control_folder"] as? String ?? projectFolder.path
        throw LoomError("Another model operation is using this project. For an overnight queue, create its STOP file in \(control) and wait for the checkpoint before starting another run.")
    }
}
