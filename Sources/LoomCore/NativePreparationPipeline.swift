import Foundation
import LoomNative

/// Prepares a bounded number of upcoming batches while the serial training
/// owner works on the GPU. Results and errors are consumed in submission order,
/// even when workers finish out of order. Work must honor run cancellation.
final class NativePreparationPipeline<Value: Sendable>: @unchecked Sendable {
    private final class Entry: @unchecked Sendable {
        // All access is protected by the pipeline's condition lock.
        var result: Result<Value, Error>?
    }

    private let cancellation: NativeCancellation
    private let worker = DispatchQueue(label: "loom-factory.image-preparation", qos: .userInitiated, attributes: .concurrent)
    private let condition = NSCondition()
    let capacity: Int
    private var pending: [Entry] = []
    private var working = 0
    private var closed = false

    init(cancellation: NativeCancellation, capacity: Int = 1) {
        self.cancellation = cancellation; self.capacity = min(2, max(1, capacity))
    }

    /// Starts preparation without waiting for it. The capacity includes running
    /// and finished batches, so neither work nor decoded results can queue up.
    func submit(_ work: @escaping @Sendable () throws -> Value) throws {
        try cancellation.check()
        condition.lock()
        guard !closed, pending.count < capacity else {
            condition.unlock()
            throw LoomError("Image preparation has reached its pending batch limit, or has been closed.")
        }
        let entry = Entry()
        pending.append(entry); working += 1
        condition.unlock()
        worker.async {
            let prepared = Result { try autoreleasepool {
                try self.cancellation.check()
                return try work()
            } }
            self.condition.lock()
            entry.result = prepared; self.working -= 1
            self.condition.broadcast()
            self.condition.unlock()
        }
    }

    /// Waits for the submitted batch, preserving its preparation error. Stop is
    /// checked while waiting, so callers can unwind before slow work finishes.
    func take() throws -> Value {
        try cancellation.check()
        condition.lock()
        defer { condition.unlock() }
        while true {
            try cancellation.check()
            guard !closed, let first = pending.first else { throw LoomError("No prepared image batch is pending.") }
            if let prepared = first.result {
                pending.removeFirst()
                return try prepared.get()
            }
            _ = condition.wait(until: Date(timeIntervalSinceNow: 0.05))
        }
    }

    /// Joins active preparation and releases the pending batch before the
    /// owner's decoded-pixel cache is cleared. It does not cancel the shared
    /// run flag. Call only from the training owner, never from a work closure.
    func close() {
        condition.lock()
        closed = true
        condition.broadcast()
        while working > 0 { condition.wait() }
        pending.removeAll()
        condition.unlock()
    }
}
