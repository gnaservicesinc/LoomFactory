import Foundation
import LoomNative

/// Prepares one upcoming batch while the serial training owner works on the GPU.
/// Only one pending result is retained; callers consume it before submitting
/// another batch. Preparation must honor the supplied run cancellation flag.
final class NativePreparationPipeline<Value: Sendable>: @unchecked Sendable {
    private let cancellation: NativeCancellation
    private let worker = DispatchQueue(label: "loom-factory.image-preparation", qos: .userInitiated)
    private let condition = NSCondition()
    private var pending = false
    private var working = false
    private var closed = false
    private var result: Result<Value, Error>?

    init(cancellation: NativeCancellation) { self.cancellation = cancellation }

    /// Starts preparation without waiting for it. Submitting a second pending
    /// batch is an error rather than an unbounded queue of decoded images.
    func submit(_ work: @escaping @Sendable () throws -> Value) throws {
        try cancellation.check()
        condition.lock()
        guard !closed, !pending else {
            condition.unlock()
            throw LoomError("Image preparation already has a pending batch, or has been closed.")
        }
        pending = true; working = true
        condition.unlock()
        worker.async {
            let prepared = Result { try self.cancellation.check(); return try work() }
            self.condition.lock()
            self.result = prepared; self.working = false
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
        guard pending, !closed else { throw LoomError("No prepared image batch is pending.") }
        while working {
            _ = condition.wait(until: Date(timeIntervalSinceNow: 0.05))
            try cancellation.check()
        }
        try cancellation.check()
        guard let prepared = result else { throw LoomError("Image preparation finished without a result.") }
        pending = false; result = nil
        return try prepared.get()
    }

    /// Joins active preparation and releases the pending batch before the
    /// owner's decoded-pixel cache is cleared. It does not cancel the shared
    /// run flag. Call only from the training owner, never from a work closure.
    func close() {
        condition.lock()
        closed = true
        while working { condition.wait() }
        result = nil; pending = false
        condition.unlock()
    }
}
