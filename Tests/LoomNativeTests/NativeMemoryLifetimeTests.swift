import Darwin
import Foundation
import Metal
import XCTest
@testable import LoomNative

final class NativeMemoryLifetimeTests: XCTestCase {
    /// A training run stays in one synchronous worker for many epochs. An outer
    /// caller pool would conceal GPU objects retained until that worker returns.
    func testRepeatedTrainingOperationsReleaseTransientGPUAllocations() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal GPU") }
        let architecture = try NativeArchitecture(mapKind: "depth", width: 12, blocks: 2,
                                                  identifier: NativeArchitecture.pyramidIdentifier)
        let session = try NativeTrainingSession(architecture: architecture,
            options: NativeTrainingOptions(weightDecay: 0, gradientClipNorm: 0), seed: 42)
        let size = 128
        let input = (0..<(3 * size * size)).map { Float(($0 * 7 + 3) % 101) / 100 }
        let target = Array(repeating: Float(0.8), count: size * size)

        func cycle() throws {
            let loss = try session.train(input: input, target: target, batchSize: 1, height: size, width: size)
            let norm = try session.applyGradients(learningRate: 0.0001)
            let validation = try session.evaluate(input: input, target: target, batchSize: 1, height: size, width: size)
            let prediction = try session.predict(input: input, batchSize: 1, height: size, width: size)
            let weights = try session.weights()
            let optimizer = try session.optimizerState()
            XCTAssertTrue(loss.isFinite && norm.isFinite && validation.isFinite)
            XCTAssertEqual(prediction.count, size * size)
            XCTAssertEqual(weights.count, architecture.parameterShapes.count)
            XCTAssertEqual(optimizer.count, 2 * architecture.parameterShapes.count)
        }

        // Compile every target variant and populate reusable framework caches
        // before testing growth. Deliberately never drain a caller-owned pool.
        for _ in 0..<12 { try cycle() }
        let warm = try MemorySnapshot(device: device)
        var peakMetal = warm.metal, peakFootprint = warm.footprint
        for _ in 0..<80 {
            try cycle()
            let current = try MemorySnapshot(device: device)
            peakMetal = max(peakMetal, current.metal)
            peakFootprint = max(peakFootprint, current.footprint)
            // Bound the damage when this regression returns in a local run.
            if current.metal > warm.metal + 512 * Self.mebibyte ||
                current.footprint > warm.footprint + 512 * Self.mebibyte { break }
        }
        let metalGrowth = peakMetal - warm.metal
        let footprintGrowth = peakFootprint - warm.footprint
        let detail = "After warmup: Metal +\(metalGrowth / Self.mebibyte) MiB; physical footprint +\(footprintGrowth / Self.mebibyte) MiB"
        XCTAssertLessThanOrEqual(metalGrowth, 64 * Self.mebibyte, detail)
        XCTAssertLessThanOrEqual(footprintGrowth, 96 * Self.mebibyte, detail)
    }

    private static let mebibyte: UInt64 = 1024 * 1024

    private struct MemorySnapshot {
        let metal: UInt64
        let footprint: UInt64

        init(device: MTLDevice) throws {
            metal = UInt64(device.currentAllocatedSize)
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let status = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                }
            }
            guard status == KERN_SUCCESS else {
                throw NSError(domain: "NativeMemoryLifetimeTests.task_info", code: Int(status))
            }
            footprint = info.phys_footprint
        }
    }
}
