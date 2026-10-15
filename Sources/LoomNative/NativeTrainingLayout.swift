import Foundation

/// Calibration uses independent sessions and never applies optimizer updates.
public struct NativeTrainingLayoutSelection: Sendable {
    public let channelsLast: Bool
    public let nchwSeconds: Double
    public let nhwcSeconds: Double
    public let gradientsAgree: Bool
    public let calibrationSeconds: Double
    public let fallbackReason: String?
}

enum NativeTrainingLayout {
    /// Compare full tensors, rather than just norms, so permutations and sign
    /// errors cannot pass. Large Float32 convolution reductions differ by order;
    /// bound both the largest error and the relative energy of all errors.
    static func gradientsAgree(reference: [String: NativeTensor], candidate: [String: NativeTensor]) -> Bool {
        guard reference.keys.sorted() == candidate.keys.sorted() else { return false }
        for (name, expected) in reference {
            guard let actual = candidate[name], actual.shape == expected.shape,
                  actual.values.count == expected.values.count else { return false }
            var scale = 0.0, maximumError = 0.0, errorEnergy = 0.0, referenceEnergy = 0.0
            for (a, b) in zip(actual.values, expected.values) {
                guard a.isFinite, b.isFinite else { return false }
                let error = Double(a) - Double(b)
                scale = max(scale, abs(Double(b)))
                maximumError = max(maximumError, abs(error))
                errorEnergy += error * error
                referenceEnergy += Double(b) * Double(b)
            }
            if referenceEnergy == 0 {
                guard errorEnergy == 0 else { return false }
            } else {
                guard maximumError <= max(2e-6, scale * 0.002),
                      errorEnergy <= referenceEnergy * 1e-6 else { return false }
            }
        }
        return true
    }

    static func select(architecture: NativeArchitecture, options: NativeTrainingOptions,
                       adapter: NativeAdapterConfiguration?,
                       input: [Float], target: [Float], batch: Int, height: Int, width: Int,
                       weights: () throws -> [String: NativeTensor],
                       adapters: () throws -> [String: NativeTensor]?,
                       cancellation: NativeCancellation) throws -> NativeTrainingLayoutSelection {
        let started = ProcessInfo.processInfo.systemUptime
        struct Measurement {
            let seconds: Double
            let loss: Double
            let gradients: [String: NativeTensor]
        }
        var nhwcSeconds = 0.0, nchwSeconds = 0.0
        do {
            let initial = try weights(), adapterWeights = try adapters()
            func measure(channelsLast: Bool) throws -> Measurement {
                try autoreleasepool {
                    let probe = try NativeTrainingSession(architecture: architecture, options: options, seed: 0,
                        weights: initial, adapter: adapter, adapterWeights: adapterWeights,
                        cancellation: cancellation, channelsLast: channelsLast)
                    var loss = 0.0
                    // Exclude graph compilation. These visits use the exact
                    // starting weights and input of this session, never a
                    // different checkpoint's cached numerical agreement.
                    for _ in 0..<2 {
                        probe.resetAccumulatedGradients()
                        loss = try probe.train(input: input, target: target, batchSize: batch, height: height, width: width)
                    }
                    return Measurement(seconds: probe.lastExecutionMeasurements!.encodingSeconds,
                        loss: loss, gradients: try probe.accumulatedGradients())
                }
            }
            let nhwc = try measure(channelsLast: true)
            nhwcSeconds = nhwc.seconds
            try cancellation.check()
            let nchw = try measure(channelsLast: false)
            nchwSeconds = nchw.seconds
            try cancellation.check()
            let agrees = abs(nchw.loss - nhwc.loss) <= max(2e-6, abs(nhwc.loss) * 1e-5)
                && gradientsAgree(reference: nhwc.gradients, candidate: nchw.gradients)
            return NativeTrainingLayoutSelection(
                channelsLast: !(agrees && nchw.seconds < nhwc.seconds * 0.97),
                nchwSeconds: nchw.seconds, nhwcSeconds: nhwc.seconds, gradientsAgree: agrees,
                calibrationSeconds: ProcessInfo.processInfo.systemUptime - started,
                fallbackReason: agrees ? nil : "The alternative layout exceeded numerical agreement limits.")
        } catch {
            // Optional profiling must not abort viable training. Cancellation
            // remains immediate; otherwise the normal NHWC train reports any
            // error that also affects the established path.
            try cancellation.check()
            if error is CancellationError { throw error }
            return NativeTrainingLayoutSelection(channelsLast: true, nchwSeconds: nchwSeconds,
                nhwcSeconds: nhwcSeconds, gradientsAgree: false,
                calibrationSeconds: ProcessInfo.processInfo.systemUptime - started,
                fallbackReason: "GPU layout comparison unavailable: \(error.localizedDescription)")
        }
    }
}
