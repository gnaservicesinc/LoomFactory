import CryptoKit
import Darwin
import Foundation
import ImageIO
import LoomNative
import Metal

private struct ProbeArguments {
    var values: [String: String] = [:]
    var synthetic = false
    var prepareOnly = false
    init() throws {
        var arguments = Array(CommandLine.arguments.dropFirst())
        while !arguments.isEmpty {
            let argument = arguments.removeFirst()
            if argument == "--synthetic" { synthetic = true; continue }
            if argument == "--prepare-only" { prepareOnly = true; continue }
            guard argument.hasPrefix("--") else { throw NativeTrainingError.invalid("Unknown probe argument: \(argument)") }
            if let equal = argument.firstIndex(of: "=") {
                values[String(argument[..<equal].dropFirst(2))] = String(argument[argument.index(after: equal)...])
            } else {
                guard !arguments.isEmpty, !arguments[0].hasPrefix("--") else { throw NativeTrainingError.invalid("Missing value for \(argument)") }
                values[String(argument.dropFirst(2))] = arguments.removeFirst()
            }
        }
        let allowed: Set<String> = ["input", "target", "map", "size", "width", "blocks", "updates", "visits", "seed", "learning-rate", "optimizer", "output", "weights-out", "weights-in", "initial-weights-out", "gradients-out", "cancel-after-ms", "architecture", "layout"]
        guard Set(values.keys).isSubset(of: allowed), synthetic || (values["input"] != nil && values["target"] != nil) else {
            throw NativeTrainingError.invalid("Use --synthetic, or --input IMAGE --target IMAGE. Supported options: \(allowed.sorted().joined(separator: ", ")).")
        }
    }
    func integer(_ name: String, _ fallback: Int, range: ClosedRange<Int>) throws -> Int {
        guard let value = Int(values[name] ?? String(fallback)), range.contains(value) else { throw NativeTrainingError.invalid("Invalid --\(name)") }
        return value
    }
}

private func emit(_ event: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: event, options: [.sortedKeys]) {
        FileHandle.standardOutput.write(data); FileHandle.standardOutput.write(Data([10]))
    }
}

private func executionJSON(_ value: NativeGraphExecutionMeasurements) -> [String: Double] {
    ["feed_seconds": value.feedSeconds, "result_setup_seconds": value.resultSetupSeconds,
     "encoding_seconds": value.encodingSeconds, "wait_seconds": value.waitSeconds,
     "gpu_span_seconds": value.gpuSpanSeconds, "scalar_read_seconds": value.scalarReadSeconds,
     "total_seconds": value.totalSeconds,
     "feed_buffer_allocations": Double(value.feedBufferAllocationCount),
     "result_buffer_allocations": Double(value.resultBufferAllocationCount)]
}
private func memory() -> [String: Any] {
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    var result: [String: Any] = ["peak_resident_bytes": usage.ru_maxrss,
        "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory]
    // GPU allocations can grow while RSS remains small on unified memory.
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    if status == KERN_SUCCESS {
        result["physical_footprint_bytes"] = info.phys_footprint
        result["resident_bytes"] = info.resident_size
    }
    if let device = MTLCreateSystemDefaultDevice() { result["metal_allocated_bytes"] = device.currentAllocatedSize }
    return result
}
private func sysctlString(_ key: String) -> String {
    var size = 0
    guard sysctlbyname(key, nil, &size, nil, 0) == 0, size > 0 else { return "unavailable" }
    var bytes = [CChar](repeating: 0, count: size)
    guard sysctlbyname(key, &bytes, &size, nil, 0) == 0 else { return "unavailable" }
    return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}
private func hash(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var digest = SHA256()
    while let chunk = try file.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty { digest.update(data: chunk) }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
}
private func hash(_ weights: [String: NativeTensor]) -> String {
    var digest = SHA256()
    for name in weights.keys.sorted() {
        digest.update(data: Data(name.utf8))
        weights[name]!.values.withUnsafeBytes { digest.update(bufferPointer: $0) }
    }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
}

private func run() throws {
    let args = try ProbeArguments()
    guard args.values["layout"] == nil || ["nchw", "nhwc"].contains(args.values["layout"]!) else {
        throw NativeTrainingError.invalid("Choose --layout nchw or nhwc.")
    }
    let size = try args.integer("size", 1024, range: 1...8192)
    let width = try args.integer("width", 164, range: 1...512)
    let blocks = try args.integer("blocks", 4, range: 1...64)
    let updates = try args.integer("updates", 3, range: 1...100_000)
    let seed = try args.integer("seed", 42, range: 0...Int(Int32.max))
    let map = args.values["map"] ?? "depth"
    guard let rate = Double(args.values["learning-rate"] ?? "0.001"), rate.isFinite, rate > 0, rate <= 1 else { throw NativeTrainingError.invalid("Invalid probe learning rate") }
    let cancellation = NativeCancellation()
    signal(SIGTERM, SIG_IGN); signal(SIGINT, SIG_IGN)
    let terminate = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
    let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    terminate.setEventHandler { cancellation.cancel() }; interrupt.setEventHandler { cancellation.cancel() }
    terminate.resume(); interrupt.resume()
    defer { terminate.cancel(); interrupt.cancel() }
    if args.values["cancel-after-ms"] != nil {
        let delay = try args.integer("cancel-after-ms", 0, range: 1...Int(Int32.max))
        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(delay)) { cancellation.cancel() }
    }
    var report: [String: Any] = ["schema": "loom-native-execution-probe-v1", "backend": "native-metal-mpsgraph",
        "machine": sysctlString("machdep.cpu.brand_string"), "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "size": size, "width": width, "blocks": blocks, "map": map, "requested_updates": updates,
        "synthetic": args.synthetic, "seed": seed, "learning_rate": rate,
        "quality_note": "This measures execution and cancellation, not learned prediction quality."]
    let start = ProcessInfo.processInfo.systemUptime
    if args.prepareOnly {
        guard let inputPath = args.values["input"], let targetPath = args.values["target"], !args.synthetic else {
            throw NativeTrainingError.invalid("Preparation measurements require --input and --target.")
        }
        let inputURL = URL(fileURLWithPath: inputPath), targetURL = URL(fileURLWithPath: targetPath)
        let sourceHashes = try [inputURL, targetURL].map { try hash($0) }
        guard let source = CGImageSourceCreateWithURL(inputURL as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let sourceWidth = properties[kCGImagePropertyPixelWidth] as? Int,
              let sourceHeight = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw NativeTrainingError.invalid("Cannot read source dimensions.")
        }
        let visits = try args.integer("visits", 3, range: 1...1000)
        let transform = NativeCropTransform(x: (sourceWidth - size) / 2, y: (sourceHeight - size) / 2, size: size)
        NativeImages.clearDecodedPixelCache(); defer { NativeImages.clearDecodedPixelCache() }
        var measurements: [[String: Any]] = []
        for visit in 1...visits {
            let visitStart = ProcessInfo.processInfo.systemUptime
            let prepared = try NativeImages.pair(inputURL: inputURL, targetURL: targetURL, transform: transform,
                mapKind: map, cancellation: cancellation)
            let measurement: [String: Any] = ["visit": visit, "seconds": ProcessInfo.processInfo.systemUptime - visitStart,
                "input_shape": prepared.input.shape, "target_shape": prepared.target.shape]
            measurements.append(measurement); emit(measurement.merging(["event": "prepared"]) { _, new in new })
        }
        report["preparation_only"] = true; report["native_source_dimensions"] = [sourceWidth, sourceHeight]
        report["measurements"] = measurements; report["elapsed_seconds"] = ProcessInfo.processInfo.systemUptime - start
        report["sources"] = try zip([inputURL, targetURL], sourceHashes).map { url, before in
            let after = try hash(url)
            guard before == after else { throw NativeTrainingError.invalid("Source pixels changed during the probe.") }
            return ["path": url.path, "before_sha256": before, "after_sha256": after, "unchanged": true] as [String: Any]
        }
        report["status"] = "completed"
        if let path = args.values["output"] {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: path), options: .atomic)
        }
        emit(report.merging(["event": "finished"]) { _, new in new }); return
    }
    var sources: [URL: String] = [:]
    var model: NativeTrainingSession?
    var stepRecords: [[String: Any]] = []
    var failure: Error?
    do {
        let architecture = try NativeArchitecture(mapKind: map, width: width, blocks: blocks,
            identifier: args.values["architecture"] ?? NativeArchitecture.identifier)
        report["architecture"] = architecture.identifier
        report["decoded_pixel_cache_byte_limit"] = NativeImages.decodedPixelCacheByteLimit
        report["parameter_count"] = architecture.parameterCount
        emit(report.merging(["event": "started", "memory": memory()]) { _, new in new })
        let input: [Float], target: [Float]
        let decodeStart = ProcessInfo.processInfo.systemUptime
        if args.synthetic {
            // Defined NCHW values make an independent PyTorch benchmark's pixels
            // identical without storing or rewriting any source image.
            input = (0..<(3 * size * size)).map { Float($0 % 251) / 250 }
            target = (0..<(architecture.outputChannels * size * size)).map { Float(($0 + 71) % 251) / 250 }
            report["synthetic_input"] = "arange(3*size*size)%251/250 in NCHW"
            report["synthetic_target"] = "(arange(channels*size*size)+71)%251/250 in NCHW"
            report["decode_seconds"] = ProcessInfo.processInfo.systemUptime - decodeStart
            report["crop_seconds"] = 0.0
        } else {
            let inputURL = URL(fileURLWithPath: args.values["input"]!)
            let targetURL = URL(fileURLWithPath: args.values["target"]!)
            sources[inputURL] = try hash(inputURL); sources[targetURL] = try hash(targetURL)
            let rgb = try NativeImages.readRGB(from: inputURL, cancellation: cancellation)
            let teaching = try NativeImages.readTarget(from: targetURL, mapKind: map, cancellation: cancellation)
            report["decode_seconds"] = ProcessInfo.processInfo.systemUptime - decodeStart
            let cropStart = ProcessInfo.processInfo.systemUptime
            let transform = NativeCropTransform(x: (rgb.width - size) / 2, y: (rgb.height - size) / 2, size: size)
            let pair = try NativeImages.pair(input: rgb, target: teaching, transform: transform, mapKind: map, cancellation: cancellation)
            input = pair.input.values; target = pair.target.values
            report["crop_seconds"] = ProcessInfo.processInfo.systemUptime - cropStart
            report["native_source_dimensions"] = [rgb.width, rgb.height]
            report["crop"] = ["x": transform.x, "y": transform.y, "size": size]
        }
        try cancellation.check()
        let initStart = ProcessInfo.processInfo.systemUptime
        let supplied = try args.values["weights-in"].map { try NativeWeights.read(from: URL(fileURLWithPath: $0)) }
        let session = try NativeTrainingSession(architecture: architecture,
            options: NativeTrainingOptions(optimizer: args.values["optimizer"] ?? "adamw"), seed: UInt64(seed),
            weights: supplied, cancellation: cancellation,
            channelsLast: args.values["layout"].map { $0 == "nhwc" })
        report["graph_execution"] = "synchronous"
        report["gpu_activation_layout"] = session.channelsLast ? "NHWC" : "NCHW"
        model = session
        report["initialization_seconds"] = ProcessInfo.processInfo.systemUptime - initStart
        report["gpu"] = session.deviceName
        report["initial_weights_sha256"] = hash(try session.weights())
        if let path = args.values["initial-weights-out"] {
            try NativeWeights.write(try session.weights(), to: URL(fileURLWithPath: path))
        }
        emit(["event": "prepared", "decode_seconds": report["decode_seconds"]!, "crop_seconds": report["crop_seconds"]!,
              "initialization_seconds": report["initialization_seconds"]!, "gpu": session.deviceName, "memory": memory()])
        for update in 1...updates {
            try cancellation.check()
            let trainStart = ProcessInfo.processInfo.systemUptime
            let loss = try session.train(input: input, target: target, batchSize: 1, height: size, width: size)
            let trainEnd = ProcessInfo.processInfo.systemUptime
            let trainingExecution = session.lastExecutionMeasurements.map(executionJSON)
            var gradientParameters: [[String: Any]] = []
            if update == 1, let path = args.values["gradients-out"] {
                let gradients = try session.accumulatedGradients()
                try NativeWeights.write(gradients, to: URL(fileURLWithPath: path))
                for name in gradients.keys.sorted() {
                    let values = gradients[name]!.values
                    gradientParameters.append(["name": name, "norm": sqrt(values.reduce(0.0) { $0 + Double($1) * Double($1) }),
                                               "minimum": values.min()!, "maximum": values.max()!])
                }
            }
            let norm = try session.applyGradients(learningRate: rate)
            let end = ProcessInfo.processInfo.systemUptime
            let optimizerExecution = session.lastExecutionMeasurements.map(executionJSON)
            var result: [String: Any] = ["event": "update", "update": update, "loss": loss, "gradient_norm": norm,
                "forward_backward_seconds": trainEnd - trainStart, "optimizer_seconds": end - trainEnd,
                "total_step_seconds": end - trainStart, "includes_first_graph_compile": update == 1, "memory": memory()]
            if !gradientParameters.isEmpty { result["gradient_parameters"] = gradientParameters }
            result["training_execution"] = trainingExecution
            result["optimizer_execution"] = optimizerExecution
            stepRecords.append(result); emit(result)
        }
        try cancellation.check()
        report["status"] = "completed"
    } catch {
        failure = error
        report["status"] = error is CancellationError ? "cancelled" : "failed"
        report["error"] = String(describing: error)
    }
    report["updates"] = model?.updateCount ?? 0
    report["steps"] = stepRecords
    report["elapsed_seconds"] = ProcessInfo.processInfo.systemUptime - start
    report["memory"] = memory()
    var sourceRecords: [[String: Any]] = []
    for (url, before) in sources {
        let after = try hash(url)
        sourceRecords.append(["path": url.path, "before_sha256": before, "after_sha256": after, "unchanged": before == after])
        if before != after { failure = NativeTrainingError.invalid("Source pixels changed during the probe."); report["status"] = "failed" }
    }
    report["sources"] = sourceRecords
    if let model {
        let weights = try model.weights()
        report["final_weights_sha256"] = hash(weights)
        report["weights_changed"] = report["initial_weights_sha256"] as? String != report["final_weights_sha256"] as? String
        if let path = args.values["weights-out"] {
            try NativeWeights.write(weights, to: URL(fileURLWithPath: path), metadata: ["architecture": model.architecture.identifier, "schema": "loom-native-probe-weights-v1"])
            report["weights_path"] = path
        }
    }
    if let path = args.values["output"] {
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: path), options: .atomic)
    }
    emit(report.merging(["event": "finished"]) { _, new in new })
    if let failure { throw failure }
}

do { try run() }
catch {
    emit(["event": "error", "message": error.localizedDescription, "cancelled": error is CancellationError])
    exit(error is CancellationError ? 130 : 1)
}
