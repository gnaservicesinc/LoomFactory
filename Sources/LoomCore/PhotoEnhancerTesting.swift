import CoreML
import Foundation
import LoomNative

public enum PhotoEnhancerTesting {
    public static func exportCheckpoint(_ file: URL, best: Bool, destination: URL) throws -> URL {
        let checkpoint = try PhotoEnhancerTrainer.loadCheckpoint(file), root = PhotoEnhancerTrainer.runRoot(file)
        try checkpoint.settings.validate()
        let source = try PhotoEnhancerTrainer.checkedFile(best ? checkpoint.bestWeights : checkpoint.weights,
            hash: best ? checkpoint.bestWeightsSHA256 : checkpoint.weightsSHA256, root: root)
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw LoomError("Choose a new photo export folder.") }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let tensors = try NativeWeights.read(from: source)
        try NativeWeights.write(tensors, to: destination.appendingPathComponent("weights.safetensors"))
        let compiled = try NativeCoreML.export(weights: tensors, architecture: checkpoint.settings.architecture,
            imageSize: 1024, to: destination.appendingPathComponent("Enhancer.mlmodel"))
        defer { try? FileManager.default.removeItem(at: compiled) }
        let updates = best ? checkpoint.bestUpdates : checkpoint.updates
        let descriptor = PhotoEnhancerModel(name: "\(checkpoint.settings.kind.title) · \(checkpoint.runID.prefix(8)) · \(best ? "best" : "latest") update \(updates)",
            product: checkpoint.settings.kind.product.rawValue, width: checkpoint.settings.width,
            modelSHA256: try NativeFileHash.sha256(destination.appendingPathComponent("Enhancer.mlmodel")),
            weightsSHA256: try NativeFileHash.sha256(destination.appendingPathComponent("weights.safetensors")), runID: checkpoint.runID, updates: updates, trainingCategories: checkpoint.trainingCategories)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(descriptor).write(to: destination.appendingPathComponent("enhancer.json"), options: .atomic)
        return destination
    }
    public static func test(export: URL, input: URL, destination: URL, cancellation: NativeCancellation = .init()) throws -> URL {
        let descriptor = try JSONDecoder().decode(PhotoEnhancerModel.self, from: Data(contentsOf: export.appendingPathComponent("enhancer.json")))
        guard descriptor.schema == "loom.photo-enhancer.v1", descriptor.architecture == NativeArchitecture.photoCompactIdentifier,
              let product = NativeModelProduct(rawValue: descriptor.product), product.isPhoto,
              descriptor.modelFile == "Enhancer.mlmodel", descriptor.weights == "weights.safetensors",
              descriptor.input == "rgb", descriptor.output == "enhanced_rgb", descriptor.pixelContract == PhotoEnhancerModel.contract,
              [16,22,32,46].contains(descriptor.width),
              try NativeFileHash.sha256(export.appendingPathComponent(descriptor.modelFile)) == descriptor.modelSHA256,
              try NativeFileHash.sha256(export.appendingPathComponent(descriptor.weights)) == descriptor.weightsSHA256 else { throw LoomError("This photo model export has an incompatible contract or changed checksum.") }
        try PhotoEnhancerData.native1K(input); try cancellation.check()
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let compiled = try MLModel.compileModel(at: export.appendingPathComponent(descriptor.modelFile))
        defer { try? FileManager.default.removeItem(at: compiled) }
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
        guard metadata?["loom.product"] == descriptor.product else { throw LoomError("Compiled model product differs from its export.") }
        let samples = try PhotoEnhancerTrainer.rgb(input)
        let array = try MLMultiArray(shape:[1,3,1024,1024], dataType:.float32)
        samples.withUnsafeBufferPointer { array.dataPointer.assumingMemoryBound(to:Float.self).update(from:$0.baseAddress!,count:samples.count) }
        let prediction = try model.prediction(from: MLDictionaryFeatureProvider(dictionary:["rgb":MLFeatureValue(multiArray:array)]))
        guard let tensor = prediction.featureValue(for:"enhanced_rgb")?.multiArrayValue, tensor.dataType == .float32,
              tensor.shape.map(\.intValue) == [1,3,1024,1024] else { throw LoomError("The photo enhancer did not return native RGB.") }
        let strides = tensor.strides.map(\.intValue), pointer = tensor.dataPointer.assumingMemoryBound(to:Float.self)
        let values = (0..<3).flatMap { c in (0..<1024*1024).map { p in pointer[c*strides[1] + (p/1024)*strides[2] + (p%1024)*strides[3]] } }
        guard values.allSatisfy(\.isFinite) else { throw LoomError("Photo model returned non-finite RGB.") }
        try cancellation.check()
        guard !FileManager.default.fileExists(atPath:destination.path) else { throw LoomError("Choose a new photo test folder.") }
        try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true)
        try FileManager.default.copyItem(at:input,to:destination.appendingPathComponent("input.png"))
        try ModelTestImage.writePNG(values,channels:3,width:1024,height:1024,to:destination.appendingPathComponent("enhanced.png"), clampPreview:true, cancellation:cancellation)
        // EXR retains numeric sRGB code values exactly, not scene-linear radiance.
        try NativeFloatEXR.write(values,channels:3,width:1024,height:1024,to:destination.appendingPathComponent("raw-srgb-numeric.exr"), cancellation:cancellation)
        let row: [String:Any] = ["schema":"loom.photo-test.v1", "model":descriptor.name, "product":descriptor.product,
            "run_id":descriptor.runID, "updates":descriptor.updates, "model_sha256":descriptor.modelSHA256,
            "input_sha256":try NativeFileHash.sha256(input), "compute":"Core ML CPU only", "approved":false,
            "preview":"16-bit PNG clamps display to [0,1]; EXR is raw numeric sRGB, not linear RGB", "raw_out_of_range_components":values.filter { $0 < 0 || $0 > 1 }.count]
        try JSONSerialization.data(withJSONObject:row,options:[.sortedKeys,.prettyPrinted]).write(to:destination.appendingPathComponent("test.json"),options:.atomic)
        let title = descriptor.name.replacingOccurrences(of:"&",with:"&amp;").replacingOccurrences(of:"<",with:"&lt;")
        try Data("<!doctype html><meta charset='utf-8'><title>Photo enhancer A/B</title><style>body{font:16px system-ui;background:#202328;color:white;margin:24px}section{display:flex;gap:16px}figure{margin:0;flex:1}img{width:100%}</style><h1>\(title)</h1><p>Unapproved candidate, best update \(descriptor.updates). Inspect original grain, shape, color, new artifacts, and clean-input identity. This comparison does not certify learned quality. Click images for native pixels.</p><section><figure><a href='input.png'><img src='input.png'></a><figcaption>Input</figcaption></figure><figure><a href='enhanced.png'><img src='enhanced.png'></a><figcaption>Enhanced, strength 1</figcaption></figure></section><p><a href='test.json'>Identity and numeric diagnostics</a></p>".utf8).write(to:destination.appendingPathComponent("Review.html"),options:.atomic)
        return destination
    }
}
