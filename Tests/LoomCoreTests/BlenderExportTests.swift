import Foundation
import ImageIO
import LoomNative
import Testing
@testable import LoomCore

struct BlenderExportTests {
    @Test func receiptMustMatchTheExactCheckpointAndExportedModelsRemainDistinct() throws {
        let f = try Fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let receipt = f.root.appendingPathComponent("test.json")
        var proof: [String: Any] = ["run_id": "my-run", "map": "depth", "weight_source": "checkpoint",
            "checkpoint_id": "another-checkpoint", "input_sha256": f.inputHash]
        try JSONSerialization.data(withJSONObject: proof).write(to: receipt)
        #expect(throws: (any Error).self) {
            try BlenderMaterialExport.prepare([f.map], sourceImages: [], includeSource: false,
                directory: f.root.appendingPathComponent("wrong-receipt"))
        }
        proof["checkpoint_id"] = NSNull(); proof["weight_source"] = "model"
        try JSONSerialization.data(withJSONObject: proof).write(to: receipt)
        let result = ModelTestResult(directory: f.root, inputURL: f.input, outputURL: f.output, runID: "my-run",
            source: .model, map: .depth, width: 32, height: 32, tileSize: 32, tileCount: 1,
            note: "", inputSHA256: f.inputHash)
        let url = try BlenderMaterialExport.prepare([.init(result: result, modelName: "Exported best model")],
            sourceImages: [], includeSource: false, directory: f.root.appendingPathComponent("exported-best"))
        let json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let generated = try #require(json["generated"] as? [String: [String: Any]])
        #expect(generated["depth"]?["weight_source"] as? String == "model")
        #expect(generated["depth"]?["model_name"] as? String == "Exported best model")
    }

    @Test func preservesFloatPixelsExactCheckpointIdentityAndOptionalSourceMaps() throws {
        let f = try Fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let images = [f.source(.diffuse, f.input), f.source(.depth, f.output)]
        let url = try BlenderMaterialExport.prepare([f.map], sourceImages: images, includeSource: true,
            directory: f.root.appendingPathComponent("with-source"))
        let json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let generated = try #require(json["generated"] as? [String: [String: Any]])
        let depth = try #require(generated["depth"])
        #expect(depth["run_id"] as? String == "my-run")
        #expect(depth["checkpoint_id"] as? String == "epoch-17-exact")
        #expect(depth["weight_source"] as? String == "checkpoint")
        #expect(depth["model_name"] as? String == "Paper height · my-run · epoch 17")
        #expect((json["source"] as? [String: Any])?.count == 1)
        let copy = url.deletingLastPathComponent().appendingPathComponent("textures/generated-depth.exr")
        #expect(try Data(contentsOf: copy) == Data(contentsOf: f.output))
        #expect(try BlenderMaterialExport.checksum(f.input) == f.inputHash)
        #expect((json["source_issues"] as? [String])?.count == 2)
        #expect(json["width"] as? Int == 32)
        #expect(FileManager.default.fileExists(atPath: url.deletingLastPathComponent().appendingPathComponent("BlenderExport.py").path))
        #expect(throws: (any Error).self) {
            try BlenderMaterialExport.prepare([f.map], sourceImages: images, includeSource: true, directory: url.deletingLastPathComponent())
        }
        let single = try BlenderMaterialExport.prepare([f.map], sourceImages: images, includeSource: false,
            directory: f.root.appendingPathComponent("single"))
        let one = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: single)) as? [String: Any])
        #expect(one["include_source"] as? Bool == false)
        #expect((one["source"] as? [String: Any])?.isEmpty == true)
    }

    @Test func rejectsChangedInputMixedInputsAndDuplicateMapRoles() throws {
        let f = try Fixture()
        defer { try? FileManager.default.removeItem(at: f.root) }
        let dest = f.root.appendingPathComponent("rejected")
        #expect(throws: (any Error).self) {
            try BlenderMaterialExport.prepare([f.map, f.map], sourceImages: [], includeSource: false, directory: dest)
        }
        let other = f.root.appendingPathComponent("other.exr")
        try FileManager.default.copyItem(at: f.input, to: other)
        let r = ModelTestResult(directory: f.root, inputURL: other, outputURL: f.output, runID: "rough",
            source: .model, map: .roughness, width: 32, height: 32, tileSize: 32, tileCount: 1, note: "")
        #expect(throws: (any Error).self) {
            try BlenderMaterialExport.prepare([f.map, .init(result: r, modelName: "Other")], sourceImages: [], includeSource: false, directory: dest)
        }
        try NativeFloatEXR.write([Float](repeating: 0.7, count: 32*32*3), channels: 3, width: 32, height: 32, to: f.input)
        #expect(throws: (any Error).self) {
            try BlenderMaterialExport.prepare([f.map], sourceImages: [], includeSource: false, directory: dest)
        }
        #expect(!FileManager.default.fileExists(atPath: dest.path))
    }

    private struct Fixture {
        let root: URL
        let input: URL
        let output: URL
        let inputHash: String
        var map: BlenderTestMap {
            .init(result: .init(directory: root, inputURL: input, outputURL: output, runID: "my-run",
                source: .checkpoint, map: .depth, width: 32, height: 32, tileSize: 32, tileCount: 1,
                note: "", checkpointID: "epoch-17-exact", inputSHA256: inputHash), modelName: "Paper height · my-run · epoch 17")
        }
        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("blender-export-test-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            input = root.appendingPathComponent("color.exr")
            output = root.appendingPathComponent("height.exr")
            try NativeFloatEXR.write([Float](repeating: 0.5, count: 32*32*3), channels: 3, width: 32, height: 32, to: input)
            let values = (0..<(32*32)).map { Float($0) / 1031.0 }
            try NativeFloatEXR.write(values, channels: 1, width: 32, height: 32, to: output)
            inputHash = try BlenderMaterialExport.checksum(input)
        }
        func source(_ role: MapRole, _ url: URL) -> SourceImage {
            .init(id: role.rawValue, url: url, displayName: url.lastPathComponent, role: role, category: "Paper",
                  verified: false, note: "", pairingKey: "paper", width: 32, height: 32)
        }
    }
}
