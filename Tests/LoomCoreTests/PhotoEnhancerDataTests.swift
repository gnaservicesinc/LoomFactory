import XCTest
import LoomNative
@testable import LoomCore

final class PhotoEnhancerDataTests: XCTestCase {
    func testBoundedRealPhoneImportWhenExplicitlyRequested() throws {
        guard let path = ProcessInfo.processInfo.environment["LOOM_PHOTO_REAL_IMPORT"] else { throw XCTSkip("Optional bounded real-phone dataset import proof.") }
        let source = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support/Loom Factory/PhotoDiffuseResearch-2026-10-16/diagnostic-split.json")
        let output = URL(fileURLWithPath: path)
        let data = try PhotoEnhancerData.importDataset(manifest: source, kind:.repair, destination: output)
        XCTAssertEqual(data.training.count,2); XCTAssertEqual(data.validation.count,2); XCTAssertEqual(data.test.count,2)
        XCTAssertEqual(Set(data.pairs.map { $0.metadata.group }).count,3)
        try Data("Six native SIDD crop pairs across three physical scenes. Diagnostic only; no model trained; original captures and metadata untouched.\n".utf8).write(to:output.appendingPathComponent("PROOF.txt"))
    }
    func testCPUPhotoABKeepsInputBytesAndRawOutOfRangeRGB() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:root) }
        let architecture = try NativeArchitecture(mapKind:NativeModelProduct.photoRepair.rawValue,width:16,blocks:1,identifier:NativeArchitecture.photoCompactIdentifier)
        var weights = architecture.parameterShapes.mapValues { NativeTensor(shape:$0,values:Array(repeating:0,count:$0.reduce(1,*))) }
        weights["output.bias"]!.values = [0.2,-0.3,0.4]
        let rawWeights = root.appendingPathComponent("weights.safetensors")
        try NativeWeights.write(weights,to:rawWeights)
        let compiled = try NativeCoreML.export(weights:weights,architecture:architecture,imageSize:1024,to:root.appendingPathComponent("Enhancer.mlmodel"))
        defer { try? FileManager.default.removeItem(at:compiled) }
        let descriptor = PhotoEnhancerModel(name:"CPU bias test fixture - untrained",product:NativeModelProduct.photoRepair.rawValue,width:16,
            modelSHA256:try NativeFileHash.sha256(root.appendingPathComponent("Enhancer.mlmodel")),weightsSHA256:try NativeFileHash.sha256(rawWeights),runID:"test-only",updates:0)
        try JSONEncoder().encode(descriptor).write(to:root.appendingPathComponent("enhancer.json"))
        let input = root.appendingPathComponent("input.png")
        try ModelTestImage.writePNG(Array(repeating:0.13,count:3*1024*1024),channels:3,width:1024,height:1024,to:input)
        let before = try Data(contentsOf:input)
        let result = try PhotoEnhancerTesting.test(export:root,input:input,destination:root.appendingPathComponent("ab"))
        XCTAssertEqual(try Data(contentsOf:input),before)
        let row = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:result.appendingPathComponent("test.json"))) as? [String:Any])
        XCTAssertEqual(row["raw_out_of_range_components"] as? Int,1024*1024)
        XCTAssertEqual(row["approved"] as? Bool,false)
        if let path = ProcessInfo.processInfo.environment["LOOM_PHOTO_BUNDLE_FIXTURE"] {
            let export = URL(fileURLWithPath:path)
            try FileManager.default.createDirectory(at:export,withIntermediateDirectories:true)
            var fixture = descriptor; fixture.updates = 1 // Contract fixture, not a real training claim.
            fixture.name = "UNTRAINED CPU TEST FIXTURE - never select for materials"
            try JSONEncoder().encode(fixture).write(to:export.appendingPathComponent("enhancer.json"))
            for file in ["Enhancer.mlmodel","weights.safetensors"] { try FileManager.default.copyItem(at:root.appendingPathComponent(file),to:export.appendingPathComponent(file)) }
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath:result.appendingPathComponent("Review.html").path))
        XCTAssertGreaterThan(try result.appendingPathComponent("raw-srgb-numeric.exr").resourceValues(forKeys:[.fileSizeKey]).fileSize ?? 0,12_582_912)
    }
    func testCheckpointRetentionKeepsManualLatestAndExternalFiles() throws {
        let run = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let root = run.appendingPathComponent("checkpoints"), external = run.appendingPathComponent("external")
        defer { try? FileManager.default.removeItem(at:run) }
        let latest = root.appendingPathComponent(UUID().uuidString), manual = root.appendingPathComponent(UUID().uuidString), old = root.appendingPathComponent(UUID().uuidString)
        for file in [latest,manual,old,external] { try FileManager.default.createDirectory(at:file,withIntermediateDirectories:true) }
        try Data().write(to:manual.appendingPathComponent("MANUAL"))
        try Data("keep".utf8).write(to:external.appendingPathComponent("important"))
        try FileManager.default.createSymbolicLink(at:root.appendingPathComponent(UUID().uuidString),withDestinationURL:external)
        PhotoCheckpointRetention.prune(run:run,keeping:URL(fileURLWithPath:latest.path,isDirectory:false))
        XCTAssertTrue(FileManager.default.fileExists(atPath:latest.path)); XCTAssertTrue(FileManager.default.fileExists(atPath:manual.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:old.path))
        XCTAssertEqual(try String(contentsOf:external.appendingPathComponent("important"),encoding:.utf8),"keep")
    }
    func testStagePresetsKeepLightingOutOfRepairAndReframingOutOfBoth() {
        let a = PhotoEnhancerKind.repair.recipe, b = PhotoEnhancerKind.lighting.recipe
        XCTAssertFalse(a.perspective || a.exposure || a.lighting); XCTAssertTrue(a.blur && a.noise)
        XCTAssertFalse(b.perspective || b.blur || b.noise); XCTAssertTrue(b.exposure && b.lighting)
    }
    func testLiveMetadataSplitIsolationIgnoredMembershipAndBytePreservingImport() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var rows: [[String: Any]] = []
        for (i, split) in ["train", "validation", "test"].enumerated() {
            let input = "\(i)-input.png", target = "\(i)-target.png"
            for (file, value) in [(input, Float(i+1)*0.13), (target, Float(i+1)*0.18)] {
                try ModelTestImage.writePNG(Array(repeating: value, count: 3*1024*1024), channels: 3, width: 1024, height: 1024, to: root.appendingPathComponent(file))
            }
            rows.append(["input":input, "target":target, "group":"physical-\(i)", "split":split])
        }
        let manifest = root.appendingPathComponent("manifest.json")
        try JSONSerialization.data(withJSONObject: ["schema":"loom.photo-import.v1", "pairs":rows]).write(to: manifest)
        let destination = root.appendingPathComponent("imported")
        let imported = try PhotoEnhancerData.importDataset(manifest: manifest, kind: .repair, destination: destination)
        XCTAssertEqual(imported.pairs.count,3); XCTAssertEqual(imported.training.count,1)
        XCTAssertEqual(imported.validation.count,1); XCTAssertEqual(imported.test.count,1)
        XCTAssertNil(try PhotoEnhancerData.read(root.appendingPathComponent("0-input.png")))
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("0-input.png")), try Data(contentsOf: imported.training[0].input))
        let pair = imported.validation[0]
        var metadata = pair.metadata; metadata.ignored = true
        try PhotoEnhancerData.write(metadata, to: pair.input)
        let changed = try PhotoEnhancerData.scan(destination, kind:.repair)
        XCTAssertEqual(changed.validation.count,0); XCTAssertNotEqual(changed.fingerprint,imported.fingerprint)
        metadata.ignored = false; metadata.group = imported.training[0].metadata.group
        try PhotoEnhancerData.write(metadata, to: pair.input)
        XCTAssertThrowsError(try PhotoEnhancerData.scan(destination, kind:.repair))
        metadata.group = "other"; metadata.target = "../0-target.png"
        try PhotoEnhancerData.write(metadata, to: pair.input)
        XCTAssertThrowsError(try PhotoEnhancerData.scan(destination, kind:.repair))
    }
    func testMixedSynthesisCannotSilentlyTeachTheWrongStage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try JSONEncoder().encode(SyntheticPhotoRecipe()).write(to: root.appendingPathComponent("recipe.json"))
        let manifest = root.appendingPathComponent("manifest.json")
        try JSONSerialization.data(withJSONObject:["schema":"loom.synthetic-photo-data.v1", "status":"completed", "pairs":[]]).write(to:manifest)
        for kind in PhotoEnhancerKind.allCases { XCTAssertThrowsError(try PhotoEnhancerData.importDataset(manifest:manifest, kind:kind, destination:root.appendingPathComponent(kind.rawValue))) }
    }
}
