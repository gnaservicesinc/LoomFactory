import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class MapTypeClassifierTests: XCTestCase {
    func testNamesAndKnownUnsupportedMapsNeverReachClassifier() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        for suffix in ["diff", "disp", "rough", "nor_gl", "bump", "bumpmap", "ao", "ambient_occlusion", "metallic",
                       "specular", "glossiness", "smoothness", "opacity", "emission", "orm", "arm", "rma", "mrao", "packed", "coll2"] {
            try image(root, "material_\(suffix).png")
        }
        let predictor = FixedMapTypePredictor(probabilities: scalarGuess)
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let snapshot = try await project.snapshot()
        XCTAssertEqual(predictor.calls, 0)
        XCTAssertEqual(snapshot.images.filter { $0.role == .bump }.count, 2)
        XCTAssertEqual(snapshot.images.filter { $0.role == .unknown && $0.ignored }.count, 13)
        let alternateDiffuse = try XCTUnwrap(snapshot.images.first { $0.url.lastPathComponent == "material_coll2.png" })
        XCTAssertEqual(alternateDiffuse.role, .diffuse)
        XCTAssertFalse(alternateDiffuse.ignored)
        XCTAssertTrue(snapshot.images.allSatisfy { $0.mapTypeInference == nil })
        XCTAssertEqual(snapshot.images.first { $0.url.lastPathComponent == "material_disp.png" }?.role, .depth)
    }

    func testConfidentFallbackPersistsGuessAndDoesNotRewriteSourceOrPairing() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let file = try image(root, "unfamiliar.png")
        let bytes = try Data(contentsOf: file)
        let tag = ImageMetadata(id: "original-id", pairingKey: "manual-pair", category: "Bark", note: "Keep detail",
                                pairingWasManuallySelected: true)
        try ImageMetadataStore.write(tag, to: file)
        let predictor = FixedMapTypePredictor(probabilities: scalarGuess)
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let first = try await project.snapshot()
        let predicted = try XCTUnwrap(first.images.first)
        XCTAssertEqual(predicted.role, .depth)
        XCTAssertEqual(predicted.mapTypeInference?.accepted, true)
        XCTAssertEqual(predicted.mapTypeInference?.confidence, 0.97)
        XCTAssertFalse(predicted.verified)
        XCTAssertEqual(predicted.id, tag.id); XCTAssertEqual(predicted.category, tag.category)
        XCTAssertEqual(predicted.note, tag.note); XCTAssertEqual(predicted.pairingKey, tag.pairingKey)
        XCTAssertEqual(try Data(contentsOf: file), bytes)
        _ = try await project.snapshot()
        XCTAssertEqual(predictor.calls, 1, "Polling must reuse a prediction for unchanged source/model")
        let reopened = try LoomProject(sourceFolder: root)
        await reopened.setMapTypePredictor(predictor)
        let afterReopen = try await reopened.snapshot()
        XCTAssertEqual(afterReopen.images.first?.mapTypeInference, predicted.mapTypeInference)
        XCTAssertEqual(predictor.calls, 1, "Metadata survives reopening without another prediction")
    }

    func testScalarAmbiguityStaysUnknownAndManualChoicesWin() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let file = try image(root, "unfamiliar.png")
        let predictor = FixedMapTypePredictor(probabilities: ["depth": 0.52, "rough": 0.47, "diffuse": 0.005, "normal": 0.005])
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let first = try await project.snapshot()
        let uncertain = try XCTUnwrap(first.images.first)
        XCTAssertEqual(uncertain.role, .unknown)
        XCTAssertEqual(uncertain.mapTypeInference?.accepted, false)
        XCTAssertEqual(uncertain.mapTypeInference?.role, .depth)
        _ = try await project.snapshot()
        XCTAssertEqual(predictor.calls, 1)
        try await project.setRole(id: uncertain.id, role: .roughness)
        let chosen = try await project.snapshot()
        XCTAssertEqual(chosen.images.first?.role, .roughness)
        XCTAssertNil(chosen.images.first?.mapTypeInference)
        XCTAssertEqual(predictor.calls, 1)
        try await project.setRole(id: uncertain.id, role: .unknown)
        let manualUnknown = try await project.snapshot()
        XCTAssertEqual(manualUnknown.images.first?.role, .unknown)
        XCTAssertNil(manualUnknown.images.first?.mapTypeInference)
        XCTAssertEqual(predictor.calls, 1)
        XCTAssertTrue(try XCTUnwrap(ImageMetadataStore.read(file)).roleWasManuallySelected == true)
    }

    func testChangingSourceOrModelInvalidatesGuessAndMeaningfulRenameTakesPriority() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let file = try image(root, "unfamiliar.png")
        let predictor = FixedMapTypePredictor(probabilities: scalarGuess)
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let initial = try await project.snapshot()
        let oldInference = try XCTUnwrap(initial.images.first?.mapTypeInference)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 100)], ofItemAtPath: file.path)
        let changed = try await project.snapshot()
        XCTAssertNotEqual(changed.images.first?.mapTypeInference?.sourceSignature, oldInference.sourceSignature)
        XCTAssertEqual(predictor.calls, 2)
        let replacement = FixedMapTypePredictor(modelID: "replacement", probabilities: ["depth": 0.01, "rough": 0.01, "diffuse": 0.97, "normal": 0.01])
        await project.setMapTypePredictor(replacement)
        let updated = try await project.snapshot()
        XCTAssertEqual(updated.images.first?.role, .diffuse)
        XCTAssertEqual(updated.images.first?.mapTypeInference?.modelID, replacement.modelID)
        let renamed = root.appendingPathComponent("material_bump.png")
        try FileManager.default.moveItem(at: file, to: renamed)
        let named = try await project.snapshot()
        XCTAssertEqual(named.images.first?.role, .bump)
        XCTAssertNil(named.images.first?.mapTypeInference)
        XCTAssertEqual(replacement.calls, 1)
    }

    func testReviewedUnknownCanReceiveSuggestionWhileKnownRolesAndIgnoreDecisionsRemainAuthoritative() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let known = try image(root, "known.png")
        let reviewed = try image(root, "reviewed.png")
        let ignored = try image(root, "ignored.png")
        try ImageMetadataStore.write(.init(pairingKey: "a", role: .roughness, category: "Leather", note: "Known stored role"), to: known)
        try ImageMetadataStore.write(.init(pairingKey: "b", verified: true), to: reviewed)
        try ImageMetadataStore.write(.init(pairingKey: "c", ignored: true), to: ignored)
        let predictor = FixedMapTypePredictor(probabilities: scalarGuess)
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let snapshot = try await project.snapshot()
        XCTAssertEqual(snapshot.images.first { $0.url == known }?.role, .roughness)
        XCTAssertEqual(snapshot.images.first { $0.url == reviewed }?.role, .depth)
        XCTAssertEqual(snapshot.images.first { $0.url == reviewed }?.mapTypeInference?.accepted, true)
        XCTAssertEqual(snapshot.images.first { $0.url == reviewed }?.verified, true)
        XCTAssertEqual(snapshot.images.first { $0.url == ignored }?.ignored, true)
        XCTAssertEqual(predictor.calls, 1)
    }

    func testFailedModelSuggestionCannotStopDiscoveryOrRepeatEveryPoll() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try image(root, "unfamiliar.png")
        let predictor = FixedMapTypePredictor(probabilities: [:])
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let first = try await project.snapshot()
        XCTAssertEqual(first.images.count, 1)
        XCTAssertEqual(first.images.first?.role, .unknown)
        XCTAssertNil(first.images.first?.mapTypeInference)
        _ = try await project.snapshot()
        XCTAssertEqual(predictor.calls, 1)
    }

    func testFailureCacheIsScopedToEachActualFile() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        for folderName in ["a", "b"] {
            let folder = root.appendingPathComponent(folderName)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let file = try image(folder, "unfamiliar.png")
            try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 100)], ofItemAtPath: file.path)
        }
        let predictor = FixedMapTypePredictor(probabilities: [:])
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let snapshot = try await project.snapshot()
        XCTAssertEqual(snapshot.images.count, 2)
        XCTAssertEqual(predictor.calls, 2, "A failure on one file must not suppress an identical-looking sibling")
        _ = try await project.snapshot()
        XCTAssertEqual(predictor.calls, 2)
    }

    func testUnavailableOrInvalidReplacementCannotRetainAnOldAcceptedRole() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try image(root, "unfamiliar.png")
        let predictor = FixedMapTypePredictor(probabilities: scalarGuess)
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let initial = try await project.snapshot()
        XCTAssertEqual(initial.images.first?.role, .depth)
        await project.setMapTypePredictor(nil)
        let unavailable = try await project.snapshot()
        XCTAssertEqual(unavailable.images.first?.role, .unknown)
        XCTAssertNil(unavailable.images.first?.mapTypeInference)
        await project.setMapTypePredictor(predictor)
        let restored = try await project.snapshot()
        XCTAssertEqual(restored.images.first?.role, .depth)
        await project.setMapTypePredictor(FixedMapTypePredictor(modelID: "invalid-replacement", probabilities: [:]))
        let invalid = try await project.snapshot()
        XCTAssertEqual(invalid.images.first?.role, .unknown)
        XCTAssertNil(invalid.images.first?.mapTypeInference)
    }

    func testManualMetadataEditDuringPredictionWins() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let file = try image(root, "unfamiliar.png")
        let initialTag = ImageMetadata(id: "source-id", pairingKey: "original-pair")
        try ImageMetadataStore.write(initialTag, to: file)
        var editedTag = initialTag
        editedTag.role = .roughness; editedTag.roleWasManuallySelected = true
        editedTag.pairingKey = "manual-pair"; editedTag.pairingWasManuallySelected = true
        editedTag.category = "Leather"; editedTag.note = "Human change during inference"
        let manual = editedTag
        let predictor = MutatingMapTypePredictor { url in try ImageMetadataStore.write(manual, to: url) }
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let snapshot = try await project.snapshot()
        XCTAssertEqual(snapshot.images.first?.role, .roughness)
        XCTAssertEqual(snapshot.images.first?.pairingKey, manual.pairingKey)
        XCTAssertEqual(snapshot.images.first?.note, manual.note)
        XCTAssertEqual(snapshot.images.first?.category, manual.category)
        XCTAssertNil(snapshot.images.first?.mapTypeInference)
        XCTAssertEqual(try ImageMetadataStore.read(file), manual)
    }

    func testChangingSourceDuringPredictionDoesNotSaveGuessForOldBytes() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let file = try image(root, "unfamiliar.png")
        let predictor = MutatingMapTypePredictor { url in
            try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 100)], ofItemAtPath: url.path)
        }
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let changed = try await project.snapshot()
        XCTAssertEqual(changed.images.first?.role, .unknown)
        XCTAssertNil(changed.images.first?.mapTypeInference)
        XCTAssertNil(try ImageMetadataStore.read(file)?.mapTypeInference)
        let stable = try await project.snapshot()
        XCTAssertEqual(stable.images.first?.role, .depth)
        XCTAssertEqual(stable.images.first?.mapTypeInference?.accepted, true)
    }

    func testExactLegacyFilenameInferenceRepairsNewExclusionsWithoutChangingHumanTags() async throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let file = try image(root, "wood_color_gloss.png")
        let bytes = try Data(contentsOf: file)
        let legacyKey = SHA256.hash(data: Data("/wood_gloss:64x64".utf8)).map { String(format: "%02x", $0) }.joined()
        let metadata = ImageMetadata(id: "existing-id", pairingKey: legacyKey, role: .diffuse, category: "Wood",
                                     verified: true, note: "Preserve review", ignored: false)
        try ImageMetadataStore.write(metadata, to: file)
        let predictor = FixedMapTypePredictor(probabilities: scalarGuess)
        let project = try LoomProject(sourceFolder: root)
        await project.setMapTypePredictor(predictor)
        let repaired = try await project.snapshot()
        let item = try XCTUnwrap(repaired.images.first)
        XCTAssertEqual(item.role, .unknown)
        XCTAssertEqual(item.id, metadata.id); XCTAssertEqual(item.category, metadata.category)
        XCTAssertEqual(item.verified, metadata.verified); XCTAssertEqual(item.note, metadata.note)
        XCTAssertFalse(item.ignored, "An existing ignore choice is not overwritten by migration")
        XCTAssertEqual(try Data(contentsOf: file), bytes)
        XCTAssertEqual(predictor.calls, 0)
        var manual = metadata; manual.roleWasManuallySelected = true
        try ImageMetadataStore.write(manual, to: file)
        let retained = try await project.snapshot()
        XCTAssertEqual(retained.images.first?.role, .diffuse)
    }

    func testScalarMapGuessesNeedMoreEvidenceAndMalformedProbabilitiesCannotPromoteRoles() {
        let diffuse = MapTypeInferencePolicy.evaluate(["diffuse": 0.94, "normal": 0.02, "rough": 0.02, "depth": 0.02],
                                                      modelID: "test", sourceSignature: "image")
        XCTAssertEqual(diffuse?.accepted, true)
        let rough = MapTypeInferencePolicy.evaluate(["rough": 0.94, "normal": 0.02, "diffuse": 0.02, "depth": 0.02],
                                                    modelID: "test", sourceSignature: "image")
        XCTAssertEqual(rough?.accepted, false)
        XCTAssertNil(MapTypeInferencePolicy.evaluate(["rough": .nan, "normal": 0, "diffuse": 0, "depth": 1],
                                                      modelID: "test", sourceSignature: "image"))
        XCTAssertNil(MapTypeInferencePolicy.evaluate(["bump": 1], modelID: "test", sourceSignature: "image"))
    }

    func testNewExclusionRepairsPreviousCompoundFilenameInference() throws {
        let root = URL(fileURLWithPath: "/source")
        let file = root.appendingPathComponent("wood_base_colour_gloss.png")
        // The immediately previous naming policy stripped both base + colour.
        // The older first-token policy retained base; both exact versions must migrate.
        let previousKey = SHA256.hash(data: Data("/wood_gloss:64x64".utf8)).map { String(format: "%02x", $0) }.joined()
        let previous = ImageMetadata(pairingKey: previousKey, role: .diffuse, category: "Wood")
        let repaired = ImageMetadataStore.repairingLegacyInference(previous, file: file, sourceRoot: root, width: 64, height: 64)
        XCTAssertEqual(repaired.role, .unknown)
        XCTAssertEqual(repaired.category, previous.category)
        var manualPair = previous; manualPair.pairingWasManuallySelected = true
        XCTAssertEqual(ImageMetadataStore.repairingLegacyInference(manualPair, file: file, sourceRoot: root, width: 64, height: 64), manualPair)
    }

    func testSuppliedCoreMLModelRunsNativelyAndLeavesOriginalBytesUnchanged() throws {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let file = try image(root, "unfamiliar.png", size: 384)
        let original = try Data(contentsOf: file)
        let asset = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Resources/MatTypeClassifier.mlmodel")
        let predictor = try NativeMapTypeClassifier(modelURL: asset)
        let start = Date()
        let probabilities = try predictor.probabilities(for: file)
        XCTAssertEqual(Set(probabilities.keys), Set(["depth", "diffuse", "normal", "rough"]))
        XCTAssertTrue(probabilities.values.allSatisfy { $0.isFinite && (0...1).contains($0) })
        XCTAssertEqual(probabilities.values.reduce(0, +), 1, accuracy: 0.0001)
        XCTAssertEqual(try Data(contentsOf: file), original)
        XCTAssertNotNil(MapTypeInferencePolicy.evaluate(probabilities, modelID: predictor.modelID, sourceSignature: "smoke"))
        print("Native map-type classifier CPU smoke: \(Date().timeIntervalSince(start)) seconds; four-class probabilities returned.")
    }

    private var scalarGuess: [String: Double] { ["depth": 0.97, "rough": 0.01, "diffuse": 0.01, "normal": 0.01] }
    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-map-types-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.resolvingSymlinksInPath()
    }
    @discardableResult private func image(_ folder: URL, _ name: String, size: Int = 64) throws -> URL {
        let file = folder.appendingPathComponent(name)
        let bytes = Data(repeating: 100, count: size * size * 3)
        let provider = try XCTUnwrap(CGDataProvider(data: bytes as CFData))
        let cg = try XCTUnwrap(CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: size * 3,
                                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [], provider: provider,
                                      decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, cg, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return file
    }
}

private final class FixedMapTypePredictor: MapTypePredictor, @unchecked Sendable {
    let modelID: String
    private let values: [String: Double]
    private let lock = NSLock()
    private var callCount = 0
    var calls: Int { lock.lock(); defer { lock.unlock() }; return callCount }
    init(modelID: String = "test-model", probabilities: [String: Double]) { self.modelID = modelID; values = probabilities }
    func probabilities(for file: URL) throws -> [String: Double] {
        lock.lock(); defer { lock.unlock() }
        callCount += 1
        return values
    }
}

private struct MutatingMapTypePredictor: MapTypePredictor {
    let modelID = "mutating-test-model"
    let mutation: @Sendable (URL) throws -> Void
    func probabilities(for file: URL) throws -> [String: Double] {
        try mutation(file)
        return ["depth": 0.97, "rough": 0.01, "diffuse": 0.01, "normal": 0.01]
    }
}
