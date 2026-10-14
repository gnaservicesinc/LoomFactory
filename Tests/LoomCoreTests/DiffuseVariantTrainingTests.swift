import CoreGraphics
import Foundation
import ImageIO
import LoomNative
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

@MainActor
final class DiffuseVariantTrainingTests: XCTestCase {
    private let root = URL(fileURLWithPath: "/source")

    func testEachDiffuseVariantUsesTheSameUniqueTargetForEveryMapProduct() throws {
        let inputs = [member("leather", name: "color-red", role: .diffuse),
                      member("leather", name: "color-tan", role: .diffuse)]
        let targets = MapRole.targets.map { member("leather", name: $0.rawValue, role: $0) }
        for map in MapRole.targets {
            let pairs = try TrainingPairSelector.select(images: inputs + targets, dataset: "Leather", map: map, size: 32)
            XCTAssertEqual(pairs.count, 2)
            XCTAssertEqual(Set(pairs.map(\.input.id)), Set(inputs.map(\.id)))
            XCTAssertEqual(Set(pairs.map(\.target.id)), [try XCTUnwrap(targets.first { $0.role == map }).id])
            XCTAssertEqual(Set(pairs.map(\.id)), Set(inputs.map { $0.pairingKey + "|" + $0.id }))
            XCTAssertTrue(pairs.allSatisfy { !$0.input.verified && !$0.target.verified },
                          "Review flags must not exclude correctly tagged variants")
        }
    }

    func testVariantIdentitySurvivesInputOrderAndRenameWhileSingletonsKeepLegacyIdentity() throws {
        let red = member("leather", name: "red", role: .diffuse)
        let tan = member("leather", name: "tan", role: .diffuse)
        let target = member("leather", name: "depth", role: .depth)
        let selected = try TrainingPairSelector.select(images: [red, tan, target], dataset: "Leather", map: .depth, size: 32)
        let reordered = try TrainingPairSelector.select(images: [target, tan, red], dataset: "Leather", map: .depth, size: 32)
        XCTAssertEqual(Set(selected.map(\.id)), Set(reordered.map(\.id)))
        let renamed = SourceImage(id: red.id, url: red.url.deletingLastPathComponent().appendingPathComponent("renamed.png"),
            displayName: "Renamed color", role: red.role, category: red.category, verified: red.verified, note: red.note,
            pairingKey: red.pairingKey, width: red.width, height: red.height)
        let renamedPairs = try TrainingPairSelector.select(images: [renamed, tan, target], dataset: "Leather", map: .depth, size: 32)
        XCTAssertEqual(renamedPairs.first { $0.input.id == red.id }?.id, selected.first { $0.input.id == red.id }?.id)
        let singleton = try TrainingPairSelector.select(images: [renamed, target], dataset: "Leather", map: .depth, size: 32)
        XCTAssertEqual(singleton.map(\.id), [red.pairingKey], "A single input retains the pre-variant sample identity")
    }

    func testMatchingDimensionsDistinguishMultipleNativeGridsInOnePairingFamily() throws {
        let images = [32, 64].flatMap { size in
            [member("leather", name: "red-\(size)", role: .diffuse, size: size, pairingKey: "shared-family"),
             member("leather", name: "tan-\(size)", role: .diffuse, size: size, pairingKey: "shared-family"),
             member("leather", name: "depth-\(size)", role: .depth, size: size, pairingKey: "shared-family")]
        }
        let pairs = try TrainingPairSelector.select(images: images, dataset: "Leather", map: .depth, size: 32)
        XCTAssertEqual(pairs.count, 4)
        XCTAssertEqual(Set(pairs.map(\.id)).count, 4)
        for pair in pairs {
            XCTAssertEqual(pair.input.width, pair.target.width)
            XCTAssertEqual(pair.input.height, pair.target.height)
        }
    }

    func testSeveralTargetsAtTheSameDimensionsStillRequireExplicitPairing() throws {
        let images = [member("leather", name: "red", role: .diffuse),
                      member("leather", name: "tan", role: .diffuse),
                      member("leather", name: "depth-a", role: .depth),
                      member("leather", name: "depth-b", role: .depth)]
        XCTAssertThrowsError(try TrainingPairSelector.select(images: images, dataset: "Leather", map: .depth, size: 32))
    }

    func testSharedVariantsCannotSpanMaterialFoldersAndLeakTheirTargetAcrossSplits() throws {
        let red = member("leather", name: "red", role: .diffuse, pairingKey: "shared-family")
        let tan = member("other-leather", name: "tan", role: .diffuse, pairingKey: "shared-family")
        let depth = member("leather", name: "depth", role: .depth, pairingKey: "shared-family")
        XCTAssertThrowsError(try TrainingPairSelector.select(images: [red, tan, depth], dataset: "Leather", map: .depth, size: 32))
        let localTan = member("leather", name: "tan", role: .diffuse, pairingKey: "shared-family")
        let distantDepth = member("other-leather", name: "depth", role: .depth, pairingKey: "shared-family")
        XCTAssertThrowsError(try TrainingPairSelector.select(images: [red, localTan, distantDepth], dataset: "Leather", map: .depth, size: 32))
        XCTAssertEqual(try TrainingPairSelector.select(images: [red, localTan, depth], dataset: "Leather", map: .depth, size: 32).count, 2)
    }

    func testManualSharedVariantPairingRejectsSeparateFoldersBeforeChangingMetadata() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("loom-variant-pairing-" + UUID().uuidString)
        let local = folder.appendingPathComponent("leather", isDirectory: true)
        let separate = folder.appendingPathComponent("other-leather", isDirectory: true)
        try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: separate, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let red = local.appendingPathComponent("red.png"), tan = separate.appendingPathComponent("tan.png")
        let depth = local.appendingPathComponent("depth.png")
        try writePNG([UInt16](repeating: 20000, count: 32 * 32 * 3), channels: 3, to: red)
        try writePNG([UInt16](repeating: 40000, count: 32 * 32 * 3), channels: 3, to: tan)
        try writePNG([UInt16](repeating: 30000, count: 32 * 32), channels: 1, to: depth)
        for (url, role) in [(red, MapRole.diffuse), (tan, .diffuse), (depth, .depth)] {
            try ImageMetadataStore.write(ImageMetadata(pairingKey: "original-family", role: role, category: "Leather",
                roleWasManuallySelected: true, pairingWasManuallySelected: true), to: url)
        }
        let project = try LoomProject(sourceFolder: folder)
        let initial = try await project.snapshot()
        do {
            try await project.pairImages(ids: initial.images.map(\.id))
            XCTFail("Shared diffuse variants in different material folders must not be paired")
        } catch {}
        let unchanged = try await project.snapshot()
        XCTAssertEqual(Set(unchanged.images.map(\.pairingKey)), ["original-family"])
        XCTAssertEqual(Set(unchanged.images.map(\.id)), Set(initial.images.map(\.id)))
        try FileManager.default.moveItem(at: tan, to: local.appendingPathComponent("tan.png"))
        let together = try await project.snapshot()
        try await project.pairImages(ids: together.images.map(\.id))
        let paired = try await project.snapshot()
        XCTAssertEqual(Set(paired.images.map(\.pairingKey)).count, 1)
        XCTAssertNotEqual(paired.images.first?.pairingKey, "original-family")
        XCTAssertEqual(try TrainingPairSelector.select(images: paired.images, dataset: "Leather", map: .depth, size: 32).count, 2)
    }

    func testCurrentCategoryIgnoreDimensionsAndNormalConventionStillControlMembership() throws {
        let red = member("leather", name: "red", role: .diffuse)
        var tan = member("leather", name: "tan", role: .diffuse)
        var target = member("leather", name: "depth", role: .depth)
        tan.ignored = true
        var selected = try TrainingPairSelector.select(images: [red, tan, target], dataset: "Leather", map: .depth, size: 32)
        XCTAssertEqual(selected.map(\.input.id), [red.id])
        tan.ignored = false; tan.category = "Fabric"
        selected = try TrainingPairSelector.select(images: [red, tan, target], dataset: "Leather", map: .depth, size: 32)
        XCTAssertEqual(selected.map(\.input.id), [red.id])
        target.ignored = true
        XCTAssertTrue(try TrainingPairSelector.select(images: [red, target], dataset: "Leather", map: .depth, size: 32).isEmpty)
        target.ignored = false; target.category = "Fabric"
        XCTAssertTrue(try TrainingPairSelector.select(images: [red, target], dataset: "Leather", map: .depth, size: 32).isEmpty)

        let mismatch = member("leather", name: "wide", role: .diffuse, size: 64, pairingKey: red.pairingKey)
        target.category = "Leather"
        XCTAssertThrowsError(try TrainingPairSelector.select(images: [red, mismatch, target], dataset: "Leather", map: .depth, size: 32))
        XCTAssertTrue(try TrainingPairSelector.select(images: [red, target], dataset: "Leather", map: .depth, size: 64).isEmpty)

        let gl = member("leather", name: "normal-gl", role: .normal, normalConvention: "OpenGL")
        let dx = member("leather", name: "normal-dx", role: .normal, normalConvention: "DirectX")
        tan.category = "Leather"
        let normals = try TrainingPairSelector.select(images: [red, tan, gl, dx], dataset: "Leather", map: .normal, size: 32)
        XCTAssertEqual(normals.count, 2)
        XCTAssertEqual(Set(normals.map(\.target.id)), [gl.id])
        XCTAssertTrue(try TrainingPairSelector.select(images: [red, tan, dx], dataset: "Leather", map: .normal, size: 32).isEmpty)
    }

    func testAllVariantsAndResolutionsStayInTheSameWholeMaterialSplitAndCountSharedTargetsOnce() throws {
        let images = (0..<5).flatMap { material in
            [32, 64, 128].flatMap { size in
                [member("material-\(material)", name: "red-\(size)", role: .diffuse, size: size),
                 member("material-\(material)", name: "tan-\(size)", role: .diffuse, size: size),
                 member("material-\(material)", name: "depth-\(size)", role: .depth, size: size)]
            }
        }
        let pairs = try TrainingPairSelector.select(images: images, dataset: "Leather", map: .depth, size: 32)
        let settings = TrainingSettings(imageSize: 32, epochs: 3,
            batchSize: 2, gradientAccumulationSteps: 2, validationFraction: 0.4, validationSeed: 7)
        let plan = TrainingDataPlan(pairs: pairs, settings: settings, sourceFolder: root)
        let repeated = TrainingDataPlan(pairs: pairs.reversed(), settings: settings, sourceFolder: root)
        let originals = Set((plan.training + plan.validation).flatMap { [$0.input.url, $0.target.url] })
        XCTAssertEqual(originals.count, 45, "Shared targets remain one physical original despite multiple sample references")
        XCTAssertEqual(plan.totalPairCount, 30)
        XCTAssertEqual(plan.trainingPairCount, 18)
        XCTAssertEqual(plan.validationPairCount, 12)
        XCTAssertEqual(plan.trainingMaterialCount, 3)
        XCTAssertEqual(plan.validationMaterialCount, 2)
        XCTAssertEqual(Set(plan.validation.map(\.id)), Set(repeated.validation.map(\.id)))
        let trainingFolders = Set(plan.training.map { $0.input.url.deletingLastPathComponent() })
        let validationFolders = Set(plan.validation.map { $0.input.url.deletingLastPathComponent() })
        XCTAssertTrue(trainingFolders.isDisjoint(with: validationFolders))
        XCTAssertEqual(Set(plan.training.map(\.target.url)).count, 9)
        XCTAssertEqual(Set(plan.validation.map(\.target.url)).count, 6)
        for size in [32, 64, 128] {
            XCTAssertEqual(plan.training.filter { $0.input.width == size }.count, 6)
            XCTAssertEqual(plan.validation.filter { $0.input.width == size }.count, 4)
        }
    }

    func testNativeTrainingVisitsBothDiffuseVariantsWithOneOriginalSixteenBitDepthTarget() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("loom-diffuse-variants-" + UUID().uuidString)
        let material = folder.appendingPathComponent("leather", isDirectory: true)
        try FileManager.default.createDirectory(at: material, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let red = material.appendingPathComponent("red.png"), tan = material.appendingPathComponent("tan.png")
        let depth = material.appendingPathComponent("depth.png")
        let redCodes: [UInt16] = (0..<(32 * 32 * 3)).map { index in
            let code: Int = (index * 127 + 971) % 65536
            return UInt16(code)
        }
        let tanCodes: [UInt16] = redCodes.map { UInt16.max - $0 }
        let depthCodes: [UInt16] = (0..<(32 * 32)).map { index in
            let code: Int = (index * 61 + 13107) % 65536
            return UInt16(code)
        }
        try writePNG(redCodes, channels: 3, to: red)
        try writePNG(tanCodes, channels: 3, to: tan)
        try writePNG(depthCodes, channels: 1, to: depth)
        for (url, id, role) in [(red, "red-input", MapRole.diffuse), (tan, "tan-input", .diffuse), (depth, "shared-depth", .depth)] {
            try ImageMetadataStore.write(ImageMetadata(id: id, pairingKey: "shared-leather", role: role, category: "Leather",
                roleWasManuallySelected: true, pairingWasManuallySelected: true), to: url)
        }
        let originals = try [red, tan, depth].map { try Data(contentsOf: $0) }
        let project = try LoomProject(sourceFolder: folder)
        let settings = TrainingSettings(imageSize: 32, epochs: 1, batchSize: 1, learningRate: 0.001,
            shuffleEachEpoch: false, seed: 7, modelWidth: 8, modelBlocks: 1,
            gradientClipPolicy: .fixed, validationFraction: 0)
        let artifact = try await project.trainMap(datasetID: ImageMetadataStore.categoryID("Leather"), map: .depth,
            settings: settings, start: .fresh, runLabel: "Shared target variants", progress: { _ in })
        let request = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("request.json"))
        let samples = try XCTUnwrap(request["samples"] as? [[String: Any]])
        XCTAssertEqual(samples.count, 2)
        XCTAssertEqual(Set(samples.compactMap { $0["id"] as? String }), ["shared-leather|red-input", "shared-leather|tan-input"])
        XCTAssertEqual(Set(samples.compactMap { ($0["input"] as? String).map { URL(fileURLWithPath: $0).lastPathComponent } }),
                       ["red.png", "tan.png"])
        XCTAssertEqual(Set(samples.compactMap { $0["target"] as? String }).count, 1)
        XCTAssertTrue(samples.allSatisfy { ($0["target"] as? String)?.hasSuffix("/depth.png") == true })
        XCTAssertTrue(samples.allSatisfy { $0["split"] as? String == "training" })
        let descriptor = try NativeFactoryTraining.json(artifact.directory.appendingPathComponent("model.json"))
        let training = try XCTUnwrap(descriptor["training"] as? [String: Any])
        let history = try XCTUnwrap(training["history"] as? [[String: Any]])
        XCTAssertEqual(history.count, 1)
        XCTAssertEqual(history.first?["optimizer_updates"] as? Int, 2)
        XCTAssertEqual(training["updates"] as? Int, 2)
        XCTAssertTrue(try XCTUnwrap(descriptor["final_loss"] as? Double).isFinite)
        XCTAssertEqual(try [red, tan, depth].map { try Data(contentsOf: $0) }, originals)
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(depth as CFURL, nil))
        XCTAssertEqual(try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil)).bitsPerComponent, 16)
        let targetPixels = try NativeImages.readTarget(from: depth, mapKind: "depth")
        XCTAssertEqual(targetPixels.values, depthCodes.map { Float($0) / 65535 },
                       "Sharing a target must preserve its original sixteen-bit numeric codes")
        let snapshot = try await project.snapshot()
        let pairs = try TrainingPairSelector.select(images: snapshot.images, dataset: "Leather", map: .depth, size: 32)
        let pair = try XCTUnwrap(pairs.first)
        XCTAssertNoThrow(try NativeFactoryTraining.checkTags(pair, dataset: "Leather", map: .depth, root: project.sourceFolder))
        var changed = try XCTUnwrap(ImageMetadataStore.read(pair.target.url))
        changed.pairingKey = "different-family"
        try ImageMetadataStore.write(changed, to: pair.target.url)
        XCTAssertThrowsError(try NativeFactoryTraining.checkTags(pair, dataset: "Leather", map: .depth, root: project.sourceFolder),
                             "A stable variant ID must not weaken the current pairing metadata check")
    }

    private func member(_ material: String, name: String, role: MapRole, size: Int = 32,
                        pairingKey: String? = nil, normalConvention: String? = nil) -> SourceImage {
        SourceImage(id: "\(material)-\(name)", url: root.appendingPathComponent(material).appendingPathComponent(name + ".png"),
            displayName: name, role: role, category: "Leather", verified: false, note: "",
            pairingKey: pairingKey ?? "\(material)-\(size)", width: size, height: size, normalConvention: normalConvention)
    }

    private func writePNG(_ codes: [UInt16], channels: Int, to url: URL) throws {
        let data = codes.withUnsafeBytes { Data($0) }
        let provider = try XCTUnwrap(CGDataProvider(data: data as CFData))
        let image = try XCTUnwrap(CGImage(width: 32, height: 32, bitsPerComponent: 16, bitsPerPixel: 16 * channels,
            bytesPerRow: 32 * channels * 2,
            space: channels == 1 ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: [.byteOrder16Little], provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }
}
