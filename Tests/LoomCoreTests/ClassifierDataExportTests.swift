import CoreML
import CreateML
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import LoomCore

final class ClassifierDataExportTests: XCTestCase {
    func testIgnoredDiffuseAndConflictingIgnoredMapDoNotEnterClassifierExport() async throws {
        let root = try fixture(), destination = try fixture()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: destination) }
        let input = try image(root, "Active/material_diff.png", size: 1024)
        let map = try image(root, "Active/material_bump.png", size: 1024)
        let ignoredInput = try image(root, "Ignored/material_diff.png", size: 1024)
        let project = try LoomProject(sourceFolder: root)
        let snapshot = try await project.snapshot()
        try tag(input, "Wood"); try tag(map, "Wrong category"); try tag(ignoredInput, "Wood")
        for file in [map, ignoredInput] {
            let id = try XCTUnwrap(snapshot.images.first { $0.url == file }?.id)
            try await project.setIgnored(id: id, ignored: true)
        }
        let exported = try await project.exportClassifierData(to: destination)
        XCTAssertEqual(exported.imageCount, 1); XCTAssertEqual(exported.folderCount, 1)
        let links = try contents(exported.directory).flatMap(contents)
        XCTAssertEqual(links.map { $0.resolvingSymlinksInPath() }, [input])
    }

    func testCreateMLExportLinksOneNative1KDiffusePerMaterialUnderItsCategory() async throws {
        let root = try fixture(), destination = try fixture()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: destination) }
        let first = try image(root, "Oak/oak_diff_1k.png", size: 1024)
        _ = try image(root, "Oak/oak_diff_2k.png", size: 2048)
        _ = try image(root, "Oak/oak_disp_1k.png", size: 1024)
        let second = try image(root, "Pine/pine_diff_1k.png", size: 1024)
        let cloth = try image(root, "Cloth/cloth_diff_1k.png", size: 1024)
        let sourceBytes = try [first, second, cloth].map { try Data(contentsOf: $0) }
        let project = try LoomProject(sourceFolder: root)
        _ = try await project.snapshot()
        try tag(first, "Wood"); try tag(second, "Wood"); try tag(cloth, "Fabric")

        let result = try await project.exportClassifierData(to: destination)

        XCTAssertEqual(result.imageCount, 3)
        XCTAssertEqual(result.folderCount, 2)
        XCTAssertTrue(result.skippedFolders.isEmpty)
        let categories = try contents(result.directory)
        XCTAssertEqual(Set(categories.map(\.lastPathComponent)), ["Wood", "Fabric"])
        let links = try categories.flatMap(contents)
        XCTAssertEqual(links.count, 3)
        XCTAssertTrue(try links.allSatisfy { try $0.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true })
        XCTAssertEqual(Set(links.map { $0.resolvingSymlinksInPath() }), Set([first, second, cloth].map { $0.resolvingSymlinksInPath() }))
        XCTAssertEqual(try [first, second, cloth].map { try Data(contentsOf: $0) }, sourceBytes)
        XCTAssertTrue(try [first, second, cloth].allSatisfy { try ImageMetadataStore.read($0)?.verified == false })
        let accepted = try MLImageClassifier.DataSource.labeledDirectories(at: result.directory).labeledImages()
        XCTAssertEqual(Set(accepted.keys), ["Wood", "Fabric"])
        XCTAssertEqual(accepted.values.reduce(0) { $0 + $1.count }, 3)
    }

    func testExportReportsMissing1KTagsAndConflictsAndNeverTouchesExistingDestinationFiles() async throws {
        let root = try fixture(), destination = try fixture()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: destination) }
        let good = try image(root, "Good/good_diff.png", size: 1024)
        let tooLarge = try image(root, "No1K/large_diff.png", size: 2048)
        _ = try image(root, "Untagged/blank_diff.png", size: 1024)
        let mixed = try image(root, "Conflicting/mixed_diff.png", size: 1024)
        let mixedMap = try image(root, "Conflicting/mixed_disp.png", size: 1024)
        let project = try LoomProject(sourceFolder: root)
        _ = try await project.snapshot()
        try tag(good, "Wood"); try tag(tooLarge, "Wood"); try tag(mixed, "Wood"); try tag(mixedMap, "Fabric")
        let existing = destination.appendingPathComponent("keep.txt")
        try Data("User's existing file".utf8).write(to: existing)

        let first = try await project.exportClassifierData(to: destination)
        let second = try await project.exportClassifierData(to: destination)

        XCTAssertEqual(first.imageCount, 1)
        XCTAssertEqual(Set(first.skippedFolders.map(\.lastPathComponent)), ["No1K", "Untagged", "Conflicting"])
        XCTAssertNotEqual(first.directory, second.directory)
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.directory.path))
        XCTAssertEqual(try String(contentsOf: existing, encoding: .utf8), "User's existing file")
        XCTAssertFalse(try contents(destination).contains { $0.lastPathComponent.hasPrefix(".create-ml-") })
    }

    func testUnsafeAndCollidingCategoryNamesProduceSafeDistinctFoldersWithSourceLabelMapping() async throws {
        let root = try fixture(), destination = try fixture()
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: destination) }
        let first = try image(root, "One/one_diff.png", size: 1024)
        let second = try image(root, "Two/two_diff.png", size: 1024)
        let project = try LoomProject(sourceFolder: root)
        _ = try await project.snapshot()
        try tag(first, "Rock/Wood"); try tag(second, "Rock:Wood")
        let exported = try await project.exportClassifierData(to: destination)
        let names = try contents(exported.directory).map(\.lastPathComponent)
        XCTAssertEqual(Set(names).count, 2)
        XCTAssertTrue(names.allSatisfy { !$0.contains("/") && !$0.contains(":") && $0.hasPrefix("Rock Wood-") })
        let catalogFolder = project.projectFolder.appendingPathComponent("classifier-data-exports")
        let descriptorURL = try XCTUnwrap(contents(catalogFolder).first)
        let descriptor = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: descriptorURL)) as? [String: Any])
        let records = try XCTUnwrap(descriptor["labels"] as? [[String: String]])
        let current = try await project.snapshot()
        let mapped = try classifierLabelCatalog(classes: names, datasets: current.datasets, exportedLabels: records)
        XCTAssertEqual(Set(mapped.compactMap { $0["id"] }), Set(current.datasets.map(\.id)))
        XCTAssertEqual(Set(mapped.compactMap { $0["sourceLabel"] }), Set(names))
    }

    func testLabelValidationRejectsUnknownDuplicateAndAmbiguousClasses() throws {
        let wood = dataset("Wood"), fabric = dataset("Fabric")
        XCTAssertThrowsError(try classifierLabelCatalog(classes: ["Ground"], datasets: [wood, fabric]))
        XCTAssertThrowsError(try classifierLabelCatalog(classes: ["Wood", "Wood"], datasets: [wood, fabric]))
        XCTAssertThrowsError(try classifierLabelCatalog(classes: ["Wood", "wood"], datasets: [wood, fabric]))
        XCTAssertThrowsError(try classifierLabelCatalog(classes: [""], datasets: [wood, fabric]))
        XCTAssertThrowsError(try classifierLabelCatalog(classes: ["Wood"], datasets: [wood, fabric]))
        let mapping = [["id": fabric.id, "name": fabric.name, "sourceLabel": "Wood"]]
        XCTAssertThrowsError(try classifierLabelCatalog(classes: ["Wood"], datasets: [wood, fabric], exportedLabels: mapping))
        let valid = try classifierLabelCatalog(classes: ["WOOD", fabric.id], datasets: [wood, fabric])
        XCTAssertEqual(valid.compactMap { $0["id"] }, [wood.id, fabric.id])
        XCTAssertEqual(valid.compactMap { $0["sourceLabel"] }, ["WOOD", fabric.id])
    }

    func testImportedNativeClassifierKeepsDeclaredLabelsInterfaceAndNoArtificialWeights() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let asset = classifierFixture()
        let compiled = asset.pathExtension == "mlmodelc" ? asset : try await MLModel.compileModel(at: asset)
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        let native = try MLModel(contentsOf: compiled, configuration: configuration)
        let classNames = try XCTUnwrap(native.modelDescription.classLabels as? [String])
        let project = try LoomProject(sourceFolder: root)
        for (index, label) in classNames.enumerated() {
            let file = try image(root, "Material-\(index)/sample_diff.png", size: 32)
            try ImageMetadataStore.write(ImageMetadata(pairingKey: "pair-\(index)", role: .diffuse,
                category: ImageMetadataStore.friendlyName(label)), to: file)
        }
        let assetBytes: Data? = asset.hasDirectoryPath || asset.pathExtension == "mlmodelc" ? nil : try Data(contentsOf: asset)
        let artifact = try await project.importClassifier(from: asset)
        XCTAssertEqual(artifact.kind, "classifier")
        XCTAssertEqual(artifact.architecture, "create-ml-image-classifier-v1")
        XCTAssertNil(artifact.weightsPath)
        XCTAssertEqual(artifact.epochs, 0)
        let imported = artifact.directory.appendingPathComponent(try XCTUnwrap(artifact.coreMLPath))
        XCTAssertTrue(FileManager.default.fileExists(atPath: imported.path))
        if let assetBytes { XCTAssertEqual(try Data(contentsOf: imported), assetBytes) }
        let descriptor = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: artifact.directory.appendingPathComponent("model.json"))) as? [String: Any])
        XCTAssertEqual(descriptor["input_name"] as? String, native.modelDescription.inputDescriptionsByName.keys.first)
        XCTAssertEqual(descriptor["output_name"] as? String, native.modelDescription.predictedFeatureName)
        XCTAssertEqual(descriptor["input_type"] as? String,
                       native.modelDescription.inputDescriptionsByName.values.first?.imageConstraint == nil ? "multiArray" : "image")
        let labels = try XCTUnwrap(descriptor["labels"] as? [[String: String]])
        XCTAssertEqual(Set(labels.compactMap { $0["sourceLabel"] }), Set(classNames))
        XCTAssertEqual(Set(labels.compactMap { $0["id"] }).count, classNames.count)
        if let image = native.modelDescription.inputDescriptionsByName.values.first?.imageConstraint {
            XCTAssertEqual(descriptor["input_width"] as? Int, image.pixelsWide)
            XCTAssertEqual(descriptor["input_height"] as? Int, image.pixelsHigh)
            XCTAssertEqual(descriptor["input_color_space"] as? String, "RGB")
        }
        let reopened = try LoomProject(sourceFolder: root)
        let restored = try await reopened.snapshot()
        XCTAssertEqual(restored.artifacts.first?.id, artifact.id)
        XCTAssertNil(restored.artifacts.first?.weightsPath)
    }

    func testImportRejectsClassifierThatOmitsOneCurrentCategory() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let asset = classifierFixture()
        let interface = try await nativeClassifierInterface(at: asset)
        let project = try LoomProject(sourceFolder: root)
        try addCategories(interface.labels.map(ImageMetadataStore.friendlyName) + ["Additional Current Dataset"], to: root)
        do {
            _ = try await project.importClassifier(from: asset)
            XCTFail("Import accepted a classifier that cannot route one current dataset")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("cover every current source dataset"))
        }
        let snapshot = try await project.snapshot()
        XCTAssertTrue(snapshot.artifacts.isEmpty)
    }

    func testPackageRejectsNativeLabelOrInterfaceDriftAndRoutingToMissingModels() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let asset = classifierFixture()
        let interface = try await nativeClassifierInterface(at: asset)
        let project = try LoomProject(sourceFolder: root)
        try addCategories(interface.labels.map(ImageMetadataStore.friendlyName), to: root)
        let classifier = try await project.importClassifier(from: asset)
        let descriptorURL = classifier.directory.appendingPathComponent("model.json")
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: descriptorURL)) as? [String: Any])

        var changed = original
        var labels = try XCTUnwrap(original["labels"] as? [[String: String]])
        labels[0]["sourceLabel"] = "A label absent from the native classifier"
        changed["labels"] = labels
        try writeDescriptor(changed, to: descriptorURL)
        await assertExportFails(project, containing: "native labels or interface changed")

        changed = original
        changed["input_name"] = "An input absent from the native classifier"
        try writeDescriptor(changed, to: descriptorURL)
        await assertExportFails(project, containing: "native labels or interface changed")

        changed = original
        changed["project_id"] = UUID().uuidString
        try writeDescriptor(changed, to: descriptorURL)
        await assertExportFails(project, containing: "different source project")

        try writeDescriptor(original, to: descriptorURL)
        await assertExportFails(project, containing: "Train at least one map model")

        let extra = try image(root, "Extra/extra_diff.png", size: 32)
        try ImageMetadataStore.write(ImageMetadata(pairingKey: "extra", role: .diffuse,
                                                  category: "An Additional Dataset"), to: extra)
        await assertExportFails(project, containing: "does not match every dataset")
        let exports = project.projectFolder.appendingPathComponent("exports")
        XCTAssertTrue(try contents(exports).isEmpty, "Rejected exports must not leave an archive or staging folder")
    }

    func testClassifierImportRejectsMismatchedSourceCategoriesWithoutCreatingRun() async throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let image = try image(root, "OnlyOne/material_diff.png", size: 32)
        let project = try LoomProject(sourceFolder: root)
        _ = try await project.snapshot()
        try tag(image, "Wood")
        do {
            _ = try await project.importClassifier(from: classifierFixture())
            XCTFail("A classifier with unknown classes must not be attached to this source project")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("does not match"))
        }
        let snapshot = try await project.snapshot()
        XCTAssertTrue(snapshot.artifacts.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: project.projectFolder.appendingPathComponent("runs").path))
    }

    private func dataset(_ name: String) -> DatasetCategory {
        DatasetCategory(id: ImageMetadataStore.categoryID(name), name: name, imageCount: 1, verifiedCount: 0, pairCounts: [:])
    }

    private func addCategories(_ categories: [String], to root: URL) throws {
        for (index, category) in categories.enumerated() {
            let file = try image(root, "Material-\(index)/sample_diff.png", size: 32)
            try ImageMetadataStore.write(ImageMetadata(pairingKey: "pair-\(index)", role: .diffuse,
                                                      category: category), to: file)
        }
    }

    private func writeDescriptor(_ descriptor: [String: Any], to file: URL) throws {
        try JSONSerialization.data(withJSONObject: descriptor, options: [.sortedKeys]).write(to: file, options: .atomic)
    }

    private func assertExportFails(_ project: LoomProject, containing message: String,
                                  file: StaticString = #filePath, line: UInt = #line) async {
        do {
            _ = try await project.exportPackage(name: "Fixture Texture Loom", version: "1.0.0")
            XCTFail("The inconsistent classifier exported a deployment", file: file, line: line)
        } catch {
            XCTAssertTrue(error.localizedDescription.contains(message), error.localizedDescription, file: file, line: line)
        }
    }

    private func classifierFixture() -> URL {
        if let configured = ProcessInfo.processInfo.environment["LOOM_FACTORY_CLASSIFIER_FIXTURE"] { return URL(fileURLWithPath: configured) }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Resources/DTTC3.mlmodel")
    }

    private func tag(_ file: URL, _ category: String) throws {
        var metadata = try XCTUnwrap(ImageMetadataStore.read(file))
        metadata.category = category
        try ImageMetadataStore.write(metadata, to: file)
    }

    private func contents(_ directory: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-classifier-export-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @discardableResult private func image(_ root: URL, _ name: String, size: Int) throws -> URL {
        let file = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let pixels = [UInt8](repeating: 120, count: size * size * 3)
        let provider = try XCTUnwrap(CGDataProvider(data: Data(pixels) as CFData))
        let image = try XCTUnwrap(CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 24,
                                        bytesPerRow: size * 3, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [],
                                        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return file
    }
}
