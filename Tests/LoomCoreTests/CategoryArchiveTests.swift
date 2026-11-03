import CryptoKit
import Foundation
import ImageIO
import LoomNative
import Synchronization
import XCTest
@testable import LoomCore

@MainActor
final class CategoryArchiveTests: XCTestCase {
    func testClosingRequiresAllThreeCurrentApprovalsAndRejectsAStalePreview() async throws {
        let f = try await Fixture(); defer { f.remove() }
        for map in MapRole.targets {
            let before = try await f.project.snapshot()
            XCTAssertFalse(before.canCloseCategory(f.categoryID))
            try await f.approve(map)
        }
        let approved = try await f.project.snapshot()
        XCTAssertTrue(approved.canCloseCategory(f.categoryID))
        let preview = try await f.preview()
        XCTAssertEqual(preview.fileCount, 1)
        try Data("changed weights".utf8).write(to: f.project.projectFolder.appendingPathComponent("runs/depth/model.safetensors"))
        do { try await f.project.closeCategory(preview); XCTFail("Stale approvals must block deletion") }
        catch { XCTAssertTrue(error.localizedDescription.contains("approvals changed")) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: f.original.path))
    }

    func testPreviewDoesNotDeleteAndCloseKeepsGeneratedEditedAndLinkedFiles() async throws {
        let f = try await Fixture(); defer { f.remove() }
        try await f.approveAll()
        let generated = try f.image("generated/art.png", pixel: 73)
        let edited = try f.image("stone/stone_rough_1k.png", pixel: 90, role: .roughness)
        let publishedOther = try Fixture.png(pixel: 91)
        let external = f.root.deletingLastPathComponent().appendingPathComponent(UUID().uuidString + ".png")
        try Fixture.png(pixel: 30).write(to: external)
        defer { try? FileManager.default.removeItem(at: external) }
        try ImageMetadataStore.write(.init(pairingKey: "external", role: .depth, category: "Tiles"), to: external)
        let linked = f.root.appendingPathComponent("linked_disp_1k.png")
        try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: external)
        let preview = try await f.preview(extra: ["stone_rough_1k.png": publishedOther])
        XCTAssertEqual(preview.fileCount, 1)
        XCTAssertEqual(preview.preservedFileCount, 3)
        XCTAssertTrue(FileManager.default.fileExists(atPath: f.original.path))
        let modelsBefore = try await f.project.snapshot().artifacts
        try await f.project.closeCategory(preview)
        XCTAssertFalse(FileManager.default.fileExists(atPath: f.original.path))
        for file in [generated, edited, linked, external] { XCTAssertTrue(FileManager.default.fileExists(atPath: file.path)) }
        let after = try await f.project.snapshot()
        XCTAssertEqual(after.artifacts, modelsBefore)
        XCTAssertEqual(after.modelCoverage.filter { $0.slot.datasetID == f.categoryID && $0.status == .approved }.count, 3)
        XCTAssertEqual(after.closedCategories.count, 1)
    }

    func testEmptyClosedCategorySurvivesReopeningProjectAndAcceptsUpdatedProviderBytes() async throws {
        let f = try await Fixture(); defer { f.remove() }
        try await f.approveAll()
        let before = try await f.project.snapshot()
        let tag = try XCTUnwrap(ImageMetadataStore.read(f.original))
        let preview = try await f.preview()
        try await f.project.closeCategory(preview)
        let reopened = try LoomProject(sourceFolder: f.root)
        let closed = try await reopened.snapshot()
        XCTAssertTrue(closed.images.isEmpty)
        XCTAssertEqual(closed.datasets.map(\.id), before.datasets.map(\.id))
        XCTAssertEqual(closed.modelCoverage.filter { $0.status == .approved }.count, 3)
        do {
            _ = try await reopened.prepareTrainingSources(datasetID: f.categoryID, map: .depth, settings: .init()) { _ in }
            XCTFail("Training source preparation must require explicit reopening")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Reopen")) }
        let updated = try Fixture.png(pixel: 155)
        try await reopened.reopenCategory(datasetID: f.categoryID, session: f.session(bytes: updated, version: "updated")) { _ in }
        XCTAssertEqual(try Data(contentsOf: f.original), updated, "Provider updates are accepted, without a historical checksum requirement")
        XCTAssertEqual(try ImageMetadataStore.read(f.original), tag)
        let after = try await reopened.snapshot()
        XCTAssertTrue(after.closedCategories.isEmpty)
        XCTAssertEqual(after.images.count, 1)
        XCTAssertEqual(after.modelCoverage.filter { $0.status == .approved }.count, 3)
    }

    func testChangedSourceAfterPreviewIsNeverDeletedAndLeaseBlocksMaintenance() async throws {
        let f = try await Fixture(); defer { f.remove() }
        try await f.approveAll()
        let preview = try await f.preview()
        do {
            let lease = try TrainingQueueLease(projectFolder: f.project.projectFolder, controlFolder: f.root)
            do { try await f.project.closeCategory(preview); XCTFail("An active training lease must block closing") }
            catch { XCTAssertTrue(error.localizedDescription.contains("already using")) }
            withExtendedLifetime(lease) {}
        }
        try Fixture.png(pixel: 50).write(to: f.original)
        do { try await f.project.closeCategory(preview); XCTFail("A changed file must block deletion") }
        catch { XCTAssertTrue(error.localizedDescription.contains("changed after the preview")) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: f.original.path))
    }

    func testChangedMetadataAfterPreviewBlocksAllDeletions() async throws {
        let f = try await Fixture(); defer { f.remove() }
        try await f.approveAll()
        let preview = try await f.preview()
        var metadata = try XCTUnwrap(ImageMetadataStore.read(f.original)); metadata.note = "Edited after preview"
        try ImageMetadataStore.write(metadata, to: f.original)
        do { try await f.project.closeCategory(preview); XCTFail("Changed files must not be deleted") }
        catch { XCTAssertTrue(error.localizedDescription.contains("changed after the preview")) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: f.original.path))
        let after = try await f.project.snapshot()
        XCTAssertTrue(after.closedCategories.isEmpty)
    }

    func testFailedReopenKeepsRecipeAndCanRetry() async throws {
        let f = try await Fixture(); defer { f.remove() }
        try await f.approveAll()
        let preview = try await f.preview()
        try await f.project.closeCategory(preview)
        do {
            try await f.project.reopenCategory(datasetID: f.categoryID, session: f.session(bytes: Data("bad image".utf8))) { _ in }
            XCTFail("Invalid image data must not reopen the category")
        } catch { XCTAssertTrue(error.localizedDescription.contains("valid image")) }
        let failed = try await f.project.snapshot()
        XCTAssertEqual(failed.closedCategories.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: f.original.path))
        try await f.project.reopenCategory(datasetID: f.categoryID, session: f.session(bytes: f.bytes)) { _ in }
        let after = try await f.project.snapshot()
        XCTAssertTrue(after.closedCategories.isEmpty)
    }

    func testAmbientCGDirectoryVerifiesOriginalsAndCurrentZipRestoresThem() async throws {
        let f = try await Fixture(ambient: true); defer { f.remove() }
        try await f.approveAll()
        let zip = try f.zip(bytes: f.bytes)
        let preview = try await f.project.previewCategoryClose(datasetID: f.categoryID, session: f.ambientSession(zip)) { _ in }
        XCTAssertEqual(preview.fileCount, 1)
        try await f.project.closeCategory(preview)
        let updated = try Fixture.png(pixel: 101)
        try await f.project.reopenCategory(datasetID: f.categoryID, session: f.ambientSession(f.zip(bytes: updated))) { _ in }
        XCTAssertEqual(try Data(contentsOf: f.original), updated)
        let after = try await f.project.snapshot()
        XCTAssertTrue(after.closedCategories.isEmpty)
    }

    func testPartialReopenRetainsCompletedFilesAndRetriesRemainingDownloads() async throws {
        let f = try await Fixture(); defer { f.remove() }
        try await f.approveAll()
        let rough = try f.image("stone/stone_rough_1k.png", pixel: 44, role: .roughness)
        let roughBytes = try Data(contentsOf: rough)
        let extra = [rough.lastPathComponent: roughBytes]
        let preview = try await f.preview(extra: extra)
        XCTAssertEqual(preview.fileCount, 2)
        try await f.project.closeCategory(preview)
        do {
            try await f.project.reopenCategory(datasetID: f.categoryID, session: f.session(bytes: f.bytes, extra: extra, failFilename: rough.lastPathComponent)) { _ in }
            XCTFail("The failed second download must leave the category closed")
        } catch { XCTAssertTrue(error.localizedDescription.contains("incomplete")) }
        XCTAssertEqual(try Data(contentsOf: f.original), f.bytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath: rough.path))
        let partial = try await f.project.snapshot()
        XCTAssertEqual(partial.closedCategories.count, 1)
        try await f.project.reopenCategory(datasetID: f.categoryID, session: f.session(bytes: try Fixture.png(pixel: 230), extra: extra)) { _ in }
        XCTAssertEqual(try Data(contentsOf: f.original), f.bytes, "Already restored files are not replaced")
        XCTAssertEqual(try Data(contentsOf: rough), roughBytes)
        let complete = try await f.project.snapshot()
        XCTAssertTrue(complete.closedCategories.isEmpty)
    }

    func testUnpublishedOrGeneratedProviderStyleNameIsPreserved() async throws {
        let f = try await Fixture(); defer { f.remove() }
        try await f.approveAll()
        let session = ArchiveURLProtocol.session { _ in (Data(), 404, [:]) }
        let preview = try await f.project.previewCategoryClose(datasetID: f.categoryID, session: session) { _ in }
        XCTAssertEqual(preview.fileCount, 0)
        XCTAssertEqual(preview.preservedFileCount, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: f.original.path))
    }

    func testReopenConflictsKeepModelsVisibleAndNeverOverwriteExternalFiles() async throws {
        let f = try await Fixture(); defer { f.remove() }
        try await f.approveAll()
        let preview = try await f.preview()
        try await f.project.closeCategory(preview)
        let external = f.root.deletingLastPathComponent().appendingPathComponent(UUID().uuidString + ".png")
        let externalBytes = try Fixture.png(pixel: 17); try externalBytes.write(to: external)
        defer { try? FileManager.default.removeItem(at: external) }
        try FileManager.default.createSymbolicLink(at: f.original, withDestinationURL: external)
        let snapshot = try await f.project.snapshot()
        XCTAssertEqual(snapshot.closedCategories.count, 1)
        XCTAssertEqual(snapshot.modelCoverage.filter { $0.status == .approved }.count, 3)
        do {
            try await f.project.reopenCategory(datasetID: f.categoryID, session: f.session(bytes: f.bytes)) { _ in }
            XCTFail("A conflicting link must be left alone")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Linked or external")) }
        XCTAssertEqual(try Data(contentsOf: external), externalBytes)
    }

    func testLiveProviderCatalogsAndAmbientZIPRangeDirectory() async throws {
        guard let path = ProcessInfo.processInfo.environment["LOOM_TEST_CATEGORY_CATALOGS"] else { throw XCTSkip("Set LOOM_TEST_CATEGORY_CATALOGS for read-only live provider checks") }
        let root = URL(fileURLWithPath: path)
        let session = CategorySourceCatalog.session()
        defer { session.invalidateAndCancel() }
        let poly = root.appendingPathComponent("Stone Tile Wall/stone_tile_wall_diff_1k.png")
        let polyIdentity = try XCTUnwrap(CategorySourceIdentity.recognize(poly))
        let polyCatalog = try await CategorySourceCatalog.catalog(polyIdentity, session: session)
        let polyFile = try XCTUnwrap(CategorySourceCatalog.polyFile(polyCatalog, filename: poly.lastPathComponent))
        XCTAssertEqual(try CategoryArchiveFiles.digest(poly).md5, polyFile.md5)
        let ambient = root.appendingPathComponent("Bricks059/Bricks059_1K-PNG_Color.png")
        let ambientIdentity = try XCTUnwrap(CategorySourceIdentity.recognize(ambient))
        let ambientCatalog = try await CategorySourceCatalog.catalog(ambientIdentity, session: session)
        let package = try XCTUnwrap(CategorySourceCatalog.ambientPackage(ambientCatalog, identity: ambientIdentity))
        let directory = try await CategorySourceCatalog.zipEntries(package, session: session)
        let entry = try XCTUnwrap(directory.first { $0.name == ambient.lastPathComponent })
        XCTAssertEqual(try CategoryArchiveFiles.digest(ambient).crc32, entry.crc32)
        print("LIVE_CATEGORY_CATALOGS=Poly Haven MD5 and ambientCG ZIP CRC matched local originals; no source files were changed")
    }

    func testRecoveryPathsRejectTraversalAndSymlinkedParents() throws {
        let root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for path in ["../outside.png", "/outside.png", ".loom-factory/project.json", "stone/../other.png", "stone//map.png"] {
            XCTAssertThrowsError(try CategoryArchiveFiles.destination(path, root: root))
        }
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("linked"), withDestinationURL: root.deletingLastPathComponent())
        XCTAssertThrowsError(try CategoryArchiveFiles.destination("linked/new.png", root: root))
        XCTAssertNoThrow(try CategoryArchiveFiles.destination("new-folder/new.png", root: root))
        XCTAssertThrowsError(try CategorySourceCatalog.parseZIPDirectory(Data(repeating: 0, count: 48), count: 1))
    }

    private struct Fixture {
        let root: URL
        let project: LoomProject
        let categoryID = ImageMetadataStore.categoryID("Tiles")
        let original: URL
        let bytes: Data
        init(ambient: Bool = false) async throws {
            root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent("loom-close-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            original = root.appendingPathComponent(ambient ? "Bricks059/Bricks059_1K-PNG_Color.png" : "stone/stone_diff_1k.png")
            try FileManager.default.createDirectory(at: original.deletingLastPathComponent(), withIntermediateDirectories: true)
            bytes = try Self.png(pixel: 120); try bytes.write(to: original)
            try ImageMetadataStore.write(.init(pairingKey: "hand-paired", role: .diffuse, category: "Tiles", verified: true, note: "keep my review", pairingWasManuallySelected: true), to: original)
            project = try LoomProject(sourceFolder: root)
            let projectID = try await project.snapshot().projectID
            for map in MapRole.targets {
                let directory = project.projectFolder.appendingPathComponent("runs/\(map.rawValue)")
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try Data([1, 2, 3]).write(to: directory.appendingPathComponent("model.safetensors"))
                try Data([4, 5, 6]).write(to: directory.appendingPathComponent("model.mlmodel"))
                let descriptor: [String: Any] = ["schema": "loom-factory-model-v1", "status": "completed", "project_id": projectID,
                    "kind": "map", "dataset_id": categoryID, "dataset_name": "Tiles", "map": map.rawValue,
                    "architecture": NativeArchitecture.identifier, "architecture_config": ["id": NativeArchitecture.identifier, "width": 8, "blocks": 1],
                    "image_size": 32, "training": ["image_size": 32], "input_name": "rgb", "output_name": "material_map",
                    "weights_path": "model.safetensors", "coreml_path": "model.mlmodel"]
                try JSONSerialization.data(withJSONObject: descriptor).write(to: directory.appendingPathComponent("model.json"))
                let artifact = ModelArtifact(id: map.rawValue, kind: "map", datasetID: categoryID, datasetName: "Tiles", map: map,
                    directory: directory, createdAt: .now, epochs: 1, loss: nil, coreMLPath: "model.mlmodel", weightsPath: "model.safetensors", architecture: NativeArchitecture.identifier)
                try JSONEncoder().encode(artifact).write(to: directory.appendingPathComponent("artifact.json"))
                try await project.selectModelVersion(runID: map.rawValue, datasetID: categoryID, map: map)
            }
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
        func approve(_ map: MapRole) async throws { _ = try await project.setModelApproval(runID: map.rawValue, approved: true) }
        func approveAll() async throws { for map in MapRole.targets { try await approve(map) } }
        func image(_ path: String, pixel: UInt8, role: MapRole = .diffuse) throws -> URL {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Self.png(pixel: pixel).write(to: url)
            try ImageMetadataStore.write(.init(pairingKey: path, role: role, category: "Tiles"), to: url)
            return url
        }
        func preview(extra: [String: Data] = [:]) async throws -> CategoryClosePreview {
            try await project.previewCategoryClose(datasetID: categoryID, session: session(bytes: bytes, extra: extra)) { _ in }
        }
        func session(bytes: Data, version: String = "original", extra: [String: Data] = [:], failFilename: String? = nil) -> URLSession {
            var files = extra; files[original.lastPathComponent] = bytes
            let catalog = try! JSONSerialization.data(withJSONObject: files.mapValues { value in
                ["size": value.count, "md5": Insecure.MD5.hash(data: value).map { String(format: "%02x", $0) }.joined()] as [String: Any]
            }.reduce(into: [String: Any]()) { result, pair in
                var file = pair.value; file["url"] = "https://dl.polyhaven.org/\(version)/\(pair.key)"; result[pair.key] = file
            })
            let publishedFiles = files
            return ArchiveURLProtocol.session { request in
                if request.url!.host == "api.polyhaven.com" { return (catalog, 200, [:]) }
                if request.url!.lastPathComponent == failFilename { return (Data(), 500, [:]) }
                return (publishedFiles[request.url!.lastPathComponent] ?? Data(), 200, [:])
            }
        }
        func zip(bytes: Data) throws -> Data {
            let folder = root.appendingPathComponent("zip-fixture-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: folder) }
            let image = folder.appendingPathComponent(original.lastPathComponent); try bytes.write(to: image)
            let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
            process.currentDirectoryURL = folder; process.arguments = ["-q", "source.zip", image.lastPathComponent]
            try process.run(); process.waitUntilExit(); XCTAssertEqual(process.terminationStatus, 0)
            return try Data(contentsOf: folder.appendingPathComponent("source.zip"))
        }
        func ambientSession(_ zip: Data) -> URLSession {
            let catalog = try! JSONSerialization.data(withJSONObject: ["assets": [["id": "Bricks059", "downloads": [["url": "https://ambientcg.com/get?file=Bricks059_1K-PNG.zip", "size": zip.count]]]]])
            return ArchiveURLProtocol.session { request in
                if request.url!.path.hasPrefix("/api/") { return (catalog, 200, [:]) }
                if let range = request.value(forHTTPHeaderField: "Range") {
                    let limits = range.dropFirst(6).split(separator: "-").map { Int($0)! }
                    return (zip.subdata(in: limits[0]..<(limits[1] + 1)), 206, ["Content-Range": "bytes \(limits[0])-\(limits[1])/\(zip.count)"])
                }
                return (zip, 200, [:])
            }
        }
        static func png(pixel: UInt8) throws -> Data {
            let data = NSMutableData()
            let provider = try XCTUnwrap(CGDataProvider(data: Data(repeating: pixel, count: 32 * 32) as CFData))
            let image = try XCTUnwrap(CGImage(width: 32, height: 32, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: 32,
                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: [], provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
            let destination = try XCTUnwrap(CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil); XCTAssertTrue(CGImageDestinationFinalize(destination))
            return data as Data
        }
    }
}

private final class ArchiveURLProtocol: URLProtocol, @unchecked Sendable {
    typealias Reply = @Sendable (URLRequest) -> (Data, Int, [String: String])
    static let reply = Mutex<Reply?>(nil)
    static func session(_ handler: @escaping Reply) -> URLSession {
        reply.withLock { $0 = handler }
        let config = URLSessionConfiguration.ephemeral; config.protocolClasses = [Self.self]
        return URLSession(configuration: config)
    }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let handler = Self.reply.withLock { $0! }
        let (data, status, headers) = handler(request)
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers.merging(["Content-Length": String(data.count)]) { first, _ in first })!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
