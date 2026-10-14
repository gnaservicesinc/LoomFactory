import Foundation
import LoomCore
import Observation
import Testing
@testable import LoomFactory

@Suite(.serialized)
struct FactoryPresentationTests {
    @Test @MainActor
    func unchangedFolderPollDoesNotPublishAnotherSnapshot() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("loom-presentation-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = FactoryModel(rememberSourceFolder: false)
        model.openFolder(root)
        for _ in 0..<250 where model.busy { try await Task.sleep(for: .milliseconds(20)) }
        #expect(!model.busy)
        #expect(model.errorMessage == nil)
        _ = try #require(model.snapshot)
        let changes = PresentationChangeCounter()
        withObservationTracking { _ = model.snapshot } onChange: { changes.increment() }
        await model.pollFolder()
        #expect(changes.value == 0)
        #expect(model.errorMessage == nil)
    }

    @Test
    func largeMaterialIndexSupportsFastSelectionAndPreservesFiltering() throws {
        let source = URL(fileURLWithPath: "/Synthetic Materials")
        let roles: [MapRole] = [.diffuse, .depth, .roughness, .normal]
        let images = (0..<500).flatMap { material in
            [512, 1024, 2048].flatMap { size in
                roles.map { role in
                    SourceImage(id: "\(material)-\(size)-\(role.rawValue)",
                        url: source.appendingPathComponent("material_\(material)/\(size)_\(role.rawValue).png"),
                        displayName: "Material \(material) \(role.displayName)", role: role,
                        category: material % 2 == 0 ? "Bark" : "Bricks", verified: material % 3 == 0,
                        note: "", pairingKey: "\(material)-\(size)", width: size, height: size)
                }
            }
        }
        let snapshot = ProjectSnapshot(sourceURL: source, projectID: "synthetic", settings: .init(),
            images: images, datasets: [], artifacts: [])
        let index = FactoryBrowserIndex(snapshot: snapshot)
        #expect(index.folders.count == 500)
        #expect(index.matching(datasetID: "bark", datasetName: "Bark", search: "").count == 3000)
        #expect(index.matching(datasetID: "", datasetName: nil, search: "2048_depth").count == 500)
        let ids = images.map { "image:" + $0.id }
        let clock = ContinuousClock()
        let elapsed = clock.measure {
            for _ in 0..<10 {
                for id in ids { _ = index.materialBrowserIndex.row(id: id) }
            }
        }
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        print("Material index: 6,000 maps; 60,000 cached selections in \(seconds) seconds")
        #expect(seconds < 0.5)
        #expect(index.materialBrowserIndex.folder(containingImageID: images.last?.id)?.imageCount == 12)
    }
}

private final class PresentationChangeCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var value: Int { lock.withLock { count } }
    func increment() { lock.withLock { count += 1 } }
}
