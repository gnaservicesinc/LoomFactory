import Foundation
import LoomNative
import XCTest
@testable import LoomCore

private final class PhotoStopRequest: @unchecked Sendable {
    private let lock = NSLock()
    private var task: Task<TrainingCheckpointReceipt, Error>?
    func afterFirstUpdate(_ event: TrainingEvent, project: LoomProject) {
        guard event.message.contains(" · update 1 · ") else { return }
        lock.withLock {
            guard task == nil else { return }
            task = Task { try await project.saveTrainingCheckpoint(stopAfterSaving: true) }
        }
    }
    func requested() -> Task<TrainingCheckpointReceipt, Error>? { lock.withLock { task } }
}

final class PhotoEnhancerLifecycleTests: XCTestCase {
    func testReal1KSaveStopExactContinuationAndBestExportWhenEnabled() async throws {
        guard ProcessInfo.processInfo.environment["LOOM_PHOTO_GPU_1K"] == "1",
              let path = ProcessInfo.processInfo.environment["LOOM_PHOTO_NATIVE_DATASET"],
              let outputPath = ProcessInfo.processInfo.environment["LOOM_PHOTO_NATIVE_OUTPUT"] else {
            throw XCTSkip("Opt-in short real1K lifecycle probe; requires explicit dataset/output and an idle material project.")
        }
        let live = URL(fileURLWithPath:ProcessInfo.processInfo.environment["LOOM_PHOTO_GPU_PROJECT"] ?? "/opt/ipde/sources_mats/.loom-factory")
        let lease: TrainingQueueLease
        do { lease = try TrainingQueueLease(projectFolder:live,controlFolder:live) }
        catch { throw XCTSkip("GPU project is busy: \(error.localizedDescription)") }
        defer { withExtendedLifetime(lease) {} }
        let source = URL(fileURLWithPath:outputPath).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:source,withIntermediateDirectories:true)
        let project = try LoomProject(sourceFolder:source), dataset = URL(fileURLWithPath:path)
        let data = try PhotoEnhancerData.scan(dataset,kind:.repair)
        var settings = PhotoEnhancerSettings(); settings.width = 16; settings.epochs = 2; settings.minutes = 1
        let request = PhotoStopRequest()
        let first = try await project.trainPhotoEnhancer(dataset:dataset,settings:settings,progress:{ request.afterFirstUpdate($0,project:project) })
        let receipt = try await XCTUnwrap(request.requested()).value
        XCTAssertTrue(receipt.stopped)
        let checkpoint = first.appendingPathComponent("checkpoint.json"), parentBytes = try Data(contentsOf:checkpoint)
        let saved = try JSONDecoder().decode(PhotoEnhancerCheckpoint.self,from:parentBytes)
        XCTAssertEqual(saved.status,"saved_stopped"); XCTAssertGreaterThan(saved.updates,0)
        XCTAssertNil(saved.parentRunID); XCTAssertEqual(saved.updates,receipt.completedUpdates)
        let child = try await project.trainPhotoEnhancer(dataset:dataset,settings:settings,resume:checkpoint,progress:{ _ in })
        let continued = try JSONDecoder().decode(PhotoEnhancerCheckpoint.self,from:Data(contentsOf:child.appendingPathComponent("checkpoint.json")))
        XCTAssertEqual(continued.parentRunID,saved.runID); XCTAssertEqual(continued.updates,2*data.training.count)
        XCTAssertEqual(try Data(contentsOf:checkpoint),parentBytes)
        let descriptor = try JSONDecoder().decode(PhotoEnhancerModel.self,from:Data(contentsOf:child.appendingPathComponent("export/enhancer.json")))
        XCTAssertEqual(descriptor.updates,continued.bestUpdates); XCTAssertFalse(descriptor.qualified)
        XCTAssertEqual(descriptor.weightsSHA256,continued.bestWeightsSHA256)
        XCTAssertEqual(try PhotoEnhancerData.scan(dataset,kind:.repair).fingerprint,data.fingerprint)
        let latest = try PhotoEnhancerTesting.exportCheckpoint(child.appendingPathComponent("checkpoint.json"),best:false,destination:child.appendingPathComponent("latest-test-export"))
        let latestDescriptor = try JSONDecoder().decode(PhotoEnhancerModel.self,from:Data(contentsOf:latest.appendingPathComponent("enhancer.json")))
        XCTAssertEqual(latestDescriptor.updates,continued.updates); XCTAssertEqual(latestDescriptor.weightsSHA256,continued.weightsSHA256)
        let proof:[String:Any] = ["schema":"loom.photo-lifecycle-proof.v1","parent":first.path,"child":child.path,
            "parent_updates":saved.updates,"child_updates":continued.updates,"best_updates":continued.bestUpdates,
            "native_grid":1024,"width":16,"physical_scenes":Set(data.pairs.map { $0.metadata.group }).count,
            "learned_quality_claim":false,"save_stop_resume_export_passed":true]
        try JSONSerialization.data(withJSONObject:proof,options:[.prettyPrinted,.sortedKeys]).write(to:source.appendingPathComponent("proof.json"),options:.atomic)
    }
}
