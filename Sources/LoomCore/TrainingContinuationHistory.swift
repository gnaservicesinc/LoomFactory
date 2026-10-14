import Foundation

/// Keep lineage separate from this run's epoch cursor. A changed training plan
/// starts its own schedule, while retaining the measured work behind its weights.
struct TrainingContinuationHistory {
    var rows: [[String: Any]] = []
    var epochs = 0
    var updates = 0
    var images = 0
    var elapsedSeconds = 0.0
    var restoredRunElapsedSeconds = 0.0

    init() {}

    init(descriptor: [String: Any], source: TrainingWeightSource, runID: String, resumingCursor: Bool,
         trainingImages: Int = 0) {
        let training = source == .model ? descriptor["training"] as? [String: Any] ?? [:] : descriptor
        let inherited = training["prior_history"] as? [[String: Any]] ?? []
        let local = training["history"] as? [[String: Any]] ?? []
        let priorEpochs = Self.count(training["prior_epochs"])
        let priorUpdates = Self.count(training["prior_updates"])
        let priorImages = Self.count(training["prior_images"])
        let priorDuration = Self.duration(training["prior_duration_seconds"])
        let ownDuration = Self.duration(training["elapsed_seconds"] ?? training["duration_seconds"] ?? descriptor["duration_seconds"])
        rows = inherited; epochs = priorEpochs; updates = priorUpdates; images = priorImages; elapsedSeconds = priorDuration
        if resumingCursor {
            restoredRunElapsedSeconds = ownDuration
        } else {
            rows += local.map { row in
                var result = row
                result["epoch"] = Self.count(row["epoch"]) + priorEpochs
                if let update = row["update"] as? Int { result["update"] = Self.count(update) + priorUpdates }
                result["run_id"] = row["run_id"] as? String ?? runID
                return result
            }
            epochs += Self.count(training["epochs_completed"] ?? training["epochs"] ?? local.count)
            updates += Self.count(training["updates"] ?? local.last?["update"])
            let cursor = training["resume_cursor"] as? [String: Any]
            let sampleCount = Self.count(training["training_sample_count"] ?? trainingImages)
            let completedEpochs = Self.count(cursor?["epoch"] ?? training["epochs_completed"] ?? training["epochs"] ?? local.count)
            let visits = sampleCount * completedEpochs + Self.count(cursor?["imageCount"])
            images += Self.count(training["completed_images"] ?? visits)
            elapsedSeconds += ownDuration
        }
    }

    var json: [String: Any] {
        ["prior_history": rows, "prior_epochs": epochs, "prior_updates": updates,
         "prior_images": images, "prior_duration_seconds": elapsedSeconds]
    }

    /// Old checkpoints did not store timers. Their native log still contains
    /// measured snapshots. Read a bounded tail and require the same saved step.
    static func measuredProgress(in directory: URL, completedUpdates: Int) -> TrainingProgress? {
        guard let file = try? TrainingRunFiles.containedFile("training.log", directory: directory),
              let handle = try? FileHandle(forReadingFrom: file) else { return nil }
        defer { try? handle.close() }
        guard let size = try? handle.seekToEnd(),
              (try? handle.seek(toOffset: size > 524_288 ? size - 524_288 : 0)) != nil,
              let data = try? handle.read(upToCount: 524_288) else { return nil }
        for row in String(decoding: data, as: UTF8.self).split(separator: "\n").reversed() {
            guard let object = try? JSONSerialization.jsonObject(with: Data(row.utf8)) as? [String: Any],
                  let value = object["training_progress"] as? [String: Any],
                  let encoded = try? JSONSerialization.data(withJSONObject: value),
                  let progress = try? JSONDecoder().decode(TrainingProgress.self, from: encoded),
                  progress.completedSteps == completedUpdates else { continue }
            return progress
        }
        return nil
    }

    private static func count(_ value: Any?) -> Int {
        guard let value = value as? Int else { return 0 }
        return min(Int.max / 1_000_000, max(0, value))
    }

    private static func duration(_ value: Any?) -> Double {
        guard let value = value as? Double, value.isFinite else { return 0 }
        return max(0, value)
    }
}
