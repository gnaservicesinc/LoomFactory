import Darwin
import Foundation
import LoomCore
import LoomNative
import Metal

struct QueueJob: Codable, Sendable {
    let id: String
    let category: String
    let datasetID: String
    let map: MapRole
    var settings: TrainingSettings
    let seconds: Double
    let parentRunID: String?
    let evaluationInput: String?
    let evaluationIsHeldOut: Bool?
}
struct QueueConfiguration: Codable, Sendable {
    let schema: String
    let sourceFolder: String
    let deadline: String
    let jobs: [QueueJob]
    let renderer: String?
}

struct MaterialReview: Codable, Sendable {
    struct Weight: Codable, Sendable {
        let runID: String
        let source: TrainingWeightSource
    }
    let schema: String
    let sourceFolder: String
    let category: String
    let datasetID: String
    let input: String
    let weights: [String: Weight]
    let renderer: String?
    let note: String
}

final class QueueStatus: @unchecked Sendable {
    let folder: URL
    private let lock = NSLock()
    private var state: [String: Any] = [:]
    private var lastPage = Date.distantPast
    init(_ folder: URL) throws {
        self.folder = folder
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    }
    func update(_ values: [String: Any]) {
        lock.withLock {
            state.merge(values) { _, new in new }
            state["updated_at"] = ISO8601DateFormatter().string(from: Date())
            if let data = try? JSONSerialization.data(withJSONObject: state, options: [.prettyPrinted, .sortedKeys]) {
                try? data.write(to: folder.appendingPathComponent("status.json"), options: .atomic)
            }
            if Date().timeIntervalSince(lastPage) >= 2 || values["status"] != nil {
                lastPage = .now
                writePage()
            }
        }
    }
    private func writePage() {
        func escaped(_ value: Any?) -> String {
            String(describing: value ?? "—").replacingOccurrences(of: "&", with: "&amp;")
                .replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: "\"", with: "&quot;")
        }
        let reviews = ((try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasSuffix("-renders") }
            .sorted { $0.path < $1.path }.filter {
                FileManager.default.fileExists(atPath: $0.appendingPathComponent("combined-material.png").path)
            }.map { directory in
                let relative = escaped(directory.lastPathComponent)
                return "<a href=\"\(relative)/combined-material.png\"><img src=\"\(relative)/combined-material.png\"></a>"
                    + "<p><a href=\"\(relative)/combined-material-neutral.png\">Neutral surface view</a></p>"
            }.joined(separator: "\n")
        let html = """
        <!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="15">
        <title>Material training progress</title><style>
        body{font:18px system-ui;background:#191d20;color:#eee;max-width:1100px;margin:40px auto;padding:0 24px}
        p{line-height:1.5}a{color:#a4caff}img{max-width:100%;margin-top:30px}small{color:#b9c2c9}
        </style><h1>Material training · \(escaped(state["status"]))</h1>
        <p>\(escaped(state["category"])) · \(escaped(state["map"]))<br>
        Job \(escaped(state["job_index"])) of \(escaped(state["jobs_total"])) · \(escaped(state["jobs_finished"])) saved</p>
        <p>\(escaped(state["message"]))</p>
        <p>Epoch \(escaped(state["epoch"])) · Update \(escaped(state["update"]))</p>
        <p><small>\(escaped(state["architecture"])) · \(escaped(state["parameters"])) parameters<br>
        Objective: \(escaped(state["objective"])) · Initial learning rate: \(escaped(state["learning_rate"]))</small></p>
        <p>Deadline: \(escaped(state["deadline"]))<br><small>Updated \(escaped(state["updated_at"])) · refreshes every 15 seconds</small></p>
        <p>To stop after saving the current update, open <a href="Stop%20Training.command">Stop Training</a>
        in Finder, or create an empty file named STOP in this folder.</p>
        <p>Compare depth, normal and roughness together below. Left: artist targets. Right: candidate.
        Both views use the same lights and relief scale. These are review candidates.</p>\(reviews)</html>
        """
        try? Data(html.utf8).write(to: folder.appendingPathComponent("Progress.html"), options: .atomic)
    }
    func event(_ event: TrainingEvent) {
        var values: [String: Any] = ["message": event.message]
        if let value = event.update { values["update"] = value }
        if let value = event.loss { values["loss"] = value }
        if let value = event.epoch { values["epoch"] = value }
        update(values)
        if ["epoch", "checkpoint", "checkpoint_saved", "warning", "completed"].contains(event.type) {
            print("\(event.type): \(event.message)")
            fflush(stdout)
        }
    }
}

func writeJSON(_ value: Any, _ url: URL) throws {
    try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        .write(to: url, options: .atomic)
}

@main struct QueueMain {
    static func main() async {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            if args.count == 3, args[0] == "--inventory" {
                try await inventory(source: args[1], output: URL(fileURLWithPath: args[2]))
            } else if args.count == 2, args[0] == "--run" {
                try await run(URL(fileURLWithPath: args[1]))
            } else if args.count == 2, args[0] == "--evaluate" {
                try await evaluate(URL(fileURLWithPath: args[1]))
            } else if args.count == 5, ["--select-model", "--select-checkpoint"].contains(args[0]),
                      let map = MapRole(rawValue: args[3]), MapRole.targets.contains(map) {
                let project = try LoomProject(sourceFolder: URL(fileURLWithPath: args[1]))
                if args[0] == "--select-checkpoint" {
                    let selected = try await project.selectCheckpointModelVersion(runID: args[4], datasetID: args[2], map: map)
                    print(selected.id)
                } else {
                    try await project.selectModelVersion(runID: args[4], datasetID: args[2], map: map)
                    print(args[4])
                }
            } else {
                throw LoomError("Use --inventory SOURCE OUTPUT.json, --run CONFIGURATION.json, --evaluate REVIEW.json, or --select-model/--select-checkpoint SOURCE DATASET MAP RUN_ID")
            }
        } catch {
            let args = Array(CommandLine.arguments.dropFirst())
            if args.count == 2, args[0] == "--run" {
                let url = URL(fileURLWithPath: args[1]).deletingLastPathComponent().appendingPathComponent("status.json")
                var value = (try? Data(contentsOf: url)).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
                value["status"] = "failed"; value["error"] = error.localizedDescription
                value["updated_at"] = ISO8601DateFormatter().string(from: Date())
                try? writeJSON(value, url)
            }
            fputs("\(error.localizedDescription)\n", stderr); exit(1)
        }
    }

    static func evaluate(_ configurationURL: URL) async throws {
        let config = try JSONDecoder().decode(MaterialReview.self, from: Data(contentsOf: configurationURL))
        guard config.schema == "loom-material-set-evaluation-v1",
              MapRole.targets.allSatisfy({ config.weights[$0.rawValue] != nil }) else {
            throw LoomError("A material review requires saved depth, normal and roughness weights.")
        }
        let project = try LoomProject(sourceFolder: URL(fileURLWithPath: config.sourceFolder))
        let lease = try TrainingQueueLease(projectFolder: project.projectFolder,
            controlFolder: configurationURL.deletingLastPathComponent())
        defer { withExtendedLifetime(lease) {} }
        let snapshot = try await project.snapshot()
        guard let diffuse = snapshot.images.first(where: {
            $0.url.path == config.input && !$0.ignored && $0.role == .diffuse && $0.category == config.category
        }) else { throw LoomError("The review input must be a current diffuse image in this category.") }
        var set: [String: Any] = ["schema": "loom-material-set-comparison-v1", "category": config.category,
            "input": config.input, "note": config.note]
        for role in MapRole.targets {
            guard let target = snapshot.images.first(where: {
                !$0.ignored && $0.role == role && $0.pairingKey == diffuse.pairingKey
                    && $0.width == diffuse.width && $0.height == diffuse.height
                    && (role != .normal || $0.normalConvention?.lowercased() != "directx")
            }), let weight = config.weights[role.rawValue] else {
                throw LoomError("Missing matched artist target or saved weight for \(role.rawValue).")
            }
            set["original_" + role.rawValue] = target.url.path
            let test = try await project.testModel(datasetID: config.datasetID, map: role, runID: weight.runID,
                source: weight.source, inputURL: diffuse.url, progress: { print($0.message) })
            set[role.rawValue] = test.outputURL.path
            set[role.rawValue + "_run_id"] = weight.runID
        }
        let output = configurationURL.deletingPathExtension().appendingPathExtension("material-set.json")
        try writeJSON(set, output)
        if let renderer = config.renderer {
            let results = try renderReview(output, renderer: renderer)
            try writeJSON(results, configurationURL.deletingPathExtension().appendingPathExtension("results.json"))
            guard results.values.allSatisfy({ $0 == 0 }) else { throw LoomError("Material review rendering failed; inspect its saved results.") }
        }
    }

    static func renderReview(_ path: URL, renderer: String) throws -> [String: Int32] {
        var results: [String: Int32] = [:]
        let scripts = URL(fileURLWithPath: renderer).deletingLastPathComponent()
        for (key, executable, arguments) in [
            ("material_metrics_exit", "/usr/bin/env", ["python3", scripts.appendingPathComponent("evaluate-material-set.py").path, path.path]),
            ("material_render_exit", "/Applications/Blender.app/Contents/MacOS/Blender",
                ["--background", "--factory-startup", "--disable-autoexec", "--threads", "4", "--python", renderer, "--", path.path]),
            ("material_preview_exit", "/usr/bin/env", ["python3", scripts.appendingPathComponent("assemble-material-review.py").path, path.path])
        ] {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            try process.run(); process.waitUntilExit()
            results[key] = process.terminationStatus
            if process.terminationStatus != 0 { break }
        }
        return results
    }

    static func inventory(source: String, output: URL) async throws {
        let project = try LoomProject(sourceFolder: URL(fileURLWithPath: source))
        let snapshot = try await project.snapshot()
        var categories: [[String: Any]] = []
        for category in snapshot.datasets {
            let members = snapshot.images.filter { $0.category == category.name && !$0.ignored }
            var plans: [[String: Any]] = []
            var heldOutInputs: Set<URL>?
            for map in MapRole.targets {
                let settings = TrainingSettings(imageSize: 1024, sourceResolutionLevels: 1,
                    validationFraction: 0.1, validationSeed: 9587)
                let plan = try snapshot.trainingPlan(datasetID: category.id, map: map, settings: settings)
                let inputs = Set(plan?.validationInputs ?? [])
                heldOutInputs = heldOutInputs.map { $0.intersection(inputs) } ?? inputs
                plans.append(["map": map.rawValue, "pairs": plan?.trainingPairCount ?? 0,
                    "validation_pairs": plan?.validationPairCount ?? 0,
                    "materials": plan?.trainingMaterialCount ?? 0])
            }
            let inputs = members.filter { $0.role == .diffuse && $0.width >= 1024 && $0.height >= 1024 }
            let complete = inputs.filter { input in
                MapRole.targets.allSatisfy { role in members.contains {
                    $0.role == role && $0.pairingKey == input.pairingKey && $0.width == input.width && $0.height == input.height
                        && (role != .normal || $0.normalConvention?.lowercased() != "directx")
                } }
            }.sorted {
                let a = $0.width * $0.height, b = $1.width * $1.height
                if a != b { return a < b }
                return $0.url.path < $1.url.path
            }
            categories.append(["id": category.id, "name": category.name, "plans": plans,
                "evaluation_input": complete.first?.url.path as Any? ?? NSNull(),
                "held_out_evaluation_input": complete.first(where: { heldOutInputs?.contains($0.url) == true })?.url.path as Any? ?? NSNull(),
                "complete_materials": Set(complete.map { $0.url.deletingLastPathComponent().path }).count])
        }
        try writeJSON(["project_id": snapshot.projectID, "source": source, "categories": categories], output)
    }

    static func run(_ configurationURL: URL) async throws {
        let config = try JSONDecoder().decode(QueueConfiguration.self, from: Data(contentsOf: configurationURL))
        guard config.schema == "loom-quality-training-queue-v1",
              let deadline = ISO8601DateFormatter().date(from: config.deadline), deadline > Date(),
              !config.jobs.isEmpty,
              config.jobs.allSatisfy({ MapRole.targets.contains($0.map) && $0.seconds.isFinite && $0.seconds > 0 && $0.settings.validationIssue == nil }),
              Set(config.jobs.map(\.id)).count == config.jobs.count else {
            throw LoomError("Invalid or expired training queue configuration.")
        }
        let folder = configurationURL.deletingLastPathComponent()
        let status = try QueueStatus(folder)
        let project = try LoomProject(sourceFolder: URL(fileURLWithPath: config.sourceFolder))
        let lease = try TrainingQueueLease(projectFolder: project.projectFolder, controlFolder: folder)
        _ = lease // Keep the lease alive until all jobs and evaluations finish.
        defer { withExtendedLifetime(lease) {} }
        let stopURL = folder.appendingPathComponent("STOP")
        signal(SIGTERM, SIG_IGN); signal(SIGINT, SIG_IGN)
        let terminate = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
        let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        for source in [terminate, interrupt] {
            source.setEventHandler { try? Data().write(to: stopURL, options: .atomic) }
            source.resume()
        }
        defer { terminate.cancel(); interrupt.cancel() }
        var results: [[String: Any]] = []
        var materialTests: [String: [String: Any]] = [:]
        // Preserve effective pre-queue choices, including the legacy fallback.
        var references: [[String: String]] = [], groups: Set<String> = []
        for job in config.jobs where groups.insert(job.datasetID + "/" + job.map.rawValue).inserted {
            if let reference = try await project.selectedModelArtifact(datasetID: job.datasetID, map: job.map) {
                try await project.selectModelVersion(runID: reference.id, datasetID: job.datasetID, map: job.map)
                references.append(["dataset_id": job.datasetID, "map": job.map.rawValue, "run_id": reference.id])
            }
        }
        try writeJSON(references, folder.appendingPathComponent("reference-models.json"))
        status.update(["status": "running", "pid": getpid(), "deadline": config.deadline,
            "jobs_total": config.jobs.count, "jobs_finished": 0])
        for (index, job) in config.jobs.enumerated() {
            if Date() >= deadline || FileManager.default.fileExists(atPath: stopURL.path) { break }
            let began = Date()
            let until = min(deadline, began.addingTimeInterval(job.seconds))
            status.update(["job": job.id, "category": job.category, "map": job.map.rawValue,
                "architecture": job.settings.modelArchitecture, "parameters": job.settings.parameterCount(map: job.map),
                "objective": job.settings.scalarLoss.rawValue, "learning_rate": job.settings.learningRate,
                "job_index": index + 1, "update": 0, "epoch": 0, "message": "Starting \(job.category) \(job.map.rawValue)"])
            let timer = Task {
                while !Task.isCancelled {
                    do { try await Task.sleep(for: .seconds(1)) } catch { return }
                    if Date() >= until || FileManager.default.fileExists(atPath: stopURL.path) {
                        do { _ = try await project.saveTrainingCheckpoint(stopAfterSaving: true); return }
                        catch { if Task.isCancelled { return } }
                    }
                }
            }
            var result: [String: Any] = ["id": job.id, "category": job.category, "map": job.map.rawValue]
            do {
                let artifact = try await project.trainMap(datasetID: job.datasetID, map: job.map, settings: job.settings,
                    start: job.parentRunID.map { .checkpoint(runID: $0) } ?? .fresh,
                    runLabel: "Overnight quality · \(job.category) · \(job.map.displayName)",
                    scalarLossOverride: job.map != .normal ? job.settings.scalarLoss : nil,
                    progress: { status.event($0) })
                result["status"] = "completed"; result["run_id"] = artifact.id
            } catch {
                result["status"] = error is TrainingSavedStop ? "saved_stopped" : "failed"
                if !(error is TrainingSavedStop) { result["error"] = String(describing: error) }
            }
            timer.cancel()
            if let checkpoint = try await project.snapshot().checkpoints.first(where: {
                $0.datasetID == job.datasetID && $0.map == job.map && $0.modifiedAt >= began
            }) {
                result["run_id"] = checkpoint.runID
                result["updates"] = checkpoint.totalUpdatesCompleted
                if let input = job.evaluationInput, Date() < deadline {
                    do {
                        let test = try await project.testModel(datasetID: job.datasetID, map: job.map, runID: checkpoint.runID,
                            source: .checkpoint, inputURL: URL(fileURLWithPath: input), progress: { status.event($0) })
                        result["test_directory"] = test.directory.path
                        var set = materialTests[job.datasetID] ?? ["category": job.category, "input": input]
                        let current = try await project.snapshot()
                        if let diffuse = current.images.first(where: { $0.url.path == input }) {
                            for role in MapRole.targets {
                                if let target = current.images.first(where: {
                                    !$0.ignored && $0.role == role && $0.pairingKey == diffuse.pairingKey
                                        && $0.width == diffuse.width && $0.height == diffuse.height
                                        && (role != .normal || $0.normalConvention?.lowercased() != "directx")
                                }) { set["original_" + role.rawValue] = target.url.path }
                            }
                        }
                        set[job.map.rawValue] = test.outputURL.path
                        set[job.map.rawValue + "_run_id"] = checkpoint.runID
                        materialTests[job.datasetID] = set
                        if MapRole.targets.allSatisfy({ set[$0.rawValue] != nil }) {
                            let name = job.datasetID + "-material-set.json"
                            let path = folder.appendingPathComponent(name)
                            set["schema"] = "loom-material-set-comparison-v1"
                            set["note"] = job.evaluationIsHeldOut == true
                                ? "Whole-material holdout from this training split; review depth, normal and roughness together before selecting a production version."
                                : "Matched candidates; this input is not certified as a whole-material holdout. Review all three maps together before selecting a production version."
                            try writeJSON(set, path)
                            if let renderer = config.renderer {
                                result.merge(try renderReview(path, renderer: renderer)) { _, new in new }
                            }
                        }
                    } catch { result["evaluation_error"] = String(describing: error) }
                }
                if Date() < deadline {
                    do {
                        let exported = try await project.exportCheckpointModelVersion(runID: checkpoint.runID,
                            datasetID: job.datasetID, map: job.map, progress: { status.event($0) })
                        result["exported_run_id"] = exported.id
                    } catch { result["export_error"] = String(describing: error) }
                }
            }
            result["elapsed_seconds"] = Date().timeIntervalSince(began)
            result["metal_allocated_bytes_after_job"] = MTLCreateSystemDefaultDevice()?.currentAllocatedSize
            results.append(result)
            try writeJSON(results, folder.appendingPathComponent("results.json"))
            status.update(["jobs_finished": results.count, "last_result": result])
        }
        let failures = results.filter { $0["status"] as? String == "failed" || $0["evaluation_error"] != nil || $0["export_error"] != nil
            || ($0["material_metrics_exit"] as? Int32).map { $0 != 0 } == true
            || ($0["material_render_exit"] as? Int32).map { $0 != 0 } == true || ($0["material_preview_exit"] as? Int32).map { $0 != 0 } == true }.count
        status.update(["status": FileManager.default.fileExists(atPath: stopURL.path) ? "stopped" : Date() >= deadline ? "time_budget_reached" : failures > 0 ? "completed_with_errors" : "completed",
            "jobs_with_errors": failures,
            "jobs_finished": results.count, "message": "Training queue ended; saved models and checkpoints remain in the source project."])
    }
}
