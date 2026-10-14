import CryptoKit
import Darwin
import Foundation
import LoomNative

// A serial worker outside LoomProject's actor keeps Stop responsive during GPU work.
enum NativeFactoryTraining {
    static func run(id: String, root: URL, output: URL, projectID: String, dataset: DatasetCategory,
                    map: MapRole, plan: TrainingDataPlan, settings: TrainingSettings, resume: URL?,
                    cancellation: NativeCancellation, progress: @escaping @Sendable (TrainingEvent) -> Void,
                    prepareAhead: Bool = true) throws -> ModelArtifact {
        let started = Date()
        func clock() -> Double { ProcessInfo.processInfo.systemUptime }
        let stepsPerEpoch = settings.optimizerStepsPerEpoch(pairCount: plan.training.count)
        let totalUpdates = settings.epochs * stepsPerEpoch
        var tracker = TrainingProgressTracker(totalEpochs: settings.epochs, trainingImages: plan.training.count,
            validationImages: plan.validation.count, stepsPerEpoch: stepsPerEpoch,
            checkpointEveryEpochs: settings.checkpointEveryEpochs, startSeconds: clock())
        let preparation = NativePreparationPipeline<FactoryPreparedBatch>(cancellation: cancellation)
        let runMetadata = try TrainingRunFiles.metadata(at: output)
        var versionJSON: [String: Any] = ["run_id": id, "run_label": runMetadata?.label ?? ""]
        if let lineage = runMetadata?.lineage {
            versionJSON["lineage"] = ["parent_run_id": lineage.parentRunID, "source": lineage.source.rawValue,
                "weights_sha256": lineage.weightsSHA256]
        }
        defer { preparation.close(); NativeImages.clearDecodedPixelCache() }
        let logURL = output.appendingPathComponent("training.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: logURL); defer { try? log.close() }
        func emit(_ event: TrainingEvent) {
            let event = event.reporting(tracker.snapshot(now: clock()))
            progress(event)
            var json: [String: Any] = ["event": event.type, "message": event.message, "backend": "swift-mpsgraph-metal"]
            if let v = event.progress { json["progress"] = v }; if let v = event.epoch { json["epoch"] = v }
            if let v = event.loss { json["loss"] = v }; if let v = event.validationLoss { json["validation_loss"] = v }
            if let v = event.update { json["update"] = v }; if let v = event.totalUpdates { json["total_updates"] = v }
            if let v = event.learningRate { json["learning_rate"] = v }; if let v = event.parameterCount { json["parameter_count"] = v }
            if let v = event.effectiveBatchSize { json["effective_batch_size"] = v }
            if let v = event.gradientNorm { json["gradient_norm"] = v }
            if let v = event.gradientClipThreshold { json["gradient_clip_threshold"] = v }
            if let value = event.trainingProgress,
               let data = try? JSONEncoder().encode(value), let object = try? JSONSerialization.jsonObject(with: data) {
                json["training_progress"] = object
            }
            if let value = event.cycleTiming,
               let data = try? JSONEncoder().encode(value), let object = try? JSONSerialization.jsonObject(with: data) {
                json["cycle_timing"] = object
            }
            json["timestamp"] = ISO8601DateFormatter().string(from: Date())
            if var data = try? JSONSerialization.data(withJSONObject: json, options: [.sortedKeys]) {
                data.append(10); try? log.write(contentsOf: data)
            }
        }
        let all = plan.training + plan.validation
        let validationIDs = Set(plan.validation.map(\.id))
        var request = settingsJSON(settings)
        request.merge(versionJSON) { _, new in new }
        request.merge(["schema": "loom-factory-training-v1", "backend": "swift-mpsgraph-metal", "source_root": root.path,
                       "output_directory": output.path, "project_id": projectID, "dataset_id": dataset.id,
                       "display_name": dataset.name, "map_kind": map.rawValue,
                       "samples": all.map { ["id": $0.id, "input": $0.input.url.path, "target": $0.target.url.path,
                           "split": validationIDs.contains($0.id) ? "validation" : "training"] }]) { _, new in new }
        if let resume { request["resume_weights"] = resume.path }
        try writeJSON(request, to: output.appendingPathComponent("request.json"))
        emit(.init(type: "status", message: "Checking source images…"))
        var proofs: [String: SourceProof] = [:]
        for (index, pair) in all.enumerated() {
            tracker.begin(stage: .checkingSources, image: pair.input.trainingDisplayName, now: clock())
            emit(.init(type: "status", message: "Checking image \(index + 1) of \(all.count): \(pair.input.trainingDisplayName)"))
            try cancellation.check(); try checkTags(pair, dataset: dataset.name, map: map, root: root)
            for image in [pair.input, pair.target] where proofs[image.url.path] == nil {
                proofs[image.url.path] = try SourceProof(image.url, cancellation: cancellation)
            }
        }
        tracker.begin(stage: .preparingModel, now: clock())
        emit(.init(type: "status", message: "Preparing model…"))
        let architecture = try NativeArchitecture(mapKind: map.rawValue, width: settings.modelWidth, blocks: settings.modelBlocks, identifier: settings.modelArchitecture)
        let options = try NativeTrainingOptions(optimizer: settings.optimizer.rawValue, weightDecay: settings.weightDecay,
            beta1: settings.beta1, beta2: settings.beta2, momentum: settings.momentum, gradientClipNorm: settings.gradientClipNorm,
            gradientClipPolicy: settings.gradientClipPolicy.rawValue,
            gradientClipWarmupStartNorm: settings.gradientClipWarmupStartNorm,
            gradientClipWarmupUpdates: settings.gradientClipWarmupUpdates)
        var initial: [String: NativeTensor]?
        if let resume {
            _ = try TrainingRunFiles.resume(weights: resume, projectFolder: root.appendingPathComponent(".loom-factory"),
                projectID: projectID, datasetID: dataset.id, map: map, settings: settings,
                expectedSHA256: runMetadata?.lineage?.weightsSHA256)
            initial = try NativeWeights.read(from: resume)
            emit(.init(type: "status", message: "Continuing saved weights with a fresh native optimizer"))
        }
        let session = try NativeTrainingSession(architecture: architecture, options: options, seed: UInt64(settings.seed),
            weights: initial, device: settings.device.rawValue, cancellation: cancellation,
            channelsLast: !architecture.isPyramid)
        var update = 0, history: [[String: Any]] = [], bestLoss: Double?, bestEpoch: Int?, stale = 0
        var stoppedEarly = false
        var bestWeights: [String: NativeTensor]?
        var augmentation = NativeAugmentationPlan(seed: settings.seed, cropRefresh: settings.cropRefresh.rawValue,
                                                  augmentationRefresh: settings.augmentationRefresh.rawValue)
        var rng = FactoryRandom(seed: UInt64(settings.seed))
        emit(.init(type: "started", message: "Training \(dataset.name) · \(settings.epochs) epochs · \(totalUpdates) steps", update: 0, totalUpdates: totalUpdates,
                   parameterCount: settings.parameterCount(map: map), effectiveBatchSize: settings.effectiveBatchSize))
        func verify(_ pair: TrainingPair) throws {
            try cancellation.check(); try checkTags(pair, dataset: dataset.name, map: map, root: root)
            for image in [pair.input, pair.target] { try proofs[image.url.path]!.check(image.url) }
        }
        func checkpoint(_ status: String, epochComplete: Bool = true) throws {
            guard update > 0 else { return }
            let checkpointStart = clock()
            tracker.begin(stage: .checkpointing,
                          index: epochComplete && !history.isEmpty ? plan.training.count : nil, now: checkpointStart)
            emit(.init(type: "status", message: "Saving checkpoint at step \(update)…", update: update, totalUpdates: totalUpdates))
            let weights = try session.weights()
            try NativeWeights.write(weights, to: output.appendingPathComponent("checkpoint.safetensors"),
                                    metadata: ["schema": "loom-factory-checkpoint-v1", "architecture": architecture.identifier])
            var descriptor: [String: Any] = ["schema": "loom-factory-checkpoint-v1", "status": status, "project_id": projectID,
                "dataset_id": dataset.id, "dataset_name": dataset.name, "kind": "map", "map": map.rawValue,
                "architecture": architecture.identifier, "architecture_config": architectureJSON(architecture),
                "settings": settingsJSON(settings), "epochs_completed": history.count, "updates": update, "history": history,
                "weights_path": "checkpoint.safetensors", "backend": "swift-mpsgraph-metal",
                "continuation": "saved weights with a fresh optimizer and current source metadata"]
            descriptor["weights_sha256"] = try sha(output.appendingPathComponent("checkpoint.safetensors"))
            descriptor.merge(versionJSON) { _, new in new }
            try writeJSON(descriptor, to: output.appendingPathComponent("checkpoint.json"))
            tracker.checkpointCompleted(durationSeconds: clock() - checkpointStart, epochComplete: epochComplete, now: clock())
            emit(.init(type: "checkpoint", message: "Checkpoint saved at step \(update)", epoch: history.count, update: update))
        }
        do {
            for epoch in 0..<settings.epochs {
                try cancellation.check()
                tracker.beginEpoch(epoch: epoch + 1, now: clock())
                var order = plan.training
                if settings.shuffleEachEpoch { order.shuffle(using: &rng) }
                var epochLoss = 0.0, count = 0, learningRate = settings.learningRate
                var epochUpdates = 0, clippedUpdates = 0, normSum = 0.0, peakNorm = 0.0
                // Resolve deterministic transforms on the serial owner. Only the
                // upcoming minibatch's pixels are prepared by the bounded worker.
                var batches: [FactoryImageBatch] = []
                for window in stride(from: 0, to: order.count, by: settings.effectiveBatchSize) {
                    let end = min(window + settings.effectiveBatchSize, order.count)
                    for offset in stride(from: window, to: end, by: settings.batchSize) {
                        let pairs = Array(order[offset..<min(offset + settings.batchSize, end)])
                        let crops = try pairs.map { pair in
                            try augmentation.transform(sampleID: pair.id, epoch: epoch,
                                width: pair.input.width, height: pair.input.height, size: settings.imageSize,
                                flip: settings.flipCrops, rotate: settings.rotateCrops)
                        }
                        batches.append(FactoryImageBatch(pairs: pairs, crops: crops, offset: offset))
                    }
                }
                func submit(_ batch: FactoryImageBatch) throws {
                    for pair in batch.pairs { try verify(pair) }
                    try preparation.submit {
                        try FactoryPreparedBatch.prepare(batch, map: map, settings: settings, cancellation: cancellation)
                    }
                }
                if prepareAhead, let first = batches.first { try submit(first) }
                var batchIndex = 0
                for window in stride(from: 0, to: order.count, by: settings.effectiveBatchSize) {
                    let cycleStart = clock()
                    var preparationSeconds = 0.0, waitingSeconds = 0.0, trainingSeconds = 0.0
                    try cancellation.check(); session.resetAccumulatedGradients()
                    let end = min(window + settings.effectiveBatchSize, order.count)
                    learningRate = NativeTrainingOptions.learningRate(base: settings.learningRate, step: update,
                        totalSteps: totalUpdates, stepsPerEpoch: stepsPerEpoch, warmupEpochs: settings.warmupEpochs,
                        schedule: settings.learningRateSchedule.rawValue, minimumRatio: settings.minimumLearningRateRatio)
                    for offset in stride(from: window, to: end, by: settings.batchSize) {
                        let batch = batches[batchIndex]
                        tracker.begin(stage: .preparingImage, image: batch.imageNames, index: offset + 1,
                                      completedImages: epoch * order.count + count, now: clock())
                        emit(.init(type: "image_preparing", message: "Preparing image \(offset + 1) of \(order.count): \(batch.imageNames)",
                                   epoch: epoch + 1, update: update, totalUpdates: totalUpdates))
                        let waitingStart = clock()
                        let prepared = try prepareAhead ? preparation.take() : FactoryPreparedBatch.prepare(batch,
                            map: map, settings: settings, cancellation: cancellation)
                        waitingSeconds += clock() - waitingStart
                        preparationSeconds += prepared.preparationSeconds
                        for pair in batch.pairs { try verify(pair) }
                        batchIndex += 1
                        if prepareAhead, batchIndex < batches.count { try submit(batches[batchIndex]) }
                        try cancellation.check()
                        tracker.begin(stage: .training, image: batch.imageNames,
                                      index: min(offset + batch.pairs.count, order.count), now: clock())
                        emit(.init(type: "image_training", message: "Training image \(offset + 1) of \(order.count): \(batch.imageNames)",
                                   epoch: epoch + 1, update: update, totalUpdates: totalUpdates))
                        let trainingStart = clock()
                        let loss = try session.train(input: prepared.input, target: prepared.target, batchSize: batch.pairs.count,
                            height: settings.imageSize, width: settings.imageSize, weight: Float(batch.pairs.count) / Float(end - window))
                        trainingSeconds += clock() - trainingStart
                        guard loss.isFinite else { throw LoomError("Native training produced a non-finite loss. The last finite checkpoint was preserved.") }
                        epochLoss += Double(loss) * Double(batch.pairs.count); count += batch.pairs.count
                    }
                    try cancellation.check()
                    let cap = options.gradientClipThreshold(updateCount: update)
                    tracker.begin(stage: .optimizing,
                                  image: order[window..<end].map { $0.input.trainingDisplayName }.joined(separator: ", "), now: clock())
                    emit(.init(type: "step_updating", message: "Updating step \(update + 1) of \(totalUpdates)", epoch: epoch + 1,
                               update: update, totalUpdates: totalUpdates))
                    let optimizerStart = clock()
                    let norm = try session.applyGradients(learningRate: learningRate); update += 1
                    let optimizerSeconds = clock() - optimizerStart
                    epochUpdates += 1; normSum += norm; peakNorm = max(peakNorm, norm)
                    if cap > 0 && norm > cap { clippedUpdates += 1 }
                    let cycleSeconds = clock() - cycleStart
                    tracker.completeStep(stepInEpoch: epochUpdates, completedSteps: update,
                        completedImages: epoch * order.count + count, cycleSeconds: cycleSeconds, now: clock())
                    emit(.init(type: "progress", message: "Epoch \(epoch + 1) of \(settings.epochs) · step \(update) of \(totalUpdates) · \(totalUpdates - update) steps left",
                        progress: Double(update) / Double(totalUpdates), epoch: epoch + 1, loss: epochLoss / Double(count),
                        learningRate: learningRate, update: update, totalUpdates: totalUpdates,
                        gradientNorm: norm, gradientClipThreshold: cap,
                        cycleTiming: TrainingCycleTiming(preparationSeconds: preparationSeconds, preparationWaitSeconds: waitingSeconds,
                            trainingSeconds: trainingSeconds, optimizerSeconds: optimizerSeconds, totalSeconds: cycleSeconds)))
                    // Preserve the first real update even when an epoch spans thousands of pairs.
                    if update == 1 { try checkpoint("training", epochComplete: false) }
                }
                var validation: Double?
                if !plan.validation.isEmpty {
                    let validationStart = clock()
                    tracker.begin(stage: .validating, index: 0, now: clock())
                    emit(.init(type: "status", message: "Validating \(plan.validation.count) held-out images…"))
                    var sum = 0.0
                    for (index, pair) in plan.validation.enumerated() {
                        tracker.begin(stage: .validating, image: pair.input.trainingDisplayName, index: index + 1, now: clock())
                        emit(.init(type: "validation_image", message: "Validating image \(index + 1) of \(plan.validation.count): \(pair.input.trainingDisplayName)"))
                        try verify(pair)
                        let crop = NativeCropTransform(x: (pair.input.width - settings.imageSize) / 2,
                            y: (pair.input.height - settings.imageSize) / 2, size: settings.imageSize)
                        let prepared = try NativeImages.pair(inputURL: pair.input.url, targetURL: pair.target.url, transform: crop,
                            mapKind: map.rawValue, normalConvention: pair.target.normalConvention ?? "OpenGL",
                            exaggerate: settings.exaggerateFeatures, cancellation: cancellation)
                        sum += Double(try session.evaluate(input: prepared.input.values, target: prepared.target.values,
                            batchSize: 1, height: settings.imageSize, width: settings.imageSize))
                    }
                    validation = sum / Double(plan.validation.count)
                    guard validation!.isFinite else { throw LoomError("Validation produced a non-finite loss.") }
                    if bestLoss == nil || validation! < bestLoss! {
                        bestLoss = validation; bestEpoch = epoch + 1; stale = 0; bestWeights = try session.weights()
                        try NativeWeights.write(bestWeights!, to: output.appendingPathComponent("best.safetensors"))
                    } else { stale += 1 }
                    tracker.completeValidation(durationSeconds: clock() - validationStart, now: clock())
                }
                var record: [String: Any] = ["epoch": epoch + 1, "loss": epochLoss / Double(count), "learning_rate": learningRate,
                    "update": update,
                    "mean_gradient_norm": normSum / Double(max(1, epochUpdates)), "peak_gradient_norm": peakNorm,
                    "clipped_updates": clippedUpdates, "optimizer_updates": epochUpdates,
                    "gradient_clip_threshold": options.gradientClipThreshold(updateCount: max(0, update - 1))]
                record["validation_loss"] = validation ?? NSNull(); history.append(record)
                emit(.init(type: "epoch_completed", message: "Epoch \(epoch + 1) complete", epoch: epoch + 1,
                    loss: epochLoss / Double(count), validationLoss: validation, update: update, totalUpdates: totalUpdates))
                let stop = settings.earlyStoppingPatience > 0 && stale >= settings.earlyStoppingPatience
                if (epoch + 1) % settings.checkpointEveryEpochs == 0 || epoch + 1 == settings.epochs || stop { try checkpoint("training") }
                try writeJSON(["status": "training", "history": history, "updates": update], to: output.appendingPathComponent("progress.json"))
                if stop { stoppedEarly = true; break }
            }
            try cancellation.check()
            tracker.begin(stage: .verifyingSources, now: clock())
            emit(.init(type: "status", message: "Verifying original images before export…"))
            for pair in all { try verify(pair) }
            for (path, proof) in proofs {
                tracker.begin(stage: .verifyingSources, image: URL(fileURLWithPath: path).lastPathComponent, now: clock())
                emit(.init(type: "status", message: "Verifying original: \(URL(fileURLWithPath: path).lastPathComponent)"))
                try proof.verifyHash(URL(fileURLWithPath: path), cancellation: cancellation)
            }
            tracker.begin(stage: .exporting, now: clock())
            emit(.init(type: "status", message: "Exporting trained model…"))
            let deployment: [String: NativeTensor]
            if settings.exportBestWeights, let bestWeights { deployment = bestWeights } else { deployment = try session.weights() }
            try NativeWeights.write(deployment, to: output.appendingPathComponent("model.safetensors"),
                metadata: ["schema": "loom-factory-model-v1", "architecture": architecture.identifier])
            try cancellation.check()
            let compiled = try NativeCoreML.export(weights: deployment, architecture: architecture, imageSize: settings.imageSize,
                to: output.appendingPathComponent("model.mlmodel"))
            defer { try? FileManager.default.removeItem(at: compiled) }
            try cancellation.check()
            tracker.begin(stage: .verifyingSources, now: clock())
            emit(.init(type: "status", message: "Verifying originals after export…"))
            for pair in all { try verify(pair) }
            for (path, proof) in proofs { try proof.verifyHash(URL(fileURLWithPath: path), cancellation: cancellation) }
            try checkpoint("completed")
            try cancellation.check()
            let exportEpoch = settings.exportBestWeights ? (bestEpoch ?? history.count) : history.count
            var training = settingsJSON(settings)
            training.merge(["epochs": history.count, "requested_epochs": settings.epochs, "updates": update, "history": history,
                "stopped_early": stoppedEarly,
                "training_sample_count": plan.training.count, "validation_sample_count": plan.validation.count,
                "export_epoch": exportEpoch, "best_epoch": bestEpoch ?? NSNull(), "best_validation_loss": bestLoss ?? NSNull(),
                "validation_crop": plan.validation.isEmpty ? NSNull() : "fixed-center-native-no-augmentation",
                "device": "metal", "backend": "swift-mpsgraph-metal", "duration_seconds": Date().timeIntervalSince(started)]) { _, new in new }
            let finalLoss = history[exportEpoch - 1]["loss"] as! Double
            var descriptor: [String: Any] = ["schema": "loom-factory-model-v1", "status": "completed", "kind": "map",
                "project_id": projectID, "dataset_id": dataset.id, "dataset_name": dataset.name, "map": map.rawValue,
                "architecture": architecture.identifier, "architecture_config": architectureJSON(architecture), "final_loss": finalLoss,
                "final_validation_loss": history[exportEpoch - 1]["validation_loss"] ?? NSNull(),
                "created_at": ISO8601DateFormatter().string(from: Date()), "duration_seconds": Date().timeIntervalSince(started),
                "image_size": settings.imageSize, "input_name": "rgb", "output_name": "material_map",
                "input_type": "multiarray", "input_shape": [1, 3, settings.imageSize, settings.imageSize],
                "coreml_path": "model.mlmodel", "weights_path": "model.safetensors", "training": training,
                "weights_sha256": try sha(output.appendingPathComponent("model.safetensors")),
                "coreml_sha256": try sha(output.appendingPathComponent("model.mlmodel")),
                "parameter_count": settings.parameterCount(map: map), "sources": proofs.map { path, proof in proof.json(path: String(path.dropFirst(root.path.count + 1))) },
                "quality_note": "Training completion does not qualify prediction quality on unseen materials."]
            descriptor.merge(versionJSON) { _, new in new }
            try writeJSON(["status": "completed", "history": history, "updates": update], to: output.appendingPathComponent("progress.json"))
            try cancellation.check()
            try writeJSON(descriptor, to: output.appendingPathComponent("model.json"))
            let artifact = ModelArtifact(id: id, kind: "map", datasetID: dataset.id, datasetName: dataset.name, map: map,
                directory: output, epochs: history.count, loss: finalLoss, coreMLPath: "model.mlmodel", weightsPath: "model.safetensors", architecture: architecture.identifier)
            try JSONEncoder().encode(artifact).write(to: output.appendingPathComponent("artifact.json"), options: .atomic)
            tracker.begin(stage: .completed, now: clock())
            emit(.init(type: "completed", message: "Training complete · model saved", progress: 1))
            return artifact
        } catch {
            let status = error is CancellationError ? "cancelled" : "stopped"
            do { try checkpoint(status) } catch { emit(.init(type: "warning", message: "Latest checkpoint could not be saved: \(error.localizedDescription)")) }
            try? writeJSON(["status": status, "history": history, "updates": update], to: output.appendingPathComponent("progress.json"))
            tracker.begin(stage: .stopped, now: clock())
            emit(.init(type: status, message: error is CancellationError ? "Training stopped; completed steps were saved" : error.localizedDescription))
            throw error
        }
    }

    static func checkTags(_ pair: TrainingPair, dataset: String, map: MapRole, root: URL) throws {
        for (image, role) in [(pair.input, MapRole.diffuse), (pair.target, map)] {
            guard image.url.standardizedFileURL.path.hasPrefix(root.path + "/"),
                  !image.url.standardizedFileURL.path.hasPrefix(root.path + "/.loom-factory/"),
                  FileManager.default.fileExists(atPath: image.url.path),
                  let metadata = try ImageMetadataStore.read(image.url), metadata.id == image.id,
                  image.pairingKey == pair.input.pairingKey,
                  metadata.pairingKey == image.pairingKey, metadata.category == dataset, metadata.role == role, metadata.ignored != true else {
                throw LoomError("A source pair changed its file, dataset, role, pairing or ignore state: \(image.url.lastPathComponent). Refresh the source folder.")
            }
            if map == .normal && role == .normal && metadata.normalConvention?.lowercased() == "directx" {
                throw LoomError("A normal source changed convention during training. Refresh the source folder.")
            }
        }
    }
    static func architectureJSON(_ a: NativeArchitecture) -> [String: Any] {
        a.descriptor
    }
    static func settingsJSON(_ s: TrainingSettings) -> [String: Any] {
        ["image_size": s.imageSize, "model_architecture": s.modelArchitecture,
         "width": s.modelWidth, "blocks": s.modelBlocks, "epochs": s.epochs,
         "batch_size": s.batchSize, "learning_rate": s.learningRate, "shuffle_each_epoch": s.shuffleEachEpoch,
         "flip_crops": s.flipCrops, "rotate_crops": s.rotateCrops, "exaggerate_features": s.exaggerateFeatures,
         "seed": s.seed, "device": s.device.rawValue, "optimizer": s.optimizer.rawValue, "weight_decay": s.weightDecay,
         "beta1": s.beta1, "beta2": s.beta2, "momentum": s.momentum, "gradient_clip_norm": s.gradientClipNorm,
         "gradient_clip_policy": s.gradientClipPolicy.rawValue,
         "gradient_clip_warmup_start_norm": s.gradientClipWarmupStartNorm,
         "gradient_clip_warmup_updates": s.gradientClipWarmupUpdates,
         "gradient_accumulation_steps": s.gradientAccumulationSteps, "lr_schedule": s.learningRateSchedule.rawValue,
         "warmup_epochs": s.warmupEpochs, "minimum_lr_ratio": s.minimumLearningRateRatio,
         "crop_refresh": s.cropRefresh.rawValue, "augmentation_refresh": s.augmentationRefresh.rawValue,
         "validation_fraction": s.validationFraction, "validation_seed": s.validationSeed,
         "early_stopping_patience": s.earlyStoppingPatience, "export_best_weights": s.exportBestWeights,
         "checkpoint_every_epochs": s.checkpointEveryEpochs]
    }
    static func writeJSON(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .prettyPrinted]).write(to: url, options: .atomic)
    }
    static func json(_ url: URL) throws -> [String: Any] {
        guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else { throw LoomError("Invalid project descriptor.") }; return value
    }
    static func sha(_ url: URL, cancellation: NativeCancellation? = nil) throws -> String {
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        var hash = SHA256()
        while let block = try handle.read(upToCount: 1024 * 1024), !block.isEmpty { try cancellation?.check(); hash.update(data: block) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

private struct SourceProof {
    let size: Int64, seconds: Int, nanos: Int, device: Int32, inode: UInt64, hash: String
    init(_ url: URL, cancellation: NativeCancellation) throws {
        var value = stat(); guard stat(url.path, &value) == 0 else { throw LoomError("A source image is missing: \(url.lastPathComponent)") }
        size = value.st_size; seconds = value.st_mtimespec.tv_sec; nanos = value.st_mtimespec.tv_nsec
        device = value.st_dev; inode = value.st_ino; hash = try NativeFactoryTraining.sha(url, cancellation: cancellation)
        try check(url)
    }
    func check(_ url: URL) throws {
        var value = stat()
        guard stat(url.path, &value) == 0, value.st_size == size, value.st_mtimespec.tv_sec == seconds,
              value.st_mtimespec.tv_nsec == nanos, value.st_dev == device, value.st_ino == inode else {
            throw LoomError("Source pixels changed during training: \(url.lastPathComponent). Refresh the source folder.")
        }
    }
    func verifyHash(_ url: URL, cancellation: NativeCancellation) throws {
        try check(url)
        guard try NativeFactoryTraining.sha(url, cancellation: cancellation) == hash else { throw LoomError("Source pixels changed during training: \(url.lastPathComponent).") }
    }
    func json(path: String) -> [String: Any] { ["path": path, "size": size, "mtime_seconds": seconds, "mtime_nanoseconds": nanos, "device": device, "inode": inode, "sha256": hash] }
}

private struct FactoryRandom: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15; var v = state
        v = (v ^ (v >> 30)) &* 0xbf58476d1ce4e5b9; v = (v ^ (v >> 27)) &* 0x94d049bb133111eb; return v ^ (v >> 31)
    }
}
