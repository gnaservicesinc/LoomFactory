import CryptoKit
import Darwin
import Foundation
import LoomNative

// A serial worker outside LoomProject's actor keeps Stop responsive during GPU work.
enum NativeFactoryTraining {
    static func run(id: String, root: URL, output: URL, projectID: String, dataset: DatasetCategory,
                    map: MapRole, plan: TrainingDataPlan, settings: TrainingSettings, resume: URL?,
                    cancellation: NativeCancellation, progress: @escaping @Sendable (TrainingEvent) -> Void,
                    prepareAhead: Bool = true, control: TrainingRunControl? = nil,
                    preparationCapacity: Int = 2) throws -> ModelArtifact {
        var runSettings = settings
        if let issue = runSettings.validationIssue { throw LoomError(issue) }
        runSettings.resolveRandomSeedForRun()
        let settings = runSettings
        let started = Date()
        func clock() -> Double { ProcessInfo.processInfo.systemUptime }
        let stepsPerEpoch = settings.optimizerStepsPerEpoch(pairCount: plan.training.count)
        let totalUpdates = settings.epochs * stepsPerEpoch
        var tracker = TrainingProgressTracker(totalEpochs: settings.epochs, trainingImages: plan.training.count,
            validationImages: plan.validation.count, stepsPerEpoch: stepsPerEpoch,
            checkpointEveryEpochs: settings.checkpointEveryEpochs, startSeconds: clock())
        let imageCache = NativeImageCache(policy: settings.cacheImagesInMemory
            ? .bounded(byteLimit: settings.imageCacheLimitMB * 1024 * 1024) : .stream)
        let preparationDepth = min(2, max(1, preparationCapacity))
        let preparation = NativePreparationPipeline<FactoryPreparedBatch>(cancellation: cancellation, capacity: preparationDepth)
        let runMetadata = try TrainingRunFiles.metadata(at: output)
        var versionJSON: [String: Any] = ["run_id": id, "run_label": runMetadata?.label ?? ""]
        if let lineage = runMetadata?.lineage {
            versionJSON["lineage"] = ["parent_run_id": lineage.parentRunID, "source": lineage.source.rawValue,
                "weights_sha256": lineage.weightsSHA256]
        }
        defer { preparation.close(); imageCache.clear(); NativeImages.clearDecodedPixelCache() }
        let logURL = output.appendingPathComponent("training.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: logURL); defer { try? log.close() }
        var inherited = TrainingContinuationHistory()
        func emit(_ event: TrainingEvent, inheritedOffsets: Bool = true) {
            let snapshot = tracker.snapshot(now: clock())
            let priorEpochs = inheritedOffsets ? inherited.epochs : 0
            let priorUpdates = inheritedOffsets ? inherited.updates : 0
            let event = TrainingEvent(type: event.type, message: event.message,
                progress: event.progress.map { $0 == 1 ? 1 : snapshot.totalSteps > 0 ? Double(snapshot.completedSteps) / Double(snapshot.totalSteps) : $0 },
                epoch: event.epoch.map { $0 + priorEpochs }, loss: event.loss, validationLoss: event.validationLoss,
                learningRate: event.learningRate, update: event.update.map { $0 + priorUpdates },
                totalUpdates: event.totalUpdates.map { $0 + priorUpdates }, parameterCount: event.parameterCount,
                effectiveBatchSize: event.effectiveBatchSize, gradientNorm: event.gradientNorm,
                gradientClipThreshold: event.gradientClipThreshold, trainingProgress: snapshot, cycleTiming: event.cycleTiming)
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
                       "source_material_count": plan.sourceMaterialCount, "source_image_count": plan.sourceImageCount,
                       "eligible_pair_count": plan.eligiblePairCount, "excluded_pair_count": plan.excludedPairCount,
                       "training_pair_count": plan.trainingPairCount, "validation_pair_count": plan.validationPairCount,
                       "planned_training_pair_uses": plan.trainingSampleVisits, "planned_optimizer_updates": plan.totalOptimizerUpdates,
                       "samples": all.map { ["id": $0.id, "input": $0.input.url.path, "target": $0.target.url.path,
                           "split": validationIDs.contains($0.id) ? "validation" : "training"] }]) { _, new in new }
        request["effective_image_cache_bytes"] = imageCache.effectiveByteLimit
        request["image_loading"] = "lazy-bounded-prefetch"
        request["resume_state_policy"] = settings.requestedSeed == -1 ? "fresh-optimizer-and-sampler" : "restore-compatible-state"
        if let resume { request["resume_weights"] = resume.path }
        try writeJSON(request, to: output.appendingPathComponent("request.json"))
        emit(.init(type: "status", message: "Checking source images…"))
        let proofs = try NativeSourceProofStore(urls: all.flatMap { [$0.input.url, $0.target.url] }, cancellation: cancellation)
        for (index, pair) in all.enumerated() {
            tracker.begin(stage: .checkingSources, image: pair.input.trainingDisplayName, now: clock())
            emit(.init(type: "status", message: "Checking image \(index + 1) of \(all.count): \(pair.input.trainingDisplayName)"))
            try cancellation.check(); try checkTags(pair, dataset: dataset.name, map: map, root: root)
        }
        tracker.begin(stage: .preparingModel, now: clock())
        emit(.init(type: "status", message: "Preparing model…"))
        let architecture = try NativeArchitecture(mapKind: map.rawValue, width: settings.modelWidth, blocks: settings.modelBlocks, identifier: settings.modelArchitecture)
        let options = try NativeTrainingOptions(optimizer: settings.optimizer.rawValue, weightDecay: settings.weightDecay,
            beta1: settings.beta1, beta2: settings.beta2, momentum: settings.momentum, gradientClipNorm: settings.gradientClipNorm,
            gradientClipPolicy: settings.gradientClipPolicy.rawValue,
            gradientClipWarmupStartNorm: settings.gradientClipWarmupStartNorm,
            gradientClipWarmupUpdates: settings.gradientClipWarmupUpdates,
            scalarLoss: settings.scalarLoss)
        var initial: [String: NativeTensor]?, optimizerState: [String: NativeTensor]?
        var restoredUpdateCount = 0
        var restoredCursor: TrainingCheckpointCursor?
        var restoredDescriptor: [String: Any]?
        var restoredBestWeights: [String: NativeTensor]?
        var sourceDescriptor: [String: Any]?
        var sourceKind: TrainingWeightSource?
        var sourceRunID: String?
        var fingerprint: String?
        func sourceFingerprint() throws -> String {
            if let fingerprint { return fingerprint }
            let value = try proofs.fingerprint(pairs: all, validationIDs: validationIDs, cancellation: cancellation)
            fingerprint = value
            return value
        }
        if let resume {
            let resolved = try TrainingRunFiles.resume(weights: resume, projectFolder: root.appendingPathComponent(".loom-factory"),
                projectID: projectID, datasetID: dataset.id, map: map, settings: settings,
                expectedSHA256: runMetadata?.lineage?.weightsSHA256)
            initial = try NativeWeights.read(from: resume)
            sourceKind = resolved.source; sourceRunID = resolved.runID
            sourceDescriptor = try TrainingRunFiles.descriptor(at: resume.deletingLastPathComponent(), source: resolved.source)
            if resolved.source == .checkpoint {
                let folder = resume.deletingLastPathComponent()
                let descriptor = try TrainingRunFiles.descriptor(at: folder, source: .checkpoint)
                let saved = descriptor["settings"] as? [String: Any] ?? [:]
                let optimizerKeys = ["optimizer", "weight_decay", "beta1", "beta2", "momentum", "scalar_loss", "seed", "exaggerate_features", "image_size"]
                if settings.requestedSeed != -1, let path = descriptor["optimizer_path"] as? String,
                   TrainingCheckpointCursor.compatibleSettings(saved, current: settingsJSON(settings), keys: optimizerKeys) {
                    let file = try TrainingRunFiles.containedFile(path, directory: folder)
                    guard descriptor["optimizer_sha256"] as? String == (try sha(file)),
                          let count = descriptor["optimizer_updates"] as? Int, count >= 0 else {
                        throw LoomError("The saved checkpoint's optimizer state or checksum is invalid. Choose another saved version.")
                    }
                    optimizerState = try NativeWeights.read(from: file); restoredUpdateCount = count
                    if descriptor["status"] as? String != "completed",
                       let object = descriptor["resume_cursor"],
                       TrainingCheckpointCursor.compatibleSettings(saved, current: settingsJSON(settings)) {
                        let cursor = try JSONDecoder().decode(TrainingCheckpointCursor.self,
                            from: JSONSerialization.data(withJSONObject: object))
                        guard try cursor.sourceFingerprint == sourceFingerprint() else {
                            throw LoomError("The source pairs or pixels changed since this checkpoint. Continue exported model weights for a new training plan, or restore the checkpoint's original source files.")
                        }
                        guard cursor.epoch >= 0, cursor.epoch <= settings.epochs,
                              cursor.order.isEmpty || cursor.epoch < settings.epochs,
                              cursor.nextWindow >= 0, cursor.nextWindow <= plan.training.count,
                              (cursor.order.isEmpty && cursor.nextWindow == 0)
                                || (cursor.order.count == plan.training.count && Set(cursor.order) == Set(plan.training.map(\.id))),
                              cursor.crops.count == cursor.order.count,
                              cursor.imageCount == cursor.nextWindow,
                              cursor.nextWindow == plan.training.count || cursor.nextWindow % settings.effectiveBatchSize == 0 else {
                            throw LoomError("This checkpoint's epoch position is incompatible with the requested training run. Use at least \(cursor.epoch + 1) epochs or choose exported model weights.")
                        }
                        guard let completed = descriptor["updates"] as? Int, completed > 0,
                              let rows = descriptor["history"] as? [[String: Any]], rows.count == cursor.epoch,
                              cursor.epochUpdates == (cursor.nextWindow + settings.effectiveBatchSize - 1) / settings.effectiveBatchSize,
                              completed == cursor.epoch * stepsPerEpoch + cursor.epochUpdates else {
                            throw LoomError("This checkpoint's saved update count or epoch history is inconsistent. Choose another saved version.")
                        }
                        restoredCursor = cursor; restoredDescriptor = descriptor
                        if let path = descriptor["best_weights_path"] as? String {
                            let file = try TrainingRunFiles.containedFile(path, directory: folder)
                            guard descriptor["best_weights_sha256"] as? String == (try sha(file)) else {
                                throw LoomError("The saved best-validation weights checksum changed.")
                            }
                            restoredBestWeights = try NativeWeights.read(from: file)
                        }
                    }
                }
            }
            emit(.init(type: "status", message: restoredCursor != nil
                ? "Resuming checkpoint with optimizer state at epoch \(restoredCursor!.epoch + 1), image \(restoredCursor!.nextWindow + 1)"
                : optimizerState != nil ? "Continuing saved weights and optimizer state with a new epoch schedule"
                : settings.requestedSeed == -1 ? "Continuing saved weights with a fresh optimizer, shuffle and augmentation state (random seed -1)"
                : "Continuing saved weights with a fresh optimizer (changed seed, targets or optimizer settings)"))
        }
        let session = try NativeTrainingSession(architecture: architecture, options: options, seed: UInt64(settings.seed),
            weights: initial, optimizerState: optimizerState, initialUpdateCount: restoredUpdateCount,
            device: settings.device.rawValue, cancellation: cancellation)
        var update = restoredCursor != nil ? (restoredDescriptor?["updates"] as? Int ?? 0) : 0
        var history: [[String: Any]] = restoredCursor != nil ? (restoredDescriptor?["history"] as? [[String: Any]] ?? []) : []
        var bestLoss = restoredCursor?.bestLoss, bestEpoch = restoredCursor?.bestEpoch, stale = restoredCursor?.staleEpochs ?? 0
        var stoppedEarly = restoredCursor?.stoppedEarly ?? false
        var bestWeights = restoredBestWeights
        var augmentation = restoredCursor?.augmentation ?? NativeAugmentationPlan(seed: settings.seed, cropRefresh: settings.cropRefresh.rawValue,
                                                  augmentationRefresh: settings.augmentationRefresh.rawValue)
        var rng = FactoryRandom(seed: UInt64(settings.seed))
        var recordedTrainingLayout = false
        if let state = restoredCursor?.randomState { rng.state = state }
        var cursor = restoredCursor
        if var sourceDescriptor, let sourceKind, let sourceRunID, let resume {
            if sourceKind == .checkpoint, sourceDescriptor["elapsed_seconds"] == nil,
               let completed = sourceDescriptor["updates"] as? Int,
               let measured = TrainingContinuationHistory.measuredProgress(in: resume.deletingLastPathComponent(),
                    completedUpdates: completed + (sourceDescriptor["prior_updates"] as? Int ?? 0)) {
                sourceDescriptor["elapsed_seconds"] = max(0, measured.elapsedSeconds - (measured.priorElapsedSeconds ?? 0))
                sourceDescriptor["epoch_elapsed_seconds"] = measured.epochElapsedSeconds
                sourceDescriptor["completed_images"] = max(0, measured.completedImages - (measured.priorImages ?? 0))
                if restoredDescriptor != nil { restoredDescriptor = sourceDescriptor }
            }
            let sourceRequest = TrainingRunFiles.optionalJSON(resume.deletingLastPathComponent().appendingPathComponent("request.json"))
            inherited = TrainingContinuationHistory(descriptor: sourceDescriptor, source: sourceKind,
                runID: sourceRunID, resumingCursor: restoredCursor != nil,
                trainingImages: sourceRequest?["training_pair_count"] as? Int ?? 0)
        }
        tracker.restore(epoch: restoredCursor.map { $0.epoch + ($0.order.isEmpty ? 0 : 1) } ?? 0,
            stepInEpoch: restoredCursor?.epochUpdates ?? 0, completedSteps: update,
            completedImages: (restoredCursor?.epoch ?? 0) * plan.training.count + (restoredCursor?.imageCount ?? 0),
            elapsedSeconds: inherited.restoredRunElapsedSeconds,
            priorEpochs: inherited.epochs, priorSteps: inherited.updates, priorImages: inherited.images,
            priorElapsedSeconds: inherited.elapsedSeconds, now: clock())
        let restoredHistory = inherited.rows + history.map { row in
            var cumulative = row
            cumulative["epoch"] = (row["epoch"] as? Int ?? 0) + inherited.epochs
            if let count = row["update"] as? Int { cumulative["update"] = count + inherited.updates }
            return cumulative
        }
        for previous in restoredHistory {
            let epoch = previous["epoch"] as? Int ?? 0
            let loss = previous["loss"] as? Double
            emit(.init(type: "restored_epoch", message: "Previous training · epoch \(epoch)\(loss.map { " · loss \($0)" } ?? "")",
                epoch: epoch, loss: previous["loss"] as? Double, validationLoss: previous["validation_loss"] as? Double,
                learningRate: previous["learning_rate"] as? Double, update: previous["update"] as? Int,
                gradientNorm: previous["mean_gradient_norm"] as? Double,
                gradientClipThreshold: previous["gradient_clip_threshold"] as? Double), inheritedOffsets: false)
        }
        emit(.init(type: "status", message: "Seed \(settings.seed) · \(settings.requestedSeed == -1 ? "new system seed, fresh optimizer and sampler" : "fixed seed, compatible state may resume") · \(imageCache.effectiveByteLimit == 0 ? "streaming images" : "decoded image cache up to \(imageCache.effectiveByteLimit / (1024 * 1024)) MiB")"))
        emit(.init(type: "started", message: "Training \(dataset.name) · \(inherited.epochs + settings.epochs) total epochs · \(inherited.updates + totalUpdates) total updates", update: update, totalUpdates: totalUpdates,
                   parameterCount: settings.parameterCount(map: map), effectiveBatchSize: settings.effectiveBatchSize))
        func verify(_ pair: TrainingPair) throws {
            try cancellation.check(); try checkTags(pair, dataset: dataset.name, map: map, root: root)
            for image in [pair.input, pair.target] { try proofs.check(image.url) }
        }
        func checkpoint(_ status: String, epochComplete: Bool = true) throws {
            guard update > 0 else { return }
            let checkpointStart = clock()
            // Preserve early weights without waiting for untouched source files.
            // Exact image-position resume is available once all source proofs
            // are ready; otherwise these weights continue with a new schedule.
            if fingerprint == nil {
                fingerprint = proofs.fingerprintIfComplete(pairs: all, validationIDs: validationIDs)
            }
            if let fingerprint { cursor?.sourceFingerprint = fingerprint }
            tracker.begin(stage: .checkpointing,
                          index: epochComplete && !history.isEmpty ? plan.training.count : nil, now: checkpointStart)
            emit(.init(type: "status", message: "Saving checkpoint at total update \(inherited.updates + update)…", update: update, totalUpdates: totalUpdates))
            var descriptor: [String: Any] = ["schema": "loom-factory-checkpoint-v1", "status": status, "project_id": projectID,
                "dataset_id": dataset.id, "dataset_name": dataset.name, "kind": "map", "map": map.rawValue,
                "architecture": architecture.identifier, "architecture_config": architectureJSON(architecture),
                "settings": settingsJSON(settings), "epochs_completed": history.count, "updates": update, "history": history,
                "optimizer_updates": session.updateCount, "backend": "swift-mpsgraph-metal",
                "continuation": fingerprint == nil ? "weights and optimizer; new epoch schedule until source proofs are complete"
                    : "weights, optimizer state and completed-update epoch position",
                "saved_at": ISO8601DateFormatter().string(from: Date())]
            descriptor.merge(inherited.json) { _, new in new }
            let measured = tracker.snapshot(now: clock())
            descriptor["completed_images"] = measured.completedImages - inherited.images
            descriptor["elapsed_seconds"] = measured.elapsedSeconds - inherited.elapsedSeconds
            descriptor["epoch_elapsed_seconds"] = measured.epochElapsedSeconds
            if let cursor, fingerprint != nil {
                descriptor["resume_cursor"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(cursor))
            }
            descriptor.merge(versionJSON) { _, new in new }
            try TrainingCheckpointFiles.write(weights: session.weights(), optimizer: session.optimizerState(),
                bestWeights: bestWeights, descriptor: descriptor, directory: output)
            tracker.checkpointCompleted(durationSeconds: clock() - checkpointStart, epochComplete: epochComplete, now: clock())
            emit(.init(type: "checkpoint", message: "Checkpoint saved at total update \(inherited.updates + update)", epoch: history.count, update: update))
        }
        func serviceCheckpointRequest() throws {
            guard update > 0, let control, let stop = control.beginRequest() else { return }
            emit(.init(type: "checkpoint_saving", message: stop ? "Saving latest checkpoint before stopping…" : "Saving requested checkpoint…", update: update))
            do {
                try checkpoint(stop ? "saved_stopped" : "training", epochComplete: false)
            } catch {
                control.complete(.failure(error))
                emit(.init(type: "checkpoint_failed", message: "Checkpoint was not saved; training continues. \(error.localizedDescription)", update: update))
                return
            }
            control.complete(.success(.init(runDirectory: output, completedUpdates: update + inherited.updates, stopped: stop)))
            emit(.init(type: "checkpoint_saved", message: "Checkpoint saved at total update \(inherited.updates + update)\(stop ? "; stopping training" : "; training continues")", update: update))
            if stop { cancellation.cancel(); throw TrainingSavedStop() }
        }
        func automaticCheckpoint(_ status: String, epochComplete: Bool = true) {
            do { try checkpoint(status, epochComplete: epochComplete) }
            catch { emit(.init(type: "warning", message: "Automatic checkpoint failed; training continues. \(error.localizedDescription)", update: update)) }
        }
        func writeProgress(_ status: String) throws {
            let measured = tracker.snapshot(now: clock())
            var value: [String: Any] = ["status": status, "history": history, "updates": update,
                "completed_images": measured.completedImages - inherited.images,
                "elapsed_seconds": measured.elapsedSeconds - inherited.elapsedSeconds,
                "total_epochs": inherited.epochs + history.count, "total_updates": inherited.updates + update,
                "total_duration_seconds": measured.elapsedSeconds]
            value.merge(inherited.json) { _, new in new }
            try writeJSON(value, to: output.appendingPathComponent("progress.json"))
        }
        do {
            for epoch in (stoppedEarly ? settings.epochs : restoredCursor?.epoch ?? 0)..<settings.epochs {
                try cancellation.check()
                tracker.beginEpoch(epoch: epoch + 1, now: clock())
                let restoring = restoredCursor?.epoch == epoch && restoredCursor?.order.isEmpty == false ? restoredCursor : nil
                if let restoring {
                    tracker.restoreEpochPosition(stepInEpoch: restoring.epochUpdates, imageIndex: restoring.imageCount,
                        elapsedSeconds: restoredDescriptor?["epoch_elapsed_seconds"] as? Double ?? 0)
                }
                var order = plan.training
                if let restoring {
                    let pairs = Dictionary(uniqueKeysWithValues: order.map { ($0.id, $0) })
                    order = restoring.order.map { pairs[$0]! }
                } else if settings.shuffleEachEpoch { order.shuffle(using: &rng) }
                var epochLoss = restoring?.epochLoss ?? 0, count = restoring?.imageCount ?? 0, learningRate = settings.learningRate
                var epochUpdates = restoring?.epochUpdates ?? 0, clippedUpdates = restoring?.clippedUpdates ?? 0
                var normSum = restoring?.gradientNormSum ?? 0, peakNorm = restoring?.peakGradientNorm ?? 0
                let firstWindow = restoring?.nextWindow ?? 0
                // Resolve deterministic transforms on the serial owner. Only the
                // bounded upcoming batches' pixels are prepared concurrently.
                var batches: [FactoryImageBatch] = []
                var epochCrops = restoring?.crops ?? []
                for window in stride(from: firstWindow, to: order.count, by: settings.effectiveBatchSize) {
                    let end = min(window + settings.effectiveBatchSize, order.count)
                    for offset in stride(from: window, to: end, by: settings.batchSize) {
                        let pairs = Array(order[offset..<min(offset + settings.batchSize, end)])
                        let crops = try restoring != nil ? Array(epochCrops[offset..<min(offset + settings.batchSize, end)]) : pairs.map { pair in
                            try augmentation.transform(sampleID: pair.id, epoch: epoch,
                                width: pair.input.width, height: pair.input.height, size: settings.imageSize,
                                flip: settings.flipCrops, rotate: settings.rotateCrops)
                        }
                        if restoring == nil { epochCrops.append(contentsOf: crops) }
                        batches.append(FactoryImageBatch(pairs: pairs, crops: crops, offset: offset))
                    }
                }
                func updateCursor(nextWindow: Int) {
                    cursor = TrainingCheckpointCursor(epoch: epoch, nextWindow: nextWindow, order: order.map(\.id), crops: epochCrops,
                        epochLoss: epochLoss, imageCount: count, epochUpdates: epochUpdates, clippedUpdates: clippedUpdates,
                        gradientNormSum: normSum, peakGradientNorm: peakNorm, augmentation: augmentation, randomState: rng.state,
                        sourceFingerprint: fingerprint ?? "", bestLoss: bestLoss, bestEpoch: bestEpoch, staleEpochs: stale)
                }
                updateCursor(nextWindow: firstWindow)
                try serviceCheckpointRequest()
                func submit(_ batch: FactoryImageBatch) throws {
                    for pair in batch.pairs { try verify(pair) }
                    try preparation.submit {
                        for pair in batch.pairs {
                            try proofs.ensure(pair.input.url, cancellation: cancellation)
                            try proofs.ensure(pair.target.url, cancellation: cancellation)
                        }
                        return try FactoryPreparedBatch.prepare(batch, map: map, settings: settings, cancellation: cancellation, imageCache: imageCache)
                    }
                }
                var nextPreparationIndex = 0
                if prepareAhead {
                    for batch in batches.prefix(preparationDepth) { try submit(batch); nextPreparationIndex += 1 }
                }
                var batchIndex = 0
                for window in stride(from: firstWindow, to: order.count, by: settings.effectiveBatchSize) {
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
                        let prepared: FactoryPreparedBatch
                        if prepareAhead {
                            prepared = try preparation.take()
                        } else {
                            for pair in batch.pairs {
                                try proofs.ensure(pair.input.url, cancellation: cancellation)
                                try proofs.ensure(pair.target.url, cancellation: cancellation)
                            }
                            prepared = try FactoryPreparedBatch.prepare(batch, map: map, settings: settings,
                                cancellation: cancellation, imageCache: imageCache)
                        }
                        waitingSeconds += clock() - waitingStart
                        preparationSeconds += prepared.preparationSeconds
                        for pair in batch.pairs { try verify(pair) }
                        batchIndex += 1
                        if prepareAhead, nextPreparationIndex < batches.count {
                            try submit(batches[nextPreparationIndex]); nextPreparationIndex += 1
                        }
                        try cancellation.check()
                        tracker.begin(stage: .training, image: batch.imageNames,
                                      index: min(offset + batch.pairs.count, order.count), now: clock())
                        emit(.init(type: "image_training", message: "Training image \(offset + 1) of \(order.count): \(batch.imageNames)",
                                   epoch: epoch + 1, update: update, totalUpdates: totalUpdates))
                        let trainingStart = clock()
                        let loss = try session.train(input: prepared.input, target: prepared.target, batchSize: batch.pairs.count,
                            height: settings.imageSize, width: settings.imageSize, weight: Float(batch.pairs.count) / Float(end - window))
                        trainingSeconds += clock() - trainingStart
                        if !recordedTrainingLayout, let selection = session.trainingLayoutSelection {
                            recordedTrainingLayout = true
                            request["gpu_activation_layout"] = session.channelsLast ? "NHWC" : "NCHW"
                            request["layout_selection"] = ["nchw_seconds": selection.nchwSeconds,
                                "nhwc_seconds": selection.nhwcSeconds, "gradients_agree": selection.gradientsAgree,
                                "calibration_seconds": selection.calibrationSeconds, "fallback_reason": selection.fallbackReason ?? ""]
                            try writeJSON(request, to: output.appendingPathComponent("request.json"))
                            emit(.init(type: "status", message: "Selected \(session.channelsLast ? "channels-last" : "channels-first") Float32 GPU training after comparing both layouts."))
                        }
                        guard loss.isFinite else { throw LoomError("Native training produced a non-finite loss. The last finite checkpoint was preserved.") }
                        epochLoss += Double(loss) * Double(batch.pairs.count); count += batch.pairs.count
                    }
                    try cancellation.check()
                    let cap = options.gradientClipThreshold(updateCount: session.updateCount)
                    tracker.begin(stage: .optimizing,
                                  image: order[window..<end].map { $0.input.trainingDisplayName }.joined(separator: ", "), now: clock())
                    emit(.init(type: "step_updating", message: "Updating total step \(inherited.updates + update + 1) of \(inherited.updates + totalUpdates)", epoch: epoch + 1,
                               update: update, totalUpdates: totalUpdates))
                    let optimizerStart = clock()
                    let norm = try session.applyGradients(learningRate: learningRate); update += 1
                    let optimizerSeconds = clock() - optimizerStart
                    epochUpdates += 1; normSum += norm; peakNorm = max(peakNorm, norm)
                    if cap > 0 && norm > cap { clippedUpdates += 1 }
                    updateCursor(nextWindow: end)
                    let cycleSeconds = clock() - cycleStart
                    tracker.completeStep(stepInEpoch: epochUpdates, completedSteps: update,
                        completedImages: epoch * order.count + count, cycleSeconds: cycleSeconds, now: clock())
                    emit(.init(type: "progress", message: "Epoch \(inherited.epochs + epoch + 1) of \(inherited.epochs + settings.epochs) · step \(inherited.updates + update) of \(inherited.updates + totalUpdates) · \(totalUpdates - update) steps left",
                        progress: Double(update) / Double(totalUpdates), epoch: epoch + 1, loss: epochLoss / Double(count),
                        learningRate: learningRate, update: update, totalUpdates: totalUpdates,
                        gradientNorm: norm, gradientClipThreshold: cap,
                        cycleTiming: TrainingCycleTiming(preparationSeconds: preparationSeconds, preparationWaitSeconds: waitingSeconds,
                            trainingSeconds: trainingSeconds, optimizerSeconds: optimizerSeconds, totalSeconds: cycleSeconds)))
                    // Preserve the first real update even when an epoch spans thousands of pairs.
                    if update == 1 && control?.hasPendingRequest != true { automaticCheckpoint("training", epochComplete: false) }
                    try serviceCheckpointRequest()
                }
                var validation: Double?
                if !plan.validation.isEmpty {
                    let validationStart = clock()
                    tracker.begin(stage: .validating, index: 0, now: clock())
                    emit(.init(type: "status", message: "Validating \(plan.validation.count) held-out images…"))
                    var sum = 0.0
                    for (index, pair) in plan.validation.enumerated() {
                        try serviceCheckpointRequest()
                        tracker.begin(stage: .validating, image: pair.input.trainingDisplayName, index: index + 1, now: clock())
                        emit(.init(type: "validation_image", message: "Validating image \(index + 1) of \(plan.validation.count): \(pair.input.trainingDisplayName)"))
                        try verify(pair)
                        let crop = NativeCropTransform(x: (pair.input.width - settings.imageSize) / 2,
                            y: (pair.input.height - settings.imageSize) / 2, size: settings.imageSize)
                        try proofs.ensure(pair.input.url, cancellation: cancellation)
                        try proofs.ensure(pair.target.url, cancellation: cancellation)
                        let prepared = try NativeImages.pair(inputURL: pair.input.url, targetURL: pair.target.url, transform: crop,
                            mapKind: map.rawValue, normalConvention: pair.target.normalConvention ?? "OpenGL",
                            exaggerate: settings.exaggerateFeatures, cancellation: cancellation, imageCache: imageCache)
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
                try serviceCheckpointRequest()
                var record: [String: Any] = ["epoch": epoch + 1, "loss": epochLoss / Double(count), "learning_rate": learningRate,
                    "update": update,
                    "mean_gradient_norm": normSum / Double(max(1, epochUpdates)), "peak_gradient_norm": peakNorm,
                    "clipped_updates": clippedUpdates, "optimizer_updates": epochUpdates,
                    "gradient_clip_threshold": options.gradientClipThreshold(updateCount: max(0, session.updateCount - 1))]
                record["validation_loss"] = validation ?? NSNull(); history.append(record)
                let stop = settings.earlyStoppingPatience > 0 && stale >= settings.earlyStoppingPatience
                cursor = TrainingCheckpointCursor(epoch: epoch + 1, nextWindow: 0, order: [], crops: [], epochLoss: 0,
                    imageCount: 0, epochUpdates: 0, clippedUpdates: 0, gradientNormSum: 0, peakGradientNorm: 0,
                    augmentation: augmentation, randomState: rng.state, sourceFingerprint: fingerprint ?? "",
                    bestLoss: bestLoss, bestEpoch: bestEpoch, staleEpochs: stale, stoppedEarly: stop)
                emit(.init(type: "epoch_completed", message: "Epoch \(inherited.epochs + epoch + 1) complete", epoch: epoch + 1,
                    loss: epochLoss / Double(count), validationLoss: validation, update: update, totalUpdates: totalUpdates))
                if (epoch + 1) % settings.checkpointEveryEpochs == 0 || epoch + 1 == settings.epochs || stop { automaticCheckpoint("training") }
                try serviceCheckpointRequest()
                try writeProgress("training")
                if stop { stoppedEarly = true; break }
            }
            try serviceCheckpointRequest()
            try cancellation.check()
            tracker.begin(stage: .verifyingSources, now: clock())
            emit(.init(type: "status", message: "Verifying original images before export…"))
            for pair in all { try serviceCheckpointRequest(); try verify(pair) }
            try proofs.verifyAll(cancellation: cancellation) { url in
                try serviceCheckpointRequest()
                tracker.begin(stage: .verifyingSources, image: url.lastPathComponent, now: clock())
                emit(.init(type: "status", message: "Verifying original: \(url.lastPathComponent)"))
            }
            try serviceCheckpointRequest()
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
            try serviceCheckpointRequest()
            tracker.begin(stage: .verifyingSources, now: clock())
            emit(.init(type: "status", message: "Verifying originals after export…"))
            for pair in all { try serviceCheckpointRequest(); try verify(pair) }
            try proofs.verifyAll(cancellation: cancellation) { _ in try serviceCheckpointRequest() }
            automaticCheckpoint("completed")
            try serviceCheckpointRequest()
            try cancellation.check()
            let exportEpoch = settings.exportBestWeights ? (bestEpoch ?? history.count) : history.count
            var training = settingsJSON(settings)
            training.merge(["epochs": history.count, "requested_epochs": settings.epochs, "updates": update, "history": history,
                "stopped_early": stoppedEarly,
                "training_sample_count": plan.training.count, "validation_sample_count": plan.validation.count,
                "export_epoch": exportEpoch, "best_epoch": bestEpoch ?? NSNull(), "best_validation_loss": bestLoss ?? NSNull(),
                "validation_crop": plan.validation.isEmpty ? NSNull() : "fixed-center-native-no-augmentation",
                "device": "metal", "backend": "swift-mpsgraph-metal", "duration_seconds": Date().timeIntervalSince(started)]) { _, new in new }
            training.merge(inherited.json) { _, new in new }
            let measured = tracker.snapshot(now: clock())
            training["elapsed_seconds"] = measured.elapsedSeconds - inherited.elapsedSeconds
            training["completed_images"] = measured.completedImages - inherited.images
            training["total_epochs"] = inherited.epochs + history.count
            training["total_updates"] = inherited.updates + update
            training["total_duration_seconds"] = measured.elapsedSeconds
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
                "parameter_count": settings.parameterCount(map: map), "sources": proofs.snapshots().map { path, proof in proof.json(path: String(path.dropFirst(root.path.count + 1))) },
                "quality_note": "Training completion does not qualify prediction quality on unseen materials."]
            descriptor.merge(versionJSON) { _, new in new }
            try writeProgress("completed")
            try cancellation.check()
            try writeJSON(descriptor, to: output.appendingPathComponent("model.json"))
            let artifact = ModelArtifact(id: id, kind: "map", datasetID: dataset.id, datasetName: dataset.name, map: map,
                directory: output, epochs: history.count, loss: finalLoss, coreMLPath: "model.mlmodel", weightsPath: "model.safetensors", architecture: architecture.identifier)
            try JSONEncoder().encode(artifact).write(to: output.appendingPathComponent("artifact.json"), options: .atomic)
            tracker.begin(stage: .completed, now: clock())
            emit(.init(type: "completed", message: "Training complete · model saved", progress: 1))
            return artifact
        } catch {
            if error is TrainingSavedStop {
                try? writeProgress("saved_stopped")
                tracker.begin(stage: .stopped, now: clock())
                emit(.init(type: "saved_stopped", message: "Training stopped after saving checkpoint at total update \(inherited.updates + update)", update: update))
                throw error
            }
            let status = error is CancellationError ? "cancelled" : "stopped"
            var savedLatest = false
            do { try checkpoint(status); savedLatest = update > 0 }
            catch { emit(.init(type: "checkpoint_failed", message: "Latest checkpoint could not be saved. The previous checkpoint was preserved: \(error.localizedDescription)")) }
            try? writeProgress(status)
            tracker.begin(stage: .stopped, now: clock())
            emit(.init(type: status, message: error is CancellationError
                ? savedLatest ? "Training stopped; completed steps were saved" : update == 0 ? "Training stopped before any optimizer update; no trained checkpoint exists" : "Training stopped; latest save failed. Check the previous checkpoint in Model Versions."
                : error.localizedDescription))
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
        ["image_size": s.imageSize, "source_resolution_levels": s.sourceResolutionLevels, "model_architecture": s.modelArchitecture,
         "scalar_loss": s.scalarLoss.rawValue,
         "width": s.modelWidth, "blocks": s.modelBlocks, "epochs": s.epochs,
         "batch_size": s.batchSize, "learning_rate": s.learningRate, "shuffle_each_epoch": s.shuffleEachEpoch,
         "flip_crops": s.flipCrops, "rotate_crops": s.rotateCrops, "exaggerate_features": s.exaggerateFeatures,
         "seed": s.seed, "requested_seed": s.requestedSeed ?? s.seed,
         "cache_images_in_memory": s.cacheImagesInMemory, "image_cache_limit_mb": s.imageCacheLimitMB, "device": s.device.rawValue, "optimizer": s.optimizer.rawValue, "weight_decay": s.weightDecay,
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
        guard JSONSerialization.isValidJSONObject(value) else {
            throw LoomError("The training descriptor contains an invalid or non-finite value; its previous saved version was preserved.")
        }
        try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .prettyPrinted]).write(to: url, options: .atomic)
    }
    static func json(_ url: URL) throws -> [String: Any] {
        guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else { throw LoomError("Invalid project descriptor.") }; return value
    }
    static func sha(_ url: URL, cancellation: NativeCancellation? = nil) throws -> String {
        try NativeFileHash.sha256(url) { try cancellation?.check() }
    }
}

private struct FactoryRandom: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15; var v = state
        v = (v ^ (v >> 30)) &* 0xbf58476d1ce4e5b9; v = (v ^ (v >> 27)) &* 0x94d049bb133111eb; return v ^ (v >> 31)
    }
}
