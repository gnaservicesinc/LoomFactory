import CryptoKit
import Foundation

extension LoomProject {
    public func exportPackage(name: String, version: String) async throws -> URL {
        guard !operationActive else { throw LoomError("Finish or stop the current project operation before exporting.") }
        operationActive = true
        defer { operationActive = false }
        let current = try snapshot()
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !version.isEmpty else {
            throw LoomError("Give Texture Loom a model name and version before exporting.")
        }
        guard let classifier = current.artifacts.first(where: { $0.kind == "classifier" }) else {
            throw LoomError("Import a classifier from Create ML before exporting Texture Loom.")
        }
        let descriptor = try readDescriptor(classifier, projectID: document.id)
        guard let labels = descriptor["labels"] as? [[String: String]], !labels.isEmpty else {
            throw LoomError("The classifier has no dataset labels.")
        }
        let labelIDs = labels.compactMap { $0["id"] }
        let sourceLabels = labels.compactMap { $0["sourceLabel"] ?? $0["id"] }
        guard labelIDs.count == labels.count, Set(labelIDs).count == labels.count,
              sourceLabels.count == labels.count, Set(sourceLabels).count == labels.count,
              labels.allSatisfy({ label in
                  guard let id = label["id"], let name = label["name"], let sourceLabel = label["sourceLabel"] ?? label["id"] else { return false }
                  return ImageMetadataStore.categoryID(name) == id && !sourceLabel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
              }) else { throw LoomError("The classifier has an inconsistent or duplicate dataset label catalog. Import it again from Create ML.") }
        let currentDatasetIDs = Set(current.datasets.map(\.id))
        guard Set(labelIDs) == currentDatasetIDs else {
            throw LoomError("The classifier does not match every dataset currently tagged in this source folder. Import a classifier for the current datasets.")
        }
        let exports = projectFolder.appendingPathComponent("exports", isDirectory: true)
        let exportID = UUID().uuidString.lowercased()
        let stage = exports.appendingPathComponent(exportID, isDirectory: true)
        let archive = exports.appendingPathComponent("texture-loom-\(safeComponent(version))-\(exportID.prefix(8)).textureloom")
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: true)
        do {
            var classifierRecord = try stageArtifact(classifier, destination: stage, prefix: "classifier", projectID: document.id)
            guard let packagedPath = classifierRecord["path"] as? String else {
                throw LoomError("The classifier is missing its native export.")
            }
            let native = try await nativeClassifierInterface(at: stage.appendingPathComponent(packagedPath))
            guard Set(native.labels) == Set(sourceLabels),
                  native.inputName == (descriptor["input_name"] as? String),
                  native.outputName == (descriptor["output_name"] as? String) else {
                throw LoomError("The classifier's native labels or interface changed after import. Import it again before exporting Texture Loom.")
            }
            classifierRecord["labels"] = labels
            classifierRecord["inputWidth"] = native.width
            classifierRecord["inputHeight"] = native.height
            classifierRecord["inputColorSpace"] = "RGB"
            classifierRecord["inputType"] = native.tensorInput ? "multiArray" : "image"
            if native.tensorInput { classifierRecord["inputShape"] = [1, 3, native.height, native.width] }
            var categories: [[String: Any]] = []
            for label in labels {
                guard let id = label["id"], let categoryName = label["name"] else { throw LoomError("The classifier label format could not be read.") }
                var models: [[String: Any]] = []
                for map in MapRole.targets {
                    guard let artifact = current.artifacts.first(where: { $0.datasetID == id && $0.map == map }) else { continue }
                    var model = try stageArtifact(artifact, destination: stage, prefix: "models/\(id)/\(map.rawValue)", projectID: document.id)
                    model["map"] = map.rawValue
                    models.append(model)
                }
                guard !models.isEmpty else { throw LoomError("Train at least one map model for \(categoryName) before exporting the classifier's datasets.") }
                categories.append(["id": id, "name": categoryName, "version": version, "models": models])
            }
            let files = try packageFiles(stage)
            let manifest: [String: Any] = ["schema": "texture-loom.package.v1", "id": document.id,
                "name": name, "version": version, "classifier": classifierRecord, "categories": categories,
                "files": files, "createdAt": ISO8601DateFormatter().string(from: .now),
                "input": ["name": "rgb", "layout": "NCHW", "color": "sRGB", "range": [0, 1]],
                "normalConvention": "OpenGL"]
            try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
                .write(to: stage.appendingPathComponent("manifest.json"), options: .atomic)
            let card = """
            ---
            library_name: coreml
            tags:
            - texture-loom
            - material-maps
            ---

            # \(name)

            Texture Loom \(version), exported by Loom Factory. The classifier routes images to the matching dataset models.
            CoreML models are used by Texture Studio. Safetensors weights and architecture descriptors are included for compatible Loom Workshop LoRA training.

            Build integration: supply this .textureloom package to Texture Studio's build. The installed application's base model is immutable.
            Training completion alone does not qualify prediction quality. Review the models on unseen materials before deployment.
            """
            try card.write(to: stage.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
            try await execute(executable: URL(fileURLWithPath: "/usr/bin/ditto"), arguments: ["-c", "-k", stage.path, archive.path],
                              environment: [:], log: exports.appendingPathComponent(exportID + "-export.log"), progress: { _ in })
            try FileManager.default.removeItem(at: stage)
            return archive
        } catch {
            try? FileManager.default.removeItem(at: stage)
            try? FileManager.default.removeItem(at: archive)
            throw error
        }
    }

    public func publishPackage(package: URL, repository: String,
                               progress: @escaping @Sendable (TrainingEvent) -> Void) async throws -> URL {
        guard !operationActive else { throw LoomError("Finish or stop the current project operation before publishing.") }
        operationActive = true
        defer { operationActive = false }
        let components = repository.split(separator: "/")
        guard components.count == 2, components.allSatisfy({ !$0.isEmpty && $0.allSatisfy({ $0.isLetter || $0.isNumber || "._-".contains($0) }) }) else {
            throw LoomError("Use a Hugging Face repository name such as your-account/Texture-Loom.")
        }
        let resolved = package.resolvingSymlinksInPath().standardizedFileURL
        guard resolved.path.hasPrefix(projectFolder.appendingPathComponent("exports").path + "/"),
              FileManager.default.fileExists(atPath: resolved.path), resolved.pathExtension == "textureloom" else {
            throw LoomError("Export a Texture Loom package from this source project before publishing.")
        }
        let python = projectFolder.appendingPathComponent("runtime/bin/python3")
        guard FileManager.default.isExecutableFile(atPath: python.path) else { throw LoomError("Train a model in this source project before publishing its package.") }
        // Credentials remain in Hugging Face's credential store; no tokens enter a project or its logs.
        let code = """
        import json,sys
        from huggingface_hub import HfApi
        api=HfApi()
        api.whoami()
        repo=sys.argv[1]; package=sys.argv[2]
        api.create_repo(repo_id=repo,repo_type='model',exist_ok=True)
        print(json.dumps({'event':'progress','message':'Uploading the Texture Loom package…'}),flush=True)
        api.upload_file(path_or_fileobj=package,path_in_repo='TextureLoom.textureloom',repo_id=repo,repo_type='model',commit_message='Publish Texture Loom package')
        card='# Texture Loom\\n\\nA complete classifier and material-model package created with Loom Factory. Download TextureLoom.textureloom for Texture Studio build integration.\\n'
        api.upload_file(path_or_fileobj=card.encode(),path_in_repo='README.md',repo_id=repo,repo_type='model',commit_message='Update Texture Loom model card')
        print(json.dumps({'event':'completed','message':'Texture Loom published','progress':1}),flush=True)
        """
        progress(.init(type: "publish", message: "Publishing Texture Loom to \(repository)…"))
        try await execute(executable: python, arguments: ["-c", code, repository, resolved.path], environment: [:],
                          log: projectFolder.appendingPathComponent("publish.log"), progress: progress)
        return URL(string: "https://huggingface.co/" + repository)!
    }
}

private func readDescriptor(_ artifact: ModelArtifact, projectID: String) throws -> [String: Any] {
    guard let result = try JSONSerialization.jsonObject(with: Data(contentsOf: artifact.directory.appendingPathComponent("model.json"))) as? [String: Any],
          result["status"] as? String == "completed" else { throw LoomError("The trained model export is incomplete. Its run remains in the source project.") }
    guard result["project_id"] as? String == projectID else {
        throw LoomError("The model belongs to a different source project. Import or train its model in this source folder before exporting.")
    }
    return result
}
private func stageArtifact(_ artifact: ModelArtifact, destination: URL, prefix: String, projectID: String) throws -> [String: Any] {
    let descriptor = try readDescriptor(artifact, projectID: projectID)
    guard let nativePath = artifact.coreMLPath else { throw LoomError("The model is missing its native export.") }
    if artifact.kind != "classifier", artifact.weightsPath == nil { throw LoomError("The map model is missing its trainable weights.") }
    let artifactPaths = [nativePath] + (artifact.weightsPath.map { [$0] } ?? [])
    for path in artifactPaths {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.split(separator: "/").contains("..") else { throw LoomError("The model export references a file outside its run.") }
    }
    let folder = destination.appendingPathComponent(prefix, isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    for path in artifactPaths + ["model.json"] {
        let source = artifact.directory.appendingPathComponent(path)
        guard FileManager.default.fileExists(atPath: source.path) else { throw LoomError("A model file was removed from this project's run: \(source.lastPathComponent).") }
        guard source.resolvingSymlinksInPath().path.hasPrefix(artifact.directory.resolvingSymlinksInPath().path + "/") else {
            throw LoomError("The model export references a file outside its run.")
        }
        try FileManager.default.copyItem(at: source, to: folder.appendingPathComponent(source.lastPathComponent))
    }
    guard let inputName = descriptor["input_name"] as? String, !inputName.isEmpty,
          let outputName = descriptor["output_name"] as? String, !outputName.isEmpty else {
        throw LoomError("The model descriptor is missing its native input or output name.")
    }
    var record: [String: Any] = ["format": "coreml", "path": prefix + "/" + URL(fileURLWithPath: nativePath).lastPathComponent,
        "architecture": artifact.architecture, "inputName": inputName,
        "outputName": outputName]
    if let width = descriptor["input_width"] as? Int, let height = descriptor["input_height"] as? Int {
        record["inputWidth"] = width; record["inputHeight"] = height
        record["inputColorSpace"] = descriptor["input_color_space"] as? String ?? "RGB"
    }
    if let type = descriptor["input_type"] as? String { record["inputType"] = type }
    if let shape = descriptor["input_shape"] as? [Int] { record["inputShape"] = shape }
    if let weightsPath = artifact.weightsPath {
        record["weightsPath"] = prefix + "/" + URL(fileURLWithPath: weightsPath).lastPathComponent
        record["sha256"] = try fileSHA(artifact.directory.appendingPathComponent(weightsPath))
    }
    return record
}
private func packageFiles(_ folder: URL) throws -> [[String: String]] {
    guard let files = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles]) else { throw LoomError("Could not inspect the model package.") }
    var result: [[String: String]] = []
    for case let file as URL in files {
        let properties = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        guard properties.isSymbolicLink != true else { throw LoomError("Deployment models must contain their files, without links to source-project files.") }
        guard properties.isRegularFile == true else { continue }
        let canonicalFile = file.standardizedFileURL.resolvingSymlinksInPath()
        let canonicalRoot = folder.standardizedFileURL.resolvingSymlinksInPath()
        result.append(["path": String(canonicalFile.path.dropFirst(canonicalRoot.path.count + 1)), "sha256": try fileSHA(file)])
    }
    return result.sorted { $0["path"]! < $1["path"]! }
}
private func fileSHA(_ file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
    var hasher = SHA256()
    while let block = try handle.read(upToCount: 1024 * 1024), !block.isEmpty { hasher.update(data: block) }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}
private func safeComponent(_ value: String) -> String {
    String(value.map { $0.isLetter || $0.isNumber || "._-".contains($0) ? $0 : "-" })
}
