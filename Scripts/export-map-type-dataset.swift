#!/usr/bin/env swift
import CryptoKit
import Darwin
import Foundation
import ImageIO

// Read-only source scan. Saved Factory tags decide the class; filenames never
// invent a map role. ImageIO reads native dimensions without decoding pixels.
let attribute = "com.gnaservicesinc.loomfactory.image.v1"
let reportSchema = "loom-map-type-export-v1"
let classForRole = ["depth": "depth", "diffuse": "diffuse", "normal": "normal", "roughness": "rough"]
let managedClasses: Set<String> = ["depth", "diffuse", "normal", "rough", "bump"]
let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "tif", "tiff", "heic", "heif", "webp", "bmp", "exr"]
let manager = FileManager.default

struct ExportError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

struct Candidate {
    let file: URL
    let material: String
    let label: String
    let width: Int
    let height: Int
    var rank: Int {
        if min(width, height) == 1024 { return 0 }
        if min(width, height) == 2048 { return 1 }
        return min(width, height) >= 1024 ? 2 : 3
    }
    var area: Int64 { Int64(width) * Int64(height) }
}

func exists(_ url: URL) -> Bool {
    var value = stat()
    return lstat(url.path, &value) == 0
}

func metadata(_ file: URL) throws -> [String: Any]? {
    let count = getxattr(file.path, attribute, nil, 0, 0, 0)
    if count < 0 {
        if errno == ENOATTR { return nil }
        throw ExportError("Could not read saved tags: \(file.path)")
    }
    guard count > 0 && count < 1024 * 1024 else { throw ExportError("Invalid saved tag size: \(file.path)") }
    var bytes = Data(count: count)
    let read = bytes.withUnsafeMutableBytes { getxattr(file.path, attribute, $0.baseAddress, count, 0, 0) }
    guard read == count, let tag = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
        throw ExportError("Saved tags changed or are malformed: \(file.path)")
    }
    return tag
}

func better(_ first: Candidate, _ second: Candidate) -> Bool {
    if first.rank != second.rank { return first.rank < second.rank }
    if first.area != second.area { return first.rank == 3 ? first.area > second.area : first.area < second.area }
    return first.file.path < second.file.path
}

func relativePath(_ file: URL, root: URL) -> String {
    String(file.path.dropFirst(root.path.count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
}

func digest(_ value: String) -> String {
    SHA256.hash(data: Data(value.utf8)).map { String(format: "%02x", $0) }.joined()
}

func managedRelativePath(_ value: String) -> Bool {
    let pieces = value.split(separator: "/", omittingEmptySubsequences: false)
    return pieces.count == 2 && managedClasses.contains(String(pieces[0])) &&
        pieces[1].hasPrefix("loom-type-") && !pieces[1].contains("..")
}

func linkTarget(_ file: URL) -> String? {
    try? manager.destinationOfSymbolicLink(atPath: file.path)
}

// Foundation sometimes abbreviates /private/var to /var. Resolve existing
// parents with POSIX realpath before comparing roots, including new outputs.
func canonicalURL(_ path: String) throws -> URL {
    var parent = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
    var suffix: [String] = []
    while true {
        if let pointer = realpath(parent.path, nil) {
            let resolved = String(cString: pointer)
            free(pointer)
            return suffix.reversed().reduce(URL(fileURLWithPath: resolved, isDirectory: true)) { $0.appendingPathComponent($1) }
        }
        guard parent.path != "/" else { throw ExportError("Could not resolve folder path: \(path)") }
        suffix.append(parent.lastPathComponent)
        parent.deleteLastPathComponent()
    }
}

func run() throws {
    var sourcePath = "/opt/ipde/sources_mats", outputPath = "/opt/ipde/mat_type_ds"
    var dryRun = false
    let args = Array(CommandLine.arguments.dropFirst())
    var index = 0
    while index < args.count {
        switch args[index] {
        case "--source", "--output":
            guard index + 1 < args.count else { throw ExportError("\(args[index]) requires a folder path.") }
            if args[index] == "--source" { sourcePath = args[index + 1] } else { outputPath = args[index + 1] }
            index += 2
        case "--dry-run": dryRun = true; index += 1
        case "--help", "-h":
            print("Usage: swift Scripts/export-map-type-dataset.swift [--source FOLDER] [--output FOLDER] [--dry-run]")
            return
        default: throw ExportError("Unknown argument: \(args[index])")
        }
    }
    let source = try canonicalURL(sourcePath)
    let output = try canonicalURL(outputPath)
    var directory: ObjCBool = false
    guard manager.fileExists(atPath: source.path, isDirectory: &directory), directory.boolValue else {
        throw ExportError("The source folder does not exist: \(source.path)")
    }
    guard source.path != output.path, !output.path.hasPrefix(source.path + "/"), !source.path.hasPrefix(output.path + "/") else {
        throw ExportError("Source and output folders must be separate and must not contain one another.")
    }
    if exists(output) {
        guard manager.fileExists(atPath: output.path, isDirectory: &directory), directory.boolValue else {
            throw ExportError("The output path is not a folder: \(output.path)")
        }
    }
    let reportURL = output.appendingPathComponent("map-type-export-report.json")
    var previousEntries: [[String: Any]] = []
    if exists(reportURL) {
        guard linkTarget(reportURL) == nil,
              let report = try JSONSerialization.jsonObject(with: Data(contentsOf: reportURL)) as? [String: Any],
              report["schema"] as? String == reportSchema, report["source_root"] as? String == source.path,
              let entries = report["entries"] as? [[String: Any]] else {
            throw ExportError("An unrelated report already exists at \(reportURL.path). Choose another output folder.")
        }
        previousEntries = entries
    }
    var selected: [String: Candidate] = [:]
    var skipped: [String: Int] = [:]
    var warnings: [[String: String]] = []
    var visitedDirectories: Set<String> = [], visitedFiles: Set<String> = []
    var scannedImages = 0, eligibleImages = 0
    func skip(_ reason: String) { skipped[reason, default: 0] += 1 }
    func visit(_ folder: URL) throws {
        guard visitedDirectories.insert(folder.resolvingSymlinksInPath().path).inserted else { return }
        for listed in try manager.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]).sorted(by: { $0.path < $1.path }) {
            let entry = folder.appendingPathComponent(listed.lastPathComponent)
            var isDirectory: ObjCBool = false
            guard manager.fileExists(atPath: entry.path, isDirectory: &isDirectory) else { continue }
            if isDirectory.boolValue { try visit(entry); continue }
            guard imageExtensions.contains(entry.pathExtension.lowercased()), visitedFiles.insert(entry.resolvingSymlinksInPath().path).inserted else { continue }
            scannedImages += 1
            do {
                guard let tag = try metadata(entry) else { skip("untagged"); continue }
                guard tag["schema"] as? String == "loom-factory.image.v1" else { skip("unsupported_metadata"); continue }
                guard tag["ignored"] as? Bool != true else { skip("ignored"); continue }
                if tag["role"] as? String == "bump" { skip("bump_map"); continue }
                guard let role = tag["role"] as? String, let label = classForRole[role] else { skip("unknown_map_type"); continue }
                let tokens = entry.deletingPathExtension().lastPathComponent.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init)
                if (tag["normalConvention"] as? String)?.lowercased() == "directx" ||
                    (role == "normal" && tokens.contains(where: { ["dx", "directx", "normaldx"].contains($0) })) {
                    skip("directx_normal"); continue
                }
                guard let image = CGImageSourceCreateWithURL(entry as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
                      let props = CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [CFString: Any],
                      let width = props[kCGImagePropertyPixelWidth] as? Int, let height = props[kCGImagePropertyPixelHeight] as? Int,
                      width > 0, height > 0 else { skip("unreadable_dimensions"); continue }
                eligibleImages += 1
                let material = relativePath(entry.deletingLastPathComponent(), root: source)
                let candidate = Candidate(file: entry, material: material.isEmpty ? "." : material, label: label, width: width, height: height)
                let key = candidate.material + "\0" + label
                if selected[key].map({ better(candidate, $0) }) ?? true { selected[key] = candidate }
            } catch {
                skip("unreadable_metadata")
                warnings.append(["path": entry.path, "message": String(describing: error)])
            }
        }
    }
    try visit(source)
    guard !selected.isEmpty else { throw ExportError("No current images have supported saved Factory map-type tags. Open the source folder in Loom Factory first.") }
    let classes = Set(classForRole.values).sorted()
    for label in classes {
        let folder = output.appendingPathComponent(label, isDirectory: true)
        if exists(folder) {
            guard linkTarget(folder) == nil, manager.fileExists(atPath: folder.path, isDirectory: &directory), directory.boolValue else {
                throw ExportError("A class path already exists and is not a regular folder: \(folder.path)")
            }
        } else if !dryRun { try manager.createDirectory(at: folder, withIntermediateDirectories: true) }
    }
    let previousByKey = Dictionary(previousEntries.compactMap { entry -> (String, [String: Any])? in
        guard let key = entry["export_key"] as? String else { return nil }
        return (key, entry)
    }, uniquingKeysWith: { first, _ in first })
    var entries: [[String: Any]] = [], actions: [[String: String]] = []
    var counts = Dictionary(uniqueKeysWithValues: classes.map { ($0, 0) })
    var chosenPaths: Set<String> = []
    for candidate in selected.values.sorted(by: { ($0.label, $0.material) < ($1.label, $1.material) }) {
        let key = digest(source.path + "\0" + candidate.material + "\0" + candidate.label)
        let slug = candidate.material.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? String($0) : "-" }.joined()
        let stem = "loom-type-" + String(slug.prefix(70)) + "__" + String(key.prefix(20))
        var relative = candidate.label + "/" + stem + "." + candidate.file.pathExtension.lowercased()
        if let old = previousByKey[key], let oldPath = old["path"] as? String, let oldTarget = old["source"] as? String,
           managedRelativePath(oldPath), oldPath.split(separator: "/").first == Substring(candidate.label),
           linkTarget(output.appendingPathComponent(oldPath)) == oldTarget {
            relative = oldPath
        }
        var target = output.appendingPathComponent(relative)
        let old = previousByKey[key]
        var replacesOwned = false
        if exists(target), linkTarget(target) != candidate.file.path {
            replacesOwned = old?["path"] as? String == relative && old?["source"] as? String == linkTarget(target)
            if !replacesOwned {
                var suffix = 1
                repeat {
                    relative = candidate.label + "/" + stem + "-\(suffix)." + candidate.file.pathExtension.lowercased()
                    target = output.appendingPathComponent(relative)
                    suffix += 1
                } while exists(target) && linkTarget(target) != candidate.file.path
            }
        }
        let action = replacesOwned ? "update_owned_link" : exists(target) ? "reuse" : "create"
        if !dryRun {
            if replacesOwned { try manager.removeItem(at: target) }
            if !exists(target) { try manager.createSymbolicLink(atPath: target.path, withDestinationPath: candidate.file.path) }
        }
        chosenPaths.insert(relative)
        counts[candidate.label, default: 0] += 1
        actions.append(["action": action, "path": relative])
        entries.append(["export_key": key, "path": relative, "source": candidate.file.path,
                        "material_folder": candidate.material, "class": candidate.label,
                        "width": candidate.width, "height": candidate.height, "resolution_priority": candidate.rank])
    }
    for old in previousEntries {
        guard let path = old["path"] as? String, let sourcePath = old["source"] as? String,
              managedRelativePath(path), !chosenPaths.contains(path), linkTarget(output.appendingPathComponent(path)) == sourcePath else { continue }
        actions.append(["action": "remove_obsolete_owned_link", "path": path])
        if !dryRun { try manager.removeItem(at: output.appendingPathComponent(path)) }
    }
    let report: [String: Any] = ["schema": reportSchema, "source_root": source.path, "output_root": output.path,
                               "dry_run": dryRun, "scanned_images": scannedImages, "eligible_images": eligibleImages,
                               "exported_images": entries.count, "class_counts": counts,
                               "skipped_images": skipped, "warnings": warnings, "entries": entries, "actions": actions,
                               "unmanaged_output_files": "Existing files not listed in this report are preserved."]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]) + Data([10])
    if !dryRun { try data.write(to: reportURL, options: [.atomic]) }
    FileHandle.standardOutput.write(data)
}

do { try run() } catch {
    FileHandle.standardError.write(Data("Error: \(error)\n".utf8))
    exit(1)
}
