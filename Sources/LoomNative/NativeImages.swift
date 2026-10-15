import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import ImageIO

/// Decoded numeric pixels in stored image orientation, interleaved HWC order.
/// No image file, crop or cache is rewritten by this module.
public struct NativeImage: Sendable {
    public let width: Int
    public let height: Int
    public let channels: Int
    public let values: [Float]
    public init(width: Int, height: Int, channels: Int, values: [Float]) {
        self.width = width; self.height = height; self.channels = channels; self.values = values
    }
}

public struct NativeCropTransform: Codable, Sendable, Equatable {
    public let x: Int
    public let y: Int
    public let size: Int
    public let horizontalFlip: Bool
    public let verticalFlip: Bool
    public let quarterTurns: Int
    public init(x: Int, y: Int, size: Int, horizontalFlip: Bool = false,
                verticalFlip: Bool = false, quarterTurns: Int = 0) {
        self.x = x; self.y = y; self.size = size
        self.horizontalFlip = horizontalFlip; self.verticalFlip = verticalFlip; self.quarterTurns = quarterTurns
    }
    enum CodingKeys: String, CodingKey {
        case x, y, size
        case horizontalFlip = "horizontal_flip"
        case verticalFlip = "vertical_flip"
        case quarterTurns = "quarter_turns"
    }
}

public struct NativeImagePair: Sendable {
    public let input: NativeTensor
    public let target: NativeTensor
    public let crop: NativeCropTransform
}

public enum NativeImages {
    /// Keep a bounded share of unified memory for unchanged decoded originals.
    /// The measured 55-pair Bricks corpus needs 7.92 GiB of original numeric
    /// storage. A quarter of RAM (capped at 16 GiB) leaves room for it and other
    /// map targets without repeatedly decoding originals between epochs.
    public static let decodedPixelCacheByteLimit = Int(min(UInt64(16 * 1024 * 1024 * 1024),
        max(UInt64(512 * 1024 * 1024), ProcessInfo.processInfo.physicalMemory / 4)))
    private static let rawPixelCache = NativeRawPixelCache(limit: decodedPixelCacheByteLimit)

    /// Releases the bounded, in-memory cache of decoded original numeric codes.
    public static func clearDecodedPixelCache() { rawPixelCache.clear() }

    public static func readRGB(from url: URL, cancellation: NativeCancellation? = nil) throws -> NativeImage {
        let image = try decode(url, cancellation: cancellation)
        return try rgb(image, url: url, cancellation: cancellation)
    }

    private static func rgb(_ image: NativeImage, url: URL, cancellation: NativeCancellation?) throws -> NativeImage {
        var values = [Float](repeating: 0, count: image.width * image.height * 3)
        let linear = url.pathExtension.lowercased() == "exr"
        for pixel in 0..<(image.width * image.height) {
            if pixel % 65536 == 0 { try cancellation?.check() }
            for channel in 0..<3 {
                var value = image.values[pixel * image.channels + (image.channels < 3 ? 0 : channel)]
                value = min(1, max(0, value))
                if linear { value = value <= 0.0031308 ? value * 12.92 : 1.055 * pow(value, 1 / 2.4) - 0.055 }
                values[pixel * 3 + channel] = value
            }
        }
        return NativeImage(width: image.width, height: image.height, channels: 3, values: values)
    }

    public static func readTarget(from url: URL, mapKind: String,
                                  normalConvention: String = "OpenGL", cancellation: NativeCancellation? = nil) throws -> NativeImage {
        try target(decode(url, cancellation: cancellation), url: url, mapKind: mapKind,
                   normalConvention: normalConvention, cancellation: cancellation)
    }

    private static func target(_ image: NativeImage, url: URL, mapKind: String,
                               normalConvention: String, cancellation: NativeCancellation?) throws -> NativeImage {
        guard ["depth", "roughness", "normal"].contains(mapKind) else {
            throw NativeIOError.invalid("Unknown target map product: \(mapKind).")
        }
        guard image.values.allSatisfy({ $0 >= 0 && $0 <= 1 }) else {
            throw NativeIOError.invalid("Target numeric pixels must be in [0,1]: \(url.lastPathComponent).")
        }
        let normal = mapKind == "normal"
        guard !normal || image.channels >= 3 else {
            throw NativeIOError.invalid("Normal targets need three color channels: \(url.lastPathComponent).")
        }
        let channels = normal ? 3 : 1
        var result = [Float](repeating: 0, count: image.width * image.height * channels)
        for pixel in 0..<(image.width * image.height) {
            if pixel % 65536 == 0 { try cancellation?.check() }
            let base = pixel * image.channels
            if !normal, image.channels >= 3,
               (image.values[base] != image.values[base + 1] || image.values[base] != image.values[base + 2]) {
                throw NativeIOError.invalid("Scalar target has different color channels: \(url.lastPathComponent).")
            }
            for channel in 0..<channels {
                let value = image.values[base + channel]
                result[pixel * channels + channel] = normal && channel == 1 && normalConvention.lowercased() == "directx" ? 1 - value : value
            }
        }
        return NativeImage(width: image.width, height: image.height, channels: channels, values: result)
    }

    /// Reads only the selected native region into Float32 training storage.
    /// The cache stores original numeric codes, rather than full-size HWC floats.
    public static func pair(inputURL: URL, targetURL: URL, transform: NativeCropTransform,
                            mapKind: String, normalConvention: String = "OpenGL", exaggerate: Bool = false,
                            cancellation: NativeCancellation? = nil) throws -> NativeImagePair {
        let inputStorage = try storedPixels(inputURL, cancellation: cancellation)
        let targetStorage = try storedPixels(targetURL, cancellation: cancellation)
        guard inputStorage.width == targetStorage.width, inputStorage.height == targetStorage.height else {
            throw NativeIOError.invalid("Paired maps need the same native dimensions.")
        }
        try validateStoredPixels(inputStorage, url: inputURL, role: "rgb", cancellation: cancellation)
        try validateStoredPixels(targetStorage, url: targetURL, role: mapKind, cancellation: cancellation)
        let input = try rgb(numericPixels(inputStorage, region: transform, url: inputURL, cancellation: cancellation),
                            url: inputURL, cancellation: cancellation)
        let teaching = try target(numericPixels(targetStorage, region: transform, url: targetURL, cancellation: cancellation),
                                  url: targetURL, mapKind: mapKind, normalConvention: normalConvention, cancellation: cancellation)
        let local = NativeCropTransform(x: 0, y: 0, size: transform.size, horizontalFlip: transform.horizontalFlip,
                                        verticalFlip: transform.verticalFlip, quarterTurns: transform.quarterTurns)
        let prepared = try pair(input: input, target: teaching, transform: local, mapKind: mapKind,
                                exaggerate: exaggerate, cancellation: cancellation)
        return NativeImagePair(input: prepared.input, target: prepared.target, crop: transform)
    }

    public static func pair(input: NativeImage, target: NativeImage, transform: NativeCropTransform,
                            mapKind: String, exaggerate: Bool = false, cancellation: NativeCancellation? = nil) throws -> NativeImagePair {
        try cancellation?.check()
        let size = transform.size
        guard input.width == target.width, input.height == target.height,
              input.channels == 3, target.channels == (mapKind == "normal" ? 3 : 1),
              input.values.count == input.width * input.height * input.channels,
              target.values.count == target.width * target.height * target.channels,
              size > 0, transform.x >= 0, transform.y >= 0,
              transform.x <= input.width - size, transform.y <= input.height - size,
              (0...3).contains(transform.quarterTurns) else {
            throw NativeIOError.invalid("Paired maps need matching native dimensions and a valid exact crop.")
        }
        let rgb = try crop(input, transform: transform, normal: false, cancellation: cancellation)
        var teaching = try crop(target, transform: transform, normal: mapKind == "normal", cancellation: cancellation)
        if exaggerate {
            guard mapKind == "depth" else { throw NativeIOError.invalid("Feature exaggeration requires a depth target.") }
            if let low = teaching.min(), let high = teaching.max(), high - low > 1e-12 {
                let span = high - low
                for index in teaching.indices {
                    if index % 65536 == 0 { try cancellation?.check() }
                    let value = (teaching[index] - low) / span
                    teaching[index] = low + ((tanh((value * 2 - 1) * 2) / tanh(Float(2)) + 1) * 0.5) * span
                }
            }
        }
        // COW avoids extra decoded-image copies. Only the current exact crop is
        // allocated in NCHW memory for the GPU; source resolution is untouched.
        let result = NativeImagePair(input: NativeTensor(shape: [1, 3, size, size], values: rgb),
                                    target: NativeTensor(shape: [1, target.channels, size, size], values: teaching),
                                    crop: transform)
        return result
    }

    private static func crop(_ image: NativeImage, transform: NativeCropTransform, normal: Bool,
                             cancellation: NativeCancellation?) throws -> [Float] {
        let size = transform.size
        var result = [Float](repeating: 0, count: size * size * image.channels)
        for outputY in 0..<size {
            if outputY % 16 == 0 { try cancellation?.check() }
            for outputX in 0..<size {
                // Invert np.rot90 after the two flips to read original pixels.
                var x = outputX, y = outputY
                for _ in 0..<transform.quarterTurns { (x, y) = (size - 1 - y, x) }
                if transform.verticalFlip { y = size - 1 - y }
                if transform.horizontalFlip { x = size - 1 - x }
                let source = ((transform.y + y) * image.width + transform.x + x) * image.channels
                for channel in 0..<image.channels {
                    result[channel * size * size + outputY * size + outputX] = image.values[source + channel]
                }
                if normal {
                    let index = outputY * size + outputX
                    var vx = result[index], vy = result[size * size + index]
                    if transform.horizontalFlip { vx = 1 - vx }
                    if transform.verticalFlip { vy = 1 - vy }
                    for _ in 0..<transform.quarterTurns { (vx, vy) = (1 - vy, vx) }
                    result[index] = vx; result[size * size + index] = vy
                }
            }
        }
        return result
    }

    private static func decode(_ url: URL, cancellation: NativeCancellation?) throws -> NativeImage {
        try numericPixels(storedPixels(url, cancellation: cancellation), region: nil, url: url, cancellation: cancellation)
    }

    private static func storedPixels(_ url: URL, cancellation: NativeCancellation?) throws -> NativeRawPixels {
        try cancellation?.check()
        let key = try NativeRawPixelKey(url)
        if let stored = rawPixelCache.read(key) { return stored }
        let supported = ["png", "jpg", "jpeg", "tif", "tiff", "exr", "heic", "heif", "bmp", "webp"]
        guard supported.contains(url.pathExtension.lowercased()) else {
            throw NativeIOError.unsupported("The native image decoder does not support \(url.pathExtension) texture sources.")
        }
        let options: [CFString: Any] = [kCGImageSourceShouldCache: false, kCGImageSourceShouldAllowFloat: true]
        guard let source = CGImageSourceCreateWithURL(url as CFURL, options as CFDictionary),
              let decoded = CGImageSourceCreateImageAtIndex(source, 0, options as CFDictionary) else {
            throw NativeIOError.invalid("The native image decoder cannot read \(url.lastPathComponent). Deep or multipart EXR files may require conversion to a flat texture map.")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        if let storedDepth = (properties?[kCGImagePropertyDepth] as? NSNumber)?.intValue,
           storedDepth > decoded.bitsPerComponent {
            throw NativeIOError.unsupported("ImageIO cannot preserve the stored \(storedDepth)-bit precision of \(url.lastPathComponent); it returned \(decoded.bitsPerComponent)-bit pixels.")
        }
        let image: CGImage
        if decoded.colorSpace?.model == .indexed || decoded.colorSpace?.model == .cmyk {
            // Palette/CMYK low-bit images require a standard RGB interpretation.
            // No gamma conversion or drawing is used for native gray/RGB data.
            guard decoded.bitsPerComponent <= 8,
                  let context = CGContext(data: nil, width: decoded.width, height: decoded.height,
                                          bitsPerComponent: 8, bytesPerRow: decoded.width * 4,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
                throw NativeIOError.unsupported("Unsupported high precision image color model: \(url.lastPathComponent).")
            }
            context.draw(decoded, in: CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
            guard let converted = context.makeImage() else { throw NativeIOError.invalid("Cannot interpret image palette.") }
            image = converted
        } else { image = decoded }
        guard let data = image.dataProvider?.data else { throw NativeIOError.invalid("The decoded image has no pixel storage.") }
        try cancellation?.check()
        let bits = image.bitsPerComponent
        let floating = image.bitmapInfo.contains(.floatComponents)
        guard [8, 16, 32].contains(bits), (!floating || bits != 8), image.bitsPerPixel % bits == 0,
              image.colorSpace?.model == .rgb || image.colorSpace?.model == .monochrome else {
            throw NativeIOError.unsupported("Unsupported native pixel layout in \(url.lastPathComponent): \(bits)-bit components.")
        }
        let components = image.bitsPerPixel / bits
        let channels = image.colorSpace?.model == .rgb ? 3 : 1
        let alpha = image.alphaInfo
        let leadingAlpha = alpha == .first || alpha == .premultipliedFirst || alpha == .noneSkipFirst
        let premultiplied = alpha == .premultipliedFirst || alpha == .premultipliedLast
        let hasAlpha = alpha != .none && alpha != .noneSkipFirst && alpha != .noneSkipLast
        guard components >= channels + (hasAlpha ? 1 : 0), components <= 4 else {
            throw NativeIOError.unsupported("Unsupported native channel layout in \(url.lastPathComponent).")
        }
        let storage = data as Data
        guard storage.count >= image.bytesPerRow * image.height else { throw NativeIOError.invalid("Decoded image pixels are truncated.") }
        let stored = NativeRawPixels(key: key, width: image.width, height: image.height, channels: channels,
            bits: bits, components: components, bitsPerPixel: image.bitsPerPixel, bytesPerRow: image.bytesPerRow,
            bitmapInfo: image.bitmapInfo.rawValue, floating: floating, leadingAlpha: leadingAlpha,
            premultiplied: premultiplied, hasAlpha: hasAlpha, storage: storage)
        guard try NativeRawPixelKey(url) == key else { throw NativeIOError.invalid("Source pixels changed while decoding \(url.lastPathComponent).") }
        rawPixelCache.insert(stored)
        return stored
    }

    private static func numericPixels(_ image: NativeRawPixels, region: NativeCropTransform?, url: URL,
                                       cancellation: NativeCancellation?) throws -> NativeImage {
        let width = region?.size ?? image.width, height = region?.size ?? image.height
        let offsetX = region?.x ?? 0, offsetY = region?.y ?? 0
        guard width > 0, height > 0, offsetX >= 0, offsetY >= 0,
              offsetX <= image.width - width, offsetY <= image.height - height else {
            throw NativeIOError.invalid("The selected exact crop is outside the native image: \(url.lastPathComponent).")
        }
        let bits = image.bits, components = image.components, channels = image.channels
        let floating = image.floating, hasAlpha = image.hasAlpha, leadingAlpha = image.leadingAlpha
        let premultiplied = image.premultiplied
        let byteOrder = image.bitmapInfo & CGBitmapInfo.byteOrderMask.rawValue
        let littleEndian = byteOrder == CGBitmapInfo.byteOrder16Little.rawValue || byteOrder == CGBitmapInfo.byteOrder32Little.rawValue
        let reversedBytes = bits == 8 && components == 4 && byteOrder == CGBitmapInfo.byteOrder32Little.rawValue
        let integerMax = Float(pow(2.0, Double(bits)) - 1)
        var values = [Float](repeating: 0, count: width * height * channels)
        try image.storage.withUnsafeBytes { bytes in
            func sample(_ offset: Int) -> Float {
                if bits == 8 { return Float(bytes[offset]) / 255 }
                if bits == 16 {
                    let raw = bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self)
                    let code = littleEndian ? UInt16(littleEndian: raw) : UInt16(bigEndian: raw)
                    return floating ? Float(Float16(bitPattern: code)) : Float(code) / integerMax
                }
                let raw = bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
                let code = littleEndian ? UInt32(littleEndian: raw) : UInt32(bigEndian: raw)
                return floating ? Float(bitPattern: code) : Float(code) / integerMax
            }
            func component(_ base: Int, _ index: Int) -> Float {
                sample(base + (reversedBytes ? components - 1 - index : index) * bits / 8)
            }
            for y in 0..<height {
                if y % 16 == 0 { try cancellation?.check() }
                for x in 0..<width {
                    let base = (y + offsetY) * image.bytesPerRow + (x + offsetX) * image.bitsPerPixel / 8
                    let opacity = hasAlpha ? component(base, leadingAlpha ? 0 : components - 1) : 1
                    for channel in 0..<channels {
                        var value = component(base, channel + (leadingAlpha ? 1 : 0))
                        if premultiplied && opacity > 0 { value /= opacity }
                        guard value.isFinite else { throw NativeIOError.invalid("The image contains non-finite numeric pixels: \(url.lastPathComponent).") }
                        values[(y * width + x) * channels + channel] = value
                    }
                }
            }
        }
        return NativeImage(width: width, height: height, channels: channels, values: values)
    }

    private static func validateStoredPixels(_ image: NativeRawPixels, url: URL, role: String,
                                             cancellation: NativeCancellation?) throws {
        guard ["rgb", "depth", "roughness", "normal"].contains(role) else { throw NativeIOError.invalid("Unknown image role: \(role).") }
        if role == "normal", image.channels < 3 { throw NativeIOError.invalid("Normal targets need three color channels.") }
        if rawPixelCache.isValidated(image.key, role: role) { return }
        // Integer gray and normal codes are always finite and within their code
        // range. Float maps and scalar RGB agreement are checked once over the
        // entire unchanged source, preserving the full-image validation contract.
        if image.floating || (role != "rgb" && role != "normal" && image.channels >= 3) {
            for top in stride(from: 0, to: image.height, by: 16) {
                try cancellation?.check()
                let rows = min(16, image.height - top)
                let region = try numericRectangle(image, x: 0, y: top, width: image.width, height: rows,
                                                  url: url, cancellation: cancellation)
                if role != "rgb" { _ = try target(region, url: url, mapKind: role, normalConvention: "OpenGL", cancellation: cancellation) }
            }
        }
        rawPixelCache.markValidated(image.key, role: role)
    }

    private static func numericRectangle(_ image: NativeRawPixels, x: Int, y: Int, width: Int, height: Int,
                                         url: URL, cancellation: NativeCancellation?) throws -> NativeImage {
        // Retain the original row stride and bit layout while restricting the
        // storage view. This creates at most sixteen rows of temporary floats.
        let start = y * image.bytesPerRow + x * image.bitsPerPixel / 8
        let end = start + (height - 1) * image.bytesPerRow + width * image.bitsPerPixel / 8
        let slice = image.storage.subdata(in: start..<end)
        let rectangle = NativeRawPixels(key: image.key, width: width, height: height, channels: image.channels,
            bits: image.bits, components: image.components, bitsPerPixel: image.bitsPerPixel, bytesPerRow: image.bytesPerRow,
            bitmapInfo: image.bitmapInfo, floating: image.floating, leadingAlpha: image.leadingAlpha,
            premultiplied: image.premultiplied, hasAlpha: image.hasAlpha, storage: slice)
        return try numericPixels(rectangle, region: nil, url: url, cancellation: cancellation)
    }
}

struct NativeRawPixelKey: Hashable, Sendable {
    let path: String
    let device: Int32
    let inode: UInt64
    let size: Int64
    let seconds: Int
    let nanoseconds: Int
    init(_ url: URL) throws {
        var value = stat()
        guard stat(url.path, &value) == 0 else { throw NativeIOError.invalid("Source image is missing: \(url.lastPathComponent).") }
        path = url.standardizedFileURL.path; device = value.st_dev; inode = value.st_ino; size = value.st_size
        seconds = value.st_mtimespec.tv_sec; nanoseconds = value.st_mtimespec.tv_nsec
    }
}

struct NativeRawPixels: Sendable {
    let key: NativeRawPixelKey
    let width: Int, height: Int, channels: Int, bits: Int, components: Int, bitsPerPixel: Int, bytesPerRow: Int
    let bitmapInfo: UInt32
    let floating: Bool, leadingAlpha: Bool, premultiplied: Bool, hasAlpha: Bool
    let storage: Data
}

final class NativeRawPixelCache: @unchecked Sendable {
    private let lock = NSLock()
    private let limit: Int
    private var storage: [NativeRawPixelKey: NativeRawPixels] = [:]
    private var order: [NativeRawPixelKey] = []
    private var validation: [NativeRawPixelKey: Set<String>] = [:]
    private var validationOrder: [NativeRawPixelKey] = []
    private let validationLimit: Int
    private var bytes = 0
    init(limit: Int, validationLimit: Int = 4096) {
        self.limit = limit; self.validationLimit = max(1, validationLimit)
    }
    func clear() { lock.withLock {
        storage.removeAll(); order.removeAll(); validation.removeAll(); validationOrder.removeAll(); bytes = 0
    } }
    func read(_ key: NativeRawPixelKey) -> NativeRawPixels? {
        lock.withLock {
            guard let value = storage[key] else { return nil }
            order.removeAll { $0 == key }; order.append(key)
            return value
        }
    }
    func insert(_ value: NativeRawPixels) {
        lock.withLock {
            guard value.storage.count <= limit else { return }
            if let previous = storage.removeValue(forKey: value.key) { bytes -= previous.storage.count }
            order.removeAll { $0 == value.key }
            while bytes + value.storage.count > limit, let old = order.first {
                if let removed = storage.removeValue(forKey: old) { bytes -= removed.storage.count }
                order.removeFirst()
            }
            storage[value.key] = value; order.append(value.key); bytes += value.storage.count
        }
    }
    func isValidated(_ key: NativeRawPixelKey, role: String) -> Bool { lock.withLock { validation[key]?.contains(role) == true } }
    func markValidated(_ key: NativeRawPixelKey, role: String) {
        lock.withLock {
            // Pixel storage can exceed the cache or be evicted between epochs.
            // Its successful whole-image validation still applies to this exact
            // unchanged file identity. Keep these small records independently
            // bounded so large float/RGB targets are not rescanned each visit.
            if validation[key] == nil {
                while validationOrder.count >= validationLimit {
                    validation.removeValue(forKey: validationOrder.removeFirst())
                }
                validationOrder.append(key)
            }
            validation[key, default: []].insert(role)
        }
    }
}

/// Crop location and transforms have independent refresh periods. A fixed seed
/// reproduces a native run; SHA256 separates each sample/epoch/visit stream.
public struct NativeAugmentationPlan: Codable, Sendable {
    private let seed: Int
    private let cropRefresh: String
    private let augmentationRefresh: String
    private var visits = 0
    public init(seed: Int, cropRefresh: String, augmentationRefresh: String) {
        self.seed = seed; self.cropRefresh = cropRefresh; self.augmentationRefresh = augmentationRefresh
    }
    public mutating func transform(sampleID: String, epoch: Int, width: Int, height: Int, size: Int,
                                   flip: Bool, rotate: Bool, centerCrop: Bool = false) throws -> NativeCropTransform {
        guard ["per_run", "per_epoch", "per_sample"].contains(cropRefresh),
              ["per_run", "per_epoch", "per_sample"].contains(augmentationRefresh),
              size > 0, width >= size, height >= size else {
            throw NativeIOError.invalid("Source is smaller than the selected exact crop, or augmentation refresh is invalid.")
        }
        var crop = generator(stream: "crop", refresh: cropRefresh, sampleID: sampleID, epoch: epoch)
        var augmentation = generator(stream: "transform", refresh: augmentationRefresh, sampleID: sampleID, epoch: epoch)
        visits += 1
        let x = centerCrop ? (width - size) / 2 : crop.integer(width - size + 1)
        let y = centerCrop ? (height - size) / 2 : crop.integer(height - size + 1)
        let horizontal = flip && augmentation.integer(2) == 1
        let vertical = flip && augmentation.integer(2) == 1
        let turns = rotate ? augmentation.integer(4) : 0
        return NativeCropTransform(x: x, y: y, size: size, horizontalFlip: horizontal, verticalFlip: vertical, quarterTurns: turns)
    }
    private func generator(stream: String, refresh: String, sampleID: String, epoch: Int) -> NativeCropRandom {
        let period = refresh == "per_run" ? 0 : epoch
        let occurrence = refresh == "per_sample" ? visits : 0
        let digest = SHA256.hash(data: Data("\(seed)|\(stream)|\(sampleID)|\(period)|\(occurrence)".utf8))
        let value = digest.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        return NativeCropRandom(state: value)
    }
}

private struct NativeCropRandom {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        return value ^ (value >> 31)
    }
    mutating func integer(_ upperBound: Int) -> Int {
        // Rejection sampling avoids biased crop positions for arbitrary bounds.
        let bound = UInt64(upperBound), limit = UInt64.max - UInt64.max % bound
        var value = next()
        while value >= limit { value = next() }
        return Int(value % bound)
    }
}
