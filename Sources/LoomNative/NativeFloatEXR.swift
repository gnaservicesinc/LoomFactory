import Darwin
import Foundation

/// A flat, uncompressed OpenEXR scanline writer for exact Float32 model output.
/// The input is planar NCHW: one Y plane or R, G, B planes. Pixel values are
/// stored by their IEEE 754 bits, without a color transform, clamp or scaling.
/// Layout reference: https://openexr.com/en/latest/OpenEXRFileLayout.html
public enum NativeFloatEXR {
    public static func write(_ values: [Float], channels: Int, width: Int, height: Int, to url: URL,
                             cancellation: NativeCancellation = NativeCancellation()) throws {
        guard [1, 3].contains(channels), width > 0, height > 0, height <= Int(Int32.max),
              width <= Int(Int32.max) / (channels * 4), width <= Int.max / height / channels,
              values.count == width * height * channels else {
            throw NativeIOError.invalid("The Float32 EXR output has invalid dimensions or channel storage.")
        }
        try cancellation.check()
        // Both the header and scanline payload must use alphabetical channel
        // order. RGB input planes therefore write in B, G, R file order.
        let channelPlanes: [(name: String, plane: Int)] = channels == 1 ? [("Y", 0)] : [("B", 2), ("G", 1), ("R", 0)]
        var header = Data()
        append(UInt32(20_000_630), to: &header)
        append(UInt32(2), to: &header)
        var channelList = Data()
        for channel in channelPlanes {
            string(channel.name, to: &channelList)
            append(Int32(2), to: &channelList) // FLOAT, never HALF or UINT.
            channelList.append(contentsOf: [0, 0, 0, 0]) // pLinear and reserved.
            append(Int32(1), to: &channelList); append(Int32(1), to: &channelList)
        }
        channelList.append(0)
        attribute("channels", type: "chlist", value: channelList, to: &header)
        attribute("compression", type: "compression", value: Data([0]), to: &header)
        var window = Data()
        for coordinate in [Int32(0), 0, Int32(width - 1), Int32(height - 1)] { append(coordinate, to: &window) }
        attribute("dataWindow", type: "box2i", value: window, to: &header)
        attribute("displayWindow", type: "box2i", value: window, to: &header)
        attribute("lineOrder", type: "lineOrder", value: Data([0]), to: &header)
        attribute("pixelAspectRatio", type: "float", value: floatData(1), to: &header)
        var center = floatData(0); center.append(floatData(0))
        attribute("screenWindowCenter", type: "v2f", value: center, to: &header)
        attribute("screenWindowWidth", type: "float", value: floatData(1), to: &header)
        header.append(0)

        let rowBytes = width * channels * 4, planePixels = width * height
        var offsets = Data(capacity: height * 8)
        let firstChunk = UInt64(header.count) + UInt64(height) * 8
        for row in 0..<height {
            if row % 256 == 0 { try cancellation.check() }
            append(firstChunk + UInt64(row) * UInt64(rowBytes + 8), to: &offsets)
        }
        let temporary = url.deletingLastPathComponent().appendingPathComponent(".pending-exr-" + UUID().uuidString)
        guard FileManager.default.createFile(atPath: temporary.path, contents: nil) else {
            throw NativeIOError.invalid("Could not create the Float32 EXR output.")
        }
        defer { try? FileManager.default.removeItem(at: temporary) }
        let handle = try FileHandle(forWritingTo: temporary)
        do {
            try handle.write(contentsOf: header)
            try handle.write(contentsOf: offsets)
            // Only one scanline of extra storage is needed for large textures.
            var row = Data(count: rowBytes + 8)
            for y in 0..<height {
                try cancellation.check()
                row.withUnsafeMutableBytes { bytes in
                    bytes.storeBytes(of: Int32(y).littleEndian, toByteOffset: 0, as: Int32.self)
                    bytes.storeBytes(of: Int32(rowBytes).littleEndian, toByteOffset: 4, as: Int32.self)
                    for (fileChannel, channel) in channelPlanes.enumerated() {
                        for x in 0..<width {
                            let value = values[channel.plane * planePixels + y * width + x]
                            bytes.storeBytes(of: value.bitPattern.littleEndian,
                                toByteOffset: 8 + (fileChannel * width + x) * 4, as: UInt32.self)
                        }
                    }
                }
                try handle.write(contentsOf: row)
            }
            try handle.close()
        } catch {
            try? handle.close()
            throw error
        }
        try cancellation.check()
        guard rename(temporary.path, url.path) == 0 else {
            throw NativeIOError.invalid("Could not publish the Float32 EXR output: \(String(cString: strerror(errno))).")
        }
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
    private static func string(_ value: String, to data: inout Data) { data.append(contentsOf: value.utf8); data.append(0) }
    private static func floatData(_ value: Float) -> Data { var data = Data(); append(value.bitPattern, to: &data); return data }
    private static func attribute(_ name: String, type: String, value: Data, to data: inout Data) {
        string(name, to: &data); string(type, to: &data); append(Int32(value.count), to: &data); data.append(value)
    }
}
