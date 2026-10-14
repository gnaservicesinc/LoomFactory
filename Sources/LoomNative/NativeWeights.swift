import Foundation

/// Tensor storage shared by native training, Safetensors and Core ML export.
/// Convolution weights retain PyTorch's OIHW ordering; image tensors use NCHW.
public struct NativeTensor: Sendable, Equatable {
    public let shape: [Int]
    public var values: [Float]

    public init(shape: [Int], values: [Float]) {
        self.shape = shape
        self.values = values
    }
}

public enum NativeIOError: Error, LocalizedError {
    case invalid(String)
    case unsupported(String)
    public var errorDescription: String? {
        switch self {
        case .invalid(let message), .unsupported(let message): return message
        }
    }
}

/// Implements the public Safetensors format, without a Python process or pickle.
public enum NativeWeights {
    public static func write(_ tensors: [String: NativeTensor], to url: URL,
                             metadata: [String: String] = [:]) throws {
        var payload = Data()
        var header: [String: Any] = [:]
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        for name in tensors.keys.sorted() {
            guard !name.isEmpty, name != "__metadata__", let tensor = tensors[name] else {
                throw NativeIOError.invalid("A Safetensors weight has an invalid name.")
            }
            try validate(tensor, name: name)
            let start = payload.count
            tensor.values.withUnsafeBytes { bytes in payload.append(contentsOf: bytes) }
            header[name] = ["dtype": "F32", "shape": tensor.shape,
                            "data_offsets": [start, payload.count]]
        }
        var encoded = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while encoded.count % 8 != 0 { encoded.append(0x20) }
        var length = UInt64(encoded.count).littleEndian
        var result = Data(bytes: &length, count: 8)
        result.append(encoded)
        result.append(payload)
        try result.write(to: url, options: .atomic)
    }

    public static func read(from url: URL) throws -> [String: NativeTensor] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 10 else { throw NativeIOError.invalid("The Safetensors file is truncated.") }
        let length = data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        guard length <= 100_000_000, length <= data.count - 8 else {
            throw NativeIOError.invalid("The Safetensors header length is invalid.")
        }
        let headerEnd = 8 + Int(length)
        guard let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<headerEnd)) as? [String: Any] else {
            throw NativeIOError.invalid("The Safetensors header must be an object.")
        }
        var result: [String: NativeTensor] = [:]
        var intervals: [(Int, Int)] = []
        for (name, raw) in header where name != "__metadata__" {
            guard let item = raw as? [String: Any], let type = item["dtype"] as? String,
                  let shape = item["shape"] as? [Int], let offsets = item["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                throw NativeIOError.invalid("Invalid Safetensors descriptor for \(name).")
            }
            let stride: Int
            switch type {
            case "F32": stride = 4
            case "F16", "BF16": stride = 2
            default: throw NativeIOError.unsupported("Safetensors weight \(name) uses unsupported dtype \(type).")
            }
            let count = try elementCount(shape, name: name)
            let (byteCount, overflow) = count.multipliedReportingOverflow(by: stride)
            guard !overflow, offsets[0] >= 0, offsets[1] >= offsets[0],
                  offsets[1] <= data.count - headerEnd, offsets[1] - offsets[0] == byteCount else {
                throw NativeIOError.invalid("Safetensors data bounds do not match the shape of \(name).")
            }
            intervals.append((offsets[0], offsets[1]))
            let base = headerEnd + offsets[0]
            let values: [Float] = data.withUnsafeBytes { bytes in
                (0..<count).map { index in
                    if type == "F32" {
                        return Float(bitPattern: bytes.loadUnaligned(fromByteOffset: base + index * 4, as: UInt32.self).littleEndian)
                    }
                    let bits = bytes.loadUnaligned(fromByteOffset: base + index * 2, as: UInt16.self).littleEndian
                    return type == "F16" ? Float(Float16(bitPattern: bits)) : Float(bitPattern: UInt32(bits) << 16)
                }
            }
            let tensor = NativeTensor(shape: shape, values: values)
            try validate(tensor, name: name)
            result[name] = tensor
        }
        var end = 0
        for interval in intervals.sorted(by: { $0.0 < $1.0 }) {
            guard interval.0 == end else { throw NativeIOError.invalid("Safetensors payload has overlapping weights or holes.") }
            end = interval.1
        }
        guard end == data.count - headerEnd else { throw NativeIOError.invalid("Safetensors payload has trailing data.") }
        return result
    }

    static func validate(_ tensor: NativeTensor, name: String) throws {
        guard try elementCount(tensor.shape, name: name) == tensor.values.count else {
            throw NativeIOError.invalid("Weight \(name) has a shape inconsistent with its values.")
        }
        guard tensor.values.allSatisfy(\.isFinite) else {
            throw NativeIOError.invalid("Weight \(name) contains non-finite values.")
        }
    }

    private static func elementCount(_ shape: [Int], name: String) throws -> Int {
        var count = 1
        for dimension in shape {
            let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard dimension >= 0, !overflow else {
                throw NativeIOError.invalid("Weight \(name) has an invalid or overflowing shape.")
            }
            count = next
        }
        return count
    }
}
