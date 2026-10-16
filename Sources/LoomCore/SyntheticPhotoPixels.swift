import Foundation
import LoomNative

/// CPU-only synthetic camera effects. These are documented approximations, not
/// a claim to reproduce a particular phone's lens, sensor or image processor.
enum SyntheticPhotoPixels {
    static func planar(_ rgb: [Float]) -> [Float] {
        let count = rgb.count/3
        var output = [Float](repeating: 0, count: rgb.count)
        for p in 0..<count { for c in 0..<3 { output[c*count+p] = rgb[p*3+c] } }
        return output
    }
    static func linear(_ value: Float) -> Float {
        value <= 0.04045 ? value/12.92 : pow((value+0.055)/1.055, 2.4)
    }
    static func srgb(_ value: Float) -> Float {
        value <= 0.0031308 ? value*12.92 : 1.055*pow(value, 1/2.4)-0.055
    }

    struct Restored {
        let values: [Float]
    }
    static func restore(_ rgb: [Float], size: Int, recipe: SyntheticPhotoRecipe,
                        random: inout SyntheticRandom, cancellation: NativeCancellation) throws -> Restored {
        let enabled = [recipe.exposure, recipe.blur, recipe.noise, recipe.lighting]
        var active = enabled.map { $0 && random.unit() < 0.65 }
        if !active.contains(true), let first = enabled.indices.first(where: { enabled[$0] }) { active[first] = true }
        let ev = active[0] ? random.range(-0.6, 1.2)*recipe.strength : 0
        let sigma = active[1] ? random.range(0.35, 2)*recipe.strength : 0
        let noise = active[2] ? random.range(0.001, 0.012)*recipe.strength : 0
        let gradient = active[3] ? random.range(0.2, 0.6)*recipe.strength : 0
        let direction = random.range(-Double.pi, Double.pi)
        let gain = Float(pow(2, ev))
        var values = [Float](repeating: 0, count: rgb.count)
        for y in 0..<size {
            if y % 16 == 0 { try cancellation.check() }
            for x in 0..<size {
                let px = Double(x)/Double(max(1, size-1))*2-1, py = Double(y)/Double(max(1, size-1))*2-1
                let illumination = Float(1 + gradient*(0.5*(px*cos(direction)+py*sin(direction)) +
                    0.4*exp(-((px-0.4)*(px-0.4)+(py+0.2)*(py+0.2))/0.2)))
                for c in 0..<3 { let p = (y*size+x)*3+c; values[p] = linear(rgb[p])*gain*illumination }
            }
        }
        if sigma > 0 { values = try gaussian(values, size: size, sigma: sigma, cancellation: cancellation) }
        for p in 0..<(size*size) {
            if p % 4096 == 0 { try cancellation.check() }
            for c in 0..<3 {
                let i = p*3+c
                // Irwin-Hall approximation to zero-mean unit Gaussian; seed recorded.
                var gaussianNoise = 0.0
                if noise > 0 { for _ in 0..<12 { gaussianNoise += random.unit() }; gaussianNoise -= 6 }
                let value = values[i] + Float(gaussianNoise*noise)
                values[i] = min(1, max(0, srgb(min(1, max(0, value)))))
            }
        }
        return .init(values: values)
    }

    static func gaussian(_ rgb: [Float], size: Int, sigma: Double, cancellation: NativeCancellation) throws -> [Float] {
        guard sigma > 0 else { return rgb }
        let radius = Int(ceil(3*sigma))
        let raw = (-radius...radius).map { exp(-Double($0*$0)/(2*sigma*sigma)) }
        let sum = raw.reduce(0, +), kernel = raw.map { Float($0/sum) }
        var intermediate = [Float](repeating: 0, count: rgb.count), result = intermediate
        for y in 0..<size {
            if y % 16 == 0 { try cancellation.check() }
            for x in 0..<size { for c in 0..<3 {
                var value: Float = 0
                for k in -radius...radius { value += rgb[(y*size+min(size-1, max(0, x+k)))*3+c]*kernel[k+radius] }
                intermediate[(y*size+x)*3+c] = value
            } }
        }
        for y in 0..<size {
            if y % 16 == 0 { try cancellation.check() }
            for x in 0..<size { for c in 0..<3 {
                var value: Float = 0
                for k in -radius...radius { value += intermediate[(min(size-1, max(0, y+k))*size+x)*3+c]*kernel[k+radius] }
                result[(y*size+x)*3+c] = value
            } }
        }
        return result
    }

    struct Perspective {
        let values: [Float]
        let inputMask: [Float]
        let targetMask: [Float]
        let homography: [Double]
    }
    static func homography(size: Int, pitch: Double, yaw: Double) -> [Double] {
        let p = pitch*Double.pi/180, y = yaw*Double.pi/180, n = Double(size), center = (n-1)/2
        // Project a plane rotated in 3D through a pinhole camera. H maps clean
        // source pixel centers to synthetic input pixel centers, top-left origin.
        let r = [[cos(y), sin(y)*sin(p)], [0, cos(p)], [-sin(y), cos(y)*sin(p)]]
        let zoom = pitch == 0 && yaw == 0 ? 1.0 : 0.94
        let focal = 2*n*zoom
        let a = r.map { [$0[0]/n, $0[1]/n, -center*($0[0]+$0[1])/n] }
        let z = [a[2][0], a[2][1], a[2][2]+2]
        var result: [Double] = []
        for column in 0..<3 { result.append(focal*a[0][column]+center*z[column]) }
        for column in 0..<3 { result.append(focal*a[1][column]+center*z[column]) }
        result += z
        return result.map { $0/result[8] }
    }

    static func project(_ h: [Double], x: Double, y: Double) -> (Double, Double) {
        let z = h[6]*x+h[7]*y+h[8]
        return ((h[0]*x+h[1]*y+h[2])/z, (h[3]*x+h[4]*y+h[5])/z)
    }
    static func inverse(_ m: [Double]) -> [Double] {
        let a=m[0], b=m[1], c=m[2], d=m[3], e=m[4], f=m[5], g=m[6], h=m[7], i=m[8]
        let determinant = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
        return [e*i-f*h,c*h-b*i,b*f-c*e,f*g-d*i,a*i-c*g,c*d-a*f,d*h-e*g,b*g-a*h,a*e-b*d].map { $0/determinant }
    }
    static func perspective(_ rgb: [Float], size: Int, pitch: Double, yaw: Double,
                            cancellation: NativeCancellation) throws -> Perspective {
        let h = homography(size: size, pitch: pitch, yaw: yaw), inverted = inverse(h)
        var output = [Float](repeating: 0.18, count: rgb.count)
        var inputMask = [Float](repeating: 0, count: size*size), targetMask = inputMask
        func inside(_ x: Double, _ y: Double) -> Bool {
            x.isFinite && y.isFinite && x >= -1e-8 && y >= -1e-8 && x <= Double(size-1)+1e-8 && y <= Double(size-1)+1e-8
        }
        for y in 0..<size {
            if y % 16 == 0 { try cancellation.check() }
            for x in 0..<size {
                let (u,v) = project(inverted, x: Double(x), y: Double(y))
                if inside(u,v) {
                    let sx = min(Double(size-1), max(0,u)), sy = min(Double(size-1), max(0,v))
                    let x0=Int(floor(sx)), y0=Int(floor(sy)), x1=min(size-1,x0+1), y1=min(size-1,y0+1)
                    let dx=Float(sx-Double(x0)), dy=Float(sy-Double(y0))
                    for c in 0..<3 {
                        let top = rgb[(y0*size+x0)*3+c]*(1-dx) + rgb[(y0*size+x1)*3+c]*dx
                        let bottom = rgb[(y1*size+x0)*3+c]*(1-dx) + rgb[(y1*size+x1)*3+c]*dx
                        output[(y*size+x)*3+c] = top*(1-dy)+bottom*dy
                    }
                    inputMask[y*size+x] = 1
                }
                let (tx,ty) = project(h, x: Double(x), y: Double(y))
                if inside(tx,ty) { targetMask[y*size+x] = 1 }
            }
        }
        return .init(values: output, inputMask: inputMask, targetMask: targetMask, homography: h)
    }
}
