import Foundation
import CoreGraphics
import MLX

/// CGImage ⇄ MLXArray (BGR `[0,1]`, NHWC-friendly `(H,W,3)`) bridging for the colorizer. Resize is
/// done by drawing into a CG bitmap context (high-quality interpolation) — the model runs at a fixed
/// `inputSize` while orig-L is sampled at source resolution.
enum DDColorImage {

    /// Draw `cg` into a `width×height` sRGB RGBA8 context and return BGR `(H,W,3)` floats in `[0,1]`.
    static func bgr(from cg: CGImage, width: Int, height: Int) -> MLXArray {
        let cs = CGColorSpaceCreateDeviceRGB()
        var buf = [UInt8](repeating: 0, count: width * height * 4)
        buf.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: width, height: height,
                                bitsPerComponent: 8, bytesPerRow: width * 4, space: cs,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        var bgr = [Float](repeating: 0, count: width * height * 3)
        for p in 0 ..< width * height {
            bgr[p * 3 + 0] = Float(buf[p * 4 + 2]) / 255   // B
            bgr[p * 3 + 1] = Float(buf[p * 4 + 1]) / 255   // G
            bgr[p * 3 + 2] = Float(buf[p * 4 + 0]) / 255   // R
        }
        return MLXArray(bgr, [height, width, 3])
    }

    /// BGR `(H,W,3)` floats `[0,1]` → opaque sRGB CGImage.
    static func cgImage(fromBGR arr: MLXArray) -> CGImage {
        let H = arr.dim(0), W = arr.dim(1)
        let bgr = MLX.clip(arr, min: 0, max: 1).asArray(Float.self)
        var buf = [UInt8](repeating: 255, count: W * H * 4)
        for p in 0 ..< W * H {
            buf[p * 4 + 0] = UInt8((bgr[p * 3 + 2] * 255).rounded())   // R
            buf[p * 4 + 1] = UInt8((bgr[p * 3 + 1] * 255).rounded())   // G
            buf[p * 4 + 2] = UInt8((bgr[p * 3 + 0] * 255).rounded())   // B
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = buf.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8,
                      bytesPerRow: W * 4, space: cs,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        }
        return ctx.makeImage()!
    }
}
