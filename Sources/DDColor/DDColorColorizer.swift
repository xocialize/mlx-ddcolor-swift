import Foundation
import CoreGraphics
import MLX

/// End-to-end DDColor colorizer: CGImage → colorized CGImage, mirroring the PyTorch
/// `ColorizationPipeline`. Holds the loaded `DDColorModel`; the engine wrapper owns its lifecycle.
///
///   orig-L  ← bgrToLab(source @ native res)
///   ab      ← model( grayRGB( bgrToLab(source @ inputSize).L ) )         [model @ inputSize]
///   out     ← labToBGR( concat[ orig-L, nearest-resize(ab → native) ] )
public final class DDColorColorizer: @unchecked Sendable {

    public let config: DDColorConfig
    private let model: DDColorModel

    public init(config: DDColorConfig, weights: [String: MLXArray]) {
        self.config = config
        self.model = DDColorModel(config: config, weights: weights)
    }

    /// Load converted MLX safetensors from disk and build the colorizer.
    public static func fromPretrained(_ weightsPath: String, config: DDColorConfig,
                                      dtype: DType = .float32) throws -> DDColorColorizer {
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath)).mapValues { $0.asType(dtype) }
        return DDColorColorizer(config: config, weights: w)
    }

    /// Colorize. The output preserves the source resolution; color is taken from the model, luminance
    /// from the source. Accepts colour input too (only its L channel is used).
    public func callAsFunction(_ source: CGImage) -> CGImage {
        let S = config.inputSize
        let H = source.height, W = source.width
        let srcNative = DDColorImage.bgr(from: source, width: W, height: H)   // (H,W,3) BGR
        let origL = Lab.bgrToLab(srcNative)[0..., 0..., 0].expandedDimensions(axis: -1)  // (H,W,1)

        let src512 = DDColorImage.bgr(from: source, width: S, height: S)
        let grayRGB = Lab.grayRGBInput(fromBGR: src512)                       // (1,S,S,3)
        let ab = model(grayRGB)                                              // (1,S,S,2)
        let abNative = Self.nearestResize(ab, toH: H, toW: W)                // (H,W,2)

        let outBGR = Lab.recombine(originalL: origL, modelAB: abNative.expandedDimensions(axis: 0))
        outBGR.eval()
        return DDColorImage.cgImage(fromBGR: outBGR)
    }

    /// F.interpolate(mode="nearest") equivalent: src = floor(dst · in/out). `ab` is `(1,S,S,2)`.
    private static func nearestResize(_ ab: MLXArray, toH H: Int, toW W: Int) -> MLXArray {
        let S = ab.dim(1)
        let a = ab[0]                                                        // (S,S,2)
        if H == S && W == S { return a }
        let rows = MLXArray((0 ..< H).map { Int32(Float($0) * Float(S) / Float(H)) })
        let cols = MLXArray((0 ..< W).map { Int32(Float($0) * Float(S) / Float(W)) })
        return MLX.take(MLX.take(a, rows, axis: 0), cols, axis: 1)           // (H,W,2)
    }
}
