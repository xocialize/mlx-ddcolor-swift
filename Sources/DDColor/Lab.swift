import Foundation
import MLX

/// cv2-exact BGR↔Lab colorspace math + the DDColor pre/post pipeline, vectorized in MLX.
///
/// Mirrors OpenCV's float `COLOR_BGR2Lab` / `COLOR_Lab2BGR` (sRGB gamma, D65 white point, the
/// `f(t)` cube-root with the 0.008856 / 16-116 linear toe) — the documented colorization parity
/// trap (a wrong gamma or white point yields sepia / washed-out output). Pre/post are kept here so
/// the model stays a pure `gray-RGB → ab` transform and every tier yields an identical result.
///
/// Operates on `(H, W, 3)` MLXArrays in `[0,1]`. Resize is handled by the caller (the model runs at
/// `config.inputSize`); when source == inputSize (the parity fixture) pre/post are resize-free.
public enum Lab {

    // OpenCV sRGB / D65 constants.
    private static let Xn: Float = 0.950456
    private static let Zn: Float = 1.088754
    private static let labThresh: Float = 0.008856          // (6/29)^3
    private static let labThreshInv: Float = 0.206897       // 6/29
    private static let labShift: Float = 16.0 / 116.0

    private static func gammaExpand(_ c: MLXArray) -> MLXArray {  // sRGB → linear
        MLX.where(c .> 0.04045, MLX.pow((c + 0.055) / 1.055, 2.4), c / 12.92)
    }
    private static func gammaCompress(_ c0: MLXArray) -> MLXArray {  // linear → sRGB
        let c = MLX.maximum(c0, 0)   // guard pow of tiny-negative linear RGB
        let s = MLX.where(c .> 0.0031308, 1.055 * MLX.pow(c, 1.0 / 2.4) - 0.055, 12.92 * c)
        return MLX.clip(s, min: 0, max: 1)
    }
    private static func fLab(_ t: MLXArray) -> MLXArray {
        MLX.where(t .> labThresh, MLX.pow(t, 1.0 / 3.0), 7.787 * t + labShift)
    }
    private static func fLabInv(_ t: MLXArray) -> MLXArray {
        MLX.where(t .> labThreshInv, MLX.pow(t, 3.0), (t - labShift) / 7.787)
    }

    /// BGR `(H,W,3)` [0,1] → Lab `(H,W,3)` (L∈[0,100], a/b unscaled, matching cv2 float).
    public static func bgrToLab(_ bgr: MLXArray) -> MLXArray {
        let B = gammaExpand(bgr[0..., 0..., 0])
        let G = gammaExpand(bgr[0..., 0..., 1])
        let R = gammaExpand(bgr[0..., 0..., 2])
        let X = (0.412453 * R + 0.357580 * G + 0.180423 * B) / Xn
        let Y = 0.212671 * R + 0.715160 * G + 0.072169 * B
        let Z = (0.019334 * R + 0.119193 * G + 0.950227 * B) / Zn
        let fx = fLab(X), fy = fLab(Y), fz = fLab(Z)
        let L = 116 * fy - 16
        let a = 500 * (fx - fy)
        let b = 200 * (fy - fz)
        return MLX.stacked([L, a, b], axis: -1)
    }

    /// Lab `(H,W,3)` → linear-then-sRGB R,G,B planes (each `(H,W)`).
    private static func labToLinearRGB(_ lab: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let L = lab[0..., 0..., 0], a = lab[0..., 0..., 1], b = lab[0..., 0..., 2]
        let fy = (L + 16) / 116
        let fx = fy + a / 500
        let fz = fy - b / 200
        let X = fLabInv(fx) * Xn
        let Y = fLabInv(fy)
        let Z = fLabInv(fz) * Zn
        let R = gammaCompress(3.240479 * X - 1.537150 * Y - 0.498535 * Z)
        let G = gammaCompress(-0.969256 * X + 1.875991 * Y + 0.041556 * Z)
        let B = gammaCompress(0.055648 * X - 0.204043 * Y + 1.057311 * Z)
        return (R, G, B)
    }

    /// Lab `(H,W,3)` → RGB `(H,W,3)` [0,1] (the model's gray-input order).
    public static func labToRGB(_ lab: MLXArray) -> MLXArray {
        let (R, G, B) = labToLinearRGB(lab)
        return MLX.stacked([R, G, B], axis: -1)
    }

    /// Lab `(H,W,3)` → BGR `(H,W,3)` [0,1] (final output order).
    public static func labToBGR(_ lab: MLXArray) -> MLXArray {
        let (R, G, B) = labToLinearRGB(lab)
        return MLX.stacked([B, G, R], axis: -1)
    }

    /// Gray-RGB model input `(1,S,S,3)` built from the source BGR `(S,S,3)` L channel.
    public static func grayRGBInput(fromBGR bgr: MLXArray) -> MLXArray {
        let L = bgrToLab(bgr)[0..., 0..., 0]                 // (S,S)
        let z = MLX.zeros(like: L)
        let grayLab = MLX.stacked([L, z, z], axis: -1)       // (S,S,3) Lab with a=b=0
        return labToRGB(grayLab).expandedDimensions(axis: 0) // (1,S,S,3)
    }

    /// Recombine the original L with the model's ab `(1,S,S,2)` → final BGR `(S,S,3)` [0,1].
    public static func recombine(originalL L: MLXArray, modelAB ab: MLXArray) -> MLXArray {
        let abHW = ab[0]                                     // (S,S,2)
        let lab = MLX.concatenated([L, abHW], axis: -1)      // (S,S,3)
        return labToBGR(lab)
    }
}
