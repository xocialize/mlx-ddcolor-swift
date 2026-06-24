import Foundation
import MLX

/// Low-level NHWC ops shared by the DDColor forward pass. Each is a 1:1 transcription of the
/// parity-verified `oracle/mlx_ref.py` reference (CPU fp32 max_abs ≤ 5e-5 vs the PyTorch oracle).
enum Ops {

    /// Last-axis LayerNorm (NHWC channels == last axis). ConvNeXt block norm, norm0..3, and the
    /// transformer norms are all this. `eps` = 1e-6 (ConvNeXt) or 1e-5 (transformer).
    static func layerNorm(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, eps: Float) -> MLXArray {
        let u = x.mean(axis: -1, keepDims: true)
        let d = x - u
        let v = (d * d).mean(axis: -1, keepDims: true)
        return d / MLX.sqrt(v + eps) * w + b
    }

    /// Eval-mode BatchNorm over the last (channel) axis, eps 1e-5 (torch default).
    static func batchNorm(_ x: MLXArray, mean: MLXArray, varr: MLXArray,
                          weight: MLXArray, bias: MLXArray, eps: Float = 1e-5) -> MLXArray {
        (x - mean) / MLX.sqrt(varr + eps) * weight + bias
    }

    /// Exact erf-GELU (ConvNeXt uses `nn.GELU()`).
    static func gelu(_ x: MLXArray) -> MLXArray {
        0.5 * x * (1 + MLX.erf(x / 1.4142135623730951))
    }

    static func relu(_ x: MLXArray) -> MLXArray { MLX.maximum(x, 0) }

    /// NHWC PixelShuffle(r): (B,H,W,C·r²) → (B,H·r,W·r,C). torch decomposes the input channel as
    /// `c_out·r² + rh·r + rw` (c_out OUTERMOST) — split as (c_out, rh, rw), NOT (rh, rw, c_out).
    static func pixelShuffle(_ x: MLXArray, _ r: Int) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let c2 = c / (r * r)
        return x.reshaped([b, h, w, c2, r, r])
            .transposed(0, 1, 4, 2, 5, 3)        // (b,h,rh,w,rw,c_out)
            .reshaped([b, h * r, w * r, c2])
    }

    /// fastai-style anti-checkerboard blur: ReplicationPad2d((1,0,1,0)) + AvgPool2d(2, stride=1).
    /// In NHWC: pad H/W front by 1 (edge), then depthwise 2×2 box filter, stride 1.
    static func blur(_ x: MLXArray) -> MLXArray {
        let c = x.dim(-1)
        let padded = MLX.padded(x, widths: [.init(0), .init((1, 0)), .init((1, 0)), .init(0)], mode: .edge)
        let kernel = MLX.ones([c, 2, 2, 1]) / 4.0   // depthwise box filter
        return MLX.conv2d(padded, kernel, stride: .init(1), padding: .init(0), groups: c)
    }

    /// Multi-head attention with a packed `in_proj_weight` (3C,C) split into contiguous q;k;v thirds
    /// (standard `nn.MultiheadAttention`, NOT interleaved). Single batch, sequences (L, C). No mask.
    static func mha(q: MLXArray, k: MLXArray, v: MLXArray,
                    inW: MLXArray, inB: MLXArray, outW: MLXArray, outB: MLXArray,
                    heads: Int) -> MLXArray {
        let c = q.dim(-1)
        let hd = c / heads
        let scale = 1.0 / Float(hd).squareRoot()
        let wq = inW[0 ..< c], wk = inW[c ..< 2 * c], wv = inW[2 * c ..< 3 * c]
        let bq = inB[0 ..< c], bk = inB[c ..< 2 * c], bv = inB[2 * c ..< 3 * c]
        var Q = MLX.matmul(q, wq.transposed()) + bq    // (Lq, C)
        var K = MLX.matmul(k, wk.transposed()) + bk
        var V = MLX.matmul(v, wv.transposed()) + bv
        let lq = Q.dim(0), lk = K.dim(0)
        Q = Q.reshaped([lq, heads, hd]).transposed(1, 0, 2)   // (H, Lq, hd)
        K = K.reshaped([lk, heads, hd]).transposed(1, 0, 2)
        V = V.reshaped([lk, heads, hd]).transposed(1, 0, 2)
        var attn = MLX.matmul(Q, K.transposed(0, 2, 1)) * scale
        attn = MLX.softmax(attn, axis: -1)
        let out = MLX.matmul(attn, V).transposed(1, 0, 2).reshaped([lq, c])  // (Lq, C)
        return MLX.matmul(out, outW.transposed()) + outB
    }

    /// DETR `PositionEmbeddingSine(num_pos_feats=128, normalize=True)` → flattened (H·W, 2·feats).
    /// Computed on the host (small feature maps) for robustness, wrapped as an MLXArray.
    static func posEmbedSine(h: Int, w: Int, numPosFeats: Int = 128, temperature: Float = 10000) -> MLXArray {
        let scale = 2 * Float.pi
        let eps: Float = 1e-6
        var dimT = [Float](repeating: 0, count: numPosFeats)
        for i in 0 ..< numPosFeats {
            dimT[i] = powf(temperature, 2 * Float(i / 2) / Float(numPosFeats))
        }
        let c = 2 * numPosFeats
        var out = [Float](repeating: 0, count: h * w * c)
        for yi in 0 ..< h {
            let yEmbed = Float(yi + 1) / (Float(h) + eps) * scale
            for xi in 0 ..< w {
                let xEmbed = Float(xi + 1) / (Float(w) + eps) * scale
                let base = (yi * w + xi) * c
                // pos_y occupies [0, numPosFeats), pos_x occupies [numPosFeats, 2*numPosFeats)
                for j in 0 ..< numPosFeats {
                    let py = yEmbed / dimT[j]
                    let px = xEmbed / dimT[j]
                    // torch: stack(sin(even), cos(odd)).flatten → even idx = sin, odd idx = cos
                    out[base + j] = (j % 2 == 0) ? sinf(py) : cosf(py)
                    out[base + numPosFeats + j] = (j % 2 == 0) ? sinf(px) : cosf(px)
                }
            }
        }
        return MLXArray(out, [h * w, c])
    }
}
