import Foundation
import MLX

/// DDColor forward pass over a flat weights dictionary loaded from the converted safetensors
/// (canonical keys = torch-flat names; conv weights already NHWC + spectral-resolved by
/// `oracle/convert.py`). Functional style mirrors the parity-verified `oracle/mlx_ref.py` 1:1 —
/// the idiomatic `Module`-tree refactor is deferred until Swift parity is locked (skill doctrine:
/// match the reference first, refactor after parity).
///
/// Input: ImageNet-normalized gray-RGB, NHWC `(1, S, S, 3)`. Output: raw ab, NHWC `(1, S, S, 2)`.
public final class DDColorModel: @unchecked Sendable {

    public let config: DDColorConfig
    private let w: [String: MLXArray]

    // ImageNet statistics — the model normalizes its gray-RGB input internally (matches the
    // PyTorch reference `DDColor.normalize`); NHWC broadcast shape (1,1,1,3).
    private let mean = MLXArray([0.485, 0.456, 0.406] as [Float], [1, 1, 1, 3])
    private let std = MLXArray([0.229, 0.224, 0.225] as [Float], [1, 1, 1, 3])

    public init(config: DDColorConfig, weights: [String: MLXArray]) {
        self.config = config
        self.w = weights
    }

    // MARK: weight accessors
    private func arr(_ k: String) -> MLXArray { w[k]! }                       // plain (LN/BN/Linear/embed)
    private func cw(_ k: String) -> MLXArray { w[k]! }                        // conv weight (already NHWC)
    private func eff(_ base: String) -> MLXArray { w[base + ".weight"]! }     // spectral-resolved conv

    private func conv(_ x: MLXArray, _ wt: MLXArray, bias: MLXArray? = nil,
                      stride: Int = 1, pad: Int = 0, groups: Int = 1) -> MLXArray {
        let y = MLX.conv2d(x, wt, stride: .init(stride), padding: .init(pad), groups: groups)
        return bias == nil ? y : y + bias!
    }

    private func bn(_ x: MLXArray, _ base: String) -> MLXArray {
        Ops.batchNorm(x, mean: arr(base + ".running_mean"), varr: arr(base + ".running_var"),
                      weight: arr(base + ".weight"), bias: arr(base + ".bias"))
    }

    // MARK: ConvNeXt encoder → [norm0..3] NHWC feature maps
    private func encoder(_ input: MLXArray) -> [MLXArray] {
        var x = input
        var feats: [MLXArray] = []
        let depths = config.tier.depths, dims = config.tier.dims
        for i in 0 ..< 4 {
            let ds = "encoder.arch.downsample_layers.\(i)"
            if i == 0 {  // stem: Conv(k4,s4) then LayerNorm
                x = conv(x, cw(ds + ".0.weight"), bias: arr(ds + ".0.bias"), stride: 4)
                x = Ops.layerNorm(x, arr(ds + ".1.weight"), arr(ds + ".1.bias"), eps: 1e-6)
            } else {     // LayerNorm then Conv(k2,s2)
                x = Ops.layerNorm(x, arr(ds + ".0.weight"), arr(ds + ".0.bias"), eps: 1e-6)
                x = conv(x, cw(ds + ".1.weight"), bias: arr(ds + ".1.bias"), stride: 2)
            }
            for j in 0 ..< depths[i] {
                let p = "encoder.arch.stages.\(i).\(j)"
                let inp = x
                var h = conv(x, cw(p + ".dwconv.weight"), bias: arr(p + ".dwconv.bias"),
                             pad: 3, groups: dims[i])                       // depthwise 7×7
                h = Ops.layerNorm(h, arr(p + ".norm.weight"), arr(p + ".norm.bias"), eps: 1e-6)
                h = MLX.matmul(h, arr(p + ".pwconv1.weight").transposed()) + arr(p + ".pwconv1.bias")
                h = Ops.gelu(h)
                h = MLX.matmul(h, arr(p + ".pwconv2.weight").transposed()) + arr(p + ".pwconv2.bias")
                h = h * arr(p + ".gamma")
                x = inp + h
            }
            feats.append(Ops.layerNorm(x, arr("encoder.arch.norm\(i).weight"),
                                       arr("encoder.arch.norm\(i).bias"), eps: 1e-6))
        }
        return feats
    }

    // MARK: UNet decoder block (UnetBlockWide)
    private func unetBlock(_ upIn: MLXArray, skip: MLXArray, idx: Int) -> MLXArray {
        let p = "decoder.layers.\(idx)"
        // CustomPixelShuffle_ICNR: conv → BN → ReLU → pixelshuffle(2) → blur
        var x = conv(upIn, eff(p + ".shuf.conv.0"))                          // bias=False
        x = bn(x, p + ".shuf.conv.1")
        x = Ops.relu(x)
        x = Ops.pixelShuffle(x, 2)
        x = Ops.blur(x)
        let sk = bn(skip, p + ".bn")
        var cat = MLX.concatenated([x, sk], axis: -1)
        cat = Ops.relu(cat)
        // custom_conv_layer: spectral conv3x3 → ReLU → BN  ([conv.0, relu.1, BN.2])
        var y = conv(cat, eff(p + ".conv.0"), pad: 1)
        y = Ops.relu(y)
        y = bn(y, p + ".conv.2")
        return y
    }

    private func lastShuf(_ x0: MLXArray) -> MLXArray {
        // extra_bn=False: conv (with bias) → ReLU → pixelshuffle(4) → blur
        var x = conv(x0, eff("decoder.last_shuf.conv.0"), bias: arr("decoder.last_shuf.conv.0.bias"))
        x = Ops.relu(x)
        x = Ops.pixelShuffle(x, 4)
        return Ops.blur(x)
    }

    // MARK: MultiScaleColorDecoder (DETR-style query transformer)
    private func colorDecoder(_ feats3: [MLXArray], imgFeatures: MLXArray) -> MLXArray {
        let cd = "decoder.color_decoder"
        var src: [MLXArray] = [], pos: [MLXArray] = []
        for i in 0 ..< 3 {
            let f = feats3[i]
            let h = f.dim(1), wd = f.dim(2)
            pos.append(Ops.posEmbedSine(h: h, w: wd))                        // (HW, 256)
            let proj = conv(f, cw("\(cd).input_proj.\(i).weight"), bias: arr("\(cd).input_proj.\(i).bias"))
            let lev = arr("\(cd).level_embed.weight")[i]                     // (256,)
            src.append(proj.reshaped([h * wd, 256]) + lev)
        }
        let queryEmbed = arr("\(cd).query_embed.weight")                     // (100,256)
        var output = arr("\(cd).query_feat.weight")                         // (100,256)
        for i in 0 ..< config.decLayers {
            let li = i % config.numScales
            let cb = "\(cd).transformer_cross_attention_layers.\(i)"
            let sb = "\(cd).transformer_self_attention_layers.\(i)"
            let fb = "\(cd).transformer_ffn_layers.\(i)"
            // cross-attention first: q = output+query_embed, k = src+pos, v = src
            var t2 = Ops.mha(q: output + queryEmbed, k: src[li] + pos[li], v: src[li],
                             inW: arr(cb + ".multihead_attn.in_proj_weight"),
                             inB: arr(cb + ".multihead_attn.in_proj_bias"),
                             outW: arr(cb + ".multihead_attn.out_proj.weight"),
                             outB: arr(cb + ".multihead_attn.out_proj.bias"), heads: config.numHeads)
            output = Ops.layerNorm(output + t2, arr(cb + ".norm.weight"), arr(cb + ".norm.bias"), eps: 1e-5)
            // self-attention: q = k = output+query_embed, v = output
            let qk = output + queryEmbed
            t2 = Ops.mha(q: qk, k: qk, v: output,
                         inW: arr(sb + ".self_attn.in_proj_weight"),
                         inB: arr(sb + ".self_attn.in_proj_bias"),
                         outW: arr(sb + ".self_attn.out_proj.weight"),
                         outB: arr(sb + ".self_attn.out_proj.bias"), heads: config.numHeads)
            output = Ops.layerNorm(output + t2, arr(sb + ".norm.weight"), arr(sb + ".norm.bias"), eps: 1e-5)
            // FFN
            var h1 = MLX.matmul(output, arr(fb + ".linear1.weight").transposed()) + arr(fb + ".linear1.bias")
            h1 = Ops.relu(h1)
            let ff = MLX.matmul(h1, arr(fb + ".linear2.weight").transposed()) + arr(fb + ".linear2.bias")
            output = Ops.layerNorm(output + ff, arr(fb + ".norm.weight"), arr(fb + ".norm.bias"), eps: 1e-5)
        }
        output = Ops.layerNorm(output, arr("\(cd).decoder_norm.weight"), arr("\(cd).decoder_norm.bias"), eps: 1e-5)
        // color_embed MLP (3 Linear, ReLU between)
        var emb = output
        for li in 0 ..< 3 {
            let lk = "\(cd).color_embed.layers.\(li)"
            emb = MLX.matmul(emb, arr(lk + ".weight").transposed()) + arr(lk + ".bias")
            if li < 2 { emb = Ops.relu(emb) }
        }
        // einsum bqc,bchw->bqhw  (single batch)
        let H = imgFeatures.dim(1), Wd = imgFeatures.dim(2), C = imgFeatures.dim(3)
        let imgf = imgFeatures.reshaped([H * Wd, C])                          // (HW, C)
        let out = MLX.matmul(emb, imgf.transposed())                         // (100, HW)
        return out.reshaped([config.numQueries, H, Wd]).transposed(1, 2, 0).reshaped([1, H, Wd, config.numQueries])
    }

    /// Full forward: gray-RGB in `[0,1]`, NHWC `(1,S,S,3)` → raw ab NHWC `(1,S,S,2)`.
    /// ImageNet-normalizes internally (the reference does this inside `forward`); the normalized
    /// input is what `refine_net` concatenates, so normalize once up front.
    public func callAsFunction(_ grayRGB: MLXArray) -> MLXArray {
        let x = (grayRGB - mean) / std
        let feats = encoder(x)
        let out0 = unetBlock(feats[3], skip: feats[2], idx: 0)
        let out1 = unetBlock(out0, skip: feats[1], idx: 1)
        let out2 = unetBlock(out1, skip: feats[0], idx: 2)
        let out3 = lastShuf(out2)
        let color = colorDecoder([out0, out1, out2], imgFeatures: out3)       // (1,S,S,100)
        let cat = MLX.concatenated([color, x], axis: -1)                      // + normalized input (3)
        return conv(cat, eff("refine_net.0.0"), bias: arr("refine_net.0.0.bias"))  // (1,S,S,2)
    }
}
