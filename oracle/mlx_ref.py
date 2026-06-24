"""Throwaway MLX-Python reference for DDColor (oracle scratchpad).

Validates the full architecture understanding against the PyTorch goldens BEFORE
transcribing to Swift. NHWC throughout (matches MLX Conv2d / the Swift target).
CPU stream, fp32. Loads resolved weights from state_resolved.npz.

    python mlx_ref.py --size tiny
"""
import argparse
import os
import math

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

DIMS = {"tiny": ([3, 3, 9, 3], [96, 192, 384, 768]),
        "large": ([3, 3, 27, 3], [192, 384, 768, 1536])}


# FROM_ST=True → weights came from the converted safetensors (already NHWC + spectral-resolved,
# head/parametrization keys dropped). FROM_ST=False → raw npz state_dict (NCHW, parametrized).
FROM_ST = False


def load_state(path):
    if path.endswith(".safetensors"):
        return {k: np.array(v) for k, v in mx.load(path).items()}
    z = np.load(path)
    return {k: z[k] for k in z.files}


def W(s, k):  # weight as mx.array fp32
    return mx.array(s[k].astype(np.float32))


def conv_w(s, k):  # (O,I,kH,kW) -> (O,kH,kW,I); safetensors already NHWC
    a = s[k].astype(np.float32)
    return mx.array(a if FROM_ST else np.transpose(a, (0, 2, 3, 1)))


def eff_or(s, base):  # prefer resolved effective weight (spectral), else plain
    if FROM_ST:
        return mx.array(s[base + ".weight"].astype(np.float32))  # already resolved + NHWC
    eff = base + ".__effective_weight__"
    key = eff if eff in s else base + ".weight"
    return mx.array(np.transpose(s[key].astype(np.float32), (0, 2, 3, 1)))


def layernorm(x, w, b, eps):  # last-axis LN
    u = x.mean(-1, keepdims=True)
    v = ((x - u) ** 2).mean(-1, keepdims=True)
    return (x - u) / mx.sqrt(v + eps) * w + b


def batchnorm(x, s, base, eps=1e-5):  # NHWC eval-mode BN over last axis
    rm = W(s, base + ".running_mean"); rv = W(s, base + ".running_var")
    g = W(s, base + ".weight"); b = W(s, base + ".bias")
    return (x - rm) / mx.sqrt(rv + eps) * g + b


def conv2d(x, w, b=None, stride=1, pad=0, groups=1):
    y = mx.conv2d(x, w, stride=stride, padding=pad, groups=groups)
    return y + b if b is not None else y


# ---------------- ConvNeXt encoder ----------------
def convnext(x, s, depths, dims):
    feats = []
    for i in range(4):
        if i == 0:
            x = conv2d(x, conv_w(s, "encoder.arch.downsample_layers.0.0.weight"),
                       W(s, "encoder.arch.downsample_layers.0.0.bias"), stride=4)
            x = layernorm(x, W(s, "encoder.arch.downsample_layers.0.1.weight"),
                          W(s, "encoder.arch.downsample_layers.0.1.bias"), 1e-6)
        else:
            x = layernorm(x, W(s, f"encoder.arch.downsample_layers.{i}.0.weight"),
                          W(s, f"encoder.arch.downsample_layers.{i}.0.bias"), 1e-6)
            x = conv2d(x, conv_w(s, f"encoder.arch.downsample_layers.{i}.1.weight"),
                       W(s, f"encoder.arch.downsample_layers.{i}.1.bias"), stride=2)
        for j in range(depths[i]):
            p = f"encoder.arch.stages.{i}.{j}"
            inp = x
            h = conv2d(x, conv_w(s, p + ".dwconv.weight"), W(s, p + ".dwconv.bias"),
                       pad=3, groups=dims[i])
            h = layernorm(h, W(s, p + ".norm.weight"), W(s, p + ".norm.bias"), 1e-6)
            h = h @ W(s, p + ".pwconv1.weight").T + W(s, p + ".pwconv1.bias")
            h = 0.5 * h * (1 + mx.erf(h / math.sqrt(2.0)))  # exact GELU
            h = h @ W(s, p + ".pwconv2.weight").T + W(s, p + ".pwconv2.bias")
            h = h * W(s, p + ".gamma")
            x = inp + h
        f = layernorm(x, W(s, f"encoder.arch.norm{i}.weight"),
                      W(s, f"encoder.arch.norm{i}.bias"), 1e-6)
        feats.append(f)
    return feats  # NHWC


# ---------------- UNet decoder pieces ----------------
def pixel_shuffle(x, r):  # NHWC (B,H,W,C*r*r) -> (B,H*r,W*r,C)
    # torch nn.PixelShuffle: in-channel = c_out*r^2 + rh*r + rw  (c_out OUTERMOST)
    b, h, w, c = x.shape
    c2 = c // (r * r)
    x = x.reshape(b, h, w, c2, r, r)        # (b,h,w,c_out,rh,rw)
    x = x.transpose(0, 1, 4, 2, 5, 3)       # (b,h,rh,w,rw,c_out)
    return x.reshape(b, h * r, w * r, c2)


def blur(x):  # ReplicationPad2d((1,0,1,0)) + AvgPool2d(2, stride=1)
    x = mx.pad(x, [(0, 0), (1, 0), (1, 0), (0, 0)], mode="edge")
    # avg pool 2x2 stride1 via conv with a 1/4 box filter, depthwise
    c = x.shape[-1]
    k = mx.full((c, 2, 2, 1), 0.25)
    return mx.conv2d(x, k, stride=1, padding=0, groups=c)


def custom_shuf(x, s, base, scale, extra_bn):
    w = eff_or(s, base + ".conv.0")
    bk = base + ".conv.0.bias"
    b = W(s, bk) if bk in s else None
    x = conv2d(x, w, b)
    if extra_bn:
        x = batchnorm(x, base + ".conv.1") if False else x  # placeholder, replaced below
    return x


def unet_block(up_in, skip, s, idx):
    p = f"decoder.layers.{idx}"
    # CustomPixelShuffle_ICNR: conv -> BN -> ReLU -> pixelshuffle(2) -> pad+blur
    x = conv2d(up_in, eff_or(s, p + ".shuf.conv.0"))           # bias=False (extra_bn)
    x = batchnorm(x, s, p + ".shuf.conv.1")
    x = mx.maximum(x, 0)                                        # self.relu, applied after self.conv
    x = pixel_shuffle(x, 2)
    x = blur(x)
    # skip bn
    sk = batchnorm(skip, s, p + ".bn")
    cat = mx.maximum(mx.concatenate([x, sk], axis=-1), 0)       # relu(cat)
    # custom_conv_layer: spectral conv3x3 -> ReLU -> BN  ([conv.0, relu.1, BN.2])
    y = conv2d(cat, eff_or(s, p + ".conv.0"), pad=1)
    y = mx.maximum(y, 0)
    y = batchnorm(y, s, p + ".conv.2")
    return y


def last_shuf(x, s):
    x = conv2d(x, eff_or(s, "decoder.last_shuf.conv.0"),
               W(s, "decoder.last_shuf.conv.0.bias"))           # bias=True, no bn
    x = mx.maximum(x, 0)
    x = pixel_shuffle(x, 4)
    x = blur(x)
    return x


# ---------------- MultiScaleColorDecoder ----------------
def pos_embed_sine(h, w, num_pos_feats=128, temperature=10000):
    # normalize=True path; returns (1, h, w, 2*num_pos_feats) NHWC
    y_embed = mx.arange(1, h + 1, dtype=mx.float32).reshape(h, 1)
    y_embed = mx.broadcast_to(y_embed, (h, w))
    x_embed = mx.arange(1, w + 1, dtype=mx.float32).reshape(1, w)
    x_embed = mx.broadcast_to(x_embed, (h, w))
    eps = 1e-6; scale = 2 * math.pi
    y_embed = y_embed / (y_embed[-1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, -1:] + eps) * scale
    dim_t = mx.arange(num_pos_feats, dtype=mx.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = mx.stack([mx.sin(pos_x[:, :, 0::2]), mx.cos(pos_x[:, :, 1::2])], axis=3).reshape(h, w, -1)
    pos_y = mx.stack([mx.sin(pos_y[:, :, 0::2]), mx.cos(pos_y[:, :, 1::2])], axis=3).reshape(h, w, -1)
    pos = mx.concatenate([pos_y, pos_x], axis=2)  # (h,w,2*npf)
    return pos.reshape(h * w, -1)  # (HW, C)


def mha(q, k, v, s, base, nheads=8):
    # q,k,v: (L, C) single batch. packed in_proj (3C, C) contiguous thirds.
    C = q.shape[-1]; hd = C // nheads
    inw = W(s, base + ".in_proj_weight"); inb = W(s, base + ".in_proj_bias")
    wq, wk, wv = inw[:C], inw[C:2 * C], inw[2 * C:]
    bq, bk, bv = inb[:C], inb[C:2 * C], inb[2 * C:]
    Q = q @ wq.T + bq; K = k @ wk.T + bk; V = v @ wv.T + bv
    Lq, Lk = Q.shape[0], K.shape[0]
    Q = Q.reshape(Lq, nheads, hd).transpose(1, 0, 2)
    K = K.reshape(Lk, nheads, hd).transpose(1, 0, 2)
    V = V.reshape(Lk, nheads, hd).transpose(1, 0, 2)
    attn = (Q @ K.transpose(0, 2, 1)) / math.sqrt(hd)
    attn = mx.softmax(attn, axis=-1)
    out = attn @ V  # (nheads, Lq, hd)
    out = out.transpose(1, 0, 2).reshape(Lq, C)
    return out @ W(s, base + ".out_proj.weight").T + W(s, base + ".out_proj.bias")


def color_decoder(feats3, img_features, s, num_layers=9, num_levels=3):
    # feats3: list of 3 NHWC maps [out0,out1,out2]; img_features: NHWC out3
    src, pos = [], []
    for i, f in enumerate(feats3):
        _, h, w, _ = f.shape
        p = pos_embed_sine(h, w)  # (HW, 256)
        proj = conv2d(f, conv_w(s, f"decoder.color_decoder.input_proj.{i}.weight"),
                      W(s, f"decoder.color_decoder.input_proj.{i}.bias"))  # NHWC ->256
        lev = W(s, "decoder.color_decoder.level_embed.weight")[i]
        srci = proj.reshape(h * w, 256) + lev
        src.append(srci); pos.append(p)
    query_embed = W(s, "decoder.color_decoder.query_embed.weight")  # (100,256)
    output = W(s, "decoder.color_decoder.query_feat.weight")        # (100,256)
    for i in range(num_layers):
        li = i % num_levels
        cb = f"decoder.color_decoder.transformer_cross_attention_layers.{i}"
        sb = f"decoder.color_decoder.transformer_self_attention_layers.{i}"
        fb = f"decoder.color_decoder.transformer_ffn_layers.{i}"
        # cross: q=output+query_embed, k=src+pos, v=src
        t2 = mha(output + query_embed, src[li] + pos[li], src[li], s, cb + ".multihead_attn")
        output = output + t2
        output = layernorm(output, W(s, cb + ".norm.weight"), W(s, cb + ".norm.bias"), 1e-5)
        # self: q=k=output+query_embed, v=output
        qk = output + query_embed
        t2 = mha(qk, qk, output, s, sb + ".self_attn")
        output = output + t2
        output = layernorm(output, W(s, sb + ".norm.weight"), W(s, sb + ".norm.bias"), 1e-5)
        # ffn
        h1 = mx.maximum(output @ W(s, fb + ".linear1.weight").T + W(s, fb + ".linear1.bias"), 0)
        t2 = h1 @ W(s, fb + ".linear2.weight").T + W(s, fb + ".linear2.bias")
        output = output + t2
        output = layernorm(output, W(s, fb + ".norm.weight"), W(s, fb + ".norm.bias"), 1e-5)
    output = layernorm(output, W(s, "decoder.color_decoder.decoder_norm.weight"),
                       W(s, "decoder.color_decoder.decoder_norm.bias"), 1e-5)
    # color_embed MLP (3 layers, relu between)
    h = output
    for li in range(3):
        wk = f"decoder.color_decoder.color_embed.layers.{li}"
        h = h @ W(s, wk + ".weight").T + W(s, wk + ".bias")
        if li < 2:
            h = mx.maximum(h, 0)
    color_embed = h  # (100,256)
    # einsum bqc,bchw->bqhw ; img_features NHWC (1,H,W,256)
    _, H, Wd, C = img_features.shape
    imgf = img_features.reshape(H * Wd, C)            # (HW, C)
    out = color_embed @ imgf.T                        # (100, HW)
    out = out.reshape(100, H, Wd).transpose(1, 2, 0)[None]  # (1,H,W,100) NHWC
    return out


def main():
    global FROM_ST
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="tiny")
    ap.add_argument("--weights", default=None, help="converted .safetensors to verify")
    args = ap.parse_args()
    depths, dims = DIMS[args.size]
    gdir = os.path.join(os.path.dirname(__file__), "goldens", args.size)
    if args.weights:
        FROM_ST = True
        s = load_state(args.weights)
    else:
        s = load_state(os.path.join(gdir, "state_resolved.npz"))

    x_nchw = np.load(os.path.join(gdir, "input_gray_rgb_nchw.npy"))
    # ImageNet normalize (model does this internally since 3-channel)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    xn = (x_nchw - mean) / std
    x = mx.array(np.transpose(xn, (0, 2, 3, 1)).astype(np.float32))  # NHWC

    def cmp(name, mx_nhwc, golden_nchw_path, chan_last=True):
        g = np.load(golden_nchw_path)
        m = np.array(mx_nhwc)
        if chan_last:  # golden NCHW -> NHWC
            g = np.transpose(g, (0, 2, 3, 1))
        d = np.max(np.abs(g - m))
        print(f"  {name:10s} mlx{m.shape} gold{g.shape}  max_abs={d:.3e}  {'OK' if d < 1e-3 else 'FAIL'}")
        return d

    feats = convnext(x, s, depths, dims)
    for i in range(4):
        cmp(f"enc_norm{i}", feats[i], os.path.join(gdir, f"enc_norm{i}.npy"))

    out0 = unet_block(feats[3], feats[2], s, 0)
    cmp("dec_out0", out0, os.path.join(gdir, "dec_out0.npy"))
    out1 = unet_block(out0, feats[1], s, 1)
    cmp("dec_out1", out1, os.path.join(gdir, "dec_out1.npy"))
    out2 = unet_block(out1, feats[0], s, 2)
    cmp("dec_out2", out2, os.path.join(gdir, "dec_out2.npy"))
    out3 = last_shuf(out2, s)
    cmp("dec_out3", out3, os.path.join(gdir, "dec_out3.npy"))

    color = color_decoder([out0, out1, out2], out3, s)
    cmp("color_out", color, os.path.join(gdir, "color_out.npy"))

    # refine_net: cat[color(100), normalized x (3)] -> spectral conv1x1 -> 2
    cat = mx.concatenate([color, x], axis=-1)
    ab = conv2d(cat, eff_or(s, "refine_net.0.0"), W(s, "refine_net.0.0.bias"))
    cmp("out_ab", ab, os.path.join(gdir, "out_ab_nchw.npy"))


if __name__ == "__main__":
    main()
