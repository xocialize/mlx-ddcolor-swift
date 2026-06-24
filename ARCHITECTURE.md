# DDColor — MLX-Swift Port Map

Oracle: `piddnad/DDColor` @ shallow clone in
`/Users/dustinnielson/Development/porting_dev_opportunities/_eval/DDColor`.
Weights: `piddnad/DDColor-models` (HF). Released tiers used here:
`ddcolor_paper_tiny.pth` (convnext-t) · `ddcolor_modelscope.pth` (convnext-l, default/best).

This is a **from-scratch architecture port** (no MLX donor), direct PyTorch → MLX-Swift
(ConvNeXt + DETR-style query decoder are both well-trodden → no Python rung; goldens localize
parity breaks instead). Mirrors `mlx-birefnet-swift` layout. **NHWC throughout** (MLX `Conv2d`).

## Top-level dataflow (`DDColor.forward`)

```
gray-RGB x (B,3,S,S, ImageNet-normalized internally)            [pipeline supplies gray-RGB from cv2 LAB L]
  └ Encoder(ConvNeXt)  ──► [f0,f1,f2,f3]  = norm0..norm3 outputs (channels_first LN per stage)
  └ Decoder
      out0 = UnetBlockWide(f3→, skip f2)         512ch @ S/16
      out1 = UnetBlockWide(out0, skip f1)        512ch @ S/8
      out2 = UnetBlockWide(out1, skip f0)        256ch @ S/4
      out3 = CustomPixelShuffle_ICNR(out2, ×4)   256ch @ S      (img_features)
      color = MultiScaleColorDecoder([out0,out1,out2], out3)  100ch @ S
  └ refine_net = Spectral Conv1x1( cat[color(100), x(3)] → 2 )  ► ab (B,2,S,S)
```

`do_normalize=False` → output is **not** denormalized (raw ab). `num_output_channels=2`,
`num_queries=100`, `nf=512`, `dec_layers=9`, `num_scales=3`, `last_norm='Spectral'`. Input
default S=512. Only ConvNeXt `depths/dims` change per tier; decoder channels `[512,512,256]`
are tier-invariant.

## I/O contract & the LAB parity trap (pipeline, NOT the network)

`ColorizationPipeline.process` (BGR uint8 in/out):
1. `img = bgr/255` → `orig_l = cv2.cvtColor(img, BGR2Lab)[...,:1]` at **original** H×W.
2. resize img→(S,S); `img_l = BGR2Lab[...,:1]`; build LAB `(L,0,0)`; `LAB2RGB` → `img_gray_rgb`.
3. model input = `img_gray_rgb` transposed to NCHW float (NO /255 — already 0..1).
4. `output_ab` = model(x); `F.interpolate(output_ab, (H,W))`; concat `(orig_l, ab)`; `LAB2RGB`→BGR; ×255 round.

> **Parity strategy:** lock the **network** at the tensor boundary — feed the golden
> `input_gray_rgb_nchw.npy`, compare `out_ab_nchw.npy` (`<1e-3` CPU fp32). Validate cv2's
> LAB↔RGB math **separately** (it is OpenCV's D65 sRGB Lab; replicate exactly in the Swift
> pipeline or precompute). This is the documented silent-failure surface (sepia/washed-out).

## Module inventory & weight keys (tiny shapes; large scales dims)

### ConvNeXt encoder  (`encoder.arch.*`)
- **stem** `downsample_layers.0` = `Sequential(Conv2d(3→d0,k4,s4)[.0], LayerNorm_cf(d0)[.1])`.
- **downsample i∈{1,2,3}** `downsample_layers.i` = `Sequential(LayerNorm_cf(d_{i-1})[.0], Conv2d(d_{i-1}→d_i,k2,s2)[.1])`. (order is LN→Conv, opposite of stem)
- **stage i** `stages.i.j` = ConvNeXt `Block`:
  `dwconv` Conv2d(d,d,k7,p3,groups=d) → permute NHWC → `norm` **LayerNorm channels_last** (eps 1e-6)
  → `pwconv1` Linear(d→4d) → GELU → `pwconv2` Linear(4d→d) → `* gamma(d)` → permute back → `+input`.
- **norm0..norm3** = `LayerNorm channels_first` (eps 1e-6) applied to each stage output; **these
  four normed maps are the hooks** the decoder consumes. In NHWC, channels_first-LN = ordinary
  last-axis LayerNorm (mean/var over C) → same as channels_last LN math; keep both as last-axis LN.
- `encoder.arch.norm` + global-avg-pool head: **computed but discarded** by DDColor → **omit** in MLX.
- dims: tiny [96,192,384,768] depths [3,3,9,3]; large [192,384,768,1536] depths [3,3,27,3].
- Conv weight layout: PyTorch `(O,I,kH,kW)` → MLX `(O,kH,kW,I)`. Depthwise: I=1, groups=d.

### UNet decoder  (`decoder.*`)
- `UnetBlockWide` (`decoder.layers.{0,1,2}`):
  - `shuf` = `CustomPixelShuffle_ICNR(up_in→n_out, ×2, extra_bn=True)`:
    `conv.0` Spectral Conv1x1 (→ n_out·4, **bias=False**) `conv.1` BatchNorm2d → PixelShuffle(2)
    → (blur: ReplicationPad2d(1,0,1,0)+AvgPool2d(2,s1)) → applied as `relu(conv)` then shuffle then blur.
    NB forward order: `shuf(self.relu(self.conv(x)))` → wait: `CustomPixelShuffle_ICNR.forward = shuf(relu(conv(x)))` then blur.
  - `bn` = BatchNorm2d(x_in_c) applied to the **skip** feature `s`.
  - cat `relu(cat[shuf_out, bn(s)])` → `conv` = `custom_conv_layer(ni→n_out, k3, extra_bn=True)` =
    Spectral Conv3x3 (**bias=False**) → ReLU → BatchNorm2d. (conv.0=conv, conv.1=BN)
- `last_shuf` = `CustomPixelShuffle_ICNR(256→256, ×4, extra_bn=False)`: `conv.0` Spectral Conv1x1
  (→256·16=4096, **bias=True**, no BN) → PixelShuffle(4) → blur. (relu before shuffle as above)
- **BatchNorm2d**: eval mode — `(x-running_mean)/sqrt(running_var+eps)*weight+bias`, eps=1e-5 (torch default). channels-first → in NHWC normalize over last axis with per-channel params.
- **Spectral norm**: `weight = weight_orig / σ`; resolved offline → `__effective_weight__` in npz.

### MultiScaleColorDecoder  (`decoder.color_decoder.*`)  — DETR-style, hidden=256, 8 heads
- `pe_layer` PositionEmbeddingSine(128, normalize=True) — **no params**, deterministic from H,W.
- `input_proj.{0,1,2}` Conv2d 1x1 (in [512,512,256]→256, **with bias**).
- `query_feat`, `query_embed`, `level_embed` = `nn.Embedding` (100,256)/(100,256)/(3,256).
- 9× layers, each: **cross-attn first**, then self-attn, then FFN (all `forward_post`, post-norm):
  - cross `multihead_attn`: q=output+query_embed, k=src+pos, v=src; +residual; LayerNorm.
  - self `self_attn`: q=k=output+query_embed, v=output; +residual; LayerNorm.
  - ffn: Linear(256→2048)→ReLU→Linear(2048→256); +residual; LayerNorm.
  - level_index = `i % 3` selects which of the 3 src/pos scales feeds layer i.
- `decoder_norm` LayerNorm → transpose to (B,Q,C) → `color_embed` MLP(256,256,256,3 layers, ReLU between).
- `out = einsum("bqc,bchw->bqhw", color_embed, img_features)` → (B,100,S,S).
- `nn.MultiheadAttention`: `in_proj_weight (3C,C)` = **contiguous [q;k;v] thirds** (NOT interleaved),
  `in_proj_bias (3C)`, `out_proj`. scale = 1/sqrt(head_dim=32). Sequence layout is (L,N,E) in torch;
  in MLX use (N,L,E) with standard SDPA — math identical, no mask.
- **src/pos construction** (`_get_src_and_pos`): `pos = pe(feat).flatten(2).permute(2,0,1)` (HW,N,C);
  `src = (input_proj(feat).flatten(2) + level_embed[i][None,:,None]).permute(2,0,1)`.

### refine_net  (`refine_net.0.0`)
Spectral Conv1x1(100+3=103 → 2, **bias=True**). Input = `cat[color_out, x]` where x = the
**ImageNet-normalized** gray-RGB input (not the raw input).

## Goldens (CPU fp32, seeded input) — `oracle/goldens/{tiny,large}/`
`input_gray_rgb_nchw` · `enc_norm0..3` · `dec_out0..3` · `color_out` · `out_ab_nchw` · `state_resolved.npz`.
tiny out_ab: (1,2,512,512) min −4.19 max 4.20 mean −0.0008. Parity target: max_abs < 1e-3 vs these.

## Parity-trap checklist (port-time)
- [ ] Conv `(O,I,kH,kW)`→`(O,kH,kW,I)`; depthwise groups=d in MLX.
- [ ] ConvNeXt `Block` LayerNorm is over **channels** (last axis in NHWC) — eps 1e-6.
- [ ] norm0..3 = last-axis LayerNorm eps 1e-6 (channels_first in torch == last-axis in NHWC).
- [ ] BatchNorm eval, eps **1e-5**, per-channel; applied to skip (`bn`) and post-conv (`conv.1`).
- [ ] Spectral convs use resolved effective weight; bias only on `last_shuf`+`refine_net`.
- [ ] PixelShuffle(r): (B,H,W,C·r²)→(B,H·r,W·r,C); verify axis order vs torch NCHW PixelShuffle.
- [ ] Blur = ReplicationPad2d(left1,top1)+AvgPool2d(2,stride1) — asymmetric pad.
- [ ] MHA in_proj split = contiguous q;k;v thirds; scale 1/√32; cross-attn precedes self-attn.
- [ ] level_index = i%3 feature-level routing across the 9 layers.
- [ ] refine_net input concatenates the **normalized** x, not raw input.
- [ ] cv2 LAB↔RGB (D65 sRGB) replicated exactly in the pipeline — validate independently.
