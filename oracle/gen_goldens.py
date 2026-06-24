"""DDColor parity oracle — dump granular per-sub-op goldens for the MLX-Swift port.

Throwaway scratchpad (mlx-porting skill: oracle/measure dir). Builds the PyTorch
DDColor, loads released weights, runs CPU fp32 forward on a *fixed seeded* input,
and writes one .npy per intermediate so a Swift parity break localizes to the exact
op without a Python twin. Also resolves spectral/weight-norm parametrizations to
plain effective weights and dumps them as a flat .npz for the WeightConverter.

Usage:
    python gen_goldens.py --size tiny      # convnext-t, ddcolor_paper_tiny.pth
    python gen_goldens.py --size large     # convnext-l, ddcolor_modelscope.pth
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = "/Users/dustinnielson/Development/porting_dev_opportunities/_eval/DDColor"
sys.path.insert(0, REPO)

from ddcolor.model import DDColor  # noqa: E402

WEIGHTS = {
    # piddnad/DDColor-models on HuggingFace
    "tiny": ("ddcolor_paper_tiny.pth", "convnext-t"),
    "large": ("ddcolor_modelscope.pth", "convnext-l"),
}


def hf_download(filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="piddnad/DDColor-models", filename=filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(WEIGHTS), default="tiny")
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    ckpt_name, encoder_name = WEIGHTS[args.size]
    out_dir = args.out or os.path.join(os.path.dirname(__file__), "goldens", args.size)
    os.makedirs(out_dir, exist_ok=True)

    model = DDColor(
        encoder_name=encoder_name,
        decoder_name="MultiScaleColorDecoder",
        input_size=(args.input_size, args.input_size),
        num_input_channels=3,
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    ).eval()

    ckpt_path = hf_download(ckpt_name)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("params", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing[:8]:", missing[:8])
    if unexpected:
        print("  unexpected[:8]:", unexpected[:8])

    # ---- granular intermediate capture via forward hooks ----
    caps = {}

    def cap(name):
        def fn(_m, _i, o):
            caps[name] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy()
        return fn

    dec = model.decoder
    dec.layers[0].register_forward_hook(cap("dec_out0"))
    dec.layers[1].register_forward_hook(cap("dec_out1"))
    dec.layers[2].register_forward_hook(cap("dec_out2"))
    dec.last_shuf.register_forward_hook(cap("dec_out3"))
    dec.color_decoder.register_forward_hook(cap("color_out"))

    # ---- fixed seeded input at the model's tensor boundary (gray-RGB, NCHW) ----
    x = torch.randn(1, 3, args.input_size, args.input_size, dtype=torch.float32)
    np.save(os.path.join(out_dir, "input_gray_rgb_nchw.npy"), x.numpy())

    out_ab = model(x)  # (1, 2, S, S)

    # ConvNeXt stage features (the hooks the model uses internally)
    for i, h in enumerate(model.encoder.hooks):
        np.save(os.path.join(out_dir, f"enc_norm{i}.npy"), h.feature.detach().numpy())
    for k, v in caps.items():
        np.save(os.path.join(out_dir, f"{k}.npy"), v)
    np.save(os.path.join(out_dir, "out_ab_nchw.npy"), out_ab.numpy())

    # ---- resolve parametrized weights → plain effective weights for the converter ----
    flat = {}
    for name, p in model.state_dict().items():
        flat[name] = p.detach().float().cpu().numpy()
    # effective (post-spectral/weight-norm) conv weights, captured after the forward
    for mod_name, m in model.named_modules():
        if hasattr(m, "weight") and isinstance(getattr(m, "weight"), torch.Tensor) \
                and getattr(m, "weight_orig", None) is not None:
            flat[mod_name + ".__effective_weight__"] = m.weight.detach().float().cpu().numpy()
    np.savez(os.path.join(out_dir, "state_resolved.npz"), **flat)

    print(f"[done] {args.size}: out_ab {tuple(out_ab.shape)} "
          f"min={out_ab.min():.4f} max={out_ab.max():.4f} mean={out_ab.mean():.4f}")
    print(f"[done] wrote goldens → {out_dir}")
    for i, h in enumerate(model.encoder.hooks):
        print(f"  enc_norm{i}: {tuple(h.feature.shape)}")
    for k, v in caps.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
