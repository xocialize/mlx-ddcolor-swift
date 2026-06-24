"""DDColor PyTorch -> canonical MLX safetensors converter.

Resolves spectral/weight-norm parametrizations to plain effective weights, transposes
conv weights to MLX NHWC layout (O,I,kH,kW) -> (O,kH,kW,I), drops the unused ConvNeXt
classification head + final norm + parametrization bookkeeping, and writes one
safetensors per tier. The emitted flat key scheme is shared by mlx_ref.py and the Swift
WeightLoading remap (single source of truth → minimal drift).

    python convert.py --size tiny    --out weights/ddcolor_tiny.safetensors
    python convert.py --size large   --out weights/ddcolor_modelscope.safetensors

Verify: re-run mlx_ref.py with --weights <safetensors> reproduces parity.
"""
import argparse
import os
import sys

import numpy as np
import torch
import mlx.core as mx

REPO = "/Users/dustinnielson/Development/porting_dev_opportunities/_eval/DDColor"
sys.path.insert(0, REPO)
from ddcolor.model import DDColor  # noqa: E402

WEIGHTS = {"tiny": ("ddcolor_paper_tiny.pth", "convnext-t"),
           "large": ("ddcolor_modelscope.pth", "convnext-l"),
           "artistic": ("ddcolor_artistic.pth", "convnext-l")}


def build_loaded(size, input_size=512):
    ckpt_name, encoder_name = WEIGHTS[size]
    from huggingface_hub import hf_hub_download
    model = DDColor(encoder_name=encoder_name, decoder_name="MultiScaleColorDecoder",
                    input_size=(input_size, input_size), num_input_channels=3,
                    num_output_channels=2, last_norm="Spectral", do_normalize=False,
                    num_queries=100, num_scales=3, dec_layers=9).eval()
    sd = torch.load(hf_hub_download("piddnad/DDColor-models", ckpt_name), map_location="cpu")
    sd = sd.get("params", sd)
    model.load_state_dict(sd, strict=False)
    with torch.no_grad():  # one forward resolves spectral-norm .weight
        model(torch.zeros(1, 3, input_size, input_size))
    return model


SKIP_SUBSTR = ("weight_orig", "weight_u", "weight_v", "num_batches_tracked",
               "encoder.arch.head_cls", "encoder.arch.norm.")  # final norm unused
# modules whose effective weight is parametrized (spectral) — use resolved .weight
PARAM_CONV = ("decoder.layers.", "decoder.last_shuf.conv.0", "refine_net.0.0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(WEIGHTS), default="tiny")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = ap.parse_args()
    out_dtype = mx.float16 if args.dtype == "float16" else mx.float32
    model = build_loaded(args.size)

    # effective (resolved) weight per parametrized conv module
    eff = {}
    for name, m in model.named_modules():
        if getattr(m, "weight_orig", None) is not None:
            eff[name + ".weight"] = m.weight.detach().float().cpu().numpy()

    out = {}

    def put(k, arr):
        if arr.ndim == 4:  # conv (O,I,kH,kW) -> NHWC (O,kH,kW,I)
            arr = np.transpose(arr, (0, 2, 3, 1))
        out[k] = mx.array(arr.astype(np.float32)).astype(out_dtype)

    for k, v in model.state_dict().items():
        if any(sub in k for sub in SKIP_SUBSTR):
            continue
        put(k, v.detach().float().cpu().numpy())
    # spectral-norm effective weights live only in `eff` (state_dict has weight_orig/u/v, no .weight)
    for k, arr in eff.items():
        put(k, arr)

    mx.eval(list(out.values()))  # materialize before save (lazy-tensor trap)
    out_path = args.out or os.path.join(os.path.dirname(__file__), "weights",
                                        f"ddcolor_{args.size}.safetensors")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mx.save_safetensors(out_path, out)
    print(f"[convert] {args.size}: {len(out)} tensors -> {out_path}")


if __name__ == "__main__":
    main()
