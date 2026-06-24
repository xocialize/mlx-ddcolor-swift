"""Dump the PyTorch ColorizationPipeline intermediates on a real image, for validating the
Swift LAB pre/post pipeline. Source is resized to 512×512 so the model-input resize and the ab
upsample are both identity → this isolates the cv2 BGR↔Lab colorspace math (the parity trap).

Writes oracle/weights/image_parity_<tier>.safetensors:
  src_bgr  (512,512,3) float [0,1] BGR  — pipeline input (Swift loads this exact array)
  gray_rgb (512,512,3) float           — model input (RGB order, from cv2 LAB2RGB of (L,0,0))
  orig_l   (512,512,1) float           — original L channel
  out_ab   (512,512,2) float           — model ab output
  out_bgr  (512,512,3) float [0,1] BGR — final pipeline output (pre ×255 round)
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import mlx.core as mx

REPO = "/Users/dustinnielson/Development/porting_dev_opportunities/_eval/DDColor"
sys.path.insert(0, REPO)
from convert import build_loaded  # reuse the torch model builder  # noqa: E402

DEF_IMG = os.path.join(REPO, "assets/test_images/Detroit circa 1915.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--image", default=DEF_IMG)
    args = ap.parse_args()
    S = 512

    model = build_loaded(args.size, input_size=S)

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    bgr = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_AREA)
    img = (bgr / 255.0).astype(np.float32)                      # BGR float [0,1]

    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]     # (S,S,1)
    img_l = orig_l                                              # already 512²
    gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)        # (S,S,3) RGB

    with torch.no_grad():
        x = torch.from_numpy(gray_rgb.transpose(2, 0, 1)).float().unsqueeze(0)
        out_ab = model(x)[0].numpy().transpose(1, 2, 0)         # (S,S,2)

    out_lab = np.concatenate((orig_l, out_ab), axis=-1).astype(np.float32)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)          # (S,S,3) BGR float

    out = os.path.join(os.path.dirname(__file__), "weights", f"image_parity_{args.size}.safetensors")
    mx.save_safetensors(out, {
        "src_bgr": mx.array(img.astype(np.float32)),
        "gray_rgb": mx.array(gray_rgb.astype(np.float32)),
        "orig_l": mx.array(orig_l.astype(np.float32)),
        "out_ab": mx.array(out_ab.astype(np.float32)),
        "out_bgr": mx.array(out_bgr.astype(np.float32)),
    })
    print(f"[image] {args.size}: {os.path.basename(args.image)}")
    print(f"  L range [{orig_l.min():.2f},{orig_l.max():.2f}]  ab [{out_ab.min():.2f},{out_ab.max():.2f}]")
    print(f"  out_bgr [{out_bgr.min():.3f},{out_bgr.max():.3f}] -> {out}")


if __name__ == "__main__":
    main()
