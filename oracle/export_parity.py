"""Export the seeded parity fixture as safetensors for the Swift Smoke gate.

Writes `oracle/weights/parity_<tier>.safetensors` with:
  input  — RAW gray-RGB, NHWC (1,S,S,3)   [DDColorModel normalizes internally]
  out_ab — golden ab output, NHWC (1,S,S,2)   [parity target, max_abs < 1e-3]
"""
import argparse
import os
import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="tiny")
args = ap.parse_args()
gdir = os.path.join(os.path.dirname(__file__), "goldens", args.size)

x_nchw = np.load(os.path.join(gdir, "input_gray_rgb_nchw.npy"))
x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).astype(np.float32)  # raw; model normalizes

ab = np.load(os.path.join(gdir, "out_ab_nchw.npy"))
ab_nhwc = np.transpose(ab, (0, 2, 3, 1)).astype(np.float32)

out = os.path.join(os.path.dirname(__file__), "weights", f"parity_{args.size}.safetensors")
mx.save_safetensors(out, {"input": mx.array(x_nhwc), "out_ab": mx.array(ab_nhwc)})
print(f"[parity] {args.size}: input{x_nhwc.shape} out_ab{ab_nhwc.shape} -> {out}")
