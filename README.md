# mlx-ddcolor-swift

DDColor automatic image colorization on Apple-Silicon MLX-Swift — a from-scratch architecture port
of [piddnad/DDColor](https://github.com/piddnad/DDColor) (Apache-2.0), destined for an MLXEngine
`imageColorize` ModelPackage backing the Forge **Colorize** capability.

> **Status:** network port **complete + parity-locked** on both tiers (tiny / large). Engine
> ModelPackage wrap + LAB pipeline + quantization + publish are the remaining steps. See
> [`PORT-STATUS.md`](PORT-STATUS.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Architecture
ConvNeXt encoder (norm0–3 stage features) → quasi-UNet decoder (spectral convs, BatchNorm,
PixelShuffle+anti-checkerboard blur) → DETR-style `MultiScaleColorDecoder` (100 learnable color
queries, 9 layers, cross-then-self attention) → `refine_net` → **ab chroma** recombined with the
source L channel. Two tiers differ only in ConvNeXt depth/width.

## Layout
- `Sources/DDColor` — MLX-Swift core (`DDColorModel`, `DDColorOps`, `DDColorConfig`).
- `Sources/Smoke` — `ddcolor-smoke` real-forward parity gate (CPU fp32 vs PyTorch golden ab).
- `oracle/` — throwaway parity scratchpad: PyTorch golden generation, MLX-Python reference,
  Python weight converter, parity-fixture export. (gitignored artifacts)

## Quick parity check
```bash
cd oracle && source .venv/bin/activate
python gen_goldens.py --size tiny && python convert.py --size tiny && python export_parity.py --size tiny
cd .. && swift run -c release ddcolor-smoke --tier tiny \
  --weights oracle/weights/ddcolor_tiny.safetensors --parity oracle/weights/parity_tiny.safetensors
# → [smoke] tiny  max_abs=3.3e-6  OK ✅
```

## Weights
The MLXEngine package (`DDColorPackage`) loads three converted checkpoints, one per Mode:
`mlx-community/DDColor-paper-tiny-fp16` (`fast`, 110 MB), `mlx-community/DDColor-modelscope-fp16` (`best`, 456 MB)
and `mlx-community/DDColor-artistic-fp16` (`artistic`, 456 MB).

Since 0.3.0 (AB-T-0222) `DDColorConfiguration` declares all three as `weightSources`, so MLXEngine sizes, previews
and downloads them into its model store before `load()`, with byte progress and resume. `load()` reads the store's
flat or snapshot layout, and also `<root>/models/<org>/<name>/`, where 0.2.x downloaded through swift-transformers'
`HubApi`, so existing installs aren't fetched again. Outside the engine it still downloads through `HubApi`. The
`*WeightsURL` overrides bypass all of it.

## License
Port code MIT. DDColor architecture + weights Apache-2.0 (piddnad/DDColor). ConvNeXt block
adapted from facebookresearch/ConvNeXt (MIT).
