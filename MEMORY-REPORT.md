# DDColor — Memory Report

Empirical peak unified memory for the `minUnifiedMemory` / `QuantFootprint` declaration (C10).
Measured via `MLX.GPU.peakMemory` around a full `DDColorColorizer` forward (CGImage → colorized
CGImage, the production path), `MLX.GPU.resetPeakMemory()` before each run. Host: M-Max, macOS 26.

## Measurements (input 1000×553)

| Tier | Arch | dtype | Peak | Latency |
|---|---|---|---|---|
| fast | convnext-t | fp32 | 2.22 GB | 0.10 s |
| best | convnext-l | fp32 | 2.91 GB | 0.13 s |
| fast | convnext-t | fp16 | **2.11 GB** | 0.36 s |
| best | convnext-l | fp16 | **2.45 GB** | 0.14 s |

(`large` native-res colorize @ 880×678: 2.27 s including weight load; steady-state forward ~0.13 s.)

## Declaration

`footprints: [QuantFootprint(quant: .fp16, residentBytes: 4_000_000_000)]` — a single 4 GB fp16
envelope covers **all three tiers**:

- The peak is **activation-dominated**, not weight-dominated: the model runs at a fixed 512² internally
  and the LAB colorspace ops run at native resolution, so weight size (large fp16 ≈ 0.4 GB) is a small
  fraction of the 2.1–2.45 GB peak. Tiers differ by only ~0.35 GB.
- Because all tiers are close and consumer-friendly (≤2.45 GB measured), declaring one footprint with
  margin does **not** lock smaller Macs out of any tier — unlike BiRefNet (6 GB fast / 18 GB best), no
  per-mode footprint or runtime memory guard is needed here.
- The 4 GB margin (vs 2.45 GB measured @ 1000×553) covers larger inputs: native-res `orig_l`, the final
  BGR, and the ab nearest-upsample scale with image area. Re-measure if targeting ≥4K envelopes.

## int4

Not pursued: int4 quantizes weights, but the working set here is activation, so it would shrink ~0.4 GB
of a 2.45 GB peak for a likely quality cost. fp16 is the shipping runtime.
