# Efficiency Adoption Brief — `mlx-ddcolor-swift` (DDColor, `imageColorize`)

> **For a session-specific agent.** Self-contained: audit + tasks to adopt the MLXEngine
> library-efficiency contract (engine 1.14.0 / 0.15.0). Load the `mlx-swift-integration` skill and read
> `references/package-efficiency.md` (incl. "Gotchas & measurement") + `references/memory-harness.md`
> first. Brief shape follows the BiRefNet/LTX template. Audited + executed 2026-06-30.

## Why this one matters
DDColor is the **colorize** link of the ForgeOptimizer chain (IQA → restore → upscale → colorize,
alongside BiRefNet matting). Single-component (fixed 512² model + native-res LAB colorspace ops), so the
footprint is activation-dominated. With the 1.14 split it co-resides with the rest of the chain on
weights while sharing ONE activation reserve.

## Package at a glance
- **Wrapper:** `DDColorPackage` (`Sources/MLXDDColor/`), core `DDColorColorizer` (`DDColor`). Capability `imageColorize`. **Single-component**, three tier Modes dispatched per-request (`fast`=convnext-t · `best`=convnext-l/modelscope · `artistic`=stylized convnext-l).
- **One declared footprint (fp16):** the tiers are close + consumer-friendly (no per-mode footprint / runtime guard, unlike BiRefNet's 6/22 GB split). HF-downloaded fp16 weights.
- **Home:** `mlxengine-image/PROD/`.

## Engine dependency status
- `Package.swift` pinned `mlx-engine-swift` **`from: "0.10.0"`**, resolved 0.10.0. **P0 = `swift package update`** → 0.15.0 (the pin admits it; no manifest edit). The `imageColorize @ 1.7.0` surface-contract comment is still accurate and left as-is. Done.

## Audit vs. the four levers

| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡→🟢 | resolved 0.10.0; re-resolved to 0.15.0 | **P0 (done)** |
| 1. Split footprint | ❌→🟢 | was flat `QuantFootprint(.fp16, 4.0 GB)` envelope; now split (resident + activation) | **P1 (done)** |
| QuantConfigured | ❌→🟢 | config already stored `var quant`; added empty conformance | **P1 (done)** |
| 2. mmap/lazy load | 🟢 | `DDColorColorizer.fromPretrained` per tier; weights ~0.4 GB (large) | note only |
| 3. Per-stage evict | ➖ | single-component; tiers built lazily (`fast`/`artistic` on first request), `unload()` clears all | note only |
| 4. BudgetAware | ➖ | fp16 is the validated runtime; int4 not pursued (weights are a small fraction of an activation-dominated peak) | defer |

## P0 — engine update (done)
`swift package update mlx-engine-swift` → resolved **0.10.0 → 0.15.0**. Builds clean against the 1.14 contract.

## QuantConfigured (done)
`DDColorConfiguration` already stored `var quant: Quant` (fp16 default), so conformance is a trivial empty
extension — the governor now charges the matching declared `QuantFootprint` for the selected quant.

## P1 — split declared (done)
All tiers run a fixed 512² model + native-res LAB colorspace, so the working set is activation-dominated.
Re-measured via the existing `ddcolor-package-smoke` (extended with a MEM readout + `--quant`) at
**1024×512**, fp16, `best` (convnext-l = the largest-resident tier):

| Mode | quant | floor (resident) | peak | activation | run |
|---|---|---|---|---|---|
| best (convnext-l) | fp16 | 435 MB | 2236 MB | ~1.8 GB | 2.2 s |

Consistent with MEMORY-REPORT.md (fast 2.11 GB / best 2.45 GB @1000×553 — tiers within ~0.35 GB). Declared:
```swift
QuantFootprint(.fp16, residentBytes: 512_000_000, peakActivationBytes: 1_800_000_000)
```
**Engine charge:** the flat 4 GB envelope becomes a **512 MB** resident floor (covers the largest tier's
weights) plus a **shared** ~1.8 GB transient reserve (one across residents) — instead of baking 4 GB into
residency. Output validated (admitted license gate, valid 1024×512 PNG). Re-measure if targeting ≥4K
(activation scales with the native-res colorspace ops).

## Already good (don't regress)
- Native HF downloader → engine model store (`modelsRootDirectory`) + `WeightDownloadProgress` forwarding; direct-URL override for CLI/pre-resolved callers.
- `ModelStorable`; lazy per-tier build; cancellation; the single-footprint-covers-all-modes decision (tiers are close — no per-mode split needed).

## Deferred — P2 (n/a single-component), P3 (n/a — lazy tier build already minimal), P4 (BudgetAware: fp16 is the runtime; int4 trades little on an activation-dominated peak).

## Definition of done
- [x] `swift package update` → engine 0.15.0.
- [x] Config conforms to `QuantConfigured` (already had `quant`).
- [x] Split declared, re-measured via `ddcolor-package-smoke` (1024×512, fp16, best), provenance comments kept; MEMORY-REPORT.md left as the detailed per-tier record.
- [x] Smoke green; valid output captured (license admitted, non-trivial PNG).
- [x] BudgetAware deferred (note).
- [x] Update the DDColor row in `mlx-engine-swift/docs/model-registry.md`: Eff ✅, Eng 0.15.0.

## Outcome (executed 2026-06-30)
P0+QuantConfigured+P1 done as above. The activation-dominated single-footprint model (no per-mode split,
unlike BiRefNet) is preserved; the flat 4 GB is split into a 512 MB resident floor + shared ~1.8 GB
transient. `ddcolor-package-smoke` extended with floor/peak + `--quant` for repeatable measurement.
