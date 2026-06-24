# mlx-ddcolor-swift — Port Status

DDColor (piddnad/DDColor, **Apache-2.0**) → MLX-Swift, for an MLXEngine `imageColorize`
ModelPackage feeding the Forge **Colorize** capability. From-scratch architecture port (no MLX
donor). Direct PyTorch → MLX-Swift; full architecture validated in throwaway MLX-Python first.

## ✅ DONE & VERIFIED (Phase 1 network port)

| Step | Result |
|---|---|
| Architecture read + mapped | `ARCHITECTURE.md` (every module, shape, config, parity trap) |
| Oracle env + granular goldens | `oracle/` venv (torch 2.12 / mlx 0.31.2 / cv2); `oracle/goldens/{tiny,large}/` per-sub-op `.npy` |
| Full MLX-Python parity (de-risk) | `oracle/mlx_ref.py` — **all** intermediates ≤ 5e-5 vs PyTorch, both tiers (CPU fp32) |
| **Weight conversion** | `oracle/convert.py` — resolves spectral/weight-norm → effective weights, NHWC transpose, drops unused head; emits canonical safetensors. **Verified reproduces parity from disk.** |
| Swift package scaffold | `Package.swift` (DDColor core = mlx-swift only; Smoke gate) — **builds clean** |
| Swift core port | `Sources/DDColor/` — `DDColorModel` + `DDColorOps` transcribed 1:1 from the verified reference |
| **Swift parity gate** | `ddcolor-smoke`: **tiny max_abs 3.3e-6 · large 1.5e-6** vs golden ab (CPU fp32) ✅ |
| Normalize inside model | public input = gray-RGB `[0,1]` (matches reference); parity re-verified ✅ |
| **cv2-exact LAB pipeline** | `Lab.swift` + `DDColorColorizer` (CGImage→CGImage, resize); validated pre 1.9e-3 / post 7.9e-5 / e2e mean 9e-4 vs PyTorch `ColorizationPipeline`; native-res colorize verified on real archival photos ✅ |
| **`imageColorize` contract** | added to MLXToolKit (contract **1.7.0**): `Capability.imageColorize` + `Colorize.swift` (Request/Response/Contract; fast/best/artistic modes) ✅ |
| **`MLXDDColor` ModelPackage** | `DDColorPackage` (@InferenceActor) + `DDColorConfiguration`; Apache/MIT license gate; fp16 4 GB measured footprint; `ddcolor-package-smoke` drives load()/run() → coherent PNG ✅ |
| **Memory report + C0–C13** | `MEMORY-REPORT.md` (fp16 tiny 2.1 / large 2.45 GB) + `CONFORMANCE.md` (C0–C13 **PASS**) ✅ |

Reproduce:
```bash
cd oracle && source .venv/bin/activate
python gen_goldens.py --size tiny && python gen_goldens.py --size large
python convert.py --size tiny && python convert.py --size large
python export_parity.py --size tiny && python export_parity.py --size large
python mlx_ref.py --size tiny --weights weights/ddcolor_tiny.safetensors   # MLX-Python parity
cd .. && swift run -c release ddcolor-smoke --tier tiny \
  --weights oracle/weights/ddcolor_tiny.safetensors --parity oracle/weights/parity_tiny.safetensors
```

## ▢ REMAINING

1. **Publish** (external — needs HF/GitHub credentials + your go-ahead): convert tiny/large/artistic to
   fp16 safetensors → `mlx-community/DDColor-*-fp16` (Collection); code → `xocialize/mlx-ddcolor-swift`.
   Until then `DDColorConfiguration` accepts direct `*WeightsURL` overrides (CLI smoke uses this).
2. **Artistic tier weights**: `ddcolor_artistic.pth` (same convnext-l arch) — run `convert.py` (add the
   artistic entry) + a footprint check. Contract/package already declare the `artistic` mode.
3. **Stage 2 — engine app integration** (`mlx-swift-integration` Stage 2): register `DDColorPackage`
   through `MLXServeEngine` in a consuming app, drive register→prepare→run via the shared harness.
4. **Forge Phase 2** (`COLORIZATION-PLAN.md`, separate repo): `ColorizeKit`/`ColorizeUI` capability
   (`ColorizeProvider` seam) + `EngineColorizeProvider` injecting `MLXDDColor`. Video = per-frame + SEA-RAFT.

int4 deliberately skipped — peak is activation-dominated, not weight-dominated (see MEMORY-REPORT.md).

## Forge wiring (Phase 2, separate — `COLORIZATION-PLAN.md`)
`ColorizeKit` (`ColorizeProvider` seam, mirrors `ExtractKit`'s `MatteProvider`) + `ColorizeUI`
(`ColorizeCapability`) + `EngineColorizeProvider` injecting `MLXDDColor`. Video (Phase 3) reuses the
per-frame + SEA-RAFT recipe.

## Notes / decisions
- **Canonical converter is Python** (not a Swift `Convert` target like BiRefNet): spectral-norm
  resolution needs torch power-iteration on `weight_orig/u/v`. Swift consumes the resolved safetensors.
- **Functional core** (weights dict + free-function forward) mirrors the verified reference for lowest
  parity risk. Idiomatic `Module`-tree refactor is a *post-parity* option (skill: match first).
- Released checkpoints use `num_queries=100`, `num_output_channels=2`, `MultiScaleColorDecoder`
  (the plan's "256 queries" was wrong — matched the reference).
- `oracle/` is a throwaway scratchpad (gitignored weights/goldens/venv).
