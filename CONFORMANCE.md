# DDColor `imageColorize` ModelPackage — C0–C13 conformance review

Reviewed 2026-06-23 against `EngineeringDocs/MLXEngineDocs/conformance.md` (contract **1.7.0**, which
introduces `.imageColorize`). Manual reviewer pass; evidence cites
`Sources/MLXDDColor/{DDColorPackage,DDColorConfiguration}.swift`. Parity is **not** a conformance item
(owned by mlx-porting) — DDColor network parity-locked CPU fp32 (tiny 3.3e-6 · large 1.4e-6 vs PyTorch),
LAB pipeline validated (pre 1.9e-3 / post 7.9e-5 / e2e mean 9e-4), package run live-verified
(`ddcolor-package-smoke` → coherent colorized PNG).

**Verdict: PASS (C0–C13).** Offline contract build green; package envelope (license gate → load → run →
Image) live-verified on real images.

| C | Item | Verdict | Evidence |
|---|---|---|---|
| **C0** | Contract version | ✅ | `manifest.contractVersion` defaults to `ContractVersion.current` (1.7.0 — the version that introduced `.imageColorize`). |
| **C1** | Capability registration | ✅ | One canonical capability `.imageColorize`; one `surfaces` entry (`ColorizeContract.descriptor`), one model. No fusion. |
| **C2** | Canonical schema | ✅ | I/O = canonical `ColorizeRequest` (`image`, `mode`) → `ColorizeResponse(image:)`; output artifact `.image`. Contract's own types verbatim. |
| **C3** | Canonical artifact I/O | ✅ | Input `Image` (PNG/JPEG bytes) → output `Image` (PNG bytes), serialized round-trip; no live-tensor/IOSurface fork. |
| **C4** | Mode-as-parameter | ✅ | `fast`/`best`/`artistic` are `ColorizeContract` **Modes** on `req.mode`; `run()` dispatches on them (same input artifact, different weights). NOT separate surfaces. |
| **C5** | metaData hygiene | ✅ | `metaData` unused; the tier is a **canonical Mode**, not smuggled. |
| **C6** | Specialty declaration | ✅ | `specialties: []` — none claimed (no governed-vocab term applies). |
| **C7** | Weight license gate | ✅ | `weightLicense: .apache2` (piddnad/DDColor Apache-2.0) — permissive; `LicensePolicy.permissiveOnly.evaluate` → `.admitted` (verified live in `ddcolor-package-smoke`). |
| **C8** | Port-code license gate | ✅ | `portCodeLicense: .mit` (from-scratch Swift port + ConvNeXt block adapted from facebookresearch/ConvNeXt MIT) — permissive. |
| **C9** | PackageConfiguration | ✅ | `DDColorConfiguration: PackageConfiguration, ModelStorable` (Codable+Sendable); session-stable values (repos/quant/weight URLs/store root). Per-request `mode`/`image` ride the request. |
| **C10** | Requirements manifest | ✅ | `footprints: [.fp16 4.0 GB]` (empirically measured — see MEMORY-REPORT.md), `requiredBackends: [.metalGPU]`, `minMacOS 26`. All three tiers are activation-dominated, close (2.1–2.45 GB), and consumer-friendly → one footprint covers all modes (no per-mode split / runtime guard needed). |
| **C11** | MCPBridge introspection | ✅ | Surface schema = canonical `ColorizeContract.descriptor(...)` (`ToolDescriptor` + `image` param + `supportedModes`) — introspectable. |
| **C12** | Forward-compat discipline | ✅ | No closed-`Capability` switch; `run()` guards `request.capability == .imageColorize` else-throws (`unsupportedCapability`). |
| **C13** | Runtime governance | ✅* | Engine-constructed via `PackageRegistration.of` + `nonisolated init` (no compute in init); `@InferenceActor`-isolated `load/run/unload`; `unload()` nils all three colorizers (working set freed); default tier warmed in `load()`, others lazy. *Cancellation checked at the forward boundaries (`Task.checkCancellation` before + after) — a single colorize forward is atomic (no intra-kernel boundary to preempt). |

## Notes
- **int4 deliberately skipped**: footprint is activation-dominated (fixed 512² model + native-res
  colorspace), not weight-dominated — int4 shrinks weights, not the activation working set, so it buys
  ~nothing here (large fp16 weights ≈ 0.4 GB of a 2.45 GB peak). fp16 is the shipping runtime.
- **Publishing pending**: converted fp16 weights ready locally; HF `mlx-community/DDColor-*-fp16` repos
  (referenced by `DDColorConfiguration` defaults) not yet uploaded — config accepts a direct
  `*WeightsURL` override in the interim.
