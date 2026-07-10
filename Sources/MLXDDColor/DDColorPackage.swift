import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXProfiling
import Hub
import DDColor

/// The conformant `imageColorize` ModelPackage over the from-scratch DDColor core. One package, three
/// tier Modes dispatched in `run(_:)` on `request.mode` (`fast` → convnext-t · `best` → convnext-l /
/// modelscope · `artistic` → the artistic checkpoint); each returns a colorized `Image` at the source
/// resolution. The engine constructs / loads / evicts it (C13) — it never constructs itself.
@InferenceActor
public final class DDColorPackage: ModelPackage {
    public typealias Configuration = DDColorConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // DDColor weights Apache-2.0 (piddnad/DDColor); our Swift port code MIT. Both permissive (C7/C8).
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/DDColor-modelscope-fp16", revision: "main", tier: 2),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14). All three tiers are **activation-dominated** (fixed 512²
                // model + native-res LAB colorspace), close, and consumer-friendly — so a single fp16
                // RE-BASELINED to in-app phys_footprint (MLXEngineImage, isolate+clearCache, engine 0.17.0
                // post-load floor). The `ddcolor-package-smoke` MLX-peak (best/convnext-l: floor 435 MB,
                // activation 1.8 GB) under-read the admission-relevant phys: the in-app `ddcolor-fast` (tiny
                // tier) measured floor 0.10 GB · peak 4.64 GB → activation 4.54 GB — already 2.5× the smoke
                // *best* estimate, despite being the LIGHTER tier. The old-harness "5.4 GB floor" was post-run
                // retention (BOUNDED — flat across 3 resident passes, freed on evict → belongs in the
                // transient, not residency). residentBytes kept at 512 MB (best/convnext-l weights, the
                // heavier default-load tier — covers all modes); peakActivationBytes raised to the in-app
                // fast number. ⚠️ FLAGGED: the BEST tier (load() default) is not yet measured in-app and is
                // likely ≥ this — re-measure a `ddcolor-best` in-app run and bump if higher (under-declaring
                // falsely admits on tight Macs). `QuantConfigured` charges the fp16 footprint.
                footprints: [QuantFootprint(quant: .fp16, residentBytes: 512_000_000, peakActivationBytes: 4_540_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0))
            ),
            surfaces: [
                ColorizeContract.descriptor(
                    name: "ddcolor",
                    summary: "DDColor automatic colorization → colorized image at source resolution. "
                        + "fast = convnext-t (~0.1s) · best = convnext-l/modelscope · artistic = stylized.",
                    modes: [ColorizeContract.fast, ColorizeContract.best, ColorizeContract.artistic])
            ])
    }

    private let configuration: Configuration
    private var fast: DDColorColorizer?
    private var best: DDColorColorizer?
    private var artistic: DDColorColorizer?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Warm the default (`best`) tier; other tiers build lazily on first request.
    public func load() async throws {
        if best == nil { best = try await build(.best) }
    }

    public func unload() async {
        fast = nil; best = nil; artistic = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before capability validation
        // (engine ≥ 0.27.0). Mid-run: the forward is ONE monolithic MLX eval (fixed 512² forward
        // + native-res LAB recombine, no iterative loop); the real seams are pre-forward (after
        // the lazy tier build / possible weight download) and post-forward/pre-encode below —
        // see CancellationTests for the cadence of record.
        try Task.checkCancellation()
        guard request.capability == .imageColorize, let req = request as? ColorizeRequest else {
            throw DDColorError.unsupportedCapability(request.capability)
        }
        let colorizer = try await colorizer(for: req.mode)
        // Pre-forward checkpoint: the tier build above can include a first-request weight
        // download, so re-check before committing to the forward.
        try Task.checkCancellation()
        let input = try Self.decode(req.image)
        // Single-shot inference is profiled (MLX_PROFILE=1). The colorizer self-evals its output
        // (fixed 512² forward + native-res LAB recombine), so the coarse region times it honestly;
        // phys/⚠PAGING readings flag when a huge native-res recombine nears the working-set ceiling.
        let prof = MLXProfiler.shared
        prof.beginRun("ddcolor imageColorize mode=\(req.mode?.rawValue ?? "best") \(input.width)x\(input.height)")
        let output = prof.region("colorize", "forward") {
            colorizer(input)                                // preprocess → forward → recombine → CGImage
        }
        prof.endRun()   // one image per run — no per-unit denominator
        try Task.checkCancellation()
        let png = try Self.encodePNG(output)
        return ColorizeResponse(image: Image(format: .png, data: png,
                                             width: output.width, height: output.height))
    }

    // MARK: - Colorizer construction (one per tier, lazy)

    private enum Tier { case fast, best, artistic }

    private func colorizer(for mode: Mode?) async throws -> DDColorColorizer {
        switch mode {
        case ColorizeContract.fast?:
            if fast == nil { fast = try await build(.fast) }
            return fast!
        case ColorizeContract.artistic?:
            if artistic == nil { artistic = try await build(.artistic) }
            return artistic!
        default:   // nil or "best" → the recommended general tier
            if best == nil { best = try await build(.best) }
            return best!
        }
    }

    private func build(_ tier: Tier) async throws -> DDColorColorizer {
        let ddTier: DDColorTier = tier == .fast ? .tiny : .large
        let url = try await weightsURL(tier)
        return try DDColorColorizer.fromPretrained(url.path,
                                                   config: DDColorConfig(tier: ddTier),
                                                   dtype: Self.dtype(configuration.quant))
    }

    /// Resolve the weights file, downloading from HF on first use via swift-transformers `HubApi`
    /// (pointed at the engine-stamped model-store root; idempotent). A direct override URL bypasses
    /// the network entirely (pre-resolved caller / CLI smoke).
    private func weightsURL(_ tier: Tier) async throws -> URL {
        let override: URL?
        let repo: String
        switch tier {
        case .fast: override = configuration.fastWeightsURL; repo = configuration.fastRepo
        case .best: override = configuration.bestWeightsURL; repo = configuration.bestRepo
        case .artistic: override = configuration.artisticWeightsURL; repo = configuration.artisticRepo
        }
        if let override {
            guard FileManager.default.fileExists(atPath: override.path) else {
                throw DDColorError.weightsMissing(override)
            }
            return override
        }
        let hub = HubApi(downloadBase: configuration.modelsRootDirectory)
        let dir = try await hub.snapshot(from: repo, matching: ["*.safetensors"]) { @Sendable progress in
            WeightDownloadProgress.report(fraction: progress.fractionCompleted)
        }
        let url = dir.appendingPathComponent(configuration.weightsFile)
        guard FileManager.default.fileExists(atPath: url.path) else { throw DDColorError.weightsMissing(url) }
        return url
    }

    private nonisolated static func dtype(_ q: Quant) -> DType {
        switch q {
        case .fp32: return .float32
        case .bf16: return .bfloat16
        default:    return .float16
        }
    }

    // MARK: - Image ⇄ canonical artifact

    private nonisolated static func decode(_ image: Image) throws -> CGImage {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw DDColorError.decodeFailed
        }
        return cg
    }

    private nonisolated static func encodePNG(_ cg: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw DDColorError.encodeFailed
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw DDColorError.encodeFailed }
        return data as Data
    }

    public enum DDColorError: Error {
        case unsupportedCapability(Capability)
        case weightsMissing(URL)
        case decodeFailed
        case encodeFailed
    }
}

public extension DDColorPackage {
    /// Engine registration entry (license/eligibility gate at register; weights lazy on first run).
    nonisolated static var registration: PackageRegistration { .of(DDColorPackage.self) }
}
