import Foundation
import ArgumentParser
import MLX
import MLXToolKit
import MLXDDColor

/// Drive the conformant `DDColorPackage` exactly as the engine would: check the license gate against
/// the manifest, construct via `init(configuration:)`, `load()`, then `run(ColorizeRequest)` and write
/// the resulting `Image`. Proves the package envelope (not just the core) does a real forward.
@main
struct PackageSmoke: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ddcolor-package-smoke",
        abstract: "Drive DDColorPackage through load()/run() with a real image.")

    @Option(name: .long, help: "Local converted weights (best/large tier) — bypasses HF download.")
    var weights: String
    @Option(name: .long, help: "Input image path.") var image: String
    @Option(name: .long, help: "Output PNG path.") var out: String
    @Option(name: .long, help: "fast | best | artistic") var mode: String = "best"
    @Option(name: .long, help: "fp16 | fp32 — inference + declared-footprint quant.") var quant: String = "fp16"

    func run() async throws {
        // C7/C8 license gate — the engine admits the package only if both layers are permissive.
        let decl = DDColorPackage.manifest.license
        let gate = LicensePolicy.permissiveOnly.evaluate(decl)
        print("[pkg] license weight=\(decl.weightLicense) port=\(decl.portCodeLicense) → \(gate)")
        guard gate.isAdmitted else { throw ExitCode(1) }

        let q: Quant = quant == "fp32" ? .fp32 : .fp16
        let cfg = DDColorConfiguration(quant: q, bestWeightsURL: URL(fileURLWithPath: weights))
        let pkg = DDColorPackage(configuration: cfg)
        try await pkg.load()

        let data = try Data(contentsOf: URL(fileURLWithPath: image))
        let req = ColorizeRequest(image: Image(format: .jpeg, data: data),
                                  mode: Mode(rawValue: mode))
        MLX.GPU.resetPeakMemory()
        let start = Date()
        let resp = try await pkg.run(req)
        let secs = Date().timeIntervalSince(start)
        guard let cr = resp as? ColorizeResponse else { throw ExitCode(1) }
        try cr.image.data.write(to: URL(fileURLWithPath: out))
        let peakMB = Double(MLX.GPU.snapshot().peakMemory) / 1_048_576
        print(String(format: "[pkg] %@ %@ run → %dx%d %@  (%.2fs) | peakGPU %.0f MB → %@",
                     mode, quant, cr.image.width ?? 0, cr.image.height ?? 0,
                     cr.image.format.rawValue, secs, peakMB, out))

        // --- Memory report (this variant; one process = a clean peak). Methodology per the
        // memory-harness: peak active during the real forward + resident floor (active after freeing
        // activations) + the device's recommendedWorkingSet ceiling. Recommend = peak×1.2 + 256 MB.
        MLX.GPU.clearCache()
        let floorMB = Double(MLX.GPU.snapshot().activeMemory) / 1_048_576
        let workingSetMB = Double(MLX.GPU.deviceInfo().maxRecommendedWorkingSetSize) / 1_048_576
        let recommendMB = peakMB * 1.2 + 256
        print(String(format: "MEM %@ %@ | floor %.0f MB · peak %.0f MB · activation %.0f MB · "
            + "recommend %.0f MB (×1.2+256) | workingSet %.0f MB · exceeds %@", mode, quant, floorMB,
            peakMB, max(0, peakMB - floorMB), recommendMB, workingSetMB,
            recommendMB > workingSetMB ? "YES" : "no"))
    }
}
