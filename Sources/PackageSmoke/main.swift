import Foundation
import ArgumentParser
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

    func run() async throws {
        // C7/C8 license gate — the engine admits the package only if both layers are permissive.
        let decl = DDColorPackage.manifest.license
        let gate = LicensePolicy.permissiveOnly.evaluate(decl)
        print("[pkg] license weight=\(decl.weightLicense) port=\(decl.portCodeLicense) → \(gate)")
        guard gate.isAdmitted else { throw ExitCode(1) }

        let cfg = DDColorConfiguration(quant: .fp32, bestWeightsURL: URL(fileURLWithPath: weights))
        let pkg = DDColorPackage(configuration: cfg)
        try await pkg.load()

        let data = try Data(contentsOf: URL(fileURLWithPath: image))
        let req = ColorizeRequest(image: Image(format: .jpeg, data: data),
                                  mode: Mode(rawValue: mode))
        let start = Date()
        let resp = try await pkg.run(req)
        let secs = Date().timeIntervalSince(start)
        guard let cr = resp as? ColorizeResponse else { throw ExitCode(1) }
        try cr.image.data.write(to: URL(fileURLWithPath: out))
        print(String(format: "[pkg] %@ run → %dx%d %@  (%.2fs) → %@",
                     mode, cr.image.width ?? 0, cr.image.height ?? 0,
                     cr.image.format.rawValue, secs, out))
    }
}
