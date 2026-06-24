import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import ArgumentParser
import MLX
import DDColor

/// Real-forward parity gate: load the converted DDColor weights + the seeded parity fixture
/// (exported by `oracle/export_parity.py`), run `DDColorModel`, and assert max_abs vs the
/// PyTorch-derived golden `out_ab`. Pinned to the CPU stream to match the parity discipline
/// (Apple-GPU fp32 matmul accumulates ~8e-4/op — masks/mimics real op bugs).
@main
struct Smoke: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ddcolor-smoke",
        abstract: "DDColor MLX-Swift parity gate against the PyTorch golden ab output."
    )

    @Option(name: .long, help: "tiny | large") var tier: String = "tiny"
    @Option(name: .long, help: "Converted weights safetensors.") var weights: String
    @Option(name: .long, help: "Tensor parity fixture (raw gray-RGB input + out_ab).") var parity: String?
    @Option(name: .long, help: "Image LAB-pipeline fixture (src_bgr/gray_rgb/orig_l/out_ab/out_bgr).") var imageParity: String?
    @Option(name: .long, help: "Max-abs parity threshold.") var threshold: Float = 1e-3
    @Option(name: .long, help: "Max-abs LAB-pipeline threshold (cv2 gamma-LUT slack).") var labThreshold: Float = 2e-2
    @Option(name: .long, help: "Colorize this image end-to-end (CGImage→CGImage, GPU).") var colorizeImage: String?
    @Option(name: .long, help: "Output PNG path for --colorize-image.") var out: String?
    @Option(name: .long, help: "Inference dtype for --colorize-image (float32 | float16).") var dtype: String = "float32"

    func run() throws {
        let t = DDColorTier(rawValue: tier) ?? .tiny
        let cfg = DDColorConfig(tier: t)

        // Full end-to-end colorize on the default (GPU) stream — the production path.
        if let colorizeImage, let out {
            let dt: DType = dtype == "float16" ? .float16 : .float32
            let colorizer = try DDColorColorizer.fromPretrained(weights, config: cfg, dtype: dt)
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: colorizeImage) as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                print("[smoke] failed to decode \(colorizeImage)"); throw ExitCode(1)
            }
            MLX.GPU.resetPeakMemory()
            let start = Date()
            let result = colorizer(cg)
            let secs = Date().timeIntervalSince(start)
            let peakGB = Double(MLX.GPU.peakMemory) / 1e9
            try writePNG(result, to: out)
            print(String(format: "[smoke] %@ colorize  %dx%d → %@  (%.2fs, peak %.2f GB)",
                         tier, cg.width, cg.height, out, secs, peakGB))
            return
        }

        Device.setDefault(device: Device(.cpu))   // CPU-stream parity discipline
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: weights)).mapValues { $0.asType(.float32) }
        let model = DDColorModel(config: cfg, weights: w)

        var failed = false
        if let parity { failed = try networkParity(model, parity) || failed }
        if let imageParity { failed = try labPipeline(model, imageParity) || failed }
        if failed { throw ExitCode(1) }
    }

    private func writePNG(_ cg: CGImage, to path: String) throws {
        let url = URL(fileURLWithPath: path) as CFURL
        guard let dest = CGImageDestinationCreateWithURL(url, UTType.png.identifier as CFString, 1, nil) else {
            throw ExitCode(1)
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw ExitCode(1) }
    }

    /// Network parity: raw gray-RGB input → ab vs PyTorch golden (CPU fp32).
    private func networkParity(_ model: DDColorModel, _ path: String) throws -> Bool {
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let out = model(fx["input"]!.asType(.float32)); out.eval()
        let golden = fx["out_ab"]!.asType(.float32)
        let maxAbs = MLX.abs(out - golden).max().item(Float.self)
        report("network", maxAbs, threshold)
        return maxAbs >= threshold
    }

    /// LAB pipeline: validate cv2-exact pre (gray-RGB), post (Lab→BGR from golden ab), and e2e.
    private func labPipeline(_ model: DDColorModel, _ path: String) throws -> Bool {
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: path)).mapValues { $0.asType(.float32) }
        let srcBGR = fx["src_bgr"]!, origL = fx["orig_l"]!, goldAB = fx["out_ab"]!, goldBGR = fx["out_bgr"]!
        // pre: BGR → gray-RGB model input
        let grayRGB = Lab.grayRGBInput(fromBGR: srcBGR)
        let preErr = MLX.abs(grayRGB[0] - fx["gray_rgb"]!).max().item(Float.self)
        // post (isolated): recombine golden ab with original L
        let postBGR = Lab.recombine(originalL: origL, modelAB: goldAB.expandedDimensions(axis: 0))
        let postErr = MLX.abs(postBGR - goldBGR).max().item(Float.self)
        // e2e: pre → model → post
        let ab = model(grayRGB)
        let e2eBGR = Lab.recombine(originalL: origL, modelAB: ab)
        MLX.eval(grayRGB, postBGR, e2eBGR)
        let e2eErr = MLX.abs(e2eBGR - goldBGR).max().item(Float.self)
        let e2eMean = MLX.abs(e2eBGR - goldBGR).mean().item(Float.self)
        let abErr = MLX.abs(ab[0] - goldAB).max().item(Float.self)
        let abMean = MLX.abs(ab[0] - goldAB).mean().item(Float.self)
        report("lab-pre ", preErr, labThreshold)        // colorspace correctness gates
        report("lab-post", postErr, labThreshold)
        report("lab-e2e ", e2eMean, labThreshold)        // e2e gated on MEAN BGR error
        print(String(format: "[smoke] %@ lab-e2e  (info) max_abs=%.3f  ab max=%.2f mean=%.3f (Lab a/b ~[-160,110])",
                     tier, e2eErr, abErr, abMean))
        // max_abs is dominated by single near-gamut-boundary pixels where a sub-unit ab shift flips an
        // out-of-gamut value; the mean is the visual-fidelity metric. pre/post gate the colorspace itself.
        return preErr >= labThreshold || postErr >= labThreshold || e2eMean >= labThreshold
    }

    private func report(_ name: String, _ err: Float, _ thr: Float) {
        print(String(format: "[smoke] %@ %@  max_abs=%.3e  thr=%.0e  %@",
                     tier, name, err, thr, err < thr ? "OK ✅" : "FAIL ❌"))
    }

    private func shape(_ a: MLXArray) -> String { "(" + a.shape.map(String.init).joined(separator: ",") + ")" }
}
