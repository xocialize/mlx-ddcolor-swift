import Foundation
import MLXToolKit

/// Configuration for the DDColor `imageColorize` package. Three converted weight sets back the three
/// tier Modes — `ColorizeContract.fast` = convnext-t · `.best` = convnext-l (modelscope) · `.artistic`
/// = the artistic convnext-l checkpoint. fast loads the tiny arch; best/artistic share the large arch
/// (zero code difference — only `DDColorTier` + weights change). Weights resolve under the engine's
/// model store (`modelsRootDirectory` + repo dir + `weightsFile`).
public struct DDColorConfiguration: PackageConfiguration, ModelStorable {
    /// HF repo id for the converted MLX `.safetensors` backing each tier.
    public var fastRepo: String
    public var bestRepo: String
    public var artisticRepo: String
    /// Weights filename within each repo's local snapshot directory.
    public var weightsFile: String
    /// Inference dtype — also the declared footprint quant. fp16 is the validated runtime.
    public var quant: Quant
    /// Engine-injected model-store root (download dir + security-scoped bookmark).
    public var modelsRootDirectory: URL?
    /// Direct paths to weights, bypassing model-store resolution (pre-resolved callers + CLI smoke).
    public var fastWeightsURL: URL?
    public var bestWeightsURL: URL?
    public var artisticWeightsURL: URL?

    public init(fastRepo: String = "mlx-community/DDColor-paper-tiny-fp16",
                bestRepo: String = "mlx-community/DDColor-modelscope-fp16",
                artisticRepo: String = "mlx-community/DDColor-artistic-fp16",
                weightsFile: String = "model.safetensors",
                quant: Quant = .fp16,
                modelsRootDirectory: URL? = nil,
                fastWeightsURL: URL? = nil,
                bestWeightsURL: URL? = nil,
                artisticWeightsURL: URL? = nil) {
        self.fastRepo = fastRepo
        self.bestRepo = bestRepo
        self.artisticRepo = artisticRepo
        self.weightsFile = weightsFile
        self.quant = quant
        self.modelsRootDirectory = modelsRootDirectory
        self.fastWeightsURL = fastWeightsURL
        self.bestWeightsURL = bestWeightsURL
        self.artisticWeightsURL = artisticWeightsURL
    }
}

/// `QuantConfigured` (engine 1.14): the config already stores `var quant`, so the `MemoryGovernor`
/// charges the matching declared `QuantFootprint` (fp16 today) for the selected quant instead of the
/// largest-that-fits heuristic. Trivial empty conformance.
extension DDColorConfiguration: QuantConfigured {}
