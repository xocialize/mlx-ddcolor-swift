import Foundation
import MLXToolKit

/// Configuration for the DDColor `imageColorize` package. Three converted weight sets back the three
/// tier Modes — `ColorizeContract.fast` = convnext-t · `.best` = convnext-l (modelscope) · `.artistic`
/// = the artistic convnext-l checkpoint. fast loads the tiny arch; best/artistic share the large arch
/// (zero code difference — only `DDColorTier` + weights change). All three repos are declared `weightSources`,
/// which the engine downloads into its model store before `load()`.
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

/// The weights the engine materializes before `load()` (contract 1.24): every tier's checkpoint, since a request can
/// name any Mode (ForgeCore's colorize provider maps fast / best / artistic). The engine sizes, previews and downloads
/// them (resumably) instead of the package fetching each on first use: 110 MB tiny + 2 × 456 MB large.
extension DDColorConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "fast", repo: fastRepo, matching: [weightsFile]),
         WeightSource(role: "best", repo: bestRepo, matching: [weightsFile]),
         WeightSource(role: "artistic", repo: artisticRepo, matching: [weightsFile])]
    }

    /// The explicit weights URLs first, then the store's layouts, including where 0.2.x downloaded them.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        defaultMissingWeightSources(storeRoot: storeRoot).filter { source in
            let override = switch source.role {
            case "fast": fastWeightsURL
            case "artistic": artisticWeightsURL
            default: bestWeightsURL
            }
            if let override, FileManager.default.fileExists(atPath: override.path) { return false }
            return !StoreWeights.installedByHubAPI(weightsFile, repo: source.repo, root: storeRoot)
        }
    }
}
