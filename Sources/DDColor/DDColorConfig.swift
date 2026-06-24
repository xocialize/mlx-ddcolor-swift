import Foundation

/// DDColor tier. Only the ConvNeXt `depths`/`dims` differ; the decoder + color-decoder
/// (nf=512, num_queries=100, dec_layers=9, num_scales=3) are tier-invariant.
public enum DDColorTier: String, Sendable, CaseIterable {
    case tiny          // convnext-t  (ddcolor_paper_tiny)  — fast
    case large         // convnext-l  (ddcolor_modelscope)  — best/default

    public var depths: [Int] { self == .tiny ? [3, 3, 9, 3] : [3, 3, 27, 3] }
    public var dims: [Int] { self == .tiny ? [96, 192, 384, 768] : [192, 384, 768, 1536] }
}

/// Fixed model hyperparameters shared by every released DDColor colorization checkpoint.
/// Matches the PyTorch oracle (piddnad/DDColor) exactly — see ARCHITECTURE.md.
public struct DDColorConfig: Sendable {
    public var tier: DDColorTier
    public var inputSize: Int = 512          // model operates at S×S; pipeline resizes around it
    public let numQueries = 100
    public let decLayers = 9
    public let numScales = 3
    public let hiddenDim = 256
    public let numHeads = 8
    public let nf = 512
    public init(tier: DDColorTier, inputSize: Int = 512) {
        self.tier = tier
        self.inputSize = inputSize
    }
}
