// swift-tools-version: 6.2
// mlx-ddcolor-swift — DDColor automatic image colorization on Swift/MLX, destined for an MLXEngine
// `imageColorize` ModelPackage (Forge "Colorize" capability). The `DDColor` core is a from-scratch
// architecture port (no MLX donor) of piddnad/DDColor (Apache-2.0); the `MLXDDColor` wrapper (added in
// the engine-integration phase) provides the conformant ModelPackage. See ARCHITECTURE.md / PORT-STATUS.md.
import PackageDescription

let mlxCore: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"),
]

let package = Package(
    name: "mlx-ddcolor-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "DDColor", targets: ["DDColor"]),              // MLX core (mlx-swift only)
        .library(name: "MLXDDColor", targets: ["MLXDDColor"]),        // engine-consumable ModelPackage
        .executable(name: "ddcolor-smoke", targets: ["DDColorSmoke"]), // real-forward parity gate
        .executable(name: "ddcolor-package-smoke", targets: ["DDColorPackageSmoke"]),  // drive the ModelPackage
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),  // Hub download
        // MLXToolKit contract (imageColorize @ 1.7.0); 0.27.0 = the CAN cancellation gate
        // (MLXServeConformance CAN-1..3). Use the released tag — a `revision:` pin can't be
        // reconciled with the version range the sibling wrappers use, which jams a consuming
        // app's graph (APP-VALIDATION BRIDGE-026).
        // ≥ 0.32.0: the engine materializes declared WeightSources before load() (contract 1.24) and probes their flat layout.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.32.0"),
        // Shared env-gated profiler (MLX_PROFILE=1); zero overhead when unset.
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "DDColor",
            dependencies: mlxCore,
            path: "Sources/DDColor",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Engine wrapper — the conformant `imageColorize` ModelPackage.
        .target(
            name: "MLXDDColor",
            dependencies: [
                "DDColor",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
            ],
            path: "Sources/MLXDDColor"
        ),
        .executableTarget(
            name: "DDColorSmoke",
            dependencies: ["DDColor", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/Smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Drives the conformant DDColorPackage through register-shaped load()/run() (the first real
        // forward through the package envelope: license gate → ColorizeRequest dispatch → Image out).
        .executableTarget(
            name: "DDColorPackageSmoke",
            dependencies: [
                "MLXDDColor",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/PackageSmoke"
        ),
        .testTarget(
            name: "MLXDDColorTests",
            dependencies: [
                "MLXDDColor",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),  // CAN gate
            ],
            path: "Tests/MLXDDColorTests"
        ),
    ]
)
