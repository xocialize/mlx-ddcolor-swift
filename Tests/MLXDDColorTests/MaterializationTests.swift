// MaterializationTests.swift — the engine materializes DDColor's three tier checkpoints before load() (contract 1.24,
// AB-T-0222). Offline: no network, no weights, no MLX kernels.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXDDColor

final class MaterializationTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "ddcolor-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private func place(_ directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appending(path: "model.safetensors")
        try Data([0]).write(to: file)
        return file
    }

    func testMATGate() throws {
        let urls = try ["fast", "best", "artistic"].map { try place(root.appending(path: "explicit/\($0)")) }
        let report = MaterializationConformance.check(
            freshConfiguration: DDColorConfiguration(),
            satisfiedConfiguration: DDColorConfiguration(fastWeightsURL: urls[0], bestWeightsURL: urls[1],
                                                         artisticWeightsURL: urls[2]))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testEveryTierIsDeclared() {
        let sources = DDColorConfiguration().weightSources
        XCTAssertEqual(sources.map(\.role), ["fast", "best", "artistic"])
        XCTAssertEqual(sources.map(\.repo), ["mlx-community/DDColor-paper-tiny-fp16", "mlx-community/DDColor-modelscope-fp16",
                                             "mlx-community/DDColor-artistic-fp16"])
        XCTAssertTrue(sources.allSatisfy { $0.matching == ["model.safetensors"] })
    }

    /// The engine's flat layout satisfies the probe, and the loader finds the file there.
    func testEngineFlatLayout() throws {
        let config = DDColorConfiguration(modelsRootDirectory: root)
        XCTAssertEqual(config.missingWeightSources(storeRoot: root).count, 3)
        let file = try place(ModelStore(root: root).directory(for: config.bestRepo)!)
        XCTAssertEqual(config.missingWeightSources(storeRoot: root).map(\.role), ["fast", "artistic"])
        XCTAssertEqual(StoreWeights.file("model.safetensors", repo: config.bestRepo, root: root)?.path, file.path)
    }

    /// 0.2.x downloaded through `HubApi(downloadBase:)` into `<root>/models/<org>/<name>/`: those installs read as
    /// present, so the engine doesn't download them again, and the loader uses them.
    func testPre030InstallIsNotDownloadedAgain() throws {
        let config = DDColorConfiguration(modelsRootDirectory: root)
        let best = try place(root.appending(path: "models/mlx-community/DDColor-modelscope-fp16"))
        XCTAssertEqual(config.missingWeightSources(storeRoot: root).map(\.role), ["fast", "artistic"])
        XCTAssertEqual(StoreWeights.file("model.safetensors", repo: config.bestRepo, root: root)?.path, best.path)
    }

    /// An explicit URL satisfies its own tier only.
    func testExplicitURLSatisfiesItsTier() throws {
        let fast = try place(root.appending(path: "explicit"))
        let config = DDColorConfiguration(modelsRootDirectory: root, fastWeightsURL: fast)
        XCTAssertEqual(config.missingWeightSources(storeRoot: root).map(\.role), ["best", "artistic"])
    }
}
