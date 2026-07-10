// CancellationTests.swift — DDColor through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before capability
// validation, weights, or image decode); CAN-3 is the document of record for the checkpoint
// cadence. DDColor is a CAN-3 judgment case: the forward is ONE monolithic MLX eval (fixed 512²
// forward + native-res LAB recombine, self-evaled before CGImage materialization — no tiling, no
// iterative loop, fast tier ~0.1 s) — but the RE-BASELINED manifest declares peakActivationBytes
// 4.54 GB ≥ 2 GB, so the manifest is long-run implied and the sub-second exemption is NOT
// available (the original smoke's 1.8 GB was an MLX-peak under-read). The declared cadence
// therefore names the REAL seams run() has, once per colorized frame each:
//   • entry checkpoint (first act of run(), DDColorPackage.run)
//   • pre-forward, after the lazy tier build — a first-request weight download can precede
//     the forward (DDColorPackage.run, after `colorizer(for:)`)
//   • post-forward / pre-PNG-encode (DDColorPackage.run, after endRun)
// No per-step checkpoints are fabricated — there is no loop to hang them on.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXDDColor

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = DDColorPackage(configuration: DDColorConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ColorizeRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // peakActivationBytes 4.54 GB ≥ 2 GB ⇒ long-run implied by the declared envelope,
        // even though the forward itself completes in well under a second on the fast tier.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: DDColorPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: DDColorPackage.manifest,
            posture: .cadence([
                // Pre-forward seam: once per frame, after the lazy tier build / possible weight
                // download, before committing to the monolithic eval.
                .init(phase: .encode, unit: .frame),
                // Post-forward seam: once per frame, between materialization and PNG encode.
                .init(phase: .postprocess, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
