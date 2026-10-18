import XCTest
@testable import LingXiStudio

final class MaskDocumentTests: XCTestCase {
    @MainActor
    func testSameImageKeepsStrokesAndUndoHistory() {
        let mask = paintedMask()
        let pixels = mask.rgbaImage()!.pixels
        mask.prepare(width: 64, height: 64, imageID: "first")
        XCTAssertEqual(mask.rgbaImage()!.pixels, pixels)
        XCTAssertTrue(mask.canUndo)
        mask.undo()
        XCTAssertTrue(mask.isEmpty)
        mask.redo()
        XCTAssertEqual(mask.rgbaImage()!.pixels, pixels)
    }

    @MainActor
    func testDifferentImageOfSameSizeClearsStrokesAndHistory() {
        let mask = paintedMask()
        mask.prepare(width: 64, height: 64, imageID: "second")
        XCTAssertTrue(mask.isEmpty)
        XCTAssertFalse(mask.canUndo)
        XCTAssertFalse(mask.canRedo)
        XCTAssertEqual(mask.coverage, 0)
    }

    @MainActor
    func testRemovingImageClearsCanvas() {
        let mask = paintedMask()
        mask.prepare(width: 0, height: 0, imageID: nil)
        XCTAssertFalse(mask.hasCanvas)
        XCTAssertTrue(mask.isEmpty)
        XCTAssertNil(mask.rgbaImage())
        XCTAssertFalse(mask.canUndo)
    }

    @MainActor
    private func paintedMask() -> MaskDocument {
        let mask = MaskDocument()
        mask.prepare(width: 64, height: 64, imageID: "first")
        mask.brushSize = 16
        mask.beginStroke()
        mask.dab(at: .init(x: 32, y: 32))
        XCTAssertFalse(mask.isEmpty)
        return mask
    }
}
