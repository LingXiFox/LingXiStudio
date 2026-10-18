import AppKit
import XCTest
@testable import LingXiStudio

final class FocusDismissalTests: XCTestCase {
    @MainActor
    func testClickInsideTextEditorKeepsFocus() {
        let (window, editor) = textEditorWindow()
        defer { window.close() }
        AppDelegate.dismissTextFocus(in: window, at: NSPoint(x: 80, y: 80))
        XCTAssertTrue(window.firstResponder === editor)
    }

    @MainActor
    func testBlankClickDismissesTextEditor() {
        let (window, editor) = textEditorWindow()
        defer { window.close() }
        AppDelegate.dismissTextFocus(in: window, at: NSPoint(x: 350, y: 250))
        XCTAssertFalse(window.firstResponder === editor)
    }

    @MainActor
    func testBlankClickCommitsFieldEditorValue() {
        _ = NSApplication.shared
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 400, height: 300), styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        defer { window.close() }
        let field = NSTextField(frame: NSRect(x: 20, y: 20, width: 120, height: 24))
        window.contentView!.addSubview(field)
        XCTAssertTrue(window.makeFirstResponder(field))
        let editor = window.firstResponder as! NSTextView
        editor.string = "1024"
        AppDelegate.dismissTextFocus(in: window, at: NSPoint(x: 350, y: 250))
        XCTAssertFalse(window.firstResponder === editor)
        XCTAssertEqual(field.stringValue, "1024")
    }

    @MainActor
    private func textEditorWindow() -> (NSWindow, NSTextView) {
        _ = NSApplication.shared
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 400, height: 300), styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let scroll = NSScrollView(frame: NSRect(x: 20, y: 20, width: 200, height: 160))
        let editor = NSTextView(frame: scroll.bounds)
        scroll.documentView = editor
        window.contentView!.addSubview(scroll)
        XCTAssertTrue(window.makeFirstResponder(editor))
        return (window, editor)
    }
}
