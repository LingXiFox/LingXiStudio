# GUI design and interaction acceptance

Date: 2026-10-18. Base commit: `6703050` plus the existing, uncommitted native UI redesign.

## Delivered behavior

- The provided `ICONs/icon.icon` is compiled by Apple `actool` during packaging; `Assets.car`, `icon.icns`, `CFBundleIconName` and `CFBundleIconFile` are included before signing. The native workspace icon renderer was checked.
- Warm cream / charcoal tokens, fox-orange actions, shared spacing and typography. Secondary text and primary-action text were adjusted for contrast; checked token pairs range from 4.56:1 to 7.01:1.
- Creation modes stay on the left. Positive and negative prompts are always side by side, following the user's explicit change to the v1.0 document. Negative prompts still require Guidance > 1 to affect inference.
- Prompt presets and prompt history open independent, searchable windows with full text previews and explicit use / append actions. Generation presets in the top toolbar retain their existing menu.
- Clicking outside an active text editor or numeric field resigns text focus. Numeric fields commit through AppKit's normal validation; the click continues to its original destination.
- Segmented controls use a thin fox-orange keyboard focus outline in place of the system blue ring. Reduce Motion disables their selection animation and progress interpolation.
- Canvas tools are compact and outside the artwork. Filmstrip visibility stays reachable during generation; thumbnails are keyboard-accessible buttons.
- LoRA rows reserve a line for the name and keep enable / scale controls compact. Rank, target count, file size and coverage are available in the action menu and tooltip.
- Generation places a translucent, static scrim over the preview only. It displays actual stage, step count, measured average s/step, denoising ETA and cancellation. It retains the previous final image until the new final image arrives. There is no intermediate decode or new GPU work for previews.
- Global resources, performance policy and diagnostics stay in Settings / Runtime Inspector. No core model, sampler, precision or LoRA execution source changed.

## Verification

| Check | Result |
|---|---|
| Release build and app bundle | Passed; ad-hoc signed; `codesign --verify --deep --strict` passed |
| App icon | Native Icon Composer compilation, resource / plist assertions, and system icon rendering passed |
| XCTest | 7 cases, 0 failures; nonempty xUnit report |
| Blank click: positive prompt | Focus changed from text editor to window |
| Blank click: negative prompt | Focus changed from text editor to window |
| Blank click: seed | Focus changed from numeric field to window |
| Field value commit | XCTest verifies `1024` is committed when leaving the field editor |
| Prompt history | Opened a separate window, selected text, searched `watercolor` |
| Prompt preset | Saved the current prompt as `霓虹招牌夜景`, selected it and used it; current prompt preserved |
| Filmstrip | Collapsed and reopened; original expanded preference restored |
| Appearance | Inspected full light and dark windows; original system-following preference restored |
| Reduced window | After the user manually resized it, both editors, both parameter rows, generate action, canvas tools and LoRA controls remained visible without overlap |
| Generation | Real 1024×1024 BF16, Turbo 6 steps, seed 42, dynamic LoRA scale 1 completed and saved |
| Cancellation | A following task was cancelled; canvas displayed cancellation and kept the completed artwork |
| Image regression | New output and the earlier manually generated output have identical decoded pixels |

The image comparison used the same ImageIO / CoreGraphics RGBA rendering for both files. SHA-256:
`ef3aaad4f284a8c95f26781a25e3ec7b9188fb75063dd36021b2fa5c917fd229`.

Compared outputs:

- Prior manual output: `20261018-134657-42.png`.
- This UI acceptance output: `20261018-143826-42.png`, with its JSON record in the configured output folder.

Local evidence is retained in `.tmp/gui-refinement-20261018/`:
`light-main.png`, `light-final.png`, `dark-main.png`, `prompt-presets.png`, `prompt-history.png`,
`generation-scrim.png`, `generation-complete.png`, `generation-cancelled.png`, `app-icon.png`, `compact-main.png`.
Build / test logs and the xUnit report are copied there. These local artifacts are not committed.

## Limits

- This was a functional UI generation run with GUI capture and concurrent compilation, not a performance benchmark. The scrim correctly showed the observed timing (9.73 s/step at step 1); this run cannot establish the normal-use 5–6 s/step target. No new four-point thermal benchmark was run. Previous clean performance evidence remains in `workbench-acceptance.md`.
- Automated resizing did not change the actual window dimensions. The user then manually reduced the window, verified in the 4718592×4292935744-pixel capture `compact-main.png`. The exact minimum supported 980×640-point boundary remains unverified.
- No new failure was injected by corrupting model files, and all five image modes were not regenerated in this UI-only iteration. Their earlier acceptance and the existing model / metadata tests remain separate evidence.
- IME composition and screen-reader navigation were not exercised. Native text validation remains in control when leaving fields.
- Pre-change source snapshots, including the pre-existing uncommitted UI work, remain in `/tmp/lingxi-ui-refinement-20261018/` until user acceptance.
