# UI performance investigation

Date: 2026-10-18. Initial tested app: local commit `1322d50`.

## Observations

- The user reports roughly 20% whole-system CPU and 40% GPU while the app is open, with pronounced system stutter during conditioning and less stutter during denoising.
- Foreground idle process samples show the app at 0–0.2% CPU in `top`. Its main thread waited in the event loop in 2053 of 2089 sampled stacks. The generation thread waited for a job. This does not explain away the user's separate generation-time observation.
- A 15-second SwiftUI recording contained nine GenerateBar updates totaling 4.199 ms. The two-second metrics refresh invalidates that view, but no continuous UI update loop was observed.
- Resetting the automation session did not stop the native window-sharing stream. After the user stopped sharing through macOS, four WindowServer CPU readings fell from 18.5% to 9.1%, 10.5%, 7.4% of one core. Whole-system CPU settled near 6%. GPU snapshots were 25%, 5%, 20%, 0%.
- GPU readings come from the accelerator's whole-system instantaneous utilization counter. They do not attribute GPU use to the app and are not a sustained average. CPU process percentages and whole-system percentages have different denominators.
- The attempted generation sample started after the user had cancelled. It captured idle stacks, not the conditioning workload, and supplies no evidence about the cause of that stutter. No new successful clean generation benchmark was obtained from it.

## Cancellation and canvas follow-up

- A running task acknowledges the first cancellation request immediately, disables repeated cancellation and resets that state for the next task. Waiting tasks remain independently cancellable.
- Cancellation checks now run before existing weight-loading batches, text-encoder layers, condition-image evaluations and prefill work. They do not introduce extra GPU evaluations or alter the model's normal arithmetic.
- Already submitted GPU work must finish before the next safe cancellation boundary. The UI explains that wait rather than requiring repeated clicks.
- The generation canvas hides the selected previous artwork from preparation until completion/cancellation. Cancelling restores that artwork without deleting it.
- Stage transitions and cancellation/finish events are logged once for correlation with process sampling. No prompt text or process environment is collected by this logging.
- Eleven XCTest cases passed, including loading-batch cancellation and a tiny real text-encoder test that cancels before its second layer and verifies unchanged uncancelled output.
- The release executable was packaged with the existing icon/Metal resources as `dist/review/LingXiStudio.app`; strict deep signature verification passed. The running original app was not overwritten. Installing this build and its manual acceptance remain pending the user's exit from the original app.

## Pending acceptance

- Run the updated app without window sharing or concurrent builds, capture the actual conditioning workload and correlate it with stage logs, process CPU and memory activity.
- Verify one-click acknowledgement/cancellation and the opaque canvas in the app. The old functional screenshots do not verify these changes.
- Complete a 1024-square / Turbo 6-step generation and retain four thermal/load observations and output timings. The target remains 5–6 s/step without throttling; nominal thermal state alone cannot prove constant GPU frequency.
- Exact minimum-window layout, screen-reader navigation and all five image modes are outside this performance follow-up.

## Evidence handling

Sanitized local measurements are retained under `.tmp/ui-performance-20261018/`, which is ignored by Git. The initial Instruments recording automatically included process environment metadata; an unredacted metadata export was mistakenly displayed in tool output. The raw recording was removed and was never committed. Affected credentials should be rotated. Subsequent sampling uses `sample` and explicit performance fields rather than Instruments environment metadata.

Pre-change source snapshots remain under `/tmp/lingxi-cancel-performance-20261018/` pending user acceptance.
