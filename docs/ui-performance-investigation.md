# UI performance investigation

Date: 2026-10-18. Initial tested app: local commit `1322d50`.

## Observations

- The user reports roughly 20% whole-system CPU and 40% GPU while the app is open, with pronounced system stutter during conditioning and less stutter during denoising.
- Foreground idle process samples show the app at 0–0.2% CPU in `top`. Its main thread waited in the event loop in 2053 of 2089 sampled stacks. The generation thread waited for a job. This does not explain away the user's separate generation-time observation.
- A 15-second SwiftUI recording contained nine GenerateBar updates totaling 4.199 ms. The two-second metrics refresh invalidates that view, but no continuous UI update loop was observed.
- Resetting the automation session did not stop the native window-sharing stream. After the user stopped sharing through macOS, four WindowServer CPU readings fell from 18.5% to 9.1%, 10.5%, 7.4% of one core. Whole-system CPU settled near 6%. GPU snapshots were 25%, 5%, 20%, 0%.
- GPU readings come from the accelerator's whole-system instantaneous utilization counter. They do not attribute GPU use to the app and are not a sustained average. CPU process percentages and whole-system percentages have different denominators.
- The first attempted generation sample started after the user had cancelled. It captured idle stacks, not the conditioning workload, and supplies no evidence about the cause of that stutter.
- A subsequent no-sharing run on `7fef9d1` captured actual weight-loading and encoding stacks. Loading the text encoder took 3.870 seconds and prompt encoding took 1.106 seconds. File-reader workers performed `pread`; the generation thread waited in existing MLX evaluations. Across the whole 45-second sample, 2992 of 3314 main-thread stacks waited for events (the interval also includes idle time after cancellation). This is not a stage-specific main-thread latency measurement.
- During loading, free pages fell and compressed memory increased, while observed app CPU reached 115.52% of one core during transformer loading/prefill. Denoising observations were about 10–18% of one core. These observations support inference/loading resource pressure as a contributor; they do not establish a UI redraw loop as the cause.
- That task was cancelled after two denoising steps. Exactly one cancellation request was logged at 15:25:49 local time and cancellation finished at 15:25:50. The user then reported that the app no longer stuttered. There is no new completed six-step output from this run, so it cannot establish the steady 5–6 s/step target.
- Four retained thermal observations were nominal. The final observation was after cancellation and is labeled with the last reported generation stage; its 9% GPU utilization is not a denoising performance measurement. Raw samples and phase logs are retained in `active-generation-*.json*` and `active-generation.sample.txt`.

## Cancellation and canvas follow-up

- A running task acknowledges the first cancellation request immediately, disables repeated cancellation and resets that state for the next task. Waiting tasks remain independently cancellable.
- Cancellation checks now run before existing weight-loading batches, text-encoder layers, condition-image evaluations and prefill work. They do not introduce extra GPU evaluations or alter the model's normal arithmetic.
- Already submitted GPU work must finish before the next safe cancellation boundary. The UI explains that wait rather than requiring repeated clicks.
- The generation canvas hides the selected previous artwork from preparation until completion/cancellation. The user's later screenshot review requested an empty preview after cancellation as well; the follow-up clears selection without deleting gallery files, auto-dismisses the notice and removes the duplicate filmstrip.
- Stage transitions and cancellation/finish events are logged once for correlation with process sampling. No prompt text or process environment is collected by this logging.
- Eleven XCTest cases passed, including loading-batch cancellation and a tiny real text-encoder test that cancels before its second layer and verifies unchanged uncancelled output.
- The release executable was packaged with the existing icon/Metal resources; strict deep signature verification passed. After the user exited the original app, `7fef9d1` was installed at `dist/LingXiStudio.app` and opened without window capture. The subsequent notice/filmstrip/empty-canvas follow-up (`ffe3e2d`) passed the 11-case suite, was signature-verified, and was installed and opened after a second user-confirmed exit. Installed and staged executable SHA-256 values matched: `2a9acdada8be46ec84f4a309cf6fa1edbbdbd7ae99125b6cf32ac3f05827590c`. Manual acceptance of these three latest behaviors remains pending.

## Pending acceptance

- The user initially reported improved responsiveness, then reproduced stutter before denoising and proposed adaptive text-encoder quantization. That supersedes the earlier responsiveness acceptance. No quantization code was changed; the user subsequently deferred performance work to prioritize workspace routing and repository publication. The cause and mitigation remain unresolved.
- The next performance investigation should correlate compressor, swap-in/out and memory pressure with the actual loading/conditioning stages. A suggested adaptive text-encoder policy must not quantize the diffusion model and must avoid materializing all full-precision encoder weights before conversion.
- Verify automatic notice dismissal and an empty preview after cancellation in the canvas-cleanup build. The older screenshots show the previous behavior.
- Complete a 1024-square / Turbo 6-step generation and retain four thermal/load observations and output timings. The target remains 5–6 s/step without throttling; nominal thermal state alone cannot prove constant GPU frequency.
- Exact minimum-window layout, screen-reader navigation and all five image modes are outside this performance follow-up.

## Evidence handling

Sanitized local measurements are retained under `.tmp/ui-performance-20261018/`, which is ignored by Git. The initial Instruments recording automatically included process environment metadata; an unredacted metadata export was mistakenly displayed in tool output. The raw recording was removed and was never committed. Affected credentials should be rotated. Subsequent sampling uses `sample` and explicit performance fields rather than Instruments environment metadata.

Pre-change source snapshots remain under `/tmp/lingxi-cancel-performance-20261018/` pending user acceptance.
