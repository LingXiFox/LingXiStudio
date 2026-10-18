"""Stage-by-stage numerical parity checks of LingXiStudio against the official Qwen-Image-2.1 implementation.

Development-only tooling: it needs PyTorch, transformers and the diffusers / PEFT / accelerate sources the official
model card points to (passed through EXTRA_SRC). The runtime itself never imports any of this.

Every stage feeds both implementations the *same* intermediate inputs (tensors written to safetensors), so random
number generators never enter a comparison. Each comparison is classified against a float-noise floor measured on the
reference itself:

* exact      - must match bit for bit (integer outputs, schedule arithmetic, the Euler update).
* float      - difference within the reference's own rounding noise (bf16 vs fp32, or a 1-ulp input perturbation).
* SEMANTIC   - difference well above the noise floor: a real behavioural mismatch.

Negative controls (deliberate semantic errors run through the reference) prove the thresholds can tell the two apart.
"""

import argparse
import json
import math
import os
import sys
import time

# Source folders of diffusers, peft and accelerate (colon-separated), e.g. unpacked wheels.
for path in os.environ.get("EXTRA_SRC", "").split(":"):
    if path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.expanduser("~/Models/Qwen-Image-2.1-BF16"))
LORA = os.environ.get(
    "LORA", os.path.expanduser("~/Models/Qwen-Image-2.1-LoRA/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors")
)
TURBO_NODES = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def cos_min_token(a, b):
    a, b = a.double().reshape(-1, a.shape[-1]), b.double().reshape(-1, b.shape[-1])
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min())


def result(stage, name, kind, value, threshold, detail=""):
    if kind == "info":
        verdict = "info"
    elif kind == "exact":
        verdict = "exact" if value == 0 else "SEMANTIC"
    elif kind == "control":
        verdict = "detected" if value > threshold else "MISSED"
    else:
        verdict = "float" if value <= threshold else "SEMANTIC"
    return {"stage": stage, "check": name, "kind": kind, "value": value, "threshold": threshold, "verdict": verdict,
            "detail": detail}


# ---------------------------------------------------------------- inputs

def make_inputs(work):
    """Shared intermediate inputs: real text embeddings come later from the reference text encoder."""
    g = torch.Generator().manual_seed(0)
    lh, lw = 16, 24  # 256 x 384 pixels
    save_file({"latents": torch.randn(1, lh * lw, 64, generator=g), "noise": torch.randn(1, lh * lw, 64, generator=g),
               "velocity": torch.randn(1, lh * lw, 64, generator=g)}, f"{work}/latents.safetensors")
    img = torch.rand(1, 64, 96, 4, generator=g) * 2.4 - 1.2
    save_file({"image": img}, f"{work}/image.safetensors")


# ---------------------------------------------------------------- text encoder

def text_encoder(work):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    te = f"{MODEL}/text_encoder"
    cfg = AutoConfig.from_pretrained(te).text_config
    cfg._attn_implementation = "sdpa"
    index = json.load(open(f"{te}/model.safetensors.index.json"))["weight_map"]
    sd = {}
    for shard in sorted(set(index.values())):
        with safe_open(f"{te}/{shard}", "pt") as f:
            for k in f.keys():
                if k.startswith("model.language_model."):
                    sd[k[len("model.language_model."):]] = f.get_tensor(k)
    with torch.device("meta"):
        model = Qwen3VLTextModel(cfg)
    model.load_state_dict(sd, assign=True, strict=True)
    model.rotary_emb = type(model.rotary_emb)(config=cfg)
    model = model.to(torch.bfloat16).eval()

    swift = load_file(f"{work}/te_default.safetensors")
    strict = load_file(f"{work}/te_strict.safetensors")
    ids = swift["ids"].long()[None]
    drop = int(swift["drop"][0])
    captured = {}
    model.norm.register_forward_hook(lambda m, a, o: captured.update(pre=a[0].float(), post=o.float()))
    with torch.no_grad():
        model(input_ids=ids)
    ref_bf16, post_bf16 = captured["pre"], captured["post"]
    model = model.float()
    with torch.no_grad():
        model(input_ids=ids)
    ref_fp32 = captured["pre"]
    del model, sd

    k = slice(drop, None)  # only the kept (non-system) tokens condition the image
    floor = rel(ref_bf16[:, k], ref_fp32[:, k])
    out = [
        result("text", "reference bf16 vs fp32 (noise floor)", "info", floor, None),
        result("text", "strict bf16 vs reference bf16", "float", rel(strict["hidden"][:, k], ref_bf16[:, k]), 2 * floor,
               f"worst-token cos {cos_min_token(strict['hidden'][:, k], ref_bf16[:, k]):.5f}"),
        result("text", "default (fp32 residual) vs reference fp32", "float",
               rel(swift["hidden"][:, k], ref_fp32[:, k]), floor,
               f"worst-token cos {cos_min_token(swift['hidden'][:, k], ref_fp32[:, k]):.5f}"),
        result("text", "control: final-normed output vs reference", "control", rel(post_bf16[:, k], ref_bf16[:, k]),
               2 * floor, "the MFLUX text-to-image conditioning; must exceed the acceptance band"),
    ]
    save_file({"embeds": ref_bf16[:, k].contiguous(), "embeds_post": post_bf16[:, k].contiguous()},
              f"{work}/embeds.safetensors")
    return out


# ---------------------------------------------------------------- transformer

def load_transformer(lora):
    from diffusers import QwenImage21Transformer2DModel

    m = QwenImage21Transformer2DModel.from_pretrained(f"{MODEL}/transformer", torch_dtype=torch.bfloat16).eval()
    if lora:
        m.load_lora_adapter(LORA, adapter_name="turbo", prefix="transformer")  # PEFT, not fused into the base
        m.set_adapters(["turbo"], weights=[1.0])
    return m


def forward(m, emb, lat, sigma, trace=None):
    T, N = emb.shape[1], lat.shape[1]
    img_mask = torch.cat([torch.zeros(1, T, dtype=torch.bool), torch.ones(1, N // 4, dtype=torch.bool)], 1)
    hooks = []
    if trace is not None:
        for i, b in enumerate(m.transformer_blocks):
            hooks.append(b.register_forward_hook(lambda mod, a, o, i=i: trace.__setitem__(i, o[:, T:].float())))
    t = torch.tensor([sigma * 1000]).to(torch.bfloat16)  # the pipeline's `t.expand(...).to(latents.dtype)`
    with torch.no_grad():
        out = m(hidden_states=lat, encoder_hidden_states=emb, timestep=t / 1000, img_shapes=[[(1, 16, 24)]],
                img_mask=img_mask, return_dict=False)[0][:, -N:].float()
    for h in hooks:
        h.remove()
    return out


def transformer(work, sigmas):
    emb_all = load_file(f"{work}/embeds.safetensors")
    emb = emb_all["embeds"].to(torch.bfloat16)
    lat = load_file(f"{work}/latents.safetensors")["latents"].to(torch.bfloat16)
    out = []

    m = load_transformer(lora=False)
    # Conditioning: timestep embedding rows [t, 0], modulation, text projection.
    cond = load_file(f"{work}/cond.safetensors")
    t = (torch.tensor([sigmas[0] * 1000]).to(torch.bfloat16) / 1000)
    rows = torch.cat([t, t.new_zeros(1)])
    with torch.no_grad():
        temb = m.time_text_embed(rows, emb)
        modulation = m.modulation(temb)
        txt = m.txt_in(emb)
    for name, ref in [("temb", temb), ("modulation", modulation), ("txt", txt)]:
        out.append(result("conditioning", f"{name} vs reference", "float", rel(cond[name], ref.float()), 1e-2))

    swift = load_file(f"{work}/dit.safetensors")
    trace = {}
    ref = [forward(m, emb, lat, s, trace if i == 0 else None) for i, s in enumerate(sigmas)]
    # Noise floor: the reference's own response to a 1-ulp perturbation of its bf16 input.
    g = torch.Generator().manual_seed(3)
    ulp = 2.0 ** (torch.floor(torch.log2(lat.float().abs().clamp_min(1e-30))) - 7)
    lat_p = (lat.float() + ulp * torch.where(torch.rand(lat.shape, generator=g) < 0.5, -1.0, 1.0)).to(torch.bfloat16)
    floor = rel(forward(m, emb, lat_p, sigmas[0]), ref[0])
    out.append(result("transformer", "reference 1-ulp input perturbation (noise floor)", "info", floor, None))
    for i, s in enumerate(sigmas):
        out.append(result("transformer", f"velocity sigma={s} vs reference", "float", rel(swift[f"velocity.s{i}"], ref[i]),
                          2 * floor, f"worst-token cos {cos_min_token(swift[f'velocity.s{i}'], ref[i]):.5f}"))
    swift_trace = load_file(f"{work}/dit_trace.safetensors")
    out.append(result("transformer", "eager (traced) vs compiled velocity", "exact",
                      float((swift_trace["velocity.s0"] - swift["velocity.s0"]).abs().max()), 0))
    growth = [rel(swift_trace[f"block{b}.s0"], trace[b]) for b in range(len(trace))]
    out.append(result("transformer", "per-block hidden rel-L2 (first, middle, last)", "info", growth[-1], None,
                      ", ".join(f"b{b}:{growth[b]:.4f}" for b in (0, 1, 7, 15, 23, 31))))
    control = forward(m, emb_all["embeds_post"].to(torch.bfloat16), lat, sigmas[0])
    out.append(result("transformer", "control: final-normed text conditioning", "control", rel(control, ref[0]),
                      2 * floor, "must exceed the acceptance band"))
    del m

    m = load_transformer(lora=True)
    swift_lora = load_file(f"{work}/dit_lora.safetensors")
    ref_lora = forward(m, emb, lat, sigmas[0])
    out.append(result("lora", "dynamic LoRA velocity vs PEFT reference", "float", rel(swift_lora["velocity.s0"], ref_lora),
                      2 * floor, f"worst-token cos {cos_min_token(swift_lora['velocity.s0'], ref_lora):.5f}"))
    out.append(result("lora", "control: LoRA missing (base model)", "control", rel(ref[0], ref_lora), 2 * floor,
                      "must exceed the acceptance band"))
    two = load_file(f"{work}/dit_lora_two_halves.safetensors")
    out.append(result("lora", "two adapters at 0.5 vs PEFT one at 1.0", "float", rel(two["velocity.s0"], ref_lora),
                      2 * floor))

    # End-to-end: 6 turbo steps from identical noise, reference pipeline loop with its KV cache.
    noise = load_file(f"{work}/latents.safetensors")["noise"].to(torch.bfloat16)
    swift_e2e = load_file(f"{work}/denoise.safetensors")
    steps, sigmas_ref = reference_loop(m, emb, noise)
    traj = [rel(swift_e2e[f"latents.{i}"], x) for i, x in enumerate(steps)]
    latents = steps[-1]
    out.append(result("end-to-end", "sigmas (turbo, no terminal stretch)", "float",
                      float((swift_e2e["sigmas"].double() - sigmas_ref.double()).abs().max()), 1e-6))
    out.append(result("end-to-end", "final latents after 6 turbo steps", "info", traj[-1], None,
                      "per step: " + ", ".join(f"{x:.4f}" for x in traj)))
    save_file({"latents": latents.float().contiguous(), "swift": swift_e2e[f"latents.{len(traj) - 1}"].contiguous()},
              f"{work}/e2e_final.safetensors")
    return out


def reference_loop(m, emb, noise):
    """The official pipeline's denoising loop (turbo schedule, prefix KV cache), returning every step's latents."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{MODEL}/scheduler", shift_terminal=None)
    N, T = noise.shape[1], emb.shape[1]
    sched.set_timesteps(sigmas=TURBO_NODES, mu=calculate_shift(N, 256, 8192, 0.5, 0.9))
    sched.set_begin_index(0)
    img_mask = torch.cat([torch.zeros(1, T, dtype=torch.bool), torch.ones(1, N // 4, dtype=torch.bool)], 1)
    cache = QwenImage21KVCache(len(m.transformer_blocks))
    latents, steps = noise, []
    for i, tstep in enumerate(sched.timesteps):
        with torch.no_grad():
            v = m(hidden_states=latents, encoder_hidden_states=emb, timestep=tstep.expand(1).to(torch.bfloat16) / 1000,
                  img_shapes=[[(1, 16, 24)]], img_mask=img_mask, kv_cache=cache,
                  kv_cache_mode="extract" if i == 0 else "cached", return_dict=False)[0][:, -N:]
        latents = sched.step(v, tstep, latents, return_dict=False)[0]
        steps.append(latents.float())
    return steps, sched.sigmas


def decode_psnr(a, b):
    """PSNR between two latent sets decoded by the same fp32 reference decoder."""
    from diffusers import AutoencoderKLQwenImage21

    cfg = json.load(open(f"{MODEL}/vae/config.json"))
    mean = torch.tensor(cfg["latents_mean"]).view(1, 64, 1, 1, 1)
    std = torch.tensor(cfg["latents_std"]).view(1, 64, 1, 1, 1)
    v = AutoencoderKLQwenImage21.from_pretrained(f"{MODEL}/vae").eval()
    imgs = []
    for lat in (a, b):
        z = lat.transpose(1, 2).reshape(1, 64, 1, 16, 24)
        with torch.no_grad():
            x = v.decode(z * std + mean, return_dict=False)[0][:, :3, 0]
        imgs.append(((x / 2 + 0.5).clamp(0, 1) * 255).round())
    mse = float(((imgs[0] - imgs[1]) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def e2e_floor(work):
    """End-to-end noise floor: the reference against itself with a 1-ulp perturbation of the initial noise."""
    emb = load_file(f"{work}/embeds.safetensors")["embeds"].to(torch.bfloat16)
    noise = load_file(f"{work}/latents.safetensors")["noise"].to(torch.bfloat16)
    g = torch.Generator().manual_seed(4)
    ulp = 2.0 ** (torch.floor(torch.log2(noise.float().abs().clamp_min(1e-30))) - 7)
    noise_p = (noise.float() + ulp * torch.where(torch.rand(noise.shape, generator=g) < 0.5, -1.0, 1.0)).to(torch.bfloat16)
    m = load_transformer(lora=True)
    steps, _ = reference_loop(m, emb, noise_p)
    del m
    ref = load_file(f"{work}/e2e_final.safetensors")["latents"]
    floor_rel = rel(steps[-1], ref)
    floor_psnr = decode_psnr(ref, steps[-1])
    mine = load_file(f"{work}/e2e_final.safetensors")["swift"]
    mine_psnr = decode_psnr(ref, mine)
    return [
        result("end-to-end", "reference vs itself, 1-ulp noise perturbation: final latents", "info", floor_rel, None,
               f"decoded PSNR {floor_psnr:.2f} dB"),
        result("end-to-end", "final latents vs reference", "float", rel(mine, ref), 2 * floor_rel,
               f"decoded PSNR {mine_psnr:.2f} dB vs the reference's own {floor_psnr:.2f} dB"),
    ]


# ---------------------------------------------------------------- scheduler / euler / postprocess / vae

def scheduler(work):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    out = []
    mu = calculate_shift(16 * 24, 256, 8192, 0.5, 0.9)
    for name, nodes, terminal, file in [("linear 40", list(np.linspace(1.0, 1 / 40, 40)), 0.02, "sched_linear"),
                                        ("viggle turbo", TURBO_NODES, None, "sched_turbo")]:
        s = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{MODEL}/scheduler", shift_terminal=terminal)
        s.set_timesteps(sigmas=nodes, mu=mu)
        mine = load_file(f"{work}/{file}.safetensors")["sigmas"]
        out.append(result("scheduler", f"{name} sigmas", "float",
                          float((mine.double() - s.sigmas.double()).abs().max()), 1e-6))

    e = load_file(f"{work}/latents.safetensors")
    lat, vel = e["latents"].to(torch.bfloat16), e["velocity"].to(torch.bfloat16)
    mine = load_file(f"{work}/euler.safetensors")
    s = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{MODEL}/scheduler", shift_terminal=None)
    s.set_timesteps(sigmas=TURBO_NODES, mu=calculate_shift(lat.shape[1], 256, 8192, 0.5, 0.9))
    worst = 0.0
    for i in range(len(TURBO_NODES)):
        s._step_index = i
        ref = s.step(vel, s.timesteps[i], lat, return_dict=False)[0].float()
        worst = max(worst, float((mine[f"next.{i}"] - ref).abs().max()))
    out.append(result("scheduler", "Euler update (bf16 in/out, all 6 steps)", "exact", worst, 0))
    return out


def postprocess(work):
    from diffusers.image_processor import VaeImageProcessor

    image = load_file(f"{work}/image.safetensors")["image"]
    proc = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    ref = proc.postprocess(image.permute(0, 3, 1, 2), output_type="np")
    ref = torch.from_numpy((ref * 255).round().astype("uint8").astype(np.int32))
    mine = load_file(f"{work}/post.safetensors")["pixels"]
    return [result("postprocess", "float image -> uint8 RGBA", "exact", int((mine - ref).abs().max()), 0)]


def vae(work, real_latent):
    from diffusers import AutoencoderKLQwenImage21

    lat = load_file(real_latent)["latents"]
    lh, lw = 16, 24
    cfg = json.load(open(f"{MODEL}/vae/config.json"))
    mean = torch.tensor(cfg["latents_mean"]).view(1, 64, 1, 1, 1)
    std = torch.tensor(cfg["latents_std"]).view(1, 64, 1, 1, 1)
    z = lat.transpose(1, 2).reshape(1, 64, 1, lh, lw)
    v = AutoencoderKLQwenImage21.from_pretrained(f"{MODEL}/vae").eval()
    refs = {}
    for dt in (torch.float32, torch.bfloat16):
        # PyTorch's CPU bf16 convolutions are single-threaded and take tens of minutes; bf16 runs on MPS instead.
        device = "mps" if dt == torch.bfloat16 and torch.backends.mps.is_available() else "cpu"
        with torch.no_grad():
            x = v.to(device=device, dtype=dt).decode((z * std + mean).to(device=device, dtype=dt), return_dict=False)[0]
            refs[dt] = x[:, :, 0].float().cpu().permute(0, 2, 3, 1)
    floor = rel(refs[torch.bfloat16], refs[torch.float32])
    f32 = load_file(f"{work}/vae_f32.safetensors")["image"]
    b16 = load_file(f"{work}/vae_bf16.safetensors")["image"]
    return [
        result("vae", "reference bf16 vs fp32 (noise floor)", "info", floor, None),
        result("vae", "fp32 decode (default) vs reference fp32", "float", rel(f32, refs[torch.float32]), 1e-3,
               f"mean |diff| {float((f32 - refs[torch.float32]).abs().mean()) * 127.5:.3f} 8-bit levels"),
        result("vae", "bf16 decode vs reference bf16", "float", rel(b16, refs[torch.bfloat16]), 2 * floor),
    ]


def e2e_decode(work):
    """Decodes the reference and LingXi final latents with the same fp32 reference decoder and compares pixels."""
    d = load_file(f"{work}/e2e_final.safetensors")
    psnr = decode_psnr(d["latents"], d["swift"])
    return [result("end-to-end", "decoded image PSNR vs reference (dB)", "info", psnr, None,
                   "same reference decoder for both latents")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inputs", "text", "transformer", "scheduler", "postprocess", "vae", "e2e-decode",
                                   "e2e-floor"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--real-latent")
    ap.add_argument("--sigmas", default="0.9,0.3")
    a = ap.parse_args()
    start = time.time()
    if a.stage == "inputs":
        make_inputs(a.work)
        sys.exit()
    fn = {"text": lambda: text_encoder(a.work),
          "transformer": lambda: transformer(a.work, [float(x) for x in a.sigmas.split(",")]),
          "scheduler": lambda: scheduler(a.work), "postprocess": lambda: postprocess(a.work),
          "vae": lambda: vae(a.work, a.real_latent), "e2e-decode": lambda: e2e_decode(a.work),
          "e2e-floor": lambda: e2e_floor(a.work)}[a.stage]
    rows = fn()
    with open(f"{a.work}/result-{a.stage}.json", "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        v = r["value"]
        print(f"[{r['verdict']:>9}] {r['stage']}: {r['check']} = {v:.4g}" + (f"  ({r['detail']})" if r["detail"] else ""))
    print(f"{a.stage} done in {time.time() - start:.0f}s")
