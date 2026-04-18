# awesome-open-video-models

**A community-maintained reference for open-weight and frontier AI video generation models in 2026.**

Tracking architecture specs, benchmark scores, hardware requirements, and release status — updated weekly. If a model ships or a leaderboard moves, this document moves with it.

> Maintained by independent researchers. Not affiliated with any model team.  
> Cloud inference for several models listed here: **[dlls5.app](https://dlls5.app)**

---

## Quick navigation

- [Leaderboard snapshot (Artificial Analysis, April 2026)](#leaderboard-snapshot)
- [Model index](#model-index)
  - [HappyHorse 1.0](#happyhorse-10--pre-release)
  - [Wan 2.2](#wan-22)
  - [LTX-2 Pro](#ltx-2-pro)
  - [HunyuanVideo-1.5](#hunyuanvideo-15)
  - [Kling 3.0](#kling-30-closed)
  - [Seedance 2.0](#seedance-20-closed)
- [Hardware & inference requirements](#hardware--inference-requirements)
- [How to contribute](#how-to-contribute)

---

## Leaderboard snapshot

*Source: [Artificial Analysis Video Arena](https://artificialanalysis.ai/video) · snapshot April 2026 · scores shift daily*

| Tier | Models | Elo range |
|---|---|---|
| Frontier closed | Seedance 2.0, SkyReels V4, Kling 3.0, PixVerse V6, Veo 3.1, Runway Gen-4.5 | ~1 200–1 275 |
| Mid-tier closed | Sora 2 Pro, Hailuo 2.3, Wan 2.6, Vidu Q2 | ~1 150–1 200 |
| Open-weights SOTA | LTX-2 Pro, LTX-2 Fast, Wan 2.2 A14B | ~1 100–1 135 |
| Earlier open-weights | HunyuanVideo-1.5, Wan 2.1 14B, Wan 2.2 5B | ~950–1 020 |
| Pending placement | **HappyHorse 1.0** | arena entry seen, not yet in public table |

Anything at or above LTX-2 Pro is considered open-source SOTA. Anything in the frontier closed tier competes directly with the best paid APIs.

---

## Model index

### HappyHorse 1.0 — pre-release

> **Status: not yet open-sourced.** Model weights, inference code, and an official repository have not been published. The information below is compiled from community architecture notes, alleged technical leaks, and the model's observed appearances in the Artificial Analysis Video Arena. Treat as plausible but unverified until the official release.

**Why it is being watched closely:** HappyHorse 1.0 is the first open-weight model (once released) to claim native joint audio-video generation — meaning dialogue, Foley, and ambient sound are produced in the same forward pass as the video frames, rather than by a separate audio model bolted on afterward. It appeared as a mystery entry in the Artificial Analysis Video Arena and was independently noted to include native audio output, which is unusual for any open-weight entry.

#### Reported specifications

| Component | Reported value |
|---|---|
| Total parameters | ~15B |
| Architecture | Unified self-attention Transformer (no dedicated cross-attention branches) |
| Total layers | 40 |
| Layer layout | Sandwich: 4 modality-specific layers at each end, 32 shared middle layers |
| Modalities | Text, image, video, and audio — concatenated into a single token sequence |
| Multimodal fusion | Per-attention-head learned scalar gates (sigmoid) for training stability |
| Conditioning | Reference image and denoising signal through one minimal unified interface |
| Timestep embeddings | Reportedly omitted — denoising state inferred from noise level of input latents |
| Distillation | DMD-2 (Distribution Matching Distillation v2) |
| Sampling steps | 8, no classifier-free guidance required |
| Inference runtime | MagiCompiler (full-graph compilation, ~1.2× end-to-end speedup reported) |
| Reference hardware | NVIDIA H100 80 GB |
| Reported 1080p time | ~38 seconds on H100 |
| Native lip-sync languages | 6: English, Mandarin, Japanese, Korean, German, French |

#### Why the reported design choices matter

**Unified self-attention instead of cross-attention.** Wan 2.2, HunyuanVideo, LTX-2, and CogVideoX all inject text conditioning via cross-attention from a separate encoder, and audio (when present) comes from a completely different model. HappyHorse reportedly puts text, image, video, and audio tokens into the same sequence and lets self-attention handle all of it. The claimed benefit: audio-video alignment is learned as a fundamental part of denoising, not as a downstream fix-up step.

**Sandwich layer layout.** First 4 and last 4 layers handle modality-specific embedding and decoding; the middle 32 layers share parameters across all modalities. Reported benefit: most of the network does cross-modal reasoning rather than being split into siloed sub-networks, improving parameter efficiency.

**Per-head sigmoid gating.** Joint multimodal training is unstable because audio-loss gradients can dominate or be dominated by video-loss gradients. The reported solution: a learned scalar gate on each attention head that can selectively dampen heads producing destructive gradients for a given modality.

**Timestep-free denoising.** Standard diffusion models feed the current timestep as an explicit embedding into every layer. HappyHorse reportedly omits this, on the basis that the noise level is already encoded in the noisy latents. This is described as a prerequisite for the aggressive 8-step DMD-2 distillation.

**DMD-2 distillation.** Standard video diffusion needs 25–50 steps plus classifier-free guidance. DMD-2 trains a student model to match the teacher's output distribution in 8 steps without CFG. This is the basis for the reported "1080p in ~38 seconds" figure.

#### Announced release scope (not yet published)

- Base model weights
- Distilled 8-step model weights
- Super-resolution module
- Inference code
- License: described as fully open source with commercial use permitted; exact terms not yet published

#### Where to follow updates

- Official platform (linked from community notes — check for current URL)
- [Artificial Analysis Text-to-Video Leaderboard](https://artificialanalysis.ai/video/text-to-video)
- [Artificial Analysis Image-to-Video Leaderboard](https://artificialanalysis.ai/video/image-to-video)

---

### Wan 2.2

**Status: open-weight, production-ready**

| Attribute | Value |
|---|---|
| Parameters | A14B variant: ~14B (MoE, 14B activated) · 5B variant also available |
| Architecture | Diffusion Transformer (DiT), MoE backbone |
| Variants | T2V-A14B · I2V-A14B · TI2V-5B |
| Max resolution | 720p (480p also supported) |
| Native audio | No |
| Diffusers integration | Yes (T2V, I2V, TI2V all integrated) |
| License | Apache 2.0 |
| Repo | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) |
| GPU requirement | 1× H100 80 GB for A14B; 4090 for 5B variant |
| Leaderboard position | Top open-weights tier (Elo ~1 100–1 135) |

**Notable ecosystem projects built on Wan 2.2:**
- ALG (Adaptive Low-Pass Guidance) — training-free +36% dynamic degree on VBench-I2V
- Wan2.2-Lightning — 4-step distilled I2V via Phased DMD
- LightX2V — optimized inference with TeaCache and SageAttention
- rCM (NVLabs, ICLR 2026) — consistency model distillation from Wan 2.1, 4-step video generation

**Community pipeline (VBench-I2V optimized):** HunyuanImage-3.0 preprocessing → ALG → Wan 2.2 I2V → super-resolution → frame interpolation. Results and code at **[dlls5.app/pipeline](https://dlls5.app)**.

---

### LTX-2 Pro

**Status: open-weight**

| Attribute | Value |
|---|---|
| Parameters | ~13B |
| Architecture | DiT |
| Max resolution | 1080p |
| Native audio | No |
| Sampling steps | ~25 |
| License | Check official repo |
| Leaderboard position | Top open-weights tier (Elo ~1 100–1 135), roughly tied with Wan 2.2 A14B |

---

### HunyuanVideo-1.5

**Status: open-weight**

| Attribute | Value |
|---|---|
| Parameters | ~13B |
| Architecture | DiT |
| Max resolution | 720p |
| Native audio | No |
| License | Check official repo |
| Leaderboard position | Earlier open-weights tier (Elo ~950–1 020) |

**HunyuanImage-3.0-Instruct** (image generation, same team) is separately useful as a preprocessing step for I2V pipelines. It is an 80B MoE model with 13B activated parameters, supports CoT-guided editing, and runs on 2× H100 with the NF4-quantized Distil variant.

---

### Kling 3.0 — closed

**Status: closed API, not open-weight**

| Attribute | Value |
|---|---|
| Architecture | DiT + 3D VAE |
| Max resolution | 4K native, 60fps |
| Max duration | 15 seconds per clip, 6-shot multi-shot storyboarding |
| Native audio | Yes — dialogue, Foley, ambient sound in 6 languages |
| Motion Control | Reference-video-driven motion transfer (viral feature) |
| Element Binding | Cross-shot character consistency |
| Leaderboard position | Frontier closed tier (Elo ~1 200–1 275) |
| Access | API via klingai.com |

---

### Seedance 2.0 — closed

**Status: closed API, not open-weight**

| Attribute | Value |
|---|---|
| Architecture | DiT |
| Native audio | Yes — first unified audio-video joint generation model from ByteDance |
| Multi-shot | Yes, from single prompt |
| Lip-sync languages | 8+ |
| Leaderboard position | Frontier closed tier, #1 Elo as of April 2026 |
| Access | API via jimeng.jianying.com |

---

## Hardware & inference requirements

| Model | Min VRAM | Recommended | Approx. generation time (720p, 5s) |
|---|---|---|---|
| Wan 2.2 A14B | 80 GB (1× H100) | 2× H100 | ~4 min unoptimized · ~60s with TeaCache |
| Wan 2.2 5B | 24 GB (1× 4090) | 1× H100 | ~2 min |
| LTX-2 Pro | ~40 GB | 1× H100 | ~2–3 min |
| HunyuanVideo-1.5 | ~40 GB | 1× H100 | ~3–5 min |
| HunyuanImage-3.0-Instruct-Distil (image) | 20 GB NF4 | 2× H100 BF16 | ~30s (8 steps) |
| HappyHorse 1.0 (reported) | H100 80 GB | H100 80 GB | ~38s (1080p, 8 steps) |

> Need cloud inference without local setup? **[dlls5.app](https://dlls5.app)** provides optimized API access to several models in this list, with super-resolution and frame interpolation built into the pipeline.

---

## Comparison: HappyHorse 1.0 vs. current open-weight leaders

| Feature | HappyHorse 1.0 (reported) | LTX-2 Pro | Wan 2.2 A14B | HunyuanVideo-1.5 |
|---|---|---|---|---|
| Parameters | ~15B | ~13B | 14B (MoE) | ~13B |
| Backbone | Unified self-attention | DiT | DiT | DiT |
| Native audio | Yes (joint with video) | No | No | No |
| Native lip-sync | 6 languages | 0 | 0 | 0 |
| Sampling steps | 8 (no CFG) | ~25 | ~50 | ~50 |
| Reported 1080p time | ~38s on H100 | minutes | minutes | minutes |
| Open weights today | **Not yet** | Yes | Yes | Yes |

---

## How to contribute

Pull requests welcome for:

- Correcting or updating any specification
- Adding new models (open-weight only, or widely discussed pre-release models)
- Hardware benchmark data from your own runs
- Broken links

For HappyHorse 1.0 specifically: if you have verified information from the official release (once it ships), please open a PR with a source link and we will update immediately.

---

## Changelog

| Date | Update |
|---|---|
| 2026-04-18 | Initial publish. HappyHorse 1.0 added as pre-release entry. Wan 2.2, LTX-2, HunyuanVideo-1.5, Kling 3.0, Seedance 2.0 added. |

---

*Star this repo to get notified when HappyHorse 1.0 ships or the leaderboard moves.*  
*Cloud pipeline (super-res + frame interpolation included): [dlls5.app](https://dlls5.app)*
