# Agent notes

## Modal skills

Official Modal skill (version-aligned docs):

```bash
modal skills install --yes --claude
modal skills show
```

This repo also vendors [modal-auto-research-skills](https://github.com/modal-projects/modal-auto-research-skills.git) under `.claude/skills/`. Use them for GPU / Modal work:

- `modal` — current SDK docs bundled by `modal skills install`
- `modal-basic-skills` — app as a package, `modal deploy -m`, lazy `from_name()`, CLI
- `modal-gpu-dev` — interactive GPU sandboxes
- `modal-gpu-experiment` — volumes, secrets, retries, checkpoints
- `sub-agents` — parallel agents across GPUs

Refresh the research pack:

```bash
git clone https://github.com/modal-projects/modal-auto-research-skills.git /tmp/modal-auto-research-skills
cp -R /tmp/modal-auto-research-skills/{modal-basic-skills,modal-gpu-dev,modal-gpu-experiment,sub-agents} \
  .claude/skills/
```


## GPU cost safety (highest priority)

Normal implementation, refactoring, tests, documentation, and local debugging **must not launch GPUs**.

- Production generation uses only `Trellis2Worker`.
- Keep `PRODUCTION_GPU=A100-80GB`, `min_containers=0`, `max_containers=1`, `buffer_containers=0`, `scaledown_window=10`, and the 10-minute per-input timeout unless the user explicitly approves a cost-policy change.
- Never add `Cls.with_options(gpu=...)`, request/env GPU selectors, `@modal.concurrent`, fallback GPU lists, or additional production GPU pools.
- Do not automatically invoke `modal-gpu-dev`, `modal-gpu-experiment`, `sub-agents`, or `modal_trellis2.modal.gpu_smoke`. Those are explicit experiment tools and can create additional billable GPUs.
- CPU `prefetch`, CPU `prefetch --status`, local `doctor`, dry-run, pytest, and CI are the default validation path.
- Live generation must run deployed CPU `prefetch_status` before looking up `Trellis2Worker`; an incomplete Volume must never be discovered after A100 launch.
- Production currently accepts only `pipeline=512`. Higher-resolution pipelines remain experimental until separately GPU-validated.
- Keep TRELLIS.2 source pinned to `75fbf0183001ed9876c8dbb35de6b68552ee08bd` and the primary HF model bundle pinned to `af44b45f2e35a493886929c6d786e563ec68364d`.
- Preserve the currently runtime-validated `flash_attn_3` backend/wheel for this TRELLIS image. Do not swap attention backends from generic hardware assumptions; change it only with an explicit live benchmark/compatibility test.
- Keep input-cost guards: upload <=20MB, decoded image <=40MP, normalize longest side <=1024 before remote calls, production `texture_size` in `{256, 512, 1024}`, and Web live batch <=20 images.
- Keep large GLB transfer on `modal-trellis2-results`: GPU writes + commits a temporary file, local client verifies/downloads it, JobStore persists the durable copy, then the remote temporary file is removed.
- Preserve VRAM telemetry (`after_load`, `before_infer`, `after_infer`, `after_export`, `after_cleanup`) so every paid run records the real memory envelope.

## Architecture

Local CLI/Web own jobs and the GLB workbench. Modal owns GPU inference.

- Contract: image bytes → `ImageTo3DGenerator` → GLB bytes
- Default path is official `microsoft/TRELLIS.2-4B` via `ModalTrellis2Generator`
- `--dry-run` / `MockGenerator` is opt-in for the upload/download loop
- Weights: `uv run modal-trellis2 prefetch` is CPU-only, writes `/models/trellis2` plus DINOv3/background-removal bundles, validates all six 512 checkpoint config+weight pairs, then commits the Volume.
- GPU is `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `block_network=True`, and has no HF secret.
- RGB uploads are bounded to 1024px before `CpuPreprocessor`; alpha uploads are cropped locally. Background removal tries a complete RMBG bundle first, then a complete BiRefNet bundle.
- GPU `Trellis2Worker` deliberately does **not** use CPU Memory Snapshots: TRELLIS.2 imports `flex_gemm`/Triton at pipeline import time, and Modal CPU snapshot hooks have no GPU driver.
- Catchable Python exceptions during GPU initialization are stored as `init_error` instead of escaping `@modal.enter()`. `health/generate` then report a normal method failure so Python-level startup errors do not become deployed-container crash loops.
- The GPU worker never late-loads a missing model after `.cuda()`. Missing production models are a hard preflight/init error.
- Generated GLBs are temporary files on `modal-trellis2-results`; the local generator downloads and verifies size before deleting the remote copy. The local JobStore is the durable user-facing copy.
- Production GPU policy is cost-first: fixed `A100-80GB`, `min_containers=0`, `max_containers=1`, `buffer_containers=0`, `scaledown_window=10`. Bursty jobs queue onto the one warm GPU container instead of scaling out.
- Production generation must not use `Cls.with_options(gpu=...)`; dynamic GPU variants create separate autoscaling pools and can bypass the one-container cost cap. Benchmark/experimentation with other GPU types belongs in a separate path.
- Live GPU needs gated `facebook/dinov3-vitl16-pretrain-lvd1689m` accepted on the HF account behind `HF_TOKEN` **before prefetch**. After that the GPU never talks to Hugging Face.
- Do **not** `modal run -m modal_trellis2.modal.worker` just to prefetch — that file registers `Trellis2Worker` and builds CUDA.
- Deploy: `uv run modal-trellis2 deploy`.
- Probe: `uv run modal-trellis2 health` is CPU-only. `uv run modal-trellis2 health --gpu` starts an A100.
- Live reuse acceptance: `uv run modal-trellis2 verify-gpu-reuse ... --confirm-cost` only when the user explicitly requests a billable GPU test.
- Local web/CLI are **not** `modal serve`.

The production prefetch secret `huggingface-secret` needs only `HF_TOKEN`. Do not mount unrelated GitHub/CivitAI credentials into this app. Never commit tokens.

Upstream clones belong in `vendor/` and stay gitignored. Refresh with `scripts/fetch-upstream.sh`.
