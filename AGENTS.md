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
- Production currently accepts only `pipeline=512`. Higher-resolution pipelines remain experimental until separately GPU-validated.
- Keep upstream TRELLIS.2 source pinned to `75fbf0183001ed9876c8dbb35de6b68552ee08bd`; do not float on `main` in production images.
- Keep input-cost guards: upload <=20MB, decoded image <=40MP, production `texture_size` in `{256, 512, 1024}`, and Web live batch <=20 images. Do not relax them without explicit user approval.

## Architecture

Local CLI/Web own jobs and the GLB workbench. Modal owns GPU inference.

- Contract: image bytes → `ImageTo3DGenerator` → GLB bytes
- Default path is official `microsoft/TRELLIS.2-4B` via `ModalTrellis2Generator`
- `--dry-run` / `MockGenerator` is opt-in for the upload/download loop
- Weights: `modal-trellis2 prefetch` (CPU image only) writes `/models/trellis2` plus HF cache copies of DINOv3 / BiRefNet, then `commit()`s. GPU is `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `block_network=True`, and has no HF secret.
- RGB uploads: `CpuPreprocessor` runs BiRefNet on CPU. Images that already have alpha are cropped locally.
- GPU `Trellis2Worker` deliberately does **not** use CPU Memory Snapshots: TRELLIS.2 imports `flex_gemm`/Triton at pipeline import time, and Modal CPU snapshot hooks have no GPU driver. The regular GPU `@modal.enter()` loads the already-prefetched local pipeline, then `.cuda()` + `run` + official `to_glb`.
- Production GPU policy is cost-first: fixed `A100-80GB`, `min_containers=0`, `max_containers=1`, `buffer_containers=0`, `scaledown_window=10`. Bursty jobs queue onto the one warm GPU container instead of scaling out.
- Production generation must not use `Cls.with_options(gpu=...)`; dynamic GPU variants create separate autoscaling pools and can bypass the one-container cost cap. Benchmark/experimentation with other GPU types belongs in a separate path.
- Live GPU needs gated `facebook/dinov3-vitl16-pretrain-lvd1689m` accepted on the HF account behind `HF_TOKEN` **before prefetch**. After that the GPU never talks to Hugging Face.
- Do **not** `modal run -m modal_trellis2.modal.worker` just to prefetch — that file registers `Trellis2Worker` and builds CUDA
- Deploy: `modal-trellis2 deploy` or `modal deploy -m modal_trellis2.modal.deploy`
- Probe: `modal-trellis2 health` (CPU Volume). `health --gpu` starts an A100.
- Live reuse acceptance: `modal-trellis2 verify-gpu-reuse ... --confirm-cost` only when the user explicitly requests a billable GPU test.
- Smoke: `modal run -m modal_trellis2.modal.smoke` (secret) / `modal run -m modal_trellis2.modal.gpu_smoke` (CUDA image + A100)
- Local web/CLI are **not** `modal serve`

The production prefetch secret `huggingface-secret` needs only `HF_TOKEN`. Do not mount unrelated GitHub/CivitAI credentials into this app. Never commit tokens.

Upstream clones belong in `vendor/` and stay gitignored. Refresh with `scripts/fetch-upstream.sh`.
