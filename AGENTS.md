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

## Architecture

Local CLI/Web own jobs and the GLB workbench. Modal owns GPU inference.

- Contract: image bytes → `ImageTo3DGenerator` → GLB bytes
- Default path is official `microsoft/TRELLIS.2-4B` via `ModalTrellis2Generator`
- `--dry-run` / `MockGenerator` is opt-in for the upload/download loop
- Weights: `modal-trellis2 prefetch` (CPU image only) writes `/models/trellis2` plus HF cache copies of DINOv3 / BiRefNet, then `commit()`s. GPU is `HF_HUB_OFFLINE=1` and has no HF secret.
- RGB uploads: `CpuPreprocessor` runs BiRefNet on CPU. Images that already have alpha are cropped locally.
- GPU `Trellis2Worker` uses a CPU memory snapshot (`@modal.enter(snap=True)`), then only `.cuda()` + `run` + official `to_glb`.
- Production GPU policy is cost-first: fixed `A100-80GB`, `min_containers=0`, `max_containers=1`, `buffer_containers=0`, `scaledown_window=10`. Bursty jobs queue onto the one warm GPU container instead of scaling out.
- Production generation must not use `Cls.with_options(gpu=...)`; dynamic GPU variants create separate autoscaling pools and can bypass the one-container cost cap. Benchmark/experimentation with other GPU types belongs in a separate path.
- Live GPU needs gated `facebook/dinov3-vitl16-pretrain-lvd1689m` accepted on the HF account behind `HF_TOKEN` **before prefetch**. After that the GPU never talks to Hugging Face.
- Do **not** `modal run -m modal_trellis2.modal.worker` just to prefetch — that file registers `Trellis2Worker` and builds CUDA
- Deploy: `modal-trellis2 deploy` or `modal deploy -m modal_trellis2.modal.deploy`
- Probe: `modal-trellis2 health` (CPU Volume). `health --gpu` starts an A100.
- Smoke: `modal run -m modal_trellis2.modal.smoke` (secret) / `modal run -m modal_trellis2.modal.gpu_smoke` (CUDA image + A100)
- Local web/CLI are **not** `modal serve`

Tokens live in Modal secret `huggingface-secret` (`HF_TOKEN`, `CIVITAI_TOKEN`, `GITHUB_TOKEN`). Never commit them.

Upstream clones belong in `vendor/` and stay gitignored. Refresh with `scripts/fetch-upstream.sh`.
