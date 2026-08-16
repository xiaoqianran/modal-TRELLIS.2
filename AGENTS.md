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
- Default path is `--dry-run` / `MockGenerator`
- Weights: CPU `prefetch_weights` writes `/models/trellis2` on Volume and `commit()`s
- GPU `Trellis2Worker` loads that snapshot when `pipeline.json` exists
- Deploy: `modal deploy -m modal_trellis2.modal.worker`
- Smoke: `modal run -m modal_trellis2.modal.smoke`
- Local web/CLI are **not** `modal serve`

Tokens live in Modal secret `huggingface-secret` (`HF_TOKEN`, `CIVITAI_TOKEN`, `GITHUB_TOKEN`). Never commit them.

Upstream clones belong in `vendor/` and stay gitignored. Refresh with `scripts/fetch-upstream.sh`.
