# Agent notes

This repo follows the same split as `modal-sana`:

- `src/modal_trellis2/core/` — image in, GLB out. No torch. No TRELLIS imports.
- `src/modal_trellis2/modal/` — the only place that may import `trellis2` / `o_voxel`.
- `src/modal_trellis2/web/` and `cli/` — thin interfaces over `GenerateService`.

Default path is `--dry-run` / `MockGenerator`. Do not make the first web loop depend on a live GPU.

Upstream clones belong in `vendor/` and stay gitignored. Refresh with `scripts/fetch-upstream.sh` and `scripts/index-upstream.sh`.

Read `docs/ARCHITECTURE.md` before adding models, post-process pipelines, or a React shell.
