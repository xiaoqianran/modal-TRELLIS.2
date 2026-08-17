# Validation

这份清单把**零 GPU 成本验证**和**显式付费 GPU 验收**分开。默认只执行前者。

## 1. 本地 / CI：零 GPU

```bash
uv run python -m compileall -q src tests
uv run ruff check src tests
uv run pytest

PYTHONPATH=src uv run python - <<'PY'
import modal_trellis2.application
import modal_trellis2.core.service
import modal_trellis2.cli.app
import modal_trellis2.web.server
print("import smoke: ok")
PY
```

本地 dry-run：

```bash
uv run modal-trellis2 generate path/to/photo.png -o /tmp/mesh.glb --dry-run
uv run modal-trellis2 web
```

这些路径不得启动 Modal GPU。

## 2. CPU 模型准备：仍不启动 GPU

```bash
uv run modal-trellis2 prefetch
uv run modal-trellis2 prefetch --status
```

`prefetch --status` 必须返回 `ok=true` 才进入 deploy/live 阶段。完整性检查包含：TRELLIS.2 `pipeline.json`、生产 512 所需六个 checkpoint 的 `.json + .safetensors`、DINOv3 完整权重，以及 RMBG/BiRefNet 至少一个完整可加载 bundle。主 TRELLIS.2 HF bundle 与源码均固定 revision。

## 3. Deploy / CPU health

```bash
uv run modal-trellis2 deploy
uv run modal-trellis2 health
```

`health` 默认只检查 CPU Volume，不启动 A100。live generator 也会在查找 `Trellis2Worker` 前再次执行 CPU-only `prefetch_status`，避免模型包问题到 A100 上才暴露。

## 4. 显式付费 GPU 验收

只有明确决定承担 GPU 费用时执行：

```bash
uv run modal-trellis2 health --gpu

uv run modal-trellis2 verify-gpu-reuse path/to/photo.png \
  --count 3 \
  --confirm-cost
```

连续请求应返回相同 `container_instance_id`，证明同一 warm GPU container 被复用。

真实 generation telemetry 还必须检查：

```text
vram.after_load
vram.before_infer
vram.after_infer
vram.after_export
vram.after_cleanup
```

每个阶段记录 `allocated_gb / reserved_gb / peak_allocated_gb / peak_reserved_gb / free_gb / total_gb`。第一次 512 live run 后，用真实 peak 判断 A100-80GB 余量。

要额外验证 10 秒 scale-to-zero：

```bash
uv run modal-trellis2 verify-gpu-reuse path/to/photo.png \
  --count 3 \
  --check-scale-down \
  --confirm-cost
```

该命令会在连续请求后等待 15 秒，再主动发起一次新的付费探针；预期新的 `container_instance_id` 与之前不同。

## 5. 输出传输验收

生产 GLB 不直接作为 GPU Function 的大 bytes 返回：

```text
GPU worker
  → /outputs/jobs/<job_id>/mesh.glb
  → output_volume.commit()
  → small metadata {output_path, size_bytes, ...}
  → local Volume.read_file()
  → verify size
  → JobStore save
  → Volume.remove_file()
```

本地最终 GLB 必须通过 `glTF` magic / length 校验。若远程临时文件删除失败，生成仍可成功，但 telemetry 的 `output_cleanup_error` 必须记录清理错误，便于后续处理存储残留。

## 6. 成本不可变量

```text
GPU = A100-80GB
min_containers = 0
max_containers = 1
buffer_containers = 0
scaledown_window = 10s
per-input timeout = 10min
pipeline = 512 only
texture_size ∈ {256, 512, 1024}
TRELLIS.2 source = 75fbf0183001ed9876c8dbb35de6b68552ee08bd
TRELLIS.2 model = af44b45f2e35a493886929c6d786e563ec68364d
no with_options(gpu=...)
no @modal.concurrent
block_network = True
no GPU Memory Snapshot import phase
no late-loading missing models after .cuda()
```

输入保护：

```text
upload <= 20MB
decoded image <= 40MP
remote model input longest side <= 1024
Web live batch <= 20 images
```

当前生产 TRELLIS image 保留已经进入真实运行验证路径的 `flash_attn_3` backend。不要在无 live 验证的情况下仅凭通用 GPU 架构假设切换 backend。

## 本次审计说明

这轮仓库审计本身没有由本流程启动新的 Modal GPU。代码/CI 侧验证覆盖 compileall、Ruff、pytest、import smoke、Python 3.10 远程源码语法和新增的模型完整性/输入边界/Job 状态/路径安全/输出 Volume/显存 telemetry 契约。实际 CUDA 显存峰值仍必须来自显式付费 live run，不能由静态测试伪造。
