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

首次顺序：

```bash
uv run modal-trellis2 prefetch
uv run modal-trellis2 prefetch --status
```

`prefetch` 使用临时 CPU App 下载并 `commit()` 到 Volume；ready 必须同时满足 TRELLIS.2、DINOv3、sparse decoder，以及 RMBG/BiRefNet 至少一个可用。它还会写 `/models/manifest.json`，记录模型 revision 与 readiness。

## 3. Deploy / CPU health

```bash
uv run modal-trellis2 deploy
uv run modal-trellis2 health
```

`health` 默认只检查 CPU Volume，不启动 A100。

## 4. 显式付费 GPU 验收

只有明确决定承担 GPU 费用时执行：

```bash
uv run modal-trellis2 health --gpu

uv run modal-trellis2 verify-gpu-reuse path/to/photo.png \
  --count 3 \
  --confirm-cost
```

连续请求应返回相同 `container_instance_id`，证明同一 warm GPU container 被复用。

真实 generation telemetry 还必须检查 `vram.after_load / before_infer / after_infer / after_export / after_cleanup`；每阶段包含 `allocated_gb / reserved_gb / peak_allocated_gb / peak_reserved_gb / free_gb / total_gb`。

要额外验证 10 秒 scale-to-zero：

```bash
uv run modal-trellis2 verify-gpu-reuse path/to/photo.png \
  --count 3 \
  --check-scale-down \
  --confirm-cost
```

该命令会在连续请求后等待 15 秒，再主动发起一次新的付费探针；预期新的 `container_instance_id` 与之前不同。

## 5. 成本不可变量

生产路径必须持续满足：

```text
GPU = A100-80GB
min_containers = 0
max_containers = 1
buffer_containers = 0
scaledown_window = 10s
per-input timeout = 10min
pipeline = 512 only
TRELLIS.2 source = 75fbf0183001ed9876c8dbb35de6b68552ee08bd
no with_options(gpu=...)
no @modal.concurrent
block_network = True
large GLB output = dedicated Modal Volume, never Function bytes
remote image payload <= 1,800,000 bytes
```

输入成本保护：

```text
upload <= 20MB
decoded image <= 40MP
model longest side <= 1024
remote image payload <= 1,800,000 bytes
texture_size ∈ {256, 512, 1024}
Web live batch <= 20 images
```

GPU Worker 没有 Hugging Face secret，且运行时网络被封锁；所有模型下载必须发生在 CPU prefetch。

## 当前修复环境说明

本仓库修复过程中已执行本地 pytest、compileall、CLI dry-run、FastAPI TestClient 和 JavaScript syntax check。当前执行环境没有 Modal CLI，因此**没有执行任何 live GPU 验收，也没有产生 GPU 费用**。live reuse / scale-to-zero 需要在有 Modal 凭证与 CLI 的环境中按上面的显式命令执行。

## 6. 输出传输防回归

零 GPU 静态/单测必须确认：

```text
modal-trellis2-results Volume 已定义
Trellis2Worker 挂载 /outputs
GLB 写入 Volume 后 commit
GPU generate 不 return glb_bytes
本地 generator 使用 Volume.read_file + remove_file
准备给 CPU/GPU 的图片 <= 1,800,000 bytes
```

真实失败日志中 TRELLIS 三段 sampling 已全部完成，错误发生在 Modal 尝试把大返回值上传 blob object storage 时。因此这一组是生产传输合同，不是性能微调。
