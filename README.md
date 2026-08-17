# modal-TRELLIS.2

本地上传一张图，对面还一个 **GLB**。Web 把网格放到转台上。  
推理走官方 [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)，跑在 [Modal](https://modal.com) GPU 上。

```text
图片 → Core → official TRELLIS.2-4B → GLB → 本地转台
         └── --dry-run 才是 tinted cube
```

CLI 和 Web 共用 `GenerateService`。拆法和 [modal-sana](https://github.com/xiaoqianran/modal-sana) 一样：Interface / Core / Modal。见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 现在就能跑

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

modal-trellis2 doctor
modal-trellis2 prefetch          # CPU 下载官方 4B + DINOv3 + BiRefNet → Volume
modal-trellis2 deploy            # 注册 CPU 抠图 + GPU worker（modal_trellis2.modal.deploy）
modal-trellis2 health            # 只查 Volume，不点 GPU
modal-trellis2 generate path/to/photo.png -o /tmp/mesh.glb
modal-trellis2 web
```

打开 http://127.0.0.1:7863 。默认 **官方 TRELLIS.2-4B**，当前生产合同只开放 `pipeline=512`。  
权重只在 CPU prefetch 下载。GPU 离线加载 Volume，空闲 **10 秒**释放。  
生产 GPU 固定为 **A100-80GB**，最多 **1 个 GPU container**；连续请求排队并优先复用同一个 warm container，不会因为突发提交横向扩成多张 GPU。  
只有勾上 dry-run / 传 `--dry-run` 才会回立方体。生产输入额外限制为：上传文件 ≤20MB、解码后 ≤4000 万像素、`texture_size` 只能是 `256/512/1024`；Web live 队列一次最多 20 张，避免单次误操作长时间占住唯一 GPU。

## 成本策略

```text
Hugging Face
    ↓
CPU prefetch → Modal Volume
                  ↓
RGB 上传 → CPU rembg（可 warm 5 分钟）
                  ↓
              GPU Queue
                  ↓
        唯一 A100-80GB container
           job1 → job2 → job3
                  ↓
             idle 10 sec
                  ↓
              scale to zero
```

生产 worker 明确使用：`min_containers=0`、`max_containers=1`、`buffer_containers=0`、`scaledown_window=10`、单输入最长 10 分钟；普通生成路径不使用动态 `with_options(gpu=...)`，也不使用 `@modal.concurrent`。GPU 设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，并启用 `block_network=True`，运行时整个外网被封死，不会在昂贵 GPU 上下载模型。

需要测试其他 GPU 时应走独立 benchmark/实验路径，不要改变生产生成请求的 GPU 配置。

## 连续任务 / 同容器复用

Web 文件选择器支持多图。多张图片不会并发提交，而是在浏览器中严格串行：

```text
job1 完成 → 立即 job2 → 立即 job3 → ...
```

配合生产 `max_containers=1`，这既避免横向扩 GPU，也尽量让连续任务落在同一个 10 秒 warm container。每次 live generation 的 telemetry 都返回 `container_instance_id`，可显式验收：

```bash
# 会真实启动并计费 GPU；没有 --confirm-cost 时命令会拒绝执行
modal-trellis2 verify-gpu-reuse image.png --count 3 --confirm-cost

# 额外等待 15 秒，再做一次冷启动探针，验证 scale-to-zero 后 container id 已变化
modal-trellis2 verify-gpu-reuse image.png --count 3 --check-scale-down --confirm-cost
```

普通 pytest / CI 永远不会执行这个 live probe。Web live 多图一次最多 20 张；如果要处理更多，分批提交即可，避免浏览器一次排入过长的付费队列。

## 接口

```http
POST /api/generate
Content-Type: multipart/form-data

image=<file>&pipeline=512&seed=42&dry_run=false
```

```json
{
  "id": "job_01…",
  "status": "completed",
  "asset_url": "/api/assets/job_01….glb",
  "glb_size_bytes": 1840
}
```

`GET /api/assets/{id}.glb` 的 `Content-Type` 是 `model/gltf-binary`。

## 凭证（不要写进 git）

```bash
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
modal secret create --force huggingface-secret \
  HF_TOKEN="$HF_TOKEN"
```

要用和 `HF_TOKEN` 同一账号同意：

- [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)（权重本身公开）
- [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)（官方图像编码器，gated）

GPU 镜像中的官方 TRELLIS.2 源码固定到 commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd`，避免未来重新 build 时随上游 `main` 漂移。

首次顺序固定为：`modal-trellis2 prefetch`（临时 CPU App）→ `modal-trellis2 prefetch --status` → `modal-trellis2 deploy`。`modal-trellis2 health` 默认仍只查 CPU Volume；只有 `health --gpu` 才会启动 A100。prefetch 还会写 `/models/manifest.json`，记录模型 revision / readiness，供 GPU health 和生成 telemetry 对照。

## 本地对照上游

```bash
./scripts/fetch-upstream.sh
./scripts/index-upstream.sh
```

会拉取官方 TRELLIS.2、fast-trellis2、Meshii。`vendor/` 不进 git。fast-trellis2 是第二刀加速，默认不用。

## 开发

完整的零 GPU / 显式付费验收清单见 [`docs/VALIDATION.md`](docs/VALIDATION.md)。

```bash
python -m compileall -q src tests
pytest
# CI 还会执行：ruff check src tests + import smoke
```
