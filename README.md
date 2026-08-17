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

打开 http://127.0.0.1:7863 。默认 **官方 TRELLIS.2-4B**，`pipeline=512`。  
权重只在 CPU prefetch 下载。GPU 离线加载 Volume，空闲 **10 秒**释放。  
只有勾上 dry-run / 传 `--dry-run` 才会回立方体。

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
  HF_TOKEN="$HF_TOKEN" CIVITAI_TOKEN="$CIVITAI_TOKEN" GITHUB_TOKEN="$GITHUB_TOKEN"
```

要用和 `HF_TOKEN` 同一账号同意：

- [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)（权重本身公开）
- [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)（官方图像编码器，gated）

然后 `modal-trellis2 prefetch`（CPU）。查 Volume：`modal-trellis2 prefetch --status` 或 `modal-trellis2 health`。  
`health --gpu` 才会启动 A100。

## 本地对照上游

```bash
./scripts/fetch-upstream.sh
./scripts/index-upstream.sh
```

会拉取官方 TRELLIS.2、fast-trellis2、Meshii。`vendor/` 不进 git。fast-trellis2 是第二刀加速，默认不用。

## 开发

```bash
pytest
```
