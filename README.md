# modal-TRELLIS.2

本地上传一张图，对面还一个 **GLB**。Web 把网格放到转台上。  
[Modal](https://modal.com) 以后跑 [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)；第一版先把合同打直。

```text
图片 → Core → GLB 文件 → 本地转台
         └── dry-run 时是 tinted cube
         └── live 时才是 microsoft/TRELLIS.2-4B
```

CLI 和 Web 共用 `GenerateService`。拆法和 [modal-sana](https://github.com/xiaoqianran/modal-sana) 一样：Interface / Core / Modal。为什么不直接搬 Meshii、fast-trellis2 何时再上，见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 现在就能跑

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

modal-trellis2 doctor
modal-trellis2 generate path/to/photo.png -o /tmp/mesh.glb
modal-trellis2 web
```

打开 http://127.0.0.1:7863 。默认 **dry-run**：不消耗 Modal 额度，回一个从原图取色的立方体。用它确认：

1. 图片能上传  
2. 对面能还文件  
3. 浏览器能转着看、能下载

## 接口

```http
POST /api/generate
Content-Type: multipart/form-data

image=<file>&pipeline=512&seed=42&dry_run=true
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
modal skills install --yes --claude
```

`TRELLIS.2-4B` 目前不需要额外 license 点击。Civitai / GitHub token 留给以后的资源，这版推理只用 `HF_TOKEN`。

## 下一步

1. `modal run -m modal_trellis2.modal.smoke` 确认账号和 secret
2. `modal-trellis2 prefetch` 把 4B 权重拉进 Volume（CPU）
3. `modal deploy -m modal_trellis2.modal.worker`
4. 工作台关掉 dry-run，**只开 `512` 管线** 打第一枪
5. 官方路径稳定后再接 [fast-trellis2](https://github.com/Archerkattri/fast-trellis2) 的 sampler

## 本地对照上游

```bash
./scripts/fetch-upstream.sh
./scripts/index-upstream.sh
```

会拉取官方 TRELLIS.2、fast-trellis2、Meshii，并用 [CodeGraph](https://github.com/colbymchenry/codegraph) 建索引。`vendor/` 不进 git。

## 开发

```bash
pytest
```
