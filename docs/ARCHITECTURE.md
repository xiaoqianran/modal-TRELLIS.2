# 怎么重建 modal-TRELLIS.2

结论先说：**不要抄 Meshii，也不要把官方 TRELLIS.2 fork 进这个仓库。**  
照 `modal-sana` 的拆法：Interface / Core / Modal 分开。第一版只保证一件事——

```text
图片字节 → ImageTo3DGenerator → GLB 文件
```

CLI 和 Web 走同一条 Core。没有 GPU 时用 `MockGenerator` 回一个合法立方体，把上传、落盘、下载、三维预览先跑通。

## 三个上游各自干什么

| 仓库 | 角色 | 这版怎么用 |
| --- | --- | --- |
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | 唯一模型真相。`Trellis2ImageTo3DPipeline.run(image)` → O-Voxel mesh，再 `o_voxel.postprocess.to_glb(...)` | Modal worker **只**调这条官方路径 |
| [Archerkattri/fast-trellis2](https://github.com/Archerkattri/fast-trellis2) | 训练无关加速：换 SS / SLaT sampler，大约 1.9× | **第二刀**。官方路径稳定后再挂 `accelerator=fast` |
| [sciences44/meshii](https://github.com/sciences44/meshii) | Modal 镜像、预编译 wheel、PBR 导出参数的现场笔记 | **只借基础设施**，不借多模型平台、游戏管线、React 壳 |

Meshii 不适合当底板：它同时接 TRELLIS 1 / TRELLIS 2 / PartPacker，还带 game-ready / print-ready。`backend/api/routes.py` 里同步生成仍是占位。你要的是一条能验证的合同，不是又一个编排器。

## 分层

```text
CLI / Web
    ↓
GenerateService          校验图片、建 job、验 GLB、落盘
    ↓
ImageTo3DGenerator
    ├── MockGenerator           --dry-run / 测试
    └── ModalTrellis2Generator  已 deploy 的 Trellis2Worker
            ↓
        prefetch_weights (CPU, Volume)
            ↓
        Trellis2Worker.generate (GPU)
            pipeline.run → o_voxel.to_glb → bytes
```

Core **不** import `trellis2` / `torch`。GPU 镜像、wheel、HuggingFace 权重只活在 `src/modal_trellis2/modal/`。

本地 Web 是 FastAPI，不是 `modal serve`，也不是官方 Gradio `app.py`。和 sana 一样：工作台在你机器上，贵的推理在 Modal。

## 为什么先 dry-run

TRELLIS.2-4B 至少 24GB 显存。512³ 官方 H100 大约 3 秒，Modal A100 会更慢、更贵。镜像还要装 flash-attn / o-voxel / nvdiffrast。

如果第一天就把 Web 绑死在真实推理上，你会同时调试：上传、CORS、GLB 查看器、CUDA 镜像、HF 许可、OOM。  
先让 Mock 回一个 `glTF` 头合法的立方体（颜色来自原图均值）。合同对了，再换 generator。

## 建议顺序

1. **合同（本仓库）**  
   `POST /api/generate` 收图，`GET /api/assets/{id}.glb` 回文件。Web 用 three.js 转台看。
2. **官方 GPU**  
   `modal deploy -m modal_trellis2.modal.worker`  
   先 `pipeline=512`、`texture_size=1024`。权重用 CPU `prefetch_weights` 写进 Volume，GPU 只加载。
3. **fast-trellis2**  
   worker 里换 sampler。API 已留 `accelerator` 字段，先不要实现。
4. **再谈 Meshii 的后处理**  
   减面、LOD、打印流形——那是第二个产品。

## Modal 镜像

官方 `setup.sh` 要本机编 CUDA 扩展。Modal 上抄 Meshii 已经跑通的路：

- `nvidia/cuda:12.4.0-devel-ubuntu22.04` + Python 3.10 + PyTorch 2.6.0
- clone `microsoft/TRELLIS.2`
- 装 [JeffreyXiang Space wheels](https://github.com/JeffreyXiang/Storages/releases/tag/Space_Wheels_251210)：`flash_attn_3`、`o_voxel`、`flex_gemm`、`cumesh`、`nvdiffrast`

默认 GPU：`A100-80GB`。512 可以降到 `A100`。1536 cascade 再考虑 H100。

## 本地索引

上游 clone 在 `vendor/`（gitignore）。CodeGraph 索引也在各自 `.codegraph/` 里。

```bash
./scripts/fetch-upstream.sh
./scripts/index-upstream.sh
```

本环境已建过一次：

| 项目 | 文件 | 节点 | 边 |
| --- | ---: | ---: | ---: |
| TRELLIS.2 | 154 | 2558 | 4728 |
| fast-trellis2 | 124 | 1533 | 2955 |
| meshii | 62 | 888 | 1867 |
