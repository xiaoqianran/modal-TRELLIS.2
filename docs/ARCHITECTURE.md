# 怎么重建 modal-TRELLIS.2

结论先说：**不要抄 Meshii，也不要把官方 TRELLIS.2 fork 进这个仓库。**  
照 `modal-sana` 的拆法：Interface / Core / Modal 分开。第一版只保证一件事——

```text
图片字节 → ImageTo3DGenerator → GLB 文件
```

CLI 和 Web 走同一条 Core。默认 generator 是官方 `microsoft/TRELLIS.2-4B`。`--dry-run` 才用 `MockGenerator` 回立方体。

## 三个上游各自干什么

| 仓库 | 角色 | 这版怎么用 |
| --- | --- | --- |
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | 唯一模型真相。`Trellis2ImageTo3DPipeline.run(image)` → O-Voxel mesh，再 `o_voxel.postprocess.to_glb(...)` | Modal worker **只**调这条官方路径 |
| [Archerkattri/fast-trellis2](https://github.com/Archerkattri/fast-trellis2) | 训练无关加速：换 SS / SLaT sampler，大约 1.9× | **第二刀**。官方生产路径稳定后，只在独立实验入口验证 sampler |
| [sciences44/meshii](https://github.com/sciences44/meshii) | Modal 镜像、预编译 wheel、PBR 导出参数的现场笔记 | **只借基础设施**，不借多模型平台、游戏管线、React 壳 |

Meshii 不适合当底板：它同时接 TRELLIS 1 / TRELLIS 2 / PartPacker，还带 game-ready / print-ready。`backend/api/routes.py` 里同步生成仍是占位。你要的是一条能验证的合同，不是又一个编排器。

## 分层

```text
CLI / Web
    ↓
GenerateService          校验图片、建 job、验 GLB、落盘
    ↓
ImageTo3DGenerator
    ├── ModalTrellis2Generator  默认：官方 TRELLIS.2-4B
    └── MockGenerator           --dry-run / 测试
            ↓
        prefetch.py        CPU：HF 下载 4B + DINOv3 + BiRefNet → Volume
        CpuPreprocessor    CPU：BiRefNet 抠图（上传已有 alpha 则本地做）
        Trellis2Worker     GPU：内存快照恢复 → .cuda() → run → to_glb
                           固定 A100-80GB，最多 1 个 container
                           scaledown_window=10，HF offline，不下载
```

GPU 只做两件事：把已经在 CPU 里的权重量到显存，以及官方推理 / `to_glb`（nvdiffrast 必须 CUDA）。  
下载、Volume 写入、抠图、crop 都不准占 GPU。prefetch 写 `/models/manifest.json` 记录 revision/readiness；GPU 只读取它用于 health/telemetry，并设置 `block_network=True` 完全禁止运行时外网。空闲 10 秒释放 A100。

生产 GPU 策略明确为：`min_containers=0`、`max_containers=1`、`buffer_containers=0`、`timeout=10min`、固定 `A100-80GB`。Web 多图使用客户端串行队列，上一任务完成后立即提交下一任务，主动提高同一 warm container 的复用概率；live 队列一次最多 20 张。输入层同时限制上传 ≤20MB、解码后 ≤40MP、`texture_size` ∈ `{256,512,1024}`。突发提交不会横向扩成多张 GPU，而是排队复用同一个 warm container。普通生成路径禁止 `with_options(gpu=...)`，因为不同动态 GPU 配置会形成独立 autoscaling pool；其他 GPU 型号只允许走独立 benchmark/实验入口。

Core **不** import `trellis2` / `torch`。GPU 镜像、wheel、HuggingFace 权重只活在 `src/modal_trellis2/modal/`。

本地 Web 是 FastAPI，不是 `modal serve`，也不是官方 Gradio `app.py`。和 sana 一样：工作台在你机器上，贵的推理在 Modal。

## 默认就是官方权重

HF 许可通过之后，不要再把 dry-run 当主路径。`prefetch` 已经把官方 4B、DINOv3、BiRefNet 写进 Volume。Web / CLI 默认 `--live`。

dry-run 只留给没有 Modal、没有 GPU、或只想测上传/下载的时候。

## 建议顺序

1. **合同**  
   `POST /api/generate` 收图，`GET /api/assets/{id}.glb` 回文件。
2. **官方 GPU（当前）**  
   `modal-trellis2 prefetch`（CPU）→ `deploy` → `pipeline=512`。  
   `health` 默认只查 Volume，不点 GPU。要探活 A100 才加 `--gpu`。
3. **fast-trellis2**  
   worker 里换 sampler。当前生产 API 不预留 accelerator 开关，避免实验配置渗入生产合同。
4. **再谈 Meshii 的后处理**  
   减面、LOD、打印流形——那是第二个产品。

## Modal 镜像

官方 `setup.sh` 要本机编 CUDA 扩展。Modal 上抄 Meshii 已经跑通的路：

- `nvidia/cuda:12.4.0-devel-ubuntu22.04` + Python 3.10 + PyTorch 2.6.0
- clone `microsoft/TRELLIS.2` 并固定到 `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- 装 [JeffreyXiang Space wheels](https://github.com/JeffreyXiang/Storages/releases/tag/Space_Wheels_251210)：`flash_attn_3`、`o_voxel`、`flex_gemm`、`cumesh`、`nvdiffrast`

生产 GPU 固定：`A100-80GB`，当前只开放 512。1024 / cascade / 其他 GPU 必须走独立实验验证，不从普通请求动态切换。

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
