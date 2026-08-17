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

Meshii 不适合当底板：它同时接 TRELLIS 1 / TRELLIS 2 / PartPacker，还带 game-ready / print-ready。你要的是一条能验证的合同，不是又一个编排器。

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
        prefetch.py        CPU：HF 下载 + 完整性校验 → model Volume
        CpuPreprocessor    CPU：RMBG/BiRefNet 抠图
        Trellis2Worker     GPU：本地 Volume 构造 pipeline → .cuda() → run → to_glb
                           ↓
                    temporary result Volume
                           ↓
        local generator    下载/校验 → JobStore → 删除远程临时副本
```

GPU 从已经预取好的本地 Volume 构造 TRELLIS pipeline、移动到显存，再做官方推理 / `to_glb`。TRELLIS.2 的 pipeline import 会初始化 `flex_gemm`/Triton，因此不能放进 Modal 的 CPU Memory Snapshot：普通 `snap=True` 阶段没有 GPU driver。模型下载、模型 Volume 写入、抠图、crop 都不占 GPU；**生成后的 GLB 会在 GPU 生命周期内写入临时 result Volume**，以避免把大二进制塞进 Function 返回 payload。远程结果下载并校验后，本地 JobStore 才是持久用户副本。

GPU 设置 `block_network=True`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。prefetch 写 `/models/manifest.json`，live generation 在查找 GPU class 前先调用 CPU-only `prefetch_status`；模型包不完整时不启动 A100。

生产 GPU 策略：`min_containers=0`、`max_containers=1`、`buffer_containers=0`、`timeout=10min`、`scaledown_window=10`、固定 `A100-80GB`。Web 多图严格串行。普通生成路径禁止 `with_options(gpu=...)` 和 `@modal.concurrent`。

当前生产镜像保留已经进入真实运行验证路径的 `flash_attn_3` backend。不要仅凭“某架构通常更适合某版本 FlashAttention”就替换这个 backend；如果要改，必须单独做 live compatibility/latency/VRAM 对比。

每次 generation 记录显存：`after_load`、`before_infer`、`after_infer`、`after_export`、`after_cleanup`，每个阶段含 allocated/reserved/peak/free/total。这个数据才是判断 512 路径在 A100-80GB 是否有余量的依据。

Core **不** import `trellis2` / `torch`。GPU 镜像、wheel、HuggingFace 权重只活在 `src/modal_trellis2/modal/`。

## 默认就是官方权重

HF 许可通过之后，不要再把 dry-run 当主路径。Web / CLI 默认 live。模型源码与主 HF bundle 都固定 revision，CPU prefetch 会验证六个 512 checkpoint 的 config + safetensors，以及 DINOv3 / background-removal bundle。

## 建议顺序

1. **CPU 合同 + 完整性**  
   `uv run modal-trellis2 prefetch` → `uv run modal-trellis2 prefetch --status`，必须 `ok=true`。
2. **Deploy**  
   `uv run modal-trellis2 deploy`；`uv run modal-trellis2 health` 仍然只查 CPU Volume。
3. **官方 GPU（显式付费）**  
   `uv run modal-trellis2 health --gpu` 或 live generate；当前只开放 `pipeline=512`。
4. **fast-trellis2 / 其他 GPU / 1024**  
   独立实验，不渗入生产请求合同。

## Modal 镜像

- `nvidia/cuda:12.4.0-devel-ubuntu22.04` + Python 3.10 + PyTorch 2.6.0
- TRELLIS.2 source 固定 `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- 主模型 HF revision 固定 `af44b45f2e35a493886929c6d786e563ec68364d`
- 当前运行路径使用 JeffreyXiang Space wheels：`flash_attn_3`、`o_voxel`、`flex_gemm`、`cumesh`、`nvdiffrast`

生产 GPU 固定 `A100-80GB`，当前只开放 512；texture 只开放 256/512/1024，默认 decimation 500k。

## 本地索引

上游 clone 在 `vendor/`（gitignore）。

```bash
./scripts/fetch-upstream.sh
./scripts/index-upstream.sh
```
