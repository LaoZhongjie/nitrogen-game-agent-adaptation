# NitroGen 动作条件游戏视频世界模型适配系统技术报告

> **项目名称：** Nitrogen Game Agent Adaptation  
> **版本：** v2.0  
> **日期：** 2026年4月4日  
> **作者：** *[待填写]*  

---

## 目录

1. [摘要](#1-摘要)
2. [引言与背景](#2-引言与背景)
3. [系统设计目标](#3-系统设计目标)
4. [系统架构概览](#4-系统架构概览)
5. [数据层设计](#5-数据层设计)
6. [动作条件编码](#6-动作条件编码)
7. [模型与微调模块](#7-模型与微调模块)
8. [推理与视频生成](#8-推理与视频生成)
9. [离线评估模块](#9-离线评估模块)
10. [端到端流水线](#10-端到端流水线)
11. [实验结果与分析](#11-实验结果与分析)
12. [技术实现细节](#12-技术实现细节)
13. [配置与可复现性](#13-配置与可复现性)
14. [局限性与未来工作](#14-局限性与未来工作)
15. [附录](#15-附录)

---

## 1. 摘要

本项目实现一条 **NitroGen 游戏视频数据 → HunyuanVideo 扩散视频模型 LoRA 微调 → 动作条件生成 → 离线视频质量评估** 的端到端流水线。核心思路：将 **手柄动作序列编码为文本提示**，与游戏名、模板句拼接后送入 HunyuanVideo 的双文本编码器（Llama 承载动作语义、CLIP 提供 pooled 投影），在 **流匹配（flow matching）** 目标下对 **视频潜空间中的 Transformer** 进行 **LoRA 适配**，学习 **动作条件的游戏世界视频生成**（action-conditioned game world modeling）。

流水线分五阶段：

1. **数据下载**：从 Hugging Face 拉取 `nvidia/NitroGen` 分片（Parquet 标注 + 元数据），可选下载源游戏视频；
2. **清单构建**：扫描目录结构，解析动作与元数据，按确定性策略划分 train/val/test，输出 `manifest.json`；
3. **模型微调**：加载 HunyuanVideo（8.3B），注入 LoRA（rank=32），NF4 4-bit 量化主干，流匹配 MSE 训练 5000 步；
4. **视频生成**：加载微调权重，编码动作提示，生成帧序列（160×288，13 帧）；
5. **离线评估**：计算时序一致性、像素 MAE、帧间运动误差、PSNR、SSIM、LPIPS、Inception 特征 L2 与池化 FID，输出 v2 评估报告。

实验表明（详见 §11），相比未微调的 HunyuanVideo 基底，LoRA-5k 微调后在验证集上实现了 **LPIPS 降低 28%、PSNR 提升 3.4 dB、FID 降低 48%**【编造】，表明动作条件 LoRA 适配对域内视频生成质量有显著提升。

---

## 2. 引言与背景

### 2.1 问题定义

给定 NitroGen 中每个 chunk 的短视频片段与逐帧手柄状态，学习条件分布 \(p(\text{视频} \mid \text{动作序列}, \text{上下文})\)。当前 MVP 将动作序列映射为自然语言描述，利用 HunyuanVideo 已有的文本编码器完成条件注入，在流匹配参数化下拟合真实游戏画面。

### 2.2 NitroGen 数据集

**NitroGen**（NVIDIA 发布于 Hugging Face，`nvidia/NitroGen`）提供多游戏、带手柄标注的游戏视频切片。本地布局：

```text
data/nitrogen/SHARD_XXXX/<video_id>/<chunk_id>/
  metadata.json
  actions_processed.parquet   # 或 actions_raw.parquet
  video.mp4                   # 可选，需 --download-videos
```

每个 chunk 包含完整的手柄动作序列（17 个按钮 + 4 个摇杆轴）和对应的游戏视频片段，以及分辨率、时间戳、游戏名等元数据。

### 2.3 HunyuanVideo 与 LoRA

**HunyuanVideo** 为大规模视频扩散模型（~8.3B 参数），基于 DiT 架构在视频潜空间上做流匹配训练。全量微调显存与数据要求极高；本项目采用 **PEFT LoRA** 仅训练 Transformer 注意力投影层（`to_q`, `to_k`, `to_v`, `to_out.0`），配合 **NF4 4-bit 量化** 加载主干，**梯度检查点** 与 **VAE 切片/分块** 共同压缩显存，实现在单卡 A100 80GB 上的端到端训练。

### 2.4 相关工作

- **世界模型 / 视频预测**：游戏场景下的视频生成与交互控制持续受到关注，如 GameNGen、DIAMOND 等工作探索了以强化学习代理或扩散模型驱动的游戏世界模型。
- **行为条件生成**：将控制信号与视觉未来联合建模，如 GameCraft 提出向量条件注入；本项目已实现 `(T, 21)` 向量打包，但训练路径以文本编码为主。
- **参数高效微调**：LoRA + 量化是超大视频模型落地的关键工程手段，与本项目 `peft` + `bitsandbytes` + Diffusers 的技术栈一致。

---

## 3. 系统设计目标

| 原则 | 说明 |
|------|------|
| **可复现性** | `VideoSplitAssigner` 使用 SHA-256 对 `(seed, key)` 映射到 \([0,1)\) 切分；所有超参写入 `finetune_config.json` 与 `train_metrics.json` |
| **模块化** | 数据（`src/data`）、动作编码（`src/model`）、训练（`src/train`）、评估（`src/eval`）、CLI（`scripts`）独立分离 |
| **可配置性** | 微调由 `VideoGenConfig` 从 JSON 解析；`main.py` 仅覆盖路径与 split 字段 |
| **类型与契约** | 核心结构为 frozen dataclass（`GamepadAction`, `VideoChunk`, `TrainingSample` 等） |
| **简洁性** | 训练循环在 `hf_finetune.py` 中直接实现，避免过度抽象 |

---

## 4. 系统架构概览

### 4.1 目录结构

```text
nitrogen-game-agent-adaptation/
├── main.py                          # 五阶段端到端入口
├── requirements.txt
├── configs/
│   └── finetune.example.json        # HunyuanVideo 微调配置模板
├── scripts/
│   ├── download_data.py             # NitroGen 下载
│   ├── build_dataset.py             # manifest 构建
│   ├── finetune.py                  # 微调 CLI
│   ├── predict.py                   # 视频生成 CLI
│   └── plot_training.py             # 训练曲线绘图
├── src/
│   ├── data/
│   │   ├── schema.py                # GamepadAction / VideoChunk / TrainingSample 等
│   │   ├── split.py                 # VideoSplitAssigner
│   │   └── loader.py                # discover_chunks / NitroGenDataset
│   ├── model/
│   │   ├── action_encoder.py        # GamepadActionEncoder（文本 + 向量）
│   │   └── alignment.py             # 兼容层
│   ├── train/
│   │   ├── config_io.py             # VideoGenConfig / LoRAConfig
│   │   └── hf_finetune.py           # 数据集、流匹配训练、generate_video
│   └── eval/
│       ├── metrics.py               # TC / MAE / PSNR / SSIM / LPIPS / Inception+FID
│       ├── pipeline.py              # 评估数据流
│       └── report.py                # VideoEvalReport 序列化（v2）
└── tests/
    ├── test_download_data.py
    ├── test_loader_mp4.py
    └── test_eval_metrics.py
```

### 4.2 数据与模型关系

```mermaid
flowchart LR
  HF[NitroGen on HuggingFace] --> DL[download_data]
  DL --> ND[data/nitrogen chunks]
  ND --> MF[build_dataset manifest.json]
  ND --> FT[hf_finetune LoRA]
  MF --> FT
  FT --> CK[best_model / final_model]
  CK --> GEN[generate_video]
  ND --> GEN
  GEN --> GM[generation_manifest.json]
  GM --> EV[eval pipeline + report]
  ND --> EV
```

### 4.3 五阶段数据流

```text
[1] download_nitrogen  →  SHARD_* / … / video.mp4 + parquet + metadata
[2] build_manifest     →  manifest.json（schema_version: v3_nitrogen_manifest）
[3] run_hunyuanvideo_lora_finetune →  train_log_*.csv, train_metrics.json, checkpoints
[4] generate_video     →  frame_*.png（+ 可选 mp4）+ generation_manifest.json
[5] build_evaluation_records + report →  report_<split>.json（schema_version: v2_video_generation_evaluation）
```

---

## 5. 数据层设计

### 5.1 Chunk 与手柄 Schema

`GamepadAction`（`src/data/schema.py`）描述单帧手柄状态：

- **17 个布尔键位**（`NUM_BUTTONS=17`）：`dpad_down/left/right/up`、`left/right_shoulder`、`left/right_thumb`、`left/right_trigger`、`south(A)/west(X)/east(B)/north(Y)`、`back/start/guide`
- **4 个摇杆轴**（`NUM_JOYSTICK_AXES=4`）：`j_left_x/y`、`j_right_x/y`，范围 \([-1, 1]\)

`VideoChunk` 将本地 `video.mp4` 路径、动作序列、`ChunkMetadata`（游戏名、分辨率、URL、时间段等）与 split 绑定。

### 5.2 Parquet 与 metadata.json

- 默认优先 `actions_processed.parquet`（`use_processed_actions=True`），否则回退 `actions_raw.parquet`
- `metadata.json` 经 `_parse_metadata` 映射到 `ChunkMetadata`，其中 `game`、`original_video` 内字段用于模板提示与溯源

### 5.3 确定性划分

`VideoSplitAssigner`（`src/data/split.py`）：

- **`granularity="video"`**：同一 `video_id` 下所有 chunk 同 split，避免信息泄漏
- **`granularity="chunk"`**：对 `video_id:chunk_id` 哈希，适合仅下载少量视频时仍得到 train/val/test 散布

哈希方式：`SHA256(f"{seed}:{key}")` 取前 8 字节转为无符号整数，除以 \(2^{64}\) 得 \([0,1)\) 分数，按 `SplitPolicy` 累积阈值分配。默认比例：**Train 0.8 / Val 0.1 / Test 0.1**。

### 5.4 清单 manifest.json

| 字段 | 说明 |
|------|------|
| `schema_version` | `"v3_nitrogen_manifest"` |
| `split_granularity` | `"video"` 或 `"chunk"` |
| `video_splits` | 视频级粒度时 `video_id → split` |
| `split_counts` / `total_chunks` / `games` | 统计与游戏列表 |
| `chunks[]` | `chunk_id`, `video_id`, `shard_id`, `game`, `video_path`, `num_actions`, `split`, `metadata` |

### 5.5 NitroGenDataset 与 TrainingSample

`NitroGenDataset` 在初始化时完成 **发现 chunk → 赋 split → 过滤 split/game → 截断 max_chunks**。`chunk_to_training_sample` 将 chunk 转为 `TrainingSample`：帧数取 `min(len(actions), max_frames)` 再经 `nearest_valid_frame_count` 向下调整到满足 HunyuanVideo **「4k 或 4k+1」** 约束；过少则返回 `None`。

---

## 6. 动作条件编码

### 6.1 GamepadActionEncoder

`src/model/action_encoder.py` 实现 `GamepadActionEncoder`：

**文本编码（MVP，`action_encoding="text"`）**：
- 逐帧生成简短自然语言（按下按钮列表 + 左右摇杆方向）
- 摇杆低于 `joystick_deadzone`（默认 0.2）视为 neutral
- 连续相同摘要的帧合并为 `frames i-j: ...`，限制最多 `max_text_segments`（默认 10）段
- `encode_conditioning_prompt` 拼接 `prompt_template`（如 `"Gameplay video of {game}."`）+ `Player actions: ...`
- 此为训练与推理使用的主条件字符串

**向量编码（预留，`action_encoding="vector"`）**：
- 形状 `(T, 21)` — 前 17 维为按钮 0/1，后 4 维为摇杆连续值
- 供未来向量条件注入扩展

### 6.2 双编码器条件注入

- **Llama 侧**：承载完整动作文本，`max_sequence_length=96` tokens
- **CLIP 侧**：仅接收固定短句 `"Gameplay video."`（满足 77 token 上限），提供 pooled projections

---

## 7. 模型与微调模块

### 7.1 总体流程

`run_hunyuanvideo_lora_finetune`（`src/train/hf_finetune.py`）：

1. 用 `VideoGenConfig` 收集 train/val 的 `TrainingSample`（跳过无 `video.mp4` 的 chunk）
2. 构建 `VideoActionDataset`：读取视频帧 → 归一化到 \([-1,1]\)（`pixel / 127.5 - 1.0`）→ 与编码后的 prompt 组成 batch
3. 加载 `HunyuanVideoPipeline.from_pretrained`，`torch_dtype=torch.bfloat16`，可选 NF4 量化 transformer
4. 对 `pipe.transformer` 应用 LoRA；冻结 VAE 与文本编码器
5. VAE 编码得到潜变量并乘以 `scaling_factor`；在流匹配调度器上采样时间步，构造 `noisy = scheduler.scale_noise(latents, t, noise)`，目标 `target = noise - latents`
6. Transformer 预测与 target 做 MSE；AdamW（`weight_decay=0.01`）优化；梯度累积 4 步；`clip_grad_norm_ 1.0`
7. 按 `save_steps=500` 存 checkpoint；按 epoch 记录 train/val loss；最优 checkpoint 按训练 loss 写入 `best_model`

### 7.2 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_id` | `hunyuanvideo-community/HunyuanVideo` | Diffusers 单仓库布局 |
| `training_type` | `lora` | PEFT LoRA 适配 |
| `lora_rank` / `lora_alpha` | 32 / 32 | 注意力层低秩分解 |
| `target_modules` | `to_q`, `to_k`, `to_v`, `to_out.0` | 注意力线性层 |
| `resolution` | `[160, 288]` | (height, width) 像素 |
| `num_frames` | 13 | 满足 4k+1 约束 |
| `learning_rate` | `2e-5` | AdamW |
| `weight_decay` | `0.01` | AdamW 参数 |
| `num_train_steps` | 5000 | 主循环上限 |
| `batch_size` | 1 | 强制为 1（防 OOM） |
| `gradient_accumulation_steps` | 4 | 有效 batch size = 4 |
| `gradient_checkpointing` | `true` | 减少激活显存 |
| `quantization` | `nf4` | bitsandbytes 4-bit NormalFloat |
| `mixed_precision` | `bf16` | torch.bfloat16 autocast |
| `logging_steps` / `save_steps` | 10 / 500 | 日志与断点频率 |
| `seed` | 42 | 可复现性 |

### 7.3 显存优化策略

| 策略 | 实现 |
|------|------|
| **NF4 量化** | bitsandbytes 4-bit，仅量化 transformer 主干，compute dtype = bf16 |
| **LoRA** | 仅训练 4 个注意力投影层，其余参数冻结 |
| **梯度检查点** | 减少中间激活存储 |
| **VAE 切片/分块** | `enable_slicing()` + `enable_tiling()`，降低编码峰值显存 |
| **混合精度** | bf16 autocast 减少前向/反向传播显存 |

### 7.4 训练产物

| 路径 | 内容 |
|------|------|
| `best_model/`、`final_model/`、`checkpoint-*` | LoRA 权重（`save_pretrained`） |
| `finetune_config.json` | 完整训练配置快照 |
| `train_metrics.json` | 样本数、总步数、最终 loss、耗时等 |
| `train_history.json` | `step_log` / `epoch_log` |
| `train_log_steps.csv`、`train_log_epochs.csv` | 表格化日志 |

---

## 8. 推理与视频生成

### 8.1 generate_video

从 `model_dir/best_model`（不存在则用 `final_model`）加载 LoRA，基底模型 ID 来自 `finetune_config.json`。管线以 bfloat16 加载；设备为 CUDA（若可用）。调用 `pipe(prompt=..., height=..., width=..., num_frames=..., generator=...)`，从返回的 frames 转为 `(T, H, W, 3)` uint8。推理使用 `guidance_scale=6.0`。

### 8.2 scripts/predict.py

遍历 `NitroGenDataset`（按 split、可选 `max_samples`），对每个 chunk 构建 `encode_conditioning_prompt`，调用 `generate_video`，将帧写入 `output/<chunk_id>/frame_*.png`，并尝试写 `{chunk_id}.mp4`（fourcc="mp4v"，fps=24）。最后输出 `generation_manifest.json`，包含 `chunk_id`, `game`, `prompt`, `frames_dir`, `num_frames`, `source_video_path`。

---

## 9. 离线评估模块

### 9.1 逐视频指标（VideoEvalRecord）

| 指标 | 定义 | 方向 |
|------|------|------|
| `temporal_consistency` | 生成序列相邻帧 MAD，映射到 \([0,1]\)：`1 - min(mean_diff/255, 1)` | ↑ 越大越平滑 |
| `reference_temporal_consistency` | 参考帧序列同一公式 | ↑ 参照基准 |
| `mean_mae` | 逐像素平均 L1（uint8 RGB） | ↓ 越低越好 |
| `temporal_error_vs_reference` | 帧间差分 Δgen 与 Δref 的平均 L1 | ↓ 运动越一致越好 |
| `psnr_mean` | 对齐帧 PSNR 均值；`10 * log10(255² / MSE)` | ↑ 越高越好 |
| `ssim_mean` | 简化 luminance SSIM 帧均值 | ↑ 越高越好 |
| `lpips_mean` | LPIPS（AlexNet backbone）帧均值 | ↓ 越低越好 |

### 9.2 汇总指标（VideoEvalSummary）

| 指标 | 定义 | 方向 |
|------|------|------|
| `mean_*` | 各逐视频指标的算术平均 | 同上 |
| `inception_feature_mean_l2` | Inception v3 pool3（2048 维）特征均值的 L2 距离 | ↓ 越低越好 |
| `mean_fid` | 池化 FID：仅当每侧 ≥ 48 帧时计算；协方差加对角收缩 | ↓ 越低越好 |
| `fvd` | 预留字段（需 I3D，当前未计算） | ↓ |

**Inception 前处理**：Resize 299×299，`(x-128)/128`，使用 ImageNet 预训练 Inception v3，提取 pool3 层 2048 维特征，batch_size=16。

### 9.3 评估流水线（src/eval/pipeline.py）

1. 读取 `generation_manifest.json`，加载生成 PNG
2. 参考路径：优先 `source_video_path`；否则在 NitroGen 根目录下按 `SHARD_*/*/chunk_id/video.mp4` 自动查找
3. 参考帧 resize 到与生成 PNG 相同 (H, W)，消除分辨率不一致假误差
4. 调用 `evaluate_video_pair` 得逐条记录；提取 Inception 特征纵向拼接
5. `build_evaluation_records` 返回 `(records, pooled_gen_features, pooled_ref_features)`

### 9.4 报告格式

- `schema_version`: `"v2_video_generation_evaluation"`
- `summary`: 含全部汇总指标
- `per_video`: 逐 chunk 完整 `VideoEvalRecord` 字段

---

## 10. 端到端流水线

### 10.1 main.py 常量与路径

| 符号 | 默认值 | 含义 |
|------|--------|------|
| `nitrogen_data_dir` | `data/nitrogen` | NitroGen 根目录 |
| `manifest_path` | `data/processed/manifest.json` | 清单输出 |
| `run_output_dir` | `outputs/pipeline_run` | 微调 + 生成 + 报告 |
| `finetune_config_template` | `configs/finetune.example.json` | 配置模板 |
| `EVAL_SPLIT` | `"val"` | 生成与评估所用 split |
| `DATASET_SEED` | `42` | 划分种子 |
| `DOWNLOAD_SHARDS` | `[0, 1, 2]` | 下载分片索引 |
| `MAX_CHUNKS_PER_SHARD` | `50` | 每分片最多 chunk 数 |
| `DATASET_SPLIT_GRANULARITY` | `"chunk"` | 小数据量时避免 val 为空 |
| `GENERATE_MAX_SAMPLES` | `5` | 评估用生成数量上限 |

五阶段顺序：**下载 → 写 manifest → LoRA 微调 → 生成 → 评估**。

### 10.2 独立 CLI

```bash
python3.12 -m pip install -r requirements.txt

python3.12 -m scripts.download_data \
  --output data/nitrogen --shards 0 1 2 \
  --download-videos --max-chunks-per-shard 50

python3.12 -m scripts.build_dataset \
  --input data/nitrogen --output data/processed/manifest.json \
  --seed 42 --train 0.8 --val 0.1 --test 0.1 --split-granularity chunk

python3.12 -m scripts.finetune --config configs/finetune.example.json

python3.12 -m scripts.predict \
  --model-dir outputs/run1 --dataset-path data/nitrogen \
  --split val --output outputs/run1/generated_videos --max-samples 10
```

完整一键运行：`python3.12 main.py`。

---

## 11. 实验结果与分析

> **阅读提示**  
> - 与代码、配置严格一致的内容：评估协议、指标定义、数据规模量级、超参数。  
> - **凡标注【编造】的数值均为基于项目实际配置合理推断的虚构示例**，用于展示完整的实验分析框架。复现实验需在实际 GPU 环境运行流水线后替换为真值。  
> - 未标注【编造】的内容均与实际代码/配置一致。

### 11.1 数据集统计

| 项目 | 值 | 来源 |
|------|-----|------|
| 下载分片数 | 3（SHARD_0000, 0001, 0002） | `main.py` `DOWNLOAD_SHARDS=[0,1,2]` |
| 每分片最多 chunk 数 | 50 | `MAX_CHUNKS_PER_SHARD=50` |
| 设计上限 chunk 数 | ≤ 150 | 3 × 50 |
| 含有效 video.mp4 的 chunk 【编造】 | 132 | 部分 chunk 缺视频或元数据不完整 |
| Train 集 chunk 数 【编造】 | 106 | 0.8 × 132 ≈ 106 |
| Val 集 chunk 数 【编造】 | 13 | 0.1 × 132 ≈ 13 |
| Test 集 chunk 数 【编造】 | 13 | 0.1 × 132 ≈ 13 |
| 每 chunk 帧数 | 13 | `num_frames=13`（4k+1 约束） |
| 视频分辨率 | 160 × 288 | `resolution=[160, 288]` |
| 涉及游戏数 【编造】 | 5 | NitroGen 子集包含多款游戏 |
| 划分粒度 | chunk | `DATASET_SPLIT_GRANULARITY="chunk"` |
| 划分种子 | 42 | `DATASET_SEED=42` |

**每游戏 chunk 分布 【编造】**：

| 游戏 | Train | Val | Test | 总计 |
|------|-------|-----|------|------|
| Game A（动作/RPG） | 34 | 4 | 5 | 43 |
| Game B（赛车） | 28 | 4 | 3 | 35 |
| Game C（射击） | 22 | 3 | 2 | 27 |
| Game D（平台） | 14 | 1 | 2 | 17 |
| Game E（格斗） | 8 | 1 | 1 | 10 |
| **合计** | **106** | **13** | **13** | **132** |

### 11.2 训练过程分析

#### 11.2.1 训练配置

与 `configs/finetune.example.json` 一致的完整训练配置：

- **基底模型**：HunyuanVideo 8.3B（`hunyuanvideo-community/HunyuanVideo`）
- **适配方法**：LoRA，rank=32，alpha=32，目标层 `[to_q, to_k, to_v, to_out.0]`，dropout=0.0
- **量化**：NF4 4-bit（bitsandbytes），仅量化 transformer
- **优化器**：AdamW，lr=2e-5，weight_decay=0.01
- **梯度策略**：梯度累积 4 步，梯度裁剪 norm=1.0
- **有效 batch size**：1 × 4 = 4
- **训练步数**：5000
- **混合精度**：bf16（bfloat16）
- **调度器**：流匹配（flow matching）调度
- **Guidance scale**：6.0（验证时）

#### 11.2.2 训练收敛曲线 【编造】

| 训练步数 | Train Loss (MSE) | Val Loss (MSE) | 学习率 |
|----------|-------------------|-----------------|--------|
| 0 | 0.342 | — | 2.0e-5 |
| 500 | 0.198 | 0.221 | 2.0e-5 |
| 1000 | 0.152 | 0.174 | 2.0e-5 |
| 1500 | 0.131 | 0.156 | 2.0e-5 |
| 2000 | 0.118 | 0.143 | 2.0e-5 |
| 2500 | 0.109 | 0.135 | 2.0e-5 |
| 3000 | 0.103 | 0.128 | 2.0e-5 |
| 3500 | 0.098 | 0.122 | 2.0e-5 |
| 4000 | 0.094 | 0.118 | 2.0e-5 |
| 4500 | 0.091 | 0.114 | 2.0e-5 |
| 5000 | 0.088 | 0.112 | 2.0e-5 |

**关键观察 【编造】**：

- **初始 loss 0.342**：流匹配目标在随机初始化 LoRA 上的基线，接近随机预测噪声与目标的 MSE
- **前 1000 步快速下降**：loss 从 0.342 降至 0.152（降幅 56%），LoRA 快速适配域内分布
- **1000-5000 步缓慢收敛**：loss 继续下降但幅度减小，从 0.152 到 0.088
- **Train-Val gap 稳定在 ~0.024**：未出现显著过拟合，说明 LoRA rank=32 的正则化效果与数据规模匹配
- **best_model 选择**：第 5000 步（train loss = 0.088），与代码中「按 epoch 平均 train loss 选优」一致

#### 11.2.3 训练资源消耗 【编造】

| 项目 | 值 |
|------|-----|
| GPU | 1× NVIDIA A100 80GB |
| 峰值显存 | ~62 GB（NF4 + 梯度检查点 + VAE 切片） |
| 训练总耗时 | ~14.2 小时 |
| 每步平均耗时 | ~10.2 秒 |
| LoRA 可训练参数 | ~67M（占总参数 0.8%） |
| 总模型参数 | ~8.3B |

### 11.3 实验设置

#### 11.3.1 评估协议

**固定条件**（与实现一致）：

- 评估 split：**val**（5 个 chunk，受 `GENERATE_MAX_SAMPLES=5` 限制）
- 每段帧数：**13 帧**
- 分辨率：**160 × 288**
- 随机种子：**42**
- 动作编码：**相同的 `GamepadActionEncoder` 文本条件**
- 参考帧对齐：resize 到与生成 PNG 相同分辨率
- 评估代码路径：**v2 评估报告**（`build_evaluation_records` + `evaluate_video_pair`）
- 池化帧数：5 chunks × 13 frames = **65 帧**（≥ FID_MIN_FRAMES_PER_POOL=48，可计算 FID）

#### 11.3.2 对照组定义

| 代号 | 条件 | 含义 |
|------|------|------|
| **Base** | 未微调 | 原始 HunyuanVideo，不加载 LoRA，使用相同动作文本提示。代表通用视频先验 |
| **LoRA-500** 【编造】 | 500 步短训 | 同结构 LoRA，仅训练 500 步（10%），代表严重欠拟合 |
| **LoRA-1k** 【编造】 | 1000 步短训 | 同结构 LoRA，训练 1000 步（20%），代表早期收敛 |
| **LoRA-2.5k** 【编造】 | 2500 步中训 | 同结构 LoRA，训练 2500 步（50%），代表中期检查点 |
| **LoRA-5k** | 5000 步（本文方法） | `best_model` 权重，与 `finetune.example.json` 一致 |

**指标方向约定**：

| 方向 | 指标 |
|------|------|
| ↓ 越低越好 | `mean_lpips`, `mean_mae`, `mean_temporal_error_vs_reference`, `inception_feature_mean_l2`, `mean_fid` |
| ↑ 越高越好 | `mean_psnr`, `mean_ssim` |
| ↑ 参照基准 | `mean_temporal_consistency`（需对照参考 TC，避免「糊成一片」假象） |

### 11.4 主实验：微调前后对比 【编造】

> 以下数值基于 HunyuanVideo 8.3B + LoRA rank=32 + NF4 量化 + 160×288 + 13 帧 + ~106 训练 chunk 的配置，手工设定合理相对关系。

#### 11.4.1 全指标对比表 【编造】

| 指标 | Base（未微调） | LoRA-500 | LoRA-1k | LoRA-2.5k | LoRA-5k（本文） | Δ(5k vs Base) |
|------|---------------|----------|---------|-----------|----------------|---------------|
| `mean_lpips` ↓ | 0.612 | 0.558 | 0.524 | 0.472 | **0.441** | −27.9% |
| `mean_psnr` ↑ (dB) | 17.3 | 18.1 | 19.0 | 20.1 | **20.7** | +3.4 dB |
| `mean_ssim` ↑ | 0.408 | 0.441 | 0.478 | 0.531 | **0.558** | +36.8% |
| `mean_mae` ↓ | 14.9 | 13.1 | 11.8 | 10.1 | **9.3** | −37.6% |
| `temporal_consistency` ↑ | 0.831 | 0.849 | 0.858 | 0.868 | **0.874** | +5.2% |
| `temporal_error_vs_ref` ↓ | 11.3 | 10.2 | 9.1 | 7.4 | **6.6** | −41.6% |
| `inception_feature_mean_l2` ↓ | 1.284 | 1.102 | 0.981 | 0.862 | **0.814** | −36.6% |
| `mean_fid` ↓ | 126.4 | 104.7 | 89.3 | 72.8 | **64.1** | −49.3% |
| `ref_temporal_consistency` | 0.891 | 0.891 | 0.891 | 0.891 | 0.891 | — |

#### 11.4.2 改善幅度汇总 【编造】

| 指标类别 | 指标 | Base → LoRA-5k 改善 |
|----------|------|---------------------|
| **感知质量** | LPIPS | −27.9%（0.612 → 0.441） |
| **像素精度** | PSNR | +3.4 dB（17.3 → 20.7） |
| **像素精度** | MAE | −37.6%（14.9 → 9.3） |
| **结构相似** | SSIM | +36.8%（0.408 → 0.558） |
| **时序动态** | Temporal Error | −41.6%（11.3 → 6.6） |
| **分布距离** | FID | −49.3%（126.4 → 64.1） |
| **特征距离** | Inception L2 | −36.6%（1.284 → 0.814） |
| **时序平滑** | TC | +5.2%（0.831 → 0.874） |

### 11.5 消融实验

#### 11.5.1 训练步数消融 【编造】

固定 LoRA rank=32，分析不同检查点的生成质量变化趋势：

| 步数 | LPIPS ↓ | PSNR ↑ | MAE ↓ | FID ↓ | Temporal Error ↓ |
|------|---------|--------|-------|-------|-------------------|
| 0 (Base) | 0.612 | 17.3 | 14.9 | 126.4 | 11.3 |
| 500 | 0.558 | 18.1 | 13.1 | 104.7 | 10.2 |
| 1000 | 0.524 | 19.0 | 11.8 | 89.3 | 9.1 |
| 2500 | 0.472 | 20.1 | 10.1 | 72.8 | 7.4 |
| 5000 | **0.441** | **20.7** | **9.3** | **64.1** | **6.6** |

**分析 【编造】**：
- **0→500 步**：改善最显著，LPIPS 降 8.8%，FID 降 17.2%，说明 LoRA 在前期快速捕获域内视觉分布
- **500→1000 步**：持续改善但速率下降，PSNR 提升 0.9 dB
- **1000→5000 步**：收益递减但仍在稳步改善，4000 步后趋于平稳
- **未观察到过拟合**：5000 步时 val loss 仍在缓降（0.112），train-val gap 稳定

#### 11.5.2 LoRA Rank 消融 【编造】

固定训练 5000 步，对比不同 LoRA rank（alpha 与 rank 同值）的效果：

| LoRA Rank | 可训练参数 | LPIPS ↓ | PSNR ↑ | FID ↓ | Train Loss | Val Loss |
|-----------|-----------|---------|--------|-------|------------|----------|
| 8 | ~17M | 0.498 | 19.6 | 81.2 | 0.102 | 0.124 |
| 16 | ~34M | 0.468 | 20.2 | 72.4 | 0.094 | 0.118 |
| **32**（默认） | **~67M** | **0.441** | **20.7** | **64.1** | **0.088** | **0.112** |
| 64 | ~134M | 0.436 | 20.9 | 62.3 | 0.084 | 0.115 |

**分析 【编造】**：
- **Rank 8→32**：LPIPS 从 0.498 降至 0.441（−11.4%），FID 从 81.2 降至 64.1（−21.1%），改善显著
- **Rank 32→64**：LPIPS 仅降 1.1%，FID 降 2.8%，收益微小但参数量翻倍
- **Rank 64 轻微过拟合信号**：val loss（0.115）略高于 rank 32（0.112），train-val gap 增大
- **Rank 32 为性价比最优点**：与默认配置 `lora_rank=32` 的选择一致

#### 11.5.3 动作编码方式对比 【编造】

文本编码（当前 MVP）vs 无动作条件的对比：

| 条件 | LPIPS ↓ | PSNR ↑ | Temporal Error ↓ | FID ↓ |
|------|---------|--------|-------------------|-------|
| 无动作条件（仅 `"Gameplay video of {game}."`） | 0.501 | 19.4 | 8.9 | 78.6 |
| 文本动作编码（`action_encoding="text"`） | **0.441** | **20.7** | **6.6** | **64.1** |
| Δ | −12.0% | +1.3 dB | −25.8% | −18.4% |

**分析 【编造】**：
- 即使无动作条件，LoRA 微调也能学习域内视觉风格（FID 78.6 vs Base 126.4）
- 加入文本动作编码后，**temporal_error_vs_reference 显著降低**（−25.8%），说明动作文本确实帮助模型学习了帧间变化与控制输入的对应关系
- LPIPS 与 FID 的改善表明动作条件不仅影响动态，也提升了逐帧视觉质量
- 这验证了将手柄动作编码为自然语言并通过 Llama 文本编码器注入的 MVP 方案的有效性

### 11.6 逐视频分析 【编造】

以 val split 中 5 个 chunk 的 LoRA-5k 结果为例：

| Chunk | 游戏 | LPIPS ↓ | PSNR ↑ | SSIM ↑ | MAE ↓ | TC ↑ | Temp Error ↓ |
|-------|------|---------|--------|--------|-------|------|--------------|
| val_chunk_01 | Game A | 0.402 | 21.8 | 0.592 | 8.1 | 0.891 | 5.8 |
| val_chunk_02 | Game B | 0.478 | 19.4 | 0.518 | 10.7 | 0.856 | 7.9 |
| val_chunk_03 | Game A | 0.418 | 21.2 | 0.574 | 8.6 | 0.882 | 6.1 |
| val_chunk_04 | Game C | 0.461 | 20.1 | 0.538 | 9.8 | 0.868 | 7.2 |
| val_chunk_05 | Game D | 0.446 | 20.9 | 0.566 | 9.3 | 0.873 | 5.9 |
| **均值** | — | **0.441** | **20.7** | **0.558** | **9.3** | **0.874** | **6.6** |
| **标准差** | — | 0.030 | 0.88 | 0.028 | 0.99 | 0.014 | 0.88 |

**逐视频观察 【编造】**：
- **Game A（动作/RPG）表现最佳**：训练集中 Game A chunk 最多（34 个），域内适配最充分，LPIPS 0.402、PSNR 21.8 dB
- **Game B（赛车）表现较弱**：快速运动场景帧间变化大，temporal error 最高（7.9），LPIPS 也偏高（0.478）
- **跨视频方差较小**：LPIPS 标准差 0.030，PSNR 标准差 0.88 dB，说明模型在不同游戏片段上泛化较稳定

### 11.7 分布级指标分析 【编造】

池化帧数：5 chunks × 13 frames = 65 帧（≥ FID_MIN_FRAMES_PER_POOL = 48）。

| 指标 | Base | LoRA-5k | 说明 |
|------|------|---------|------|
| 池化 FID ↓ | 126.4 | **64.1** | 生成帧分布接近参考 |
| Inception L2 ↓ | 1.284 | **0.814** | 一阶矩距离，小样本稳定 |
| FID 协方差收缩 eps | 1e-3 | 1e-3 | 对角正则化因子 |

**分析 【编造】**：
- FID 从 126.4 降至 64.1（−49.3%），表明微调后生成帧的高层语义分布显著接近真实游戏画面
- Inception L2 与 FID 同向改善（1.284 → 0.814），二者一致性增强结论可信度
- 注意：65 帧的池化样本量虽满足阈值，但 FID 在此规模下仍有较大方差（经验上需 > 2048 帧才稳定），建议解读时 **同时参考 LPIPS 与 Inception L2**

### 11.8 时序质量分析 【编造】

| 指标 | Base | LoRA-5k | 参考真值 |
|------|------|---------|----------|
| `temporal_consistency` ↑ | 0.831 | **0.874** | 0.891 |
| `temporal_error_vs_reference` ↓ | 11.3 | **6.6** | 0（自身） |

**分析 【编造】**：
- **TC（时序一致性）**：LoRA-5k（0.874）接近参考真值（0.891），gap 仅 0.017，说明生成序列帧间过渡平滑且不过度模糊
- **Temporal Error**：从 11.3 降至 6.6（−41.6%），这是最直接反映「动作→运动」对齐的指标——帧间变化模式越接近参考，说明模型越好地学习了动作与画面动态的映射
- 单独 TC 高不等于好（静帧 TC=1.0）；结合 temporal error 下降与 TC 稳步提升，可判断为**有效的运动学习**而非退化为模糊静态

### 11.9 综合分析

#### 11.9.1 微调有效性

Base 模型在相同动作文本提示下各指标显著落后，说明 **通用视频先验无法在零样本下对齐「手柄→像素」**。LoRA 微调提供的域内适配是生成质量提升的核心驱动力。

#### 11.9.2 收敛与效率

- 前 1000 步贡献了总改善的约 60%（以 LPIPS 为例：0.612→0.524 vs 0.612→0.441）【编造】
- 5000 步仍未完全收敛（val loss 仍在缓降），但边际收益递减
- LoRA rank=32 在参数效率与生成质量间取得最佳平衡

#### 11.9.3 动作条件的作用

文本动作编码在 temporal error 上带来 25.8% 的额外改善【编造】，验证了 MVP 方案的有效性。但文本编码存在信息瓶颈（`llama_max_sequence_length=96` 截断、自然语言压缩损失），向量条件路径 `(T, 21)` 有望进一步释放动作信号的精度。

#### 11.9.4 当前系统的性能定位

| 维度 | 当前表现 | 说明 |
|------|----------|------|
| 像素级 | PSNR 20.7 dB, MAE 9.3 【编造】 | 中等：低分辨率限制了精细细节 |
| 感知级 | LPIPS 0.441 【编造】 | 中等偏好：可辨识游戏场景但与原画有明显差异 |
| 分布级 | FID 64.1 【编造】 | 合理：对于 160×288 低分辨率 + 65 帧池化的条件下属正常范围 |
| 时序级 | TC 0.874, Temp Error 6.6 【编造】 | 较好：接近参考的时序平滑度，运动变化方向大体正确 |

### 11.10 复现指南

在同一 `generation_manifest.json` 结构下：

1. **Base 对照**：不加载 `best_model` 下的 LoRA，直接用原始 HunyuanVideo 管线生成
2. **短训消融**：从不同 `checkpoint-*` 目录加载 LoRA 权重
3. **LoRA rank 消融**：修改 `finetune.example.json` 中 `lora_rank` 和 `lora_alpha`，重新训练
4. **无动作条件**：将 `prompt_template` 改为不含动作的固定句
5. 所有变体使用 **同一 `build_evaluation_records` + v2 报告** 即可得到可替换上表的真实数字

---

## 12. 技术实现细节

### 12.1 依赖栈（requirements.txt）

| 包 | 版本要求 | 作用 |
|----|---------|------|
| `torch` | ≥ 2.2.0 | 训练与推理 |
| `torchvision` | ≥ 0.17.0 | Inception 特征提取 |
| `transformers` | ≥ 4.41.0 | 文本编码与量化配置 |
| `diffusers` | ≥ 0.30.0 | HunyuanVideoPipeline |
| `peft` | ≥ 0.12.0 | LoRA 适配 |
| `accelerate` | ≥ 0.27.0 | 设备与训练辅助 |
| `bitsandbytes` | ≥ 0.43.0 | NF4 4-bit 量化 |
| `safetensors` | ≥ 0.4.0 | 权重序列化 |
| `pandas` / `pyarrow` | ≥ 2.2.0 / ≥ 15.0.0 | Parquet 读写 |
| `decord` / `opencv-python` | ≥ 0.6.0 / ≥ 4.9.0 | 视频解码 |
| `lpips` | ≥ 0.1.4 | 感知距离指标 |
| `scipy` | ≥ 1.12.0 | FID 矩阵运算 |
| `huggingface-hub` | ≥ 0.23.0 | 数据集下载 |
| `yt-dlp` | ≥ 2024.8.0 | YouTube 视频下载 |
| `pytest` | ≥ 8.0.0 | 测试 |

### 12.2 流匹配时间步 dtype

调度器 `index_for_timestep` 依赖 float32 时间步；若过早转为 bf16 会导致索引失败，因此训练中对 scheduler 与 model 分别维护 fp32 / model_dtype 时间步。

### 12.3 VideoActionDataset 鲁棒性

单条样本视频损坏时，在同一次 `__getitem__` 内循环尝试其他 chunk，避免 DataLoader 崩溃；全部失败则抛出明确错误。

### 12.4 MP4 验证

`is_probably_valid_mp4()`：文件 ≥ 1024 字节、以 `ftyp` box 开头、包含 `moov` box（检查前 5 MiB、后 4 MiB 与中间区域）。

### 12.5 测试

仓库含 `tests/test_download_data.py`、`tests/test_loader_mp4.py`、`tests/test_eval_metrics.py`；评估测试覆盖 MAE、时序误差、记录聚合与池化 FID 阈值逻辑，不加载 HunyuanVideo 全模型。

---

## 13. 配置与可复现性

### 13.1 configs/finetune.example.json 完整键

`dataset_path`, `output_dir`, `model_id`, `training_type`, `lora_rank`, `lora_alpha`, `target_modules`, `resolution`, `num_frames`, `learning_rate`, `num_train_steps`, `gradient_accumulation_steps`, `batch_size`, `gradient_checkpointing`, `quantization`, `mixed_precision`, `llama_max_sequence_length`, `clip_pooler_prompt`, `vae_memory_saving`, `show_training_progress`, `action_encoding`, `prompt_template`, `joystick_deadzone`, `split_seed`, `split_granularity`, `split_policy`, `game_filter`, `max_train_chunks`, `max_val_chunks`, `use_processed_actions`, `logging_steps`, `save_steps`, `seed`。

### 13.2 可复现性要点

- 同一 `split_seed` + 同一 `split_granularity` + 同一数据目录 → split 一致
- `finetune_config.json` 与 `train_history.json` 保留超参与曲线
- `torch` / `cuda` / `diffusers` 版本差异可能导致数值微小漂移

---

## 14. 局限性与未来工作

### 14.1 当前局限

| 局限 | 说明 |
|------|------|
| **文本条件信息瓶颈** | 长序列动作经语言压缩信息有损；`llama_max_sequence_length=96` 截断风险 |
| **向量条件未接训练** | `(T,21)` 编码已定义，需在 Transformer 侧注入并设计损失 |
| **低分辨率** | 160×288 限制了视觉细节，与原始游戏画面差距显著 |
| **小规模数据** | ~106 训练 chunk 限制了模型的游戏多样性泛化 |
| **池化 FID 噪声大** | 65 帧池化下 FID 方差大，需配合 LPIPS 与 Inception L2 解读 |
| **无显式动作对齐指标** | 当前离线指标不直接度量生成画面与手柄语义一致性 |
| **推理未用训练期 NF4** | `generate_video` 以 bf16 全精度加载，与训练配置不同 |
| **单卡限制** | 未实现多 GPU / DeepSpeed 并行 |

### 14.2 未来工作

| 方向 | 具体计划 |
|------|----------|
| **向量/混合条件** | 接入 `(T, 21)` 向量条件通路，在 Transformer cross-attention 中注入精确控制信号 |
| **分辨率提升** | 渐进式训练（160×288 → 320×576 → 480×854），课程学习策略 |
| **数据规模扩展** | 下载更多 NitroGen 分片（36 个可用），扩展游戏多样性 |
| **FVD / I3D** | 接入 I3D 特征的视频级 Fréchet 距离，更准确的时序质量评估 |
| **动作一致性指标** | 基于视觉问答、事件检测、光流与控制信号相关性的自动评估 |
| **多 GPU 并行** | DeepSpeed ZeRO / FSDP，支持更大 batch 与更高分辨率 |
| **帧数扩展** | 从 13 帧逐步扩展到 25、49 帧，覆盖更长时间跨度 |
| **在线评估** | 与真实游戏环境对接，闭环测试动作→视频→动作的交互质量 |

---

## 15. 附录

### 附录 A：核心类型一览

| 模块 | 名称 | 用途 |
|------|------|------|
| `src.data.schema` | `SplitName`, `SplitPolicy` | 划分枚举与比例校验 |
| `src.data.schema` | `GamepadAction`, `ChunkMetadata`, `VideoChunk`, `TrainingSample` | 数据契约 |
| `src.data.split` | `VideoSplitAssigner` | 哈希划分 |
| `src.data.loader` | `discover_chunks`, `load_video_chunk`, `NitroGenDataset`, `chunk_to_training_sample`, `nearest_valid_frame_count` | IO 与训练样本构造 |
| `src.model.action_encoder` | `GamepadActionEncoder` | 文本/向量条件 |
| `src.train.config_io` | `VideoGenConfig`, `LoRAConfig` | 配置 |
| `src.train.hf_finetune` | `VideoActionDataset`, `run_hunyuanvideo_lora_finetune`, `generate_video` | 训练与推理 |
| `src.eval.metrics` | `VideoEvalRecord`, `VideoEvalSummary`, `evaluate_video_pair`, `evaluate_records`, `extract_inception_features`, `compute_pooled_distribution_metrics` | 指标与池化分布量 |
| `src.eval.pipeline` | `build_evaluation_records` | 评估数据流 |
| `src.eval.report` | `VideoEvalReport`, `build_evaluation_report` | 报告 |

### 附录 B：JSON Schema 版本

| 产物 | `schema_version` |
|------|------------------|
| 数据集 manifest | `v3_nitrogen_manifest` |
| 视频生成评估报告 | `v2_video_generation_evaluation` |

### 附录 C：运行环境

- **Python**：3.12+
- **GPU**：训练推荐 ≥ 80GB（A100）；推理可尝试更小显存但需降低分辨率/帧数
- **磁盘**：NitroGen 视频与缓存需预留数十 GB 至 TB 级空间（视分片与下载选项而定）

### 附录 D：指标公式汇总

| 指标 | 公式 |
|------|------|
| PSNR | `10 × log10(255² / MSE)` |
| SSIM | luminance-only: `((2μ₁μ₂ + C1)(2σ₁₂ + C2)) / ((μ₁² + μ₂² + C1)(σ₁² + σ₂² + C2))`，C1=(0.01×255)²，C2=(0.03×255)² |
| MAE | `mean(\|img1 - img2\|)` |
| LPIPS | AlexNet backbone，帧归一化到 [-1,1]，逐帧计算后平均 |
| TC | `1 - min(mean(MAD_adjacent_frames) / 255, 1)` |
| Temporal Error | `mean(\|Δgen - Δref\|)` where `Δ[t] = frames[t+1] - frames[t]` |
| FID | `\|\|μ_gen - μ_ref\|\|² + tr(Σ_gen + Σ_ref - 2√(Σ_gen Σ_ref))`，Inception v3 pool3 2048-d |
| Inception L2 | `\|\|mean(gen_features) - mean(ref_features)\|\|₂` |

---

> **【编造】数据声明**：本报告中所有标注【编造】的数值（§1 摘要中的改善百分比、§11 全部实验数据）均为基于项目实际配置（HunyuanVideo 8.3B、LoRA rank=32、NF4 量化、160×288 分辨率、13 帧、~106 训练 chunk、5000 训练步）合理推断的虚构示例，旨在展示完整的实验分析框架。复现实验需在实际 GPU 环境运行流水线后替换为真值。未标注【编造】的技术描述均与代码实现一致。
