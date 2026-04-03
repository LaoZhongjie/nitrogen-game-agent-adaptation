# Nitrogen：面向世界模型的游戏智能体行为适配系统技术报告

> **项目名称：** Nitrogen Game Agent Adaptation  
> **版本：** v1.0  
> **日期：** 2026年3月31日  
> **作者：** *[待填写]*  

---

## 目录

1. [摘要](#1-摘要)
2. [引言与背景](#2-引言与背景)
3. [系统设计目标](#3-系统设计目标)
4. [系统架构概览](#4-系统架构概览)
5. [数据层设计](#5-数据层设计)
6. [动作对齐模块](#6-动作对齐模块)
7. [模型微调模块](#7-模型微调模块)
8. [推理预测模块](#8-推理预测模块)
9. [离线评估模块](#9-离线评估模块)
10. [端到端流水线](#10-端到端流水线)
11. [实验结果与分析](#11-实验结果与分析)
12. [技术实现细节](#12-技术实现细节)
13. [配置与可复现性](#13-配置与可复现性)
14. [局限性与未来工作](#14-局限性与未来工作)
15. [附录](#15-附录)

---

## 1. 摘要

**Nitrogen** 是一个在大规模游戏视觉数据上预训练的世界模型（World Model）。本项目（**Nitrogen Game Agent Adaptation**）构建了一套完整的行为适配流水线，旨在将 Nitrogen 预训练模型高效地微调适配到特定下游游戏场景，使其能够从游戏画面帧中准确预测对应的规范化动作标签，从而驱动游戏智能体的行为决策。

系统采用四阶段流水线架构：

1. **数据集构建**：从原始游戏演示回合（Episode）数据构建标准化的片段级清单（Clip Manifest）；
2. **动作对齐**：将原始异构动作标注映射至规范化动作词汇表；
3. **模型微调**：在 Nitrogen 预训练权重的基础上，通过 Hugging Face Transformers 框架进行有监督微调，将其适配到目标游戏的动作空间；
4. **推理与评估**：在验证/测试集上执行批量推理，并计算动作准确率和时序一致性等离线评估指标。

本项目以 Python 3.12 实现，遵循模块化、可配置、可复现的工程原则，为基于 Nitrogen 世界模型的游戏智能体研究提供可靠的下游适配基础设施。

---

## 2. 引言与背景

### 2.1 Nitrogen 世界模型

**Nitrogen** 是一个在大规模多游戏视觉数据上预训练的世界模型。其核心架构基于 Vision Transformer (ViT)，通过在 ⚠️ 海量游戏画面数据上进行预训练，学习了丰富的跨游戏视觉表征。一个具备世界模型能力的智能体可以：

- **感知**：从原始视觉观测中提取高层语义特征
- **预测**：基于当前状态和候选动作，预测未来的状态变化
- **规划**：在内部模拟中搜索最优动作序列

Nitrogen 在预训练阶段已经习得了强大的游戏视觉理解能力。然而，要将这种通用的视觉理解转化为特定游戏中的精准行为决策，还需要一个**下游适配**的过程——这正是本项目所解决的核心问题。

### 2.2 问题定义

本项目聚焦于 Nitrogen 世界模型的**行为适配**（Behavioral Adaptation）环节。给定人类专家在目标游戏中的演示数据——包括逐帧截图和对应的操作标注——通过在 Nitrogen 预训练权重的基础上进行有监督微调，使模型适配到目标游戏的动作空间。

该过程可形式化为：给定观测-动作对数据集 $\mathcal{D} = \{(o_t, a_t)\}_{t=1}^{N}$，其中 $o_t$ 为第 $t$ 帧的游戏画面，$a_t$ 为对应的规范化动作标签，在 Nitrogen 预训练参数 $\theta_0$ 的基础上学习策略函数 $\pi_\theta(a | o)$，使得预测动作与专家行为的匹配度最大化。预训练权重 $\theta_0$ 提供了良好的参数初始化，使得模型仅需少量下游数据和训练轮次即可快速适配。

### 2.3 技术背景

- **世界模型**：Ha & Schmidhuber (2018) 提出的「World Models」开创了通过 VAE + RNN 构建环境内部模型的研究方向。后续工作如 Dreamer (Hafner et al., 2020) 和 IRIS (Micheli et al., 2023) 将该范式推进至更复杂的游戏与仿真环境。Nitrogen 属于该研究脉络下的视觉世界模型，通过大规模游戏数据预训练获得通用视觉表征能力。
- **Vision Transformer (ViT)**：Google 于 2020 年提出的将图像分割为固定大小 Patch 并通过 Transformer 编码器处理的模型架构。Nitrogen 的视觉编码器主干基于 ViT 架构构建，继承了 Transformer 在大规模数据上的强扩展性。
- **Hugging Face Transformers**：开源深度学习框架，提供了统一的预训练模型下载、微调和推理接口。本项目基于其 `AutoModelForImageClassification` 和 `Trainer` API 构建 Nitrogen 的下游适配训练流程。
- **行为克隆**：监督学习范式下的模仿学习方法，将状态-动作对视为 $(x, y)$ 监督信号，训练分类器或回归器。本项目采用行为克隆方法作为 Nitrogen 下游适配的训练范式。

### 2.4 相关工作

| 方法/项目 | 核心思路 | 与 Nitrogen 的关系 |
|-----------|---------|-------------------|
| **World Models** (Ha & Schmidhuber, 2018) | VAE + MDN-RNN 学习环境动态 | Nitrogen 同为世界模型，但采用 ViT 架构并在更大规模数据上预训练 |
| **Dreamer** (Hafner et al., 2020) | 在学习的世界模型中进行策略梯度优化 | Nitrogen 的下游适配采用离线行为克隆而非在线策略梯度，面向低数据量场景 |
| **IRIS** (Micheli et al., 2023) | 基于离散 Token 的 Transformer 世界模型 | Nitrogen 使用连续视觉特征而非离散 Token |
| **GATO** (Reed et al., 2022) | 多模态通用智能体 | Nitrogen 聚焦视觉模态的世界建模，下游适配面向特定游戏 |
| **VPT** (Baker et al., 2022) | Minecraft 视频预训练 + 微调 | 与 Nitrogen 思路相近（预训练 + 微调），但 VPT 仅针对 Minecraft |
| **GameGen-X** (Che et al., 2024) | 开放域游戏视频生成式世界模型 | Nitrogen 关注判别式动作预测而非生成式视频合成，两者在世界模型框架中互补 |

---

## 3. 系统设计目标

Nitrogen 遵循以下核心设计原则：

| 原则 | 说明 |
|------|------|
| **可复现性** | 数据划分采用确定性哈希算法，所有随机种子可配置，训练超参数通过 JSON 配置文件管理 |
| **模块化** | 数据处理、模型对齐、训练、评估严格分离，每个模块可独立测试和替换 |
| **可配置性** | 不硬编码路径，所有关键参数通过配置文件或 CLI 参数传递 |
| **类型安全** | 全面使用 Python 类型标注，关键数据结构采用 `dataclass` 并附带运行时验证 |
| **简洁性** | 优先使用简单、可测试的模块，避免过度工程化的抽象 |

---

## 4. 系统架构概览

### 4.1 目录结构

```
nitrogen-game-agent-adaptation/
├── main.py                        # 端到端流水线入口
├── requirements.txt               # Python 依赖声明
├── configs/
│   └── finetune.example.json      # 微调配置模板
├── docs/
│   ├── dataset_spec.md            # 数据集规范文档
│   └── project_plan.md            # 项目规划文档
├── scripts/
│   ├── build_dataset.py           # 数据集清单构建 CLI
│   ├── finetune.py                # 模型微调 CLI
│   └── predict.py                 # 推理预测 CLI
└── src/
    ├── data/
    │   ├── schema.py              # 数据 Schema 契约
    │   ├── split.py               # 确定性数据划分
    │   └── loader.py              # 清单加载与重窗口化
    ├── model/
    │   └── alignment.py           # 动作对齐接口与实现
    ├── train/
    │   ├── config_io.py           # 配置 I/O 辅助
    │   └── hf_finetune.py         # HF 微调核心逻辑
    └── eval/
        ├── metrics.py             # 评估指标计算
        ├── pipeline.py            # 评估数据拼接流水线
        └── report.py              # 评估报告生成与序列化
```

### 4.2 Nitrogen 预训练与下游适配的关系

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Nitrogen 世界模型                                  │
│                                                                          │
│  ┌────────────────────────┐       ┌──────────────────────────────────┐  │
│  │    预训练阶段（上游）     │       │     行为适配阶段（本项目）          │  │
│  │                        │       │                                  │  │
│  │  大规模多游戏视觉数据    │──────▶│  特定游戏演示数据 + 微调           │  │
│  │  → 通用视觉表征学习      │ 权重   │  → 目标动作空间映射               │  │
│  │  (ViT 架构主干)         │ 迁移   │  → 离线评估与行为预测              │  │
│  └────────────────────────┘       └──────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.3 数据流向图

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  原始回合数据  │────▶│  清单构建器    │────▶│  动作对齐器    │────▶│  微调训练器    │
│  (episodes/)  │     │ build_dataset │     │  alignment   │     │  hf_finetune │
└──────────────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                            │                     │                     │
                            ▼                     │                     ▼
                    ┌──────────────┐              │           ┌──────────────┐
                    │ manifest.json│              │           │  hf_model/   │
                    │ (片段清单)    │              │           │  label2id.json│
                    └──────┬───────┘              │           └──────┬───────┘
                           │                      │                  │
                           │         ┌────────────┘                  │
                           ▼         ▼                               ▼
                    ┌──────────────────────┐                ┌──────────────┐
                    │   推理预测器           │◀──────────────│   预测 CLI    │
                    │   predict_clip_actions│                │   predict.py │
                    └──────────┬───────────┘                └──────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   离线评估流水线       │
                    │  metrics + report    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   report_val.json    │
                    │  (评估报告)           │
                    └──────────────────────┘
```

---

## 5. 数据层设计

### 5.1 原始数据格式

原始数据按如下目录结构组织，每个子目录代表一个完整的演示回合（Episode）：

```
data/raw/
├── episode_001/
│   ├── frames/
│   │   ├── frame_000.png
│   │   ├── frame_001.png
│   │   └── ...
│   └── actions.json          # 或 actions.csv
├── episode_002/
│   ├── frames/
│   │   └── ...
│   └── actions.csv
└── ...
```

**动作标注格式**（二选一，同一回合不可同时存在）：

- `actions.json`：JSON 数组，每个元素为 `{"frame": "<文件名>", "action": "<动作文本>"}`
- `actions.csv`：CSV 文件，含 `frame` 和 `action` 两列表头

### 5.2 数据 Schema 契约

Nitrogen 定义了严格的多级数据模型（`src/data/schema.py`），通过 `dataclass` 实现运行时类型校验：

| 数据类 | 核心字段 | 验证约束 |
|--------|---------|---------|
| `SplitName` | `train` / `val` / `test` | 枚举类型，限制为三种合法值 |
| `ActionLabel` | `action_id`, `action_text`, `confidence` | ID 非空、置信度 ∈ [0, 1] |
| `FrameRecord` | `frame_idx`, `frame_path`, `timestamp_sec` | 索引 ≥ 0、路径非空、时间戳 ≥ 0 |
| `ClipRecord` | `clip_id`, `frames`, `action_labels` | 帧数与动作标签数一致且 > 0 |
| `EpisodeRecord` | `episode_id`, `game`, `demonstrator_id`, `clips`, `split` | 所有 ID 非空、片段列表非空 |
| `SplitPolicy` | `train`, `val`, `test` | 三个比例均 > 0 且和为 1.0（容差 1e-6） |

### 5.3 确定性数据划分

数据划分在**回合级别**执行（非帧级或片段级），确保同一回合的所有片段归属同一数据集。划分算法（`src/data/split.py`）基于 SHA-256 哈希实现确定性映射：

```python
score = SHA256(f"{seed}:{episode_id}") → 取前 8 字节 → 转整数 → 除以 2^64
```

得到 [0.0, 1.0) 区间内的均匀分布分数，再根据 `SplitPolicy` 的累积阈值分配至 `train`、`val` 或 `test`。

**默认划分比例**：Train 80% / Val 10% / Test 10%

### 5.4 片段清单（Clip Manifest）

构建脚本 `scripts/build_dataset.py` 扫描原始回合目录，将逐帧数据通过滑动窗口切分为固定长度片段，输出结构化的 JSON 清单：

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | `str` | 当前为 `v2_clip_manifest` |
| `input_root` | `str` | 标准化的源数据根目录路径 |
| `split_policy` | `object` | 含 `train`、`val`、`test` 比例和 `seed` |
| `episode_splits` | `object` | 回合 ID → 划分名称的映射 |
| `clip_length` | `int` | 固定片段长度（默认 16 帧） |
| `stride` | `int` | 滑动窗口步长（默认 16 帧） |
| `split_counts` | `object` | 各划分的片段数量统计 |
| `clips` | `array` | 片段列表，每项含 `episode_id`、`clip_id`、`frame_paths`、`action_labels`、`split` |

**滑动窗口机制**：对每个回合的有序帧序列，以 `clip_length` 为窗口大小、`stride` 为步长进行切分。当 `stride == clip_length` 时为无重叠切分，`stride < clip_length` 时窗口之间存在重叠。帧数不足 `clip_length` 的回合不生成任何片段。

### 5.5 清单加载器

`ManifestDataset`（`src/data/loader.py`）提供对构建好的清单的只读访问接口：

- 实现 `Sequence[ManifestSample]` 协议，支持下标访问和迭代
- 支持按 `split` 过滤
- 支持运行时**重窗口化**（re-windowing）：在加载后对已有片段进行二次滑动窗口切分，新 `clip_id` 带 `_rw_` 后缀

---

## 6. 动作对齐模块

### 6.1 设计动机

原始数据中的动作标注往往是自由格式文本（如 `"walk left"`、`"Walk Left"`、`"go left"`），存在大小写不一致、同义表达、拼写变体等问题。动作对齐模块（`src/model/alignment.py`）负责将这些异构标注统一映射为规范化的动作 ID（如 `move_left`）。

### 6.2 对齐协议

Nitrogen 定义了 `ActionAligner` Protocol 接口：

```python
class ActionAligner(Protocol):
    def align(self, raw_action: RawActionRecord) -> AlignedActionRecord: ...
```

任何实现 `align` 方法的类均可作为对齐器插入流水线。

### 6.3 词汇表对齐器

默认实现 `VocabularyActionAligner` 执行四级对齐策略：

| 优先级 | 来源 | `alignment_source` | 说明 |
|-------|------|-------------------|------|
| 1 | 直接映射 | `direct` | 标准化后的文本直接命中 `mapping` |
| 2 | 别名映射 | `alias` | 经 `aliases` 表转换后命中 `mapping` |
| 3 | 置信度过滤 | `confidence_floor` | 虽命中映射但原始置信度低于阈值，降级为未知 |
| 4 | 未知回退 | `unknown` | 所有未命中的动作映射至 `unknown_action_id` |

**标准化规则**：所有文本在查找前经 `strip().lower()` 处理。

### 6.4 示例配置

```json
{
  "action_mapping": {
    "walk left": "move_left",
    "walk right": "move_right",
    "jump": "jump"
  },
  "action_aliases": {},
  "unknown_action_id": "unknown",
  "confidence_floor": 0.0
}
```

### 6.5 批量对齐与统计

`align_many` 方法支持批量对齐，返回 `AlignmentBatch` 对象，包含 `AlignmentSummary`：

- `total_count`：总动作数
- `known_count`：成功映射数
- `unknown_count`：回退至未知的动作数
- `mean_confidence`：平均置信度

---

## 7. 模型微调模块

### 7.1 整体架构

模型微调模块（`src/train/hf_finetune.py`）负责在 Nitrogen 预训练权重的基础上进行下游行为适配。将游戏帧的动作预测建模为图像分类任务——Nitrogen 的视觉编码器提取帧级特征，分类头将其映射至目标游戏的动作空间：

- **输入**：单帧 RGB 游戏画面（视觉观测）
- **输出**：规范化动作 ID 对应的类别索引
- **模型主干**：Nitrogen 预训练世界模型（基于 ViT 架构），通过配置文件中的 `model_id` 指定预训练权重路径或 Hugging Face Hub 标识符

### 7.2 数据准备流程

```
manifest.json → ManifestDataset (按 split 过滤)
    ↓
逐片段遍历 → align_manifest_sample (动作对齐)
    ↓
逐帧展开 → FrameLabelRow (image_path, action_id)
    ↓
build_label2id → 确定性标签映射 (sorted, 含 unknown)
    ↓
FrameClassificationDataset (pixel_values, labels)
```

- `collect_aligned_frame_rows`：展开清单片段为帧级 `(图像路径, 动作ID)` 行
- `build_label2id`：对训练集出现的动作 ID 排序去重，始终包含 `unknown_action_id`，生成确定性 `{action_id: int}` 映射
- `FrameClassificationDataset`：继承 `torch.utils.data.Dataset`，对每帧加载 RGB 图像，通过 `AutoImageProcessor` 编码为 `pixel_values`，标签转为 `torch.long` 型整数

### 7.3 训练配置

训练超参数通过 `HFFinetuneParams` 封装，关键参数如下：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `model_id` | Nitrogen 预训练权重路径 | 预训练模型标识（HF Hub ID 或本地路径） |
| `num_train_epochs` | 2.0 | 微调训练轮次 |
| `per_device_train_batch_size` | 4 | 每设备训练批大小 |
| `per_device_eval_batch_size` | 4 | 每设备评估批大小 |
| `learning_rate` | 5e-5 | 微调学习率 |
| `seed` | 42 | 随机种子 |
| `logging_steps` | 10 | 日志记录步频 |

### 7.4 训练器配置

Nitrogen 使用 Hugging Face `Trainer` API，核心策略：

- **评估策略**：当提供验证集时按 epoch 评估；否则不评估
- **保存策略**：有验证集时按 epoch 保存最优模型（最多保留 2 个 checkpoint）；否则不中间保存
- **最优模型选择**：`metric_for_best_model="accuracy"`，`greater_is_better=True`
- **报告**：禁用外部日志平台（`report_to="none"`）

### 7.5 输出产物

训练完成后生成以下文件：

| 文件 | 说明 |
|------|------|
| `hf_model/` | 完整的 HF 模型权重、配置和图像处理器 |
| `label2id.json` | 动作 ID → 整数标签的映射字典 |
| `finetune_config.json` | 本次训练所用的完整配置 |
| `train_metrics.json` | 训练摘要（帧数、标签数、最终验证准确率等） |

---

## 8. 推理预测模块

### 8.1 推理流程

推理模块 `predict_clip_actions`（`src/train/hf_finetune.py`）加载适配后的 Nitrogen 模型，对给定 split 的每个片段执行如下流程：

1. 加载微调后的 Nitrogen 模型权重（`hf_model/`）和标签映射（`label2id.json`）
2. 自动选择推理设备（优先级：CUDA > MPS > CPU）
3. 对每个片段的帧进行动作对齐（保持与训练时相同的对齐配置）
4. 分批加载帧图像，经 `AutoImageProcessor` 编码
5. 模型前向推理（`torch.no_grad()`），取 `logits.argmax` 得到预测类别
6. 通过 `id2label` 反映射为动作 ID 字符串

### 8.2 输出格式

预测结果以 JSON 数组形式存储：

```json
[
  {
    "clip_id": "episode_001_clip_000000",
    "predicted_action_ids": ["move_left", "move_left", "jump", ...]
  },
  ...
]
```

---

## 9. 离线评估模块

### 9.1 评估指标

Nitrogen 实现两个核心离线评估指标（`src/eval/metrics.py`）：

#### 9.1.1 动作准确率（Action Accuracy）

逐帧精确匹配的正确率：

$$\text{Action Accuracy} = \frac{\sum_{i=1}^{N} \mathbb{1}[\hat{a}_i = a_i^*]}{N}$$

其中 $\hat{a}_i$ 为第 $i$ 帧的预测动作，$a_i^*$ 为目标动作，$N$ 为总帧数。

#### 9.1.2 平均时序一致性（Mean Temporal Consistency）

衡量预测动作序列在时间维度上的平滑程度。对于每个片段，计算相邻帧预测动作相同的比例：

$$\text{TC}(\text{clip}) = \frac{\sum_{t=1}^{T-1} \mathbb{1}[\hat{a}_t = \hat{a}_{t+1}]}{T - 1}$$

最终取所有片段的算术平均：

$$\text{Mean TC} = \frac{1}{C} \sum_{c=1}^{C} \text{TC}(\text{clip}_c)$$

该指标不依赖真实标签，仅反映模型预测的时间稳定性。高时序一致性通常表明模型不会在相邻帧间产生大量抖动预测。

### 9.2 评估流水线

评估流水线（`src/eval/pipeline.py`）执行以下步骤：

1. **加载预测记录**：解析预测 JSON 为 `PredictionRecord` 列表
2. **加载清单目标**：从原始清单中按 split 过滤获取目标片段
3. **目标对齐**：若指定了 `target_aligner`，则对清单中的原始动作标注执行与训练时相同的对齐，确保评估时 target 与 predict 处于同一词汇空间
4. **完整性校验**：
   - 检查预测中是否包含清单中不存在的 `clip_id`
   - 当 `require_all_clips=True` 时，检查是否覆盖了清单中所有片段
   - 验证每个片段的预测长度与目标长度一致
5. **生成评估记录**：构建 `EvaluationRecord` 列表

### 9.3 评估报告

评估报告（`src/eval/report.py`）封装为 `EvaluationReport` 数据结构，序列化为 JSON：

```json
{
  "schema_version": "v1_offline_evaluation_report",
  "evaluated_split": "val",
  "input_records_path": "outputs/pipeline_run/predictions_val.json",
  "output_report_path": "outputs/pipeline_run/report_val.json",
  "summary": {
    "total_clip_count": 188,
    "total_action_count": 3008,
    "total_correct_action_count": 2286,
    "action_accuracy": 0.76,
    "mean_temporal_consistency": 0.852,
    "split_metrics": {
      "val": {
        "split": "val",
        "clip_count": 188,
        "action_count": 3008,
        "correct_action_count": 2286,
        "action_accuracy": 0.76,
        "mean_temporal_consistency": 0.852
      }
    }
  }
}
```

> **注**：以上 JSON 中的数值为 ⚠️ 编造示意，实际值取决于数据和训练结果。

---

## 10. 端到端流水线

Nitrogen 的 `main.py` 提供一键式端到端执行接口，依次调用四个阶段：

| 阶段 | 步骤 | 输入 | 输出 |
|------|------|------|------|
| [1/4] 构建清单 | `build_manifest` | 原始回合目录 | `manifest.json` |
| [2/4] 模型微调 | `run_finetune` | 清单 + 配置模板 | `hf_model/` + `label2id.json` |
| [3/4] 推理预测 | `predict_clip_actions` | 清单 + 模型 | `predictions_val.json` |
| [4/4] 离线评估 | `build_evaluation_report` | 清单 + 预测 | `report_val.json` |

### 10.1 关键全局参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `raw_episodes_dir` | `data/raw` | 原始数据根目录 |
| `manifest_path` | `data/processed/manifest.json` | 清单输出路径 |
| `run_output_dir` | `outputs/pipeline_run` | 本次运行的输出根目录 |
| `finetune_config_template` | `configs/finetune.example.json` | 微调配置模板 |
| `EVAL_SPLIT` | `val` | 推理与评估所用的数据划分 |
| `CLIP_LENGTH` | 16 | 片段长度（帧数） |
| `STRIDE` | 16 | 滑动窗口步长 |
| `PREDICT_BATCH_SIZE` | 8 | 推理时每批处理的帧数 |

---

## 11. 实验结果与分析

> **[注意：以下实验数据为编造数据，仅用于展示报告格式。实际结果需在具体数据集上运行 Nitrogen 流水线后获取。所有编造数据以 ⚠️ 标注。]**

### 11.1 数据集概况

| 统计项 | 数值 |
|--------|------|
| ⚠️ 目标游戏 | 2D 横版平台跳跃游戏 |
| ⚠️ 总回合数 | 150 |
| ⚠️ 训练集回合数 | 120 |
| ⚠️ 验证集回合数 | 15 |
| ⚠️ 测试集回合数 | 15 |
| ⚠️ 总片段数 | 1,875 |
| ⚠️ 训练集片段数 | 1,500 |
| ⚠️ 验证集片段数 | 188 |
| ⚠️ 测试集片段数 | 187 |
| ⚠️ 总帧数 | 30,000 |
| ⚠️ 动作类别数（含 unknown） | 4（`move_left`, `move_right`, `jump`, `unknown`） |
| 片段长度 | 16 帧 |
| 窗口步长 | 16 帧（无重叠） |

> ⚠️ **数据来源说明**：演示数据由人类玩家在目标游戏中录制，帧率约为 10 FPS，每回合平均时长约 20 秒（~200 帧）。动作标注由游戏输入日志逐帧导出。

### 11.2 训练过程

本实验在 Nitrogen 预训练世界模型的权重基础上进行下游行为适配微调。Nitrogen 的视觉编码器（基于 ViT-Base 架构）在预训练阶段已通过 ⚠️ 大规模多游戏视觉数据学习了通用的游戏画面表征，在此基础上仅需少量目标游戏数据和极少训练轮次即可完成动作空间的适配。

| 指标 | 数值 |
|------|------|
| 预训练模型 | Nitrogen 世界模型（ViT-Base 架构主干） |
| ⚠️ 预训练数据规模 | ⚠️ ~500K 游戏画面（多游戏混合） |
| 微调范围 | 全参数微调（Nitrogen 编码器 + 新增分类头） |
| ⚠️ 下游训练帧总数 | 24,000 |
| ⚠️ 下游验证帧总数 | 3,008 |
| 微调轮次 | 2 |
| 学习率 | 5e-5 |
| 每设备批大小 | 4 |
| ⚠️ 训练总步数 | 12,000 |
| ⚠️ 训练总时长 | ~25 分钟（单卡 NVIDIA A100 80GB） |
| ⚠️ Epoch 1 验证准确率 | 68.3% |
| ⚠️ Epoch 2 验证准确率（最终） | 76.0% |

> ⚠️ **训练动态说明**：得益于 Nitrogen 世界模型在预训练阶段习得的游戏视觉表征，模型在微调第 1 轮即达到 68.3% 的验证准确率，远高于随机基线（25%）。第 2 轮进一步提升至 76.0%，说明仅 2 轮微调即可使 Nitrogen 快速适配到目标游戏的动作空间，验证了「预训练世界模型 + 少量下游数据微调」这一适配策略的高效性。

### 11.3 评估结果

#### 验证集（val）

| 指标 | 数值 |
|------|------|
| ⚠️ 片段数 | 188 |
| ⚠️ 帧数 | 3,008 |
| ⚠️ 正确帧数 | 2,286 |
| ⚠️ **动作准确率** | **76.0%** |
| ⚠️ **平均时序一致性** | **0.852** |

#### 测试集（test）

| 指标 | 数值 |
|------|------|
| ⚠️ 片段数 | 187 |
| ⚠️ 帧数 | 2,992 |
| ⚠️ 正确帧数 | 2,214 |
| ⚠️ **动作准确率** | **74.0%** |
| ⚠️ **平均时序一致性** | **0.838** |

#### ⚠️ 基线对比

| 方法 | Val 准确率 | Test 准确率 |
|------|-----------|-----------|
| ⚠️ 随机预测 | 25.0% | 25.0% |
| ⚠️ 多数类预测（始终预测 `move_right`） | 38.2% | 37.5% |
| ⚠️ ViT-Base 从头训练（随机初始化，无预训练） | 52.1% | 49.8% |
| ⚠️ ViT-Base + ImageNet 预训练微调 | 67.5% | 65.2% |
| ⚠️ **Nitrogen 预训练 + 微调（本项目）** | **76.0%** | **74.0%** |

> ⚠️ **分析**：
>
> - **Nitrogen vs. 从头训练**：Nitrogen 预训练权重相比随机初始化带来约 24 个百分点的提升，验证了预训练世界模型的知识迁移价值。
> - **Nitrogen vs. ImageNet 预训练**：相比通用 ImageNet 预训练的 ViT-Base，Nitrogen 的游戏领域预训练额外贡献了约 8.5 个百分点的提升，说明 Nitrogen 在游戏视觉数据上学到的领域特定表征（如游戏角色、场景布局、动态物体等）对下游行为适配至关重要。
> - Val 与 Test 之间 2 个百分点的差距属于正常波动范围，表明模型未出现严重过拟合。

### 11.4 动作类别分析

| 动作 | ⚠️ 样本占比 | ⚠️ 精确率 | ⚠️ 召回率 | ⚠️ F1 |
|------|-----------|----------|----------|------|
| `move_left` | 28.5% | 0.78 | 0.81 | 0.79 |
| `move_right` | 38.2% | 0.80 | 0.76 | 0.78 |
| `jump` | 26.1% | 0.69 | 0.65 | 0.67 |
| `unknown` | 7.2% | 0.42 | 0.38 | 0.40 |

> **⚠️ 编造分析**：
>
> - `move_left` 和 `move_right` 是最常见的动作，合计占比约 67%，模型在这两类上的表现最为稳定（F1 ≈ 0.78-0.79），这是因为水平移动在视觉上具有明确的场景位移特征，单帧分类器较容易捕获。
> - `jump` 动作的识别率较低（F1 = 0.67），推测原因是跳跃动作在视觉上跨越多帧过程，单帧分类器难以准确判断跳跃的起始、滞空和落地阶段的区别，且跳跃过程中的帧容易与移动帧混淆。
> - `unknown` 类别性能最差（F1 = 0.40），这是预期之中的——该类别作为异质的兜底类别，包含了待机、特殊交互等多种低频行为，视觉特征缺乏一致性。其低样本占比（7.2%）也限制了模型的学习。

### 11.5 时序一致性分析

> **⚠️ 编造分析**：
>
> 平均时序一致性（Mean TC）为 0.852（val），表示模型在约 85% 的相邻帧对上预测了相同的动作。在 16 帧片段中，这意味着平均约 2.2 次动作切换。
>
> 作为参考，基于真实标注的 Ground Truth 时序一致性约为 ⚠️ 0.91，即人类玩家平均每 ~7 帧（0.7 秒）切换一次动作。模型的 TC 略低于 Ground Truth，说明模型引入了一定程度的预测抖动——在动作过渡区域（如从移动切换到跳跃的瞬间），模型在相邻帧上可能给出不一致的预测。这一现象符合单帧分类器的固有局限，也为未来引入时序建模（如在 Nitrogen 世界模型的潜空间中进行序列决策）提供了动机。

---

## 12. 技术实现细节

### 12.1 依赖栈

| 库 | 最低版本 | 用途 |
|----|---------|------|
| `torch` | ≥ 2.2.0 | 深度学习框架 |
| `transformers` | ≥ 4.41.0 | 预训练模型加载、微调、推理 |
| `accelerate` | ≥ 0.27.0 | 分布式训练与混合精度支持 |
| `Pillow` | ≥ 10.0.0 | 图像加载与预处理 |
| `numpy` | ≥ 1.26.0 | 数值计算 |
| `pytest` | ≥ 8.0.0 | 单元测试框架 |

### 12.2 设备自动检测

推理时自动选择最优计算设备：

1. 检查 CUDA 可用性 → 使用 GPU
2. 检查 MPS 可用性 → 使用 Apple Silicon GPU
3. 回退至 CPU

### 12.3 标签映射的确定性

`build_label2id` 函数通过对所有动作 ID 进行**排序去重**后顺序编号，确保相同训练数据始终产生相同的标签映射。`unknown_action_id` 如不在训练集中，会自动追加后参与排序。

### 12.4 配置体系

配置加载支持两种动作映射来源：

- **内联映射**（`action_mapping` 字段直接写在配置 JSON 中）
- **外部文件映射**（`action_mapping_path` 指向独立的 JSON 文件）

两者互斥，同时指定会抛出异常。别名映射（`action_aliases` / `action_aliases_path`）同理。

### 12.5 HF Trainer 兼容性

训练参数中的评估策略键名存在 Transformers 版本差异（旧版 `evaluation_strategy`，新版 `eval_strategy`）。系统通过运行时反射 `TrainingArguments.__init__` 的签名自动选择正确的键名：

```python
ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
```

---

## 13. 配置与可复现性

### 13.1 完整配置模板

```json
{
  "manifest_path": "data/processed/manifest.json",
  "output_dir": "outputs/run1",
  "model_id": "google/vit-base-patch16-224",   // Nitrogen 预训练权重的 HF Hub ID 或本地路径
  "action_mapping": {
    "walk left": "move_left",
    "walk right": "move_right",
    "jump": "jump"
  },
  "action_aliases": {},
  "unknown_action_id": "unknown",
  "confidence_floor": 0.0,
  "num_train_epochs": 2.0,
  "per_device_train_batch_size": 4,
  "per_device_eval_batch_size": 4,
  "learning_rate": 5e-5,
  "logging_steps": 10,
  "seed": 42,
  "run_validation": true,
  "max_train_clips": null,
  "max_val_clips": null
}
```

> **注**：`model_id` 字段指定 Nitrogen 预训练权重的来源。当前示例中使用 `google/vit-base-patch16-224` 作为占位符；实际使用时应替换为 Nitrogen 预训练权重在 Hugging Face Hub 上的标识符或本地路径。

### 13.2 可复现性保障措施

| 维度 | 机制 |
|------|------|
| 数据划分 | SHA-256 哈希 + seed，同一 seed 下回合级划分完全确定 |
| 标签映射 | 排序后顺序编号，同数据同映射 |
| 训练随机性 | `seed` 参数控制 Transformers Trainer 的所有随机源 |
| 配置溯源 | 每次运行将完整配置写入输出目录（`finetune_config.json`） |
| 评估溯源 | 评估报告记录输入记录路径和输出路径 |

### 13.3 CLI 命令速查

```bash
# 1. 安装依赖
python3.12 -m pip install -r requirements.txt

# 2. 构建数据集清单
python3.12 -m scripts.build_dataset \
  --input data/raw \
  --output data/processed/manifest.json \
  --seed 0 --clip-length 16 --stride 16 \
  --train 0.8 --val 0.1 --test 0.1

# 3. 微调模型
python3.12 -m scripts.finetune --config configs/finetune.example.json

# 4. 生成预测
python3.12 -m scripts.predict \
  --model-dir outputs/run1 \
  --manifest data/processed/manifest.json \
  --split val \
  --finetune-config outputs/run1/finetune_config.json \
  --output outputs/run1/predictions_val.json

# 5. 端到端一键运行
python3.12 main.py
```

---

## 14. 局限性与未来工作

### 14.1 当前局限性

| 局限 | 说明 |
|------|------|
| **单帧分类** | 当前适配方案仅基于单帧预测动作，未充分利用 Nitrogen 世界模型的时序建模潜力 |
| **动作空间有限** | 示例配置仅包含 3 个动作 + 1 个未知类，实际游戏可能有更复杂的动作空间 |
| **判别式适配** | 当前仅使用 Nitrogen 的视觉编码器进行判别式分类，未利用其生成/预测能力 |
| **无在线评估** | 系统仅支持离线评估，不包含游戏环境内的闭环执行与回报评估 |
| **无多帧建模** | 缺乏对帧序列的时序建模能力（如 LSTM、Temporal Transformer 等） |
| **无测试套件** | 项目尚未包含单元测试，虽然框架支持 pytest |

### 14.2 未来工作方向

1. **Nitrogen 全能力利用**：当前仅使用 Nitrogen 的视觉编码器做帧级分类，未来应探索利用 Nitrogen 世界模型的完整能力——包括环境动态预测、状态转移建模等——进行更高级的策略学习
2. **时序建模**：在 Nitrogen 编码器特征之上叠加时序建模层（如 LSTM/GRU 或 Temporal Transformer），利用多帧上下文进行动作预测，更好地契合 Nitrogen 对时序动态的表征
3. **潜空间策略学习**：在 Nitrogen 世界模型学习的潜在表征空间中训练策略网络，而非直接在像素空间进行分类
4. **多游戏泛化**：评估 Nitrogen 预训练的跨游戏迁移能力，扩展数据集规范支持多游戏场景
5. **在线评估**：集成游戏仿真环境，评估适配后的 Nitrogen 模型在闭环执行中的累计回报
6. **数据增强**：在微调阶段引入随机裁剪、颜色扰动等图像增强策略
7. **更丰富的评估指标**：增加 per-class F1、混淆矩阵可视化、预测错误案例分析等
8. **单元测试**：为数据 schema 验证、对齐逻辑、指标计算等核心模块补充 pytest 测试
9. **分布式训练**：利用 `accelerate` 支持多 GPU / 多节点训练

---

## 15. 附录

### 附录 A：核心数据类一览

| 模块 | 数据类 | 用途 |
|------|--------|------|
| `src.data.schema` | `SplitName` | 数据集划分枚举 |
| `src.data.schema` | `ActionLabel` | 帧级动作标签 |
| `src.data.schema` | `FrameRecord` | 帧引用记录 |
| `src.data.schema` | `ClipRecord` | 片段记录 |
| `src.data.schema` | `EpisodeRecord` | 回合记录 |
| `src.data.schema` | `SplitPolicy` | 划分策略 |
| `src.data.loader` | `ManifestSample` | 清单中的单个片段样本 |
| `src.data.loader` | `ManifestDataset` | 清单数据集（`Sequence`） |
| `src.data.split` | `EpisodeSplitAssigner` | 确定性划分分配器 |
| `src.model.alignment` | `RawActionRecord` | 对齐前的原始动作 |
| `src.model.alignment` | `AlignedActionRecord` | 对齐后的规范化动作 |
| `src.model.alignment` | `VocabularyActionAligner` | 词汇表对齐器实现 |
| `src.model.alignment` | `AlignmentSummary` | 对齐批量统计 |
| `src.model.alignment` | `AlignedManifestSample` | 对齐后的清单样本 |
| `src.train.hf_finetune` | `FrameLabelRow` | 帧-标签训练行 |
| `src.train.hf_finetune` | `FrameClassificationDataset` | PyTorch 图像分类数据集 |
| `src.train.hf_finetune` | `HFFinetuneParams` | 微调超参数 |
| `src.eval.metrics` | `EvaluationRecord` | 单片段评估记录 |
| `src.eval.metrics` | `SplitEvaluationMetrics` | 按划分聚合的指标 |
| `src.eval.metrics` | `EvaluationSummary` | 总体评估摘要 |
| `src.eval.pipeline` | `PredictionRecord` | 预测记录 |
| `src.eval.report` | `EvaluationReport` | 评估报告产物 |

### 附录 B：JSON Schema 版本

| Schema | 版本标识 | 用途 |
|--------|---------|------|
| 片段清单 | `v2_clip_manifest` | 数据集构建输出 |
| 评估报告 | `v1_offline_evaluation_report` | 离线评估输出 |

### 附录 C：运行环境要求

- **Python**：3.12+
- **操作系统**：Linux / macOS（含 Apple Silicon）/ Windows
- **GPU**（推荐）：NVIDIA CUDA 兼容 GPU 或 Apple M 系列
- **磁盘空间**：⚠️ 视数据集大小而定，典型实验约需 5-20 GB
- **内存**：⚠️ 建议 ≥ 16 GB RAM

---

> **文档说明**：本报告中所有以 ⚠️ 标注的数据均为演示用途的编造数据，实际值需在具体数据集和硬件上运行 Nitrogen 流水线后获取。报告的代码分析部分基于仓库实际源码编写，准确反映系统的设计与实现。
