# Track 1（文本语境 CoT-TTS）完整路线规划：数据处理 + 训练微调

> 基于《ISCSLP 2026 CoT-TTS 数据集》文档 + 本仓库 baseline 代码（`cot_tts_text_history_inference.py`）
> 目标：教你从"原始数据集"一路规划到"训练/微调出自己的 Track 1 模型"。
> 配套：《Track1任务文档》《架构与运作流程解析》《投递ICASSP备战指南》

---

## 第一部分：数据集文档解读（先把数据看明白）

### 1.1 数据集是什么
HKUSTAudio/ISCSLP2026-CoT-TTS，约 **16K 小时、~3M 片段**，中英双语（英 54% / 中 46%），取材自电影/电视剧/广播剧/短剧。每个样本围绕一个**目标话语(target utterance)**组织，附带它的**前文对话语境、参考语音、声学特征、情绪标签和 CoT 分析**。

⚠️ **一个关键设计**：音频**故意不做激进降噪/标准化**（只统一成 FLAC），保留真实声学条件——所以不同文件采样率、声道、响度、噪声都可能不同。**但元数据/标注是基于"降噪归一化版"生成的。** 这意味着：你拿到的标注（CoT、情绪、时长、响度）是"干净版"的，而音频是"原始版"的——**预处理时这点很重要**（见 2.3）。

### 1.2 文件结构（六个文件夹，结构相同）
```
HKUSTAudio/ISCSLP2026-CoT-TTS/
├── movie-en1/  (movie-zh2/ ... 共六个，按语言+分区拆分)
│   ├── metadata.json          ← 所有标注在这里
│   ├── dialogue_segments/*.flac    ← 话语级片段（按说话人日记时间戳切，较"干净"）
│   └── continuous_segments/*.flac  ← 连续片段（每句前后延长，保留语境连续性）
```

**`dialogue_segments` vs `continuous_segments` 怎么选？**
- `dialogue_segments`：只含检测到的语音区间（如 0–3s、6–7s），更干净、更短。
- `continuous_segments`：把每句前后延长、连续覆盖（如 0–4.5s、4.5–9.5s），更保留上下文连续性。
- **Track 1 建议**：目标音频和参考音频用 **`dialogue_segments`**（干净、利于学音色和内容）；如果你想让历史语境更自然连续，可对照实验试试 `continuous_segments`。

### 1.3 metadata.json 字段（Track 1 要用哪些）
每条样本一个 JSON 对象，核心字段：

| 字段 | 含义 | Track 1 用途 |
|---|---|---|
| `target_segment.text` | 目标话语文本 | **目标文本**（要合成的句子） |
| `target_segment.segment_id` | 目标音频 ID | 定位**目标音频**（训练时的"标准答案"语音） |
| `target_segment.ref_segment_id` | 参考音频 ID | 定位**参考音频**（决定输出音色） |
| `target_segment.cot_text` | CoT 风格分析 | **CoT 监督目标**（模型要学着生成它） |
| `target_segment.emotion_tag` | 情绪描述 | 可拼进 CoT / 历史标注 |
| `target_segment.features` | duration / active_duration / loudness / expressive_intensity | 拼进 CoT 的时长·响度·表现强度字段 |
| `dialog_segments[]` | 历史对话片段(按时间排序) | **历史语境**：取 `normalized_speaker` + `text` 拼成历史文本 |
| `normalized_speaker` | 场景内重新索引的说话人标签 | Track 1 历史文本的 speaker 标签 |
| `index` | 样本唯一索引 | 数据加载/文件匹配 |

**匹配规则**：音频文件通过 `segment_id` / `ref_segment_id` 匹配；`dialog_segments` 已按时间顺序排好。

### 1.4 合规红线（写论文/训练都受约束）
- **仅限非商业学术研究**（CC BY-NC 4.0 精神）；禁止商用、转售、再分发、重建源媒体、识别原始说话人。
- 论文展示输出时**须声明是合成语音**。
- 说话人标签是自动生成的匿名标签，**不是真实身份**。
- 标注（转写/情绪/CoT/时间戳）**可能有错**，是参考而非绝对真值——做实验时要心里有数。

---

## 第二部分：Track 1 需要什么数据，怎么处理

### 2.1 先想清楚：Track 1 模型要学的"输入→输出"映射

回顾 baseline 的 prompt（`cot_tts_text_history_inference.py`），训练一条样本就是教模型补全下面这段：

```
【输入(prompt，不算 loss)】
<bos><task_start>COT-TTS<task_end>
<history_start><under_start>{历史对话文本}<under_end><history_end>
<target_start>
  <text_start>{目标文本}<text_end>
  <audio_ref_start>{参考音频的 BiCodec global token}<audio_ref_end>
<target_end>
<output_start>

【输出(监督目标，算 loss)】
<cot_start>{CoT 分析}<cot_end>
<audio_tar_start>{目标音频的 BiCodec global + semantic token}<audio_tar_end>
<output_end><eos>
```

> 所以你要从数据集里凑齐**5 样东西**：历史文本、目标文本、参考音频 token、CoT 文本、目标音频 token。前两个是纯文本，后三个要么来自标注、要么要**离线用 BiCodec 编码音频**。

### 2.2 数据处理流水线（离线预处理）

```
for 每条 metadata 样本:
  1. 历史文本 = 把 dialog_segments 按时间拼成
       "speaker-1: ...\nspeaker-2: ..."（用 normalized_speaker；可选附 [emotion_tag]）
  2. 目标文本 = target_segment.text
  3. 参考音频 = 用 ref_segment_id 找到 .flac → BiCodec.tokenize() → 取 global 前 32 个
  4. 目标音频 = 用 segment_id 找到 .flac → BiCodec.tokenize() → global(32) + semantic(变长)
  5. CoT 目标 = 由 cot_text 组装（必要时补 features 的 duration/loudness/expressive_intensity 行）
  6. 拼成上面的完整序列字符串 → 用 LLM tokenizer 编码成 input_ids
  7. 设置 labels：prompt 部分置 -100(不算loss)，输出部分=真实 token id
  8. 存成 parquet / arrow（VeOmni 训练读取的格式）
```

**为什么离线编码音频 token？** BiCodec 编码很慢，训练时每 step 现编码会拖垮速度。标准做法是**预处理阶段一次性把所有音频编码成 token 存盘**，训练时直接读 token。这是这类工作最重要的工程决定之一。

**实现要点**：
- 直接复用 baseline 里的 `BiCodecTokenizer`、`to_tokens()`、`build_prompt()`（文本历史版）——把推理代码改造成"造训练数据"的脚本，**保证训练/推理 prompt 格式完全一致**（否则训出来推理时对不上）。
- 全部音频 BiCodec 前要**重采样到 16kHz、转单声道**（数据集采样率不统一，见 1.1）。
- `loudness`/`expressive_intensity`/`duration` 这些标注是基于"降噪归一化版"算的，拼进 CoT 没问题；但**音频 token 是从你拿到的原始音频编码的**——若噪声太大可考虑做轻度降噪后再编码（这本身可以是个对照实验）。

### 2.3 数据清洗与筛选（直接影响效果）
数据集明确说标注可能有错、音频有噪声/重叠/音乐。建议建一套**过滤规则**，例如：
- 丢弃 `best_ref_similarity` 过低的样本（参考音色和目标对不上）；
- 丢弃 `active_duration` 过短/过长、或 semantic token 数异常的样本；
- 丢弃历史为空、或文本明显是乱码/错转写的样本；
- 中英分开统计，保证两种语言都有足够干净样本。

> **这一步是新人最容易忽视、却最能拉开差距的地方。** "数据质量 > 模型技巧"在 TTS 里尤其成立。

### 2.4 划分数据集
- **train / val / test** 按 `movie_id` 或 `scene_index` 划分，**确保同一部影视不跨集**（防止信息泄漏，让评测可信）。
- 留一个小而干净的 **dev 集**（几十~几百条）用于快速看指标、调超参。
- 中英分别留测试集，最终都要报。

---

## 第三部分：怎么训练 / 微调

### 3.0 先认清现实：这个仓库只有"推理代码"
README 写得很清楚——本仓库是 **inference code**。`veomni/` 里有训练需要的运行时（optim、schedulers、distributed、checkpoint、data、lora_utils），**但没有训练启动脚本**（没有 `train.py`），而且 `veomni_cli.yaml`（训练配置）是随模型包下载的、不在 git 里。

**结论**：要真正训练/微调，你需要**完整的 VeOmni 训练框架**（开源仓库），用它的训练入口 + 本 baseline 的数据格式和特殊 token。下面给两条路线。

### 3.1 路线 A（推荐，省算力）：在官方 baseline 上做 LoRA 微调
官方 baseline 本身就是 **Spark-TTS-0.5B + LoRA(rank 8)** 训出来的（`hf_checkpoint_has_lora`、`lora_config_from_run` 可证）。最划算的路线是**继续在它基础上 LoRA 微调**：

1. **基座**：Spark-TTS-0.5B（LLM 主干，约 0.5B 参数）+ 已有 baseline LoRA 权重（可选择从它继续训，或重训 LoRA）。
2. **方法**：用 `veomni.utils.lora_utils.add_lora_to_model` 注入 LoRA（rank 8、alpha 16、target = q/k/v/o/gate/up/down_proj，和 `veomni_cli.yaml` 一致）。
3. **数据**：第二部分产出的 parquet。
4. **目标**：自回归语言建模 loss，只在输出段（CoT + 音频 token）算 loss。
5. **算力**：0.5B + LoRA 很轻，**单卡 24GB（如 3090/4090/A5000）即可起步**；想快就多卡。比起训 16K 小时全量，LoRA 微调几个 epoch/几十小时量级就能见效。

> 为什么推荐 LoRA：省显存、省时间、便于多检查点对比，最契合你"个人/小团队 + ICASSP 截止只剩 ~11 周"的现实。

### 3.2 路线 B（全量微调 / 从头训）：算力大、不建议新人首选
- 全参数微调 0.5B 主干，需要更多显存与数据工程，收益对一篇 4 页论文未必更高。
- 从头训 codec/LLM 更是大工程，**不在你当前目标范围内**，除非有团队和集群。

### 3.3 训练流程（路线 A 的标准步骤）
```
1. 准备环境：VeOmni 训练框架 + 本 baseline 的 sparktts/ + 依赖
2. 预处理：第二部分流水线 → train/val parquet（含离线 BiCodec token）
3. 配置 veomni_cli.yaml：
     - 基座路径(Spark-TTS-0.5B)、LoRA 超参、特殊 token 词表
     - batch size / lr / warmup / epoch / 梯度累积 / bf16 / flash-attn
4. 启动训练（VeOmni 的分布式训练入口，DCP 格式存检查点）
5. 定期在 dev 集上评估，存 global_step_* 检查点
6. 用本仓库推理脚本加载你的检查点验证：
     python infer/cot_tts_text_history_inference.py --checkpoint_path <你的检查点> ...
     （脚本会自动 DCP→HF、自动探测 LoRA）
```

**关键超参（起点参考，再据 dev 集调）**：
- 学习率：LoRA 常用 1e-4 ~ 3e-4；warmup 数百 step。
- batch：按显存定，配合梯度累积凑有效 batch。
- epoch：先训 1–3 个 epoch 看趋势，别一上来训很久。
- 精度：bf16 + flash_attention_2（装不上换 sdpa）。
- 固定随机种子，记录每次配置（见写论文指南）。

### 3.4 训练中要盯什么
- **loss 下降是否健康**（CoT 文本和音频 token 的 loss 可分开看）。
- 定期**真的跑一遍推理听 wav**——loss 低不代表语音好，TTS 必须耳朵验收。
- 监控 dev 集客观指标：说话人相似度、WER、（你定义的）情绪/推理一致性。
- 早停：dev 指标不再涨就停，省算力。

---

## 第四部分：把"训练"接到你的 ICASSP 选题

你不一定要"训一个更强的模型"才能发论文。结合《ICASSP备战指南》的选题：

- **选题 A（一致性度量与重排）**：**几乎不用训练**！只在 baseline 推理上加一个重排器，最省算力、最快出结果——**强烈推荐时间紧的你走这条**。
- **选题 B（CoT 消融）**：可能需要**轻量 LoRA 重训**几个变体（有 CoT / 无 CoT / 去某字段），用第三部分路线 A 即可，算力可控。
- 若你确实想训一个"改进版 Track 1 模型"当主贡献：用路线 A 的 LoRA 微调 + 你的数据清洗策略/损失改进，作为方法创新点。

**给你的明确建议**：
> **先用选题 A 走通"无需训练即可出论文"的最短路径**；如果时间和算力允许，再用路线 A 的 LoRA 微调做选题 B 的消融作为加分。**别一上来就奔着"训一个 SOTA 大模型"——那是团队+集群的活，不是 11 周个人项目的最优解。**

---

## 第五部分：行动清单（本周可做）

- [ ] 申请/下载数据集（注意 license），先只下 **一个文件夹**（如 movie-en1）跑通流程，别一次拉 16K 小时。
- [ ] 写一个脚本：读 `metadata.json` → 打印一条样本的历史文本/目标文本/CoT/各 segment_id，**确认你看懂字段映射**。
- [ ] 复用 `BiCodecTokenizer` + 文本历史 `build_prompt`，把 1 条样本造成完整训练序列，肉眼核对格式与推理一致。
- [ ] 写离线 BiCodec 编码 + parquet 落盘脚本，先处理 100 条验证管线。
- [ ] 制定数据过滤规则（best_ref_similarity、时长、token 数、空历史）。
- [ ] **决定路线**：先做选题 A（免训练）出第一版结果；并行准备路线 A 的 LoRA 微调环境。

---

## 一页速记

- **数据**：metadata.json 给文本/CoT/情绪/特征/segment_id；音频在 `dialogue_segments`(干净)/`continuous_segments`(连续)。Track 1 取历史文本+目标文本+参考/目标音频。
- **处理**：拼历史文本 → 离线 BiCodec 把音频编成 token → 按 baseline prompt 格式拼序列 → labels 只算输出段 → 存 parquet。**清洗+按影视划分防泄漏**是关键。
- **训练**：仓库只有推理代码，训练靠 **VeOmni 框架**；推荐 **Spark-TTS-0.5B + LoRA 微调**（单卡 24GB 可起步），别从头训。
- **接 ICASSP**：选题 A 免训练最快；选题 B 用 LoRA 轻量重训消融。**先出结果，再谈改进。**
