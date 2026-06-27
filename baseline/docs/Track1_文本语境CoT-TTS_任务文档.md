# Track 1：文本语境感知 CoT-TTS（Text-Context-Aware CoT-TTS）任务文档

> 来源：ISCSLP 2026 CoT-TTS Challenge 官方赛题说明 + 本仓库对应基线代码
> 对应基线：`infer/cot_tts_text_history_inference.py`（文本历史基线，checkpoint `global_step_24500`）
> 配套阅读：《架构与运作流程解析》（讲音频历史基线原理，与本赛道共享同一套 token/prompt 体系）、《新手快速上手教程》

---

## 1. 赛题总览（Challenge Description）

> **给定对话语境(dialogue context)、目标文本(target text)、参考语音样本(reference speech)，系统需要先推理出"表演分析(reasoning analysis)"，再生成一段语音波形——这段语音既要与推理分析一致，又要与参考说话人的音色(timbre)一致。**

挑战的关键特征：

| 维度 | 内容 |
|---|---|
| 赛道数 | **2 个 Track**：Text Context（Track 1）与 Audio Context（Track 2） |
| 语言 | **双语：英文 + 中文** |
| 训练数据 | **约 16K 小时** |
| 数据片段 | **约 3M 段** |
| 评测构成 | **30% 客观指标 + 20% LLM 评判 + 50% 人工评测** |

### 三段式任务流水线（Task Pipeline）
1. **Context Input（语境输入）**：提供对话语境、目标文本、参考语音样本，作为风格推断的条件信号。
2. **CoT Reasoning（链式思维推理）**：合成之前，先分析"前因后果"，并总结目标音频应有的说话风格。
3. **Speech Generation（语音生成）**：生成语音，既保留参考说话人音色，又匹配推理出的说话方式与场景语境。

---

## 2. Track 1 定义：Text-Context-Aware CoT-TTS

**一句话**：模型读取**带说话人标签的对话历史（文本形式）**，据此推断目标句的**情绪、语气、节奏、交际意图**，再合成与之一致的语音。

> 这正是本赛道与 Track 2 的唯一本质区别：**Track 1 的对话历史以"文本"提供，Track 2 以"音频"提供。** 其余（目标文本、参考语音、CoT 推理、输出波形）完全相同。

### 2.1 输入（INPUT）
| 输入 | 说明 |
|---|---|
| **Dialogue context（对话语境）** | 带说话人标签的历史对话文本，例如 `speaker-0: I cannot believe this happened.` |
| **Target text（目标文本）** | 本次要合成的句子，例如 `Then we must act now.` |
| **Reference speech（参考语音）** | 决定输出音色身份的参考录音，例如 `ref_speaker.wav` |

### 2.2 期望输出（EXPECTED OUTPUT）
系统必须**同时产出两样东西**：
1. **Reasoning analysis（推理分析 / CoT）**：解释"为什么该这样说"；
2. **Output audio waveform（输出语音波形）**：与推理出的说话风格对齐。

### 2.3 官方示例（输入/输出格式）
```json
// 输入
{
  "context": ["speaker-0: I cannot believe this happened."],
  "target_text": "Then we must act now.",
  "reference_audio": "ref_speaker.wav"
}

// 输出
{
  "reasoning": "The previous turn signals shock and urgency, so the target ...",
  "output_audio": "track1_prediction.wav"
}
```

---

## 3. 这个赛道难在哪 / 与众不同之处

官方强调三个 FOCUS：

| 焦点 | 含义 | 对系统的要求 |
|---|---|---|
| **Contextual Understanding（语境理解）** | 必须理解**动态演进的对话场景**，而不是依赖孤立的风格标签 | 不能只看 target text，要"读懂前文" |
| **Explicit Reasoning（显式推理）** | 必须通过推理分析**显式暴露"为什么这句话该这么说"** | CoT 不是黑盒，要可读、可评判 |
| **Speech-Reasoning Consistency（语音-推理一致性）** | 生成的波形应**忠实反映推理输出**，而不只是"听起来自然" | 说出来的语气要和推理结论对得上 |

> 核心思想：传统 TTS 追求"自然"，本赛道在自然之上额外要求 **"语境正确 + 推理可解释 + 言行一致"**。这也是为什么评测把人工和 LLM 评判占到 70%。

---

## 4. 评测体系（Evaluation Snapshot）

官方评测在一条统一流水线里同时衡量**语音质量**与**推理可靠性**，最终排名反映自然度、语境贴合度、推理质量、语音-推理一致性四方面。三块权重：

| 权重 | 评测方式 | 考察点 |
|---|---|---|
| **30%** | **客观指标（Objective）** | 语音质量、可懂度、说话人相似度、韵律、表现力、效率 |
| **20%** | **LLM 评判（LLM-Based）** | 语境理解、推理的内部逻辑一致性、信息量 |
| **50%** | **人工评测（Human）** | 语境连贯性、推理准确度、信息量、自然度、语音-推理一致性 |

**给参赛者的启示**：
- 客观指标只占 30%，**别只盯着 WER/说话人相似度刷分**；
- 50% 人工 + 20% LLM = **70% 在评判"推理质量"和"言行一致"**，所以 CoT 写得对、且语音真的体现出来，比单纯音质更重要；
- "说话人相似度"在客观分里有明确位置 → **保住参考音色是基本盘**（对应基线的 `--use_ref_global_tokens`）。

---

## 5. Track 1 在本仓库里的对应实现

Track 1 对应基线脚本 **`infer/cot_tts_text_history_inference.py`**，与音频历史基线共享同一套 token 协议，差异仅在"历史段"。

### 5.1 Prompt 结构（源码 `build_prompt`，文本历史版）
```text
<bos>
<task_start>COT-TTS<task_end>
<history_start>
  <under_start>{对话历史文本}<under_end>      ← Track 1 用文本，直接放进 <under>
<history_end>
<target_start>
  <text_start>{目标文本}<text_end>
  <audio_ref_start>{参考音频 global token}<audio_ref_end>
<target_end>
<output_start>                                  ← prompt 到此为止，模型从这里续写 CoT + 语音 token
```

> 对比 Track 2（音频历史）：历史段是 `<audio_his_start>{global+semantic token}<audio_his_end>`。
> **Track 1 = 文本进 `<under>`；Track 2 = 音频编码成 token 进 `<audio_his>`。** 这是两套脚本唯一的结构性差别。

### 5.2 端到端流程
1. 读对话历史文本 + 目标文本；
2. 参考音频经 **BiCodec** 编码出 global token（音色指纹，前 32 个）；
3. 拼成上面的 prompt；
4. **LoRA 微调的 LLM** 自回归生成：`<cot_start>` 推理分析 + `<audio_tar_start>` 语音 token；
5. 解析出 BiCodec global/semantic token，用 **参考音色 global + 生成的 semantic** 经 BiCodec 解码还原 wav。

### 5.3 CoT 推理分析包含什么
模型输出的推理分析是结构化字段（见音频历史样例 `sample_case.raw.txt`，两赛道一致）：
`Act`（动作）/`Scene`（场景）/`Motivation`（动机）/`Goal`（目标）/`Emotion`（情绪走向）/`Valid·Total duration`（时长）/`Loudness·Expressive Intensity`（响度·表现强度）/`[Summary]`（总结："因为…所以带着…情绪说了…"）。
—— 这正好回应官方对 Explicit Reasoning 的要求。

### 5.4 跑通命令（官方启动方式）
```bash
CUDA_VISIBLE_DEVICES=0 python infer/cot_tts_text_history_inference.py \
  --checkpoint_path model/cot_tts_text_history_baseline/checkpoints/global_step_24500 \
  --history_text_path infer/cases/sample_case/history.txt \
  --target_text_path infer/cases/sample_case/target.txt \
  --ref_audio_path  infer/cases/sample_case/reference.wav \
  --output_prefix   infer/results/text_history_demo \
  --spark_model_dir model/Spark-TTS \
  --model_architecture lora \
  --device cuda:0 \
  --do_sample --temperature 0.6 --top_p 0.75 \
  --min_new_tokens 256 --max_new_tokens 2000 \
  --use_ref_global_tokens \
  --num_candidates 1 --rerank_metric none
```
或一键：`CUDA_VISIBLE_DEVICES=0 bash infer/run_cot_tts_text_history_inference.sh`

> 注意采样默认与 Track 2 略有不同：Track 1 用 `top_p 0.75`、`min_new_tokens 256`（音频历史基线是 `top_p 0.8`、`min_new_tokens 32`）。

### 5.5 输入数据格式映射
官方示例 JSON ↔ 仓库文件：
| 官方字段 | 仓库对应 |
|---|---|
| `context`（带 speaker 标签的历史） | `infer/cases/sample_case/history.txt`（如 `speaker-1: ...` 每行一句） |
| `target_text` | `target.txt` |
| `reference_audio` | `reference.wav` |
| 输出 `reasoning` | 结果里的 `*.cot_thinking.txt` |
| 输出 `output_audio` | 结果里的 `*.wav` |

---

## 6. 参赛打分策略建议（结合评测权重）

1. **先保基本盘（客观 30%）**：开 `--use_ref_global_tokens` 锁住参考音色（说话人相似度）；目标文本要读全、读清（可懂度/WER）。
2. **重攻推理质量（LLM 20%）**：CoT 要逻辑自洽、信息量足、与对话历史的因果对得上——这块直接被 LLM 评判打分。
3. **死磕一致性（人工 50% 的核心）**：确保"推理说要愤怒/急促"，合成语音听起来真的愤怒/急促。可用消融或重排（如把 `mel_similarity` 重排换成"CoT 一致性打分"）提升。
4. **双语都要测**：英文+中文都在评测范围，别只调一种语言。
5. **善用多候选**：`--num_candidates N` 生成多条 + 自定义重排选最优，是低成本提分手段。

---

## 7. 一页速记

- **Track 1 = 文本对话历史 → 推断说话风格 → 合成贴合语境且保持参考音色的语音。**
- 输入：对话历史文本(带 speaker 标签) + 目标文本 + 参考音频。
- 输出：CoT 推理分析 + 语音波形（两者都要交，且要一致）。
- 代码：`cot_tts_text_history_inference.py`，历史文本进 `<under>`，参考音色进 `<audio_ref>`，模型续写 `<cot>` + 语音 token。
- 评测：30% 客观 / 20% LLM / 50% 人工，**七成在评"推理对不对、言行一不一致"**。
- 与 Track 2 区别：**仅历史输入形式不同（文本 vs 音频）**，其余完全一致。
