# COT-TTS 音频历史记录基线 —— 架构与运作流程解析

> 对象：ISCSLP 2026 Challenge Baseline 仓库中的 **audio-history COT-TTS**（对话历史以"音频"形式提供的链式思维 TTS 基线）
> 核心入口：`infer/cot_tts_inference.py`
> 编写目的：梳理该基线从"音频对话历史 + 目标文本 + 参考说话人"到"合成语音"的完整数据流、模型结构与关键工程实现。

---

## 0. 一句话概括

这是一个**"语境感知 + 链式思维(Chain-of-Thought) + 语音离散 Token 自回归生成"的表现力 TTS 系统**。它把对话历史音频、目标文本、参考说话人音色全部编码成离散 token 拼进一个统一的 prompt，喂给一个经过 LoRA 微调的大语言模型(LLM)；模型先"推理"出说话人的动机/情绪/表现强度等 CoT 字段，再生成一段 BiCodec 语音 token，最后由声码器还原成波形。换句话说：**让 LLM 像"读懂上下文的配音演员"一样，先想清楚该用什么情绪说，再把这句话"说"出来。**

---

## 1. 任务背景：什么是 CoT-TTS

传统 TTS 只看"目标文本"决定怎么读，缺乏语境，读出来往往平淡、情绪不对。CoT-TTS（链式思维 TTS）要解决的是 **上下文感知、富表现力的语音合成**：

- 给定一段**对话历史**（前文若干句话），
- 给定本句要合成的**目标文本**，
- 给定一个**参考说话人音频**（决定音色/音色身份），
- 模型需要**先从语境推断出合适的说话风格**（情绪、动机、目标、表现强度……），再据此合成语音。

本仓库提供两条基线，区别只在"对话历史以什么形式提供"：

| 基线 | 历史形式 | 入口脚本 |
|---|---|---|
| **audio-history**（本文重点） | 历史是**音频** `history.wav` | `infer/cot_tts_inference.py` |
| text-history | 历史是**文本** `history.txt` | `infer/cot_tts_text_history_inference.py` |

两者共享同一套 prompt 协议和 token 体系，audio-history 多了一步"把历史音频也编码成 BiCodec token"。

---

## 2. 顶层架构

整个系统由**三个模型组件**串联而成：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        audio-history COT-TTS                          │
│                                                                       │
│   历史音频 history.wav ──┐                                            │
│   参考音频 reference.wav ─┤                                           │
│                          ▼                                            │
│              ┌───────────────────────┐                               │
│              │  (A) BiCodec 音频分词器 │  ← Spark-TTS / BiCodec        │
│              │   wav → 离散 token      │                              │
│              └───────────┬───────────┘                               │
│  目标文本 target.txt      │ global / semantic tokens                  │
│         │                ▼                                            │
│         │     ┌──────────────────────────┐                           │
│         └────►│  (B) 拼装统一 Prompt 字符串 │                          │
│               └───────────┬──────────────┘                           │
│                           ▼                                          │
│              ┌──────────────────────────────┐                       │
│              │  (C) LLM (LoRA 微调) 自回归生成 │  ← VeOmni 加载         │
│              │  CoT 文本 + 语音 token 序列     │                       │
│              └───────────┬──────────────────┘                       │
│                          │ 生成的 bicodec_semantic / global token     │
│                          ▼                                           │
│              ┌───────────────────────┐                              │
│              │  (A) BiCodec 反分词器   │  token → wav                 │
│              └───────────┬───────────┘                              │
│                          ▼                                          │
│                  输出 sample_case.wav + 元数据/CoT/转写              │
└─────────────────────────────────────────────────────────────────────┘
```

三大组件：

1. **BiCodec 音频分词器（Spark-TTS / `sparktts/`）**：双码本语音编解码器，把任意 wav 编码成两类离散 token，并能把 token 还原回 wav。是连接"波形世界"和"LLM token 世界"的桥梁。
2. **文本 LLM 主干（`veomni/` 加载，LoRA 微调）**：真正的"大脑"，吃下拼好的 prompt，自回归地生成 CoT 推理文本和语音 token。
3. **推理编排脚本（`infer/cot_tts_inference.py`）**：把上面两者粘起来——做数据准备、prompt 拼装、解码、多候选重排、结果落盘。

仓库目录对应关系：

- `sparktts/` —— BiCodec / Spark-TTS 运行时（组件 A）
- `veomni/` —— VeOmni 运行时，负责构建 tokenizer、foundation model、加 LoRA（组件 B 的加载层）
- `infer/` —— 推理脚本、启动脚本、样例数据（组件 C）
- `model/` —— 模型权重（不入 git，需另行下载：Spark-TTS、audio-history 基线、text-history 基线三个包）

---

## 3. 核心概念：BiCodec 的两类 token

理解整套系统的关键，是理解 BiCodec 把语音拆成的**两类离散 token**（见 `sparktts/models/audio_tokenizer.py` 与 `bicodec.py`）：

| Token 类型 | prompt 里前缀 | 物理含义 | 数量特征 |
|---|---|---|---|
| **Global token** | `<\|bicodec_global_*\|>` | **音色 / 说话人身份**（who is speaking） | 固定长度，本仓库取前 **32** 个（`--global_prefix_len 32`） |
| **Semantic token** | `<\|bicodec_semantic_*\|>` | **内容 / 韵律 / 怎么说**（what & how） | 随语音时长变长，几十~几百个 |

直觉理解：
- **global token = 声音的"指纹"**，决定听起来像谁；
- **semantic token = 声音的"内容轨"**，决定说了什么、什么节奏情绪。

`BiCodecTokenizer.tokenize(wav)` 返回 `(global_ids, semantic_ids)`；`detokenize(global_ids, semantic_ids)` 把两者合起来还原波形。BiCodec 内部用了 Wav2Vec2-large-xlsr-53 提特征、FSQ/残差量化、说话人编码器(ECAPA-TDNN/Perceiver)等模块（见 `sparktts/modules/`）。

> 这套"global 管音色 / semantic 管内容"的解耦，正是后面**音色可替换**（用参考音色的 global + 模型生成的 semantic）的工程基础。

---

## 4. 数据准备：把三路输入编码成 token

样例 `infer/cases/sample_case/`：

```
history.wav      ← 对话历史音频（audio-history 用）
history.txt      ← 对话历史文本（text-history 用，本基线不用）
reference.wav    ← 参考说话人音频
target.txt       ← 目标文本 "It is so unfair!"
```

`infer_case()` 里三路输入各自的处理（`cot_tts_inference.py:603`）：

### 4.1 目标文本
直接读字符串：`read_text(target.txt)` → `"It is so unfair!"`。

### 4.2 历史音频 → 历史 token（`build_history_tokens`）
对每个历史音频文件，按 `--history_mode` 决定编码方式：

- `full`（默认）：编码出 **global + semantic** 两类 token 都放进 prompt；
- `semantic`：只放 semantic；
- `full_first_then_semantic`：第一个历史文件给 full，后续只给 semantic。

样例里历史音频被编码成 **32 个 global + 214 个 semantic**（见 `sample_case.meta.json` 的 `history_global_tokens / history_semantic_tokens`）。

### 4.3 参考音频 → 参考 global token（`encode_audio_global`）
只取参考音频的 **global token 前 32 个**（音色指纹），用于：
1. 写进 prompt 的 `<audio_ref_start>...<audio_ref_end>` 段，告诉模型"目标音色长这样"；
2. 在 `--use_ref_global_tokens` 开启时，**最终解码用的就是参考音色的 global**，保证输出音色稳定贴合参考人。

一个工程细节：参考音频编码前会先 `maybe_trim_ref_audio()` **去掉首尾静音**（`remove_silence`，阈值随峰值自适应），避免静音污染音色提取。

---

## 5. 关键：统一 Prompt 协议

所有信息最终拼成一个**带特殊标记的长字符串**（`build_prompt`，`cot_tts_inference.py:383`）：

```text
<bos>
<task_start>COT-TTS<task_end>
<history_start>
  <spk_his_start><spk_his_end>
  <audio_his_start>{历史 global+semantic token}<audio_his_end>
<history_end>
<target_start>
  <text_start>{目标文本}<text_end>
  <spk_tar_start><spk_tar_end>
  <audio_ref_start>{参考 global token}<audio_ref_end>
<target_end>
<output_start>
<cot_start>          ← prompt 到此结束，模型从这里开始"接着写"
```

要点：
- 这是一套**结构化、可解析的标签协议**。历史段、目标段、输出段被清晰分区。
- prompt 以 `<cot_start>` 结尾，意味着**模型被引导先生成 CoT（链式思维）**，再生成语音 token。
- 对比 **text-history 版本**：历史段用 `<under_start>{历史文本}<under_end>` 而不是音频 token——这是两条基线唯一的结构性差异（`cot_tts_text_history_inference.py:323`）。

拼好的字符串再经 `tokenizer.apply_chat_template(...)` 套上 chat 模板，转成 `input_ids` 喂给 LLM（`encode_chat_prompt`）。

---

## 6. 模型加载：VeOmni + LoRA

`load_text_model_and_tokenizer()`（`cot_tts_inference.py:784`）负责把 LLM 准备好：

1. **架构自动判定**：`hf_checkpoint_has_lora()` 扫描 `model.safetensors`，若发现 `.lora_A. / .lora_B. / .base_layer.` 键，则判定为 LoRA 检查点（也可用 `--model_architecture lora/full/auto` 强制）。
2. **LoRA 路径**（本基线默认走这条，见启动脚本 `--model_architecture lora`）：
   - `veomni.models.build_foundation_model` 先按 `hf_ckpt/config.json` 建出**基座模型骨架**（不加载权重，`weights_path=None`，CPU 初始化）；
   - `veomni.utils.lora_utils.add_lora_to_model` 按 `veomni_cli.yaml` 里的配置（`lora_rank` 默认 8、`lora_alpha` 16、target_modules 为 `q/k/v/o/gate/up/down_proj`）**注入 LoRA 适配层**；
   - 再用 `safetensors.load_file` 把检查点权重 `load_state_dict(strict=False)` 灌进去，并打印 missing/unexpected key 统计做健全性检查（容忍 `rotary_emb.inv_freq` 等非持久缓冲）。
3. **DCP→HF 自动转换**：若检查点还是分布式格式(DCP)、缺少 `hf_ckpt/config.json`，`ensure_hf_model_dir()` 会调用 `veomni/scripts/merge_dcp_to_hf.py` **自动把训练态 DCP 检查点合并成 HuggingFace 格式**再加载（`--auto_convert_dcp` 默认开启）。

> 工程亮点：这套加载层对"训练产出的 DCP 检查点 / 合并后的 HF 检查点 / 全量 vs LoRA"三种情况都做了**自动探测与兜底**，使用者一条命令即可，不用手动转换。

辅助的 `install_sklearn_stub()` 还会塞一个轻量 sklearn 桩，避免仅为一个未用到的依赖而强装 sklearn——是个减少环境摩擦的小技巧。

---

## 7. 生成与解析：从 token 到波形

### 7.1 自回归生成（`infer_case` 主循环）
用 `model.generate(...)` 自回归采样，关键超参（`generation_kwargs`）：

- 采样：`do_sample=True, temperature=0.6, top_p=0.8`（启动脚本里设定）；
- 长度：`max_new_tokens=2000, min_new_tokens=32`；
- **停止条件**：在 eos 之外，额外把 `<audio_tar_end> / <output_end> / <eos>` 注册为停止 token（`resolve_stop_token_ids`）——即**生成完语音段就停**。

模型实际生成的完整序列（见 `sample_case.raw.txt`）长这样：

```text
<under_start>Just listen. I am the queen. You listen to me.[自信坚定]<under_end>
<cot_start>
  <Act>: 直接表达不满
  <Scene>: 感到被忽视或不被尊重
  <Motivation>: 因被忽视而感到不满
  <Goal>: 希望引起注意或改变现状
  <Emotion>: 情绪从平静转向不满
  <Valid duration | Total duration>: 1.450000s | 1.586313s
  <Loudness | Expressive Intensity>: -31.365203dBFS | 0.831543
  [Summary] 因为感到被忽视和不满，所以带着失望又生气的情绪说了'It is so unfair.'
<cot_end>
<audio_tar_start>
  <|bicodec_global_3719|> ... (32 个 global)
  <|bicodec_semantic_4596|> ... (76 个 semantic)
<audio_tar_end>
```

这段输出非常说明问题，它体现了 CoT-TTS 的核心思想——模型在"说话"之前，**先把历史音频转写出来（`<under>` 段，含对历史音色风格的标注 `[自信坚定]`），再做一段结构化的表演推理（CoT 7 个字段），最后才生成目标语音的 token**。

CoT 的字段语义：
- `Act/Scene/Motivation/Goal/Emotion`：表演分析（在做什么动作、什么场景、动机、目标、情绪走向）；
- `Valid/Total duration`：预测的语音时长；
- `Loudness/Expressive Intensity`：响度与"表现强度"——量化这句话要说得多用力；
- `[Summary]`：把"因为…所以用…情绪说了…"串成一句总结，作为合成的最终指导。

### 7.2 解析语音 token（`extract_audio_ids`）
从生成文本里抠出语音 token，做了**三级兜底**：
1. 先取 `<audio_tar_start>...<audio_tar_end>` 块里的 token（最规范，样例命中此路 `parse_source=audio_tar_block`）；
2. 不行就在整段生成文本里正则搜 `bicodec_global_/semantic_`；
3. 再不行就回退到原始 token id 序列解析。
若一个 semantic token 都没解析到，就把这次生成判为失败，落盘 `*.failed.*` 便于排查。

### 7.3 选 global（音色来源）
- 若开 `--use_ref_global_tokens`（启动脚本默认开）或模型没生成 global → 用**参考音频的 32 个 global**，`global_source=ref`；
- 否则用模型自己生成的 global，`global_source=generated`。
样例 meta 显示 `global_source=ref`，即**输出音色严格来自 reference.wav**，模型生成的 semantic 只负责"内容与情绪韵律"。

### 7.4 反分词成波形（`detokenize`）
```python
wav = audio_tokenizer.detokenize(global_ids, semantic_ids)  # → 16kHz 波形
```
global（音色）+ semantic（内容韵律）一起送进 BiCodec 解码器还原 wav，写出 `sample_case.wav`。

---

## 8. 多候选与重排（可选）

`--num_candidates N` 可一次生成 N 条候选；`--rerank_metric` 决定如何选：
- `none`：直接取第 1 条（默认快速路径，样例即此）；
- `mel_similarity`：对每条候选算 log-mel 统计特征（均值+方差，80 维 mel），与参考音频做**余弦相似度**，选最像参考音色的一条（`mel_similarity / log_mel_stats`）。

这是一个**轻量、无需额外大模型的重排兜底**，在采样有随机性时提升音色一致性。

---

## 9. 输出产物

每个 case 在 `infer/results/` 落盘一组文件（见 `write_candidate_files` + `infer_case` 收尾）：

| 文件 | 内容 |
|---|---|
| `sample_case.wav` | **最终合成语音**（16kHz） |
| `sample_case.pt` | 选中候选的 global/semantic token 张量 |
| `sample_case.raw.txt` | 模型完整原始生成文本（含 CoT + 语音 token） |
| `sample_case.cot.txt` / `.cot_thinking.txt` | 抠出的 CoT 推理段 |
| `sample_case.history_transcript.txt` / `.under.txt` | 模型对历史音频的转写（`<under>` 段） |
| `sample_case.target_text.txt` | 目标文本 |
| `sample_case.meta.json` | 全流程元数据（各类 token 数、音色来源、候选信息、解析路径等） |

`meta.json` 设计得相当完善，把"用了多少历史 token、参考 global 多少、最终选了哪条候选、音色来自 ref 还是 generated、解析走了哪条兜底路径"全记录下来，**可复现、可审计**。

---

## 10. 端到端流程总览（按调用顺序）

```
main()
 ├─ parse_args / set_seed / resolve_device
 ├─ ensure_hf_model_dir         # 必要时 DCP→HF 自动转换
 ├─ resolve_tokenizer_dir
 ├─ load_text_model_and_tokenizer  # 建骨架→注入LoRA→灌权重 (VeOmni)
 ├─ BiCodecTokenizer(spark_model_dir)  # 加载音频编解码器
 └─ for case in case_names:
      infer_case()
       ├─ resolve history/ref/text 路径
       ├─ read_text(target)                       # 目标文本
       ├─ build_history_tokens(history.wav)       # 历史音频→global+semantic token
       ├─ encode_audio_global(reference.wav)      # 参考音色→32 global token
       ├─ build_prompt(...)                       # 拼统一 prompt
       ├─ encode_chat_prompt → input_ids
       ├─ for cand in 1..N:
       │    ├─ model.generate(...)                # 自回归生成 CoT+语音token
       │    ├─ extract_audio_ids(...)             # 三级兜底解析 token
       │    ├─ 选 global（ref vs generated）
       │    └─ audio_tokenizer.detokenize(...)    # token→wav
       ├─ select_candidate(rerank_metric)         # 多候选重排
       └─ 写 wav / pt / cot / meta.json ...
```

---

## 11. 设计亮点小结（值得在简历/面试里讲的点）

1. **统一离散 token 表征跨模态**：文本、音色、内容韵律全部映射到同一 LLM 词表的离散 token，用一个自回归模型统一建模——这是"LLM-as-TTS-backbone"的现代范式。
2. **音色/内容解耦**：BiCodec 的 global(音色) 与 semantic(内容) 分离，使"用参考音色 + 模型生成内容"成为可能，输出音色可控、可替换。
3. **链式思维(CoT)引导表现力**：模型先输出 Act/Emotion/Loudness/Expressive Intensity 等结构化推理，再合成语音——把"为什么这么读"显式化，提升语境契合度与可解释性。
4. **语境感知**：把对话历史（音频或文本）纳入条件，模型据上下文推断说话风格，而非只看孤立的目标文本。
5. **稳健工程**：DCP↔HF 自动转换、LoRA/全量自动探测、三级 token 解析兜底、参考音去静音、多候选 mel 重排、完善的可审计 meta —— 是一套**面向真实落地**的推理管线，而不只是 demo 脚本。
6. **LoRA 高效微调**：在大基座上用 rank-8 LoRA 适配 TTS 任务，训练/存储成本低，便于多检查点切换。

---

## 12. 与 text-history 基线的对比

| 维度 | audio-history（本文） | text-history |
|---|---|---|
| 历史输入 | `history.wav`（音频） | `history.txt`（文本） |
| 历史在 prompt 中 | `<audio_his_start>{global+semantic token}` | `<under_start>{文本}<under_end>` |
| 多一步处理 | 历史音频要 BiCodec 编码 | 无需，直接放文本 |
| 适用场景 | 上游只有录音、想保留历史说话人的语气/情绪信息 | 已有可靠转写，或想要更轻量的语境 |
| 入口/checkpoint | `cot_tts_inference.py` / `global_step_37000` | `cot_tts_text_history_inference.py` / `global_step_24500` |

audio-history 的优势在于历史音频里**保留了原始的情绪、语气、韵律信息**（文本会丢掉这些），理论上能让模型对语境的"表演判断"更准。

---

## 附：数据集与规模（来自 README）

训练/评测数据来自 HKUSTAudio/chall（ISCSLP 2026 CoT-TTS Challenge 数据集），取材自电影、电视剧、广播剧、短剧等富对话语境的语音：

| 语言 | 时长 | 占比 | 片段数 |
|---|---:|---:|---:|
| 英语 | ~8.6K 小时 | 54% | ~1.62M |
| 中文 | ~7.4K 小时 | 46% | ~1.38M |
| 合计 | ~16K 小时 | 100% | ~3.0M |

数据刻意**不做激进降噪/标准化**（仅统一为 FLAC），保留真实声学条件；但元数据/标注是基于降噪归一化版本生成的，以保证标注可靠性。
