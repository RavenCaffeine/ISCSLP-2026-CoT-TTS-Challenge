# ISCSLP 2026 CoT-TTS 数据集

地址：https://huggingface.co/datasets/HKUSTAudio/ISCSLP2026-CoT-TTS

## 数据集概述

该数据集为 ISCSLP 2026 CoT-TTS 挑战赛准备，旨在支持上下文感知、表达性和 CoT 引导语音生成的研究。它由富含语音的媒体来源构成，包括电影、电视剧、广播剧和短剧，其中对话通常包含丰富的对话语境、说话者互动、场景转换和情感变化。每个样本都围绕目标话语组织，连同其前的对话语境、引用语、元数据和相应的注释，使模型能够从语境中推断出合适的说话风格，而不仅仅是依赖目标文本。

| Language | Duration   | Ratio | Segments |
| :------: | ---------- | ----- | -------- |
|   英语   | ~8.6K 小时 | 54%   | ~1.62 万 |
|   中文   | ~7.4K 小时 | 46%   | ~1.38 米 |
|   总计   | ~16K 小时  | 100%  | ~3.00    |

发布的音频文件是从原始源录音中提取的片段。为了减少存储使用，只有音频文件格式被标准化为 FLAC，其他声学特性则尽可能保留。音频并未被严格标准化或去噪。因此，不同文件可能有不同的采样率、通道配置、响度等级、背景声音或环境噪声。这一设计是有意为之：数据集旨在保持真实的声学条件，避免过度的预处理假设，让用户和参与者根据自己的方法决定如何处理、归一化、增强或过滤音频。然而，该数据集中提供的元数据和注释是基于相应音频片段的归一化和去噪版本生成的，以提高注释的可靠性和准确性。

## 文件结构

```text
HKUSTAudio/ISCSLP2026-CoT-TTS/
├── movie-en1/
│   ├── metadata.json
│   ├── continuous_segments/
│   │   └── *.flac
│   └── dialogue_segments/
│       └── *.flac
└── movie-zh2/
    ├── metadata.json
    ├── continuous_segments/
    │   └── *.flac
    └── dialogue_segments/
        └── *.flac
```



数据集分为六个文件夹。这些文件夹遵循相同的内部结构，主要区别在于语言和数据分区。分拆用于更便捷的上传、存储和处理。

每个文件夹包含一个元数据 JSON 文件和两个音频目录：`dialogue_segments/` 和 `continuous_segments/`。

- `dialogue_segments/` 包含元数据要求的所有话语级音频片段，包括目标语音、参考语音和对话上下文语音。这些片段是根据说话者日记后获得的时间戳提取的。
- `continuous_segments/` 还包含所有必要的音频片段，包括目标语音、参考语音和对话语境语音。与 `dialogue_segments/` 相比，这些片段在每次发言前后都被延长，以更好地保持原始音频语境的连续性。

例如，如果原始源覆盖 0-20 秒，`dialogue_segments/` 可能只包含检测到的语音间隔，如 0-3 秒、6-7 秒和 13-15 秒。相比之下，`continuous_segments/` 可能包含延长且连续的间隔，如 0-4.5 秒、4.5-9.5 秒和 9.5-20 秒。这种设计允许用户在更清晰的话语级片段和更保留上下文的连续音频片段之间进行选择。

## 数据格式

```json
{
  "movie_id": "ID of the source media item",
  "movie_name": "Name of the source media file",
  "scene_index": "Index of the scene",
  "start_time": "Start time of the dialogue context",
  "end_time": "End time of the dialogue context",
  "dialog_count": "Number of historical dialogue segments",
  "target_segment": {
    "segment_id": "Unique ID of the target utterance",
    "ref_segment_id": "Segment ID of the reference speech",
    "text": "Text content of the target utterance",
    "speaker": "Original speaker label",
    "normalized_speaker": "Normalized speaker label",
    "emotion_tag": "Emotional description of audio clips",

    "features": {
      "duration": "Total duration of the target audio segment",
      "active_duration": "Valid duration of audio clip (excluding silence)",
      "loudness": "Loudness of the target audio segment",
      "expressive_intensity": "Expressive intensity score of the target utterance"
    },
    "cot_text": "Chain-of-thought style analysis",
    "start": "Start time of the target utterance",
    "end": "End time of the target utterance",
    "best_ref_similarity": "Similarity between the target utterance and the reference segment"
  },
  "dialog_segments": [
    {
      "segment_id": "Unique ID of a historical dialogue segment",
      "text": "Text content of the historical utterance",
      "speaker": "Original speaker label",
      "normalized_speaker": "Normalized speaker label",
      "emotion_tag": "Emotional description of audio clips",
      "features": {
      "duration": "Total duration of the target audio segment",
      "active_duration": "Valid duration of audio clip (excluding silence)",
      "loudness": "Loudness of the target audio segment",
      "expressive_intensity": "Expressive intensity score of the target utterance"
     },

      "start": "Start time of the historical utterance",
      "end": "End time of the historical utterance"
    },
      ...
  ],
  "index": "Unique sample index used for data loading and file matching"
}
```



注释：

- 所有时间数值均以秒为单位。
- `说话者 `：原始日记标签在完整源集内分配。
- `normalized_speaker`：在当前场景或对话语境中重新索引说话者标签。
- 音频文件可以通过 `segment_id`、`ref_segment_id` 进行匹配。
- `dialog_segments` 按原始媒体中的时间顺序排列。

## 如何使用

每个样本都由 JSON 元数据条目描述。用户可以阅读 JSON 文件，获取目标话语、历史对话上下文、说话者标签、情绪标签、CoT 风格分析、声学特征和时机信息。

对应的音频文件可以通过元数据中提供的段 ID 找到。具体来说，`segment_id` 识别每个话语级音频片段，而 `ref_segment_id` 用于表示说话音色的参考语音片段。

一个典型的使用过程是：

1. 加载 JSON 元数据。
2. `dialog_segments` 可以理解为历史对话的背景。
3. 把 `target_segment.text` 当作目标文本。
4. 使用 `target_segment.ref_segment_id` 来定位参考音频。
5. 使用 `target_segment.segment_id` 定位目标音频。
6. 如有需要，请使用 `target_segment.cot_text` 作为推理风格的注释。

## 超出范围的使用

该数据集仅供与上下文感知语音生成和 CoT 引导 TTS 相关的非商业学术研究、开发和评估发布。以下用途不被允许：

- 数据集的商业用途，包括商业模型训练、商业语音生成服务或商业语音克隆产品。
- 数据集或其任何部分的再分发、重新托管、转许可、销售或重新打包。
- 重建、修复或再分发原始源媒体，包括原创电影、戏剧、音频节目或其他版权材料。
- 尝试识别、追踪或披露已发布片段中的原始演讲者、演员、角色、来源标题或版权持有人。
- 利用该数据集进行冒充、身份伪造、欺骗性语音生成、虚假信息、欺诈、骚扰或其他有害应用。
- 该数据集可用于监控、发言人识别、生物特征画像或侵犯隐私的分析系统。
- 任何违反适用法律、版权法规、平台政策或数据集维护者提供的数据使用条款的使用。

## 局限性

该数据集由丰富的语音媒体来源构建，因此存在若干用户应注意的限制：

- 数据集可能反映媒体内容的风格、分布和戏剧性特征，可能无法完全反映自然的日常对话。
- 发布的音频片段保留了原始录音的许多声学特性，因此可能存在背景声音、音乐、环境噪音、重叠语音、声道差异、响度变化和采样率差异。
- 说话者标签是自动生成的，应被视为本地匿名标签，而非真实的发言者身份。
- 转录、说话者日记结果、时间戳、情绪标签和 CoT 风格的注释可能包含错误。
- 情感标签和推理注释仅作为参考注释，不应被视为绝对的真实事实。
- 部分样本可能包含文化特有的表达、戏剧性对话或依赖语境的说话风格。
- 该数据集不赋予用户对原始版权媒体的任何权利，仅限于有限的研究使用。
- 用户有责任确保其使用数据集符合适用法律、机构政策及数据集使用条款。

## 许可与版权

该数据集仅供非商业性学术研究和评估目的准备。资料来源于公开可访问的媒体。数据集维护者不声称拥有原始媒体内容的所有权。原始视频、电影、剧集、音频节目或其他源材料的版权仍归其各自的版权持有人所有。

发布的数据集不包含完整的源媒体、完整视频、全长音频程序或原始媒体文件。取而代之的是，发布的数据由经过处理的语音片段及其相关的元数据或注释组成，这些都通过数据准备管道生成，包括分割、剪裁、规范化、过滤、转录、说话者标注、情感注释以及 CoT 风格的注释。

用户仅可将数据集用于与预期研究范围相关的非商业性研究、开发和评估。严禁商业使用、再分发、转售、转许可、公众重新托管，或任何试图重建、识别或再分发原始资料的行为。

该数据集以非商业研究使用许可证发布，遵循知识共享署名-非商业4.0许可的精神。用户必须遵守数据集维护者提供的数据使用条款。维护者保留在涉及版权、隐私、许可或伦理问题时删除、修改或限制访问任何数据项的权利。

访问该数据集不转移版权所有权、邻近权利、表演者权利、公开权或与原始源材料相关的任何其他权利。用户需完全负责确保其在所在司法管辖区内使用数据集合法且适当。

## 伦理考量

用户应负责任地处理该数据集，尊重原创作者、表演者、演讲者和版权持有者的权利。

- 请勿使用该数据集模仿、冒充或误导任何真实人物、演员、说话者或角色。
- 请勿利用该数据集生成欺骗性、有害性、诽谤性或误导性的音频内容。
- 请勿尝试识别发言者、恢复来源身份，或将已发布片段与特定个人或版权作品关联。
- 请勿利用该数据集创建支持未经授权的语音克隆、生物识别、监控或侵犯隐私的系统。
- 在展示基于该数据集训练模型产生的输出时，应明确披露使用生成或合成语音。
- 尊重与数据集及其源材料相关的所有版权、隐私和许可问题。
- 如果任何数据项目涉及版权、隐私或伦理问题，请联系数据集维护者进行审核或删除。

该数据集旨在支持上下文感知和推理引导语音生成的研究。该权利应以合法、合乎道德、非商业性且尊重原始权利人的方式使用。