# Search-R1 Beacon 使用说明

第一阶段已实现完整轨迹冻结 teacher + Beacon 输出 KL（默认全有效 loss token），两种模型的全参数 plain 基线与本地模型接续训练见 [第一阶段说明](stage1_distillation.md)。Qwen3.5 默认模型为 `Qwen/Qwen3.5-4B`。

第二阶段文档依赖加权已实现，默认关闭：`beacon.beacon_output_distill_doc_weighting=true` 开启，计算公式、逐轮文档删除规则和额外开销见 [第二阶段说明](stage2_distillation.md)。

## 数据范围

训练仅支持 `data_mode: searchr1`，数据为 Search-R1 `messages` JSONL。Beacon Trainer、原生 Trainer与 plain SFT Trainer 都会校验数据模式；旧 `interactive` 和 `single` 格式不再接受。Beacon SFT 统一使用标准 `search_comp.trainer.beacon_trainer` 入口。

已移除旧交互轨迹 Dataset、轨迹构建器、手写轨迹序列拼接和对应 YAML，以及旧单轮 RAG 训练数据管线。在线多轮检索推理继续保留：推理模块和脚本名称中的 `interactive` 表示 SearchAgent 在线搜索行为，不是已删除的训练数据模式。

默认数据路径：

```text
outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl
```

文件不包含在本目录中，可通过 `--set train_data_path=/path/to/file.jsonl` 使用已有文件。Search-R1 轨迹保留完整路径，不截断；模型标签只监督 assistant 内容，information 续写监督由模型独立控制。

## 默认 Qwen3 配置

默认基础模型为 `Qwen/Qwen3-4B-Thinking-2507`。`configs/train/beacon_qwen3_searchr1.yaml` 明确配置：

```yaml
data_mode: searchr1
model_name_or_path: Qwen/Qwen3-4B-Thinking-2507
beacon:
  enable_beacon: true
  beacon_question_memory_v1: false
  beacon_readout_distill_weight: 0.0
  beacon_output_distill_weight: 0.1
  beacon_output_distill_strategy: all
  beacon_doc_region_split: false
  beacon_information_mode: whole
  beacon_window_enabled: false
  beacon_keep_window: 1024
  beacon_ratio: 16
  beacon_pos: append
  beacon_information_loss: false
  beacon_continuation_loss_weight: 0.1
```

| 参数 | 当前行为 |
| --- | --- |
| `beacon_question_memory_v1: false` | 暂停 Qwen3.5 线性问题记忆功能；Qwen3 不创建线性状态 |
| `beacon_readout_distill_weight: 0.0` | 暂停线性 readout 蒸馏；Qwen3 没有 Writer/Reader |
| `beacon_doc_region_split: false` | 每轮整个 information 为一个压缩区，不区分子文档 |
| `beacon_information_mode: whole` | 整个正文插入 beacon 后只做一次压缩前向 |
| `beacon_window_enabled: false` | 暂停 `beacon_window` / `beacon_stride`，运行时不读取这两个旧窗口参数 |
| `beacon_keep_window: 1024` | 单独控制普通提示词 / 生成内容的前向长度，不影响 information 整段处理 |

Qwen3 YAML 不再列出 `beacon_window` / `beacon_stride`。暂停窗口时，即使旧 checkpoint 中带有这些字段，也不使用它们；若 `beacon_keep_window: null`，keep 段整段前向，`0` 同样整段前向，正整数为 keep 段上限。暂停窗口必须与 whole 配合。

直接加载不含 Beacon 字段的原生 Qwen3 checkpoint，也采用上述整段处理和暂停策略。已有 Beacon checkpoint 的显式配置继续保留，避免静默改变已训练模型的行为。

## information 布局与缓存隔离

每轮正文有 `N` 个原文 token 时，插入 `ceil(N/ratio)` 个 beacon。正文与所有 beacon 同一次前向；不同搜索轮次分别处理。

```text
原文 a b c d e，ratio=2：
append:    a b c d e B B B
intersect: a b B c d B e B
```

`append` 在整段末尾追加全部 beacon；`intersect` 每 ratio 个原文 token 插入一个，最后不足 ratio 的尾部也有一个。交错间隔只决定插入位置，不是子文档或独立 forward 的 chunk 边界。

当前整段 forward 中使用因果注意力。information 处理完成后，每个全注意力层从当前增量 KV 中只选取 beacon 行，并丢弃原文行。后续 token 可以读取 keep 区的历史 KV 和 beacon KV，无法直接读取原始 information KV。信息通过 beacon 表征传递给后续生成。

Qwen3 没有线性注意力层，不创建循环缓存、卷积缓存或 Beacon Writer。Qwen3.5 保留独立混合注意力建模与线性状态隔离逻辑。

## information 续写损失

`beacon_information_loss: false` 暂停 information 续写损失；设为 true 时必须提供正的 `beacon_continuation_loss_weight`。null / 缺省仍兼容旧的权重开关。

whole 的续写项使用 information 内的因果 next-token CE：每个原文 token 的隐状态预测下一个原文 token，共 `N-1` 个目标。beacon 和 information 外的 token 不作为该项的目标；两种插入布局都支持。它不需要额外的模型 forward。

append 布局的 beacon 全在正文之后，因此原文内部 CE 不直接训练这些 beacon；提交后的 assistant 损失和可选输出 KL 蒸馏负责向 beacon 压缩参数传递梯度。分段损失继续控制 assistant 的 thinking/search/answer 监督。

主 CE 与续写 CE 两项实际参与且权重均大于 0 时按份额归一；只有一项参与时直接乘其权重。线性 readout / 输出 KL 是独立加项：

```text
L_main = (w_ce * CE + w_cont * cont) / (w_ce + w_cont)  # 两项参与
L_main = w_i * CE_i                                  # 仅一项参与
L_main = 0                                          # 两项均关闭或缺席
L = L_main + w_readout * readout + w_output * output_KL
```

例如 `beacon_ce_loss_weight: 0.1`、`beacon_continuation_loss_weight: 0.0` 时，总损失为 `0.1 * CE + w_readout * readout + w_output * output_KL`。

Qwen3 和 Qwen3.5 均支持 `beacon_ce_loss_weight: 0.0`，此时跳过主 CE 计算，日志中的 `ce_loss` 为 0。只训练输出 KD 时，在训练配置的 `beacon` 下设置：

```yaml
beacon_ce_loss_weight: 0.0
beacon_continuation_loss_weight: 0.0
beacon_information_loss: false
beacon_readout_distill_weight: 0.0
beacon_output_distill_weight: 1.0
beacon_output_distill_strategy: all
```

此时总损失为 `beacon_output_distill_weight * output_KL`，CE 与续写均关闭时不进行份额归一。仍需提供有效 `labels` 来选择 KD 目标位置，不要为了关闭 CE 将它们全部设为 `-100`；没有有效 KD 目标的样本返回可反向的零损失。

## 环境与配置

依赖见 `requirements.txt`，参考环境为 Python 3.11、Torch 2.7.0+cu126、Transformers 5.9.0、Accelerate 1.14.0。可通过 `PYTHON=/path/to/python` 或 `CONDA_ENV=your-env` 指定环境。

| 配置 | 用途 |
| --- | --- |
| `beacon_qwen3_searchr1.yaml` | 默认 Qwen3 整段模式，窗口暂停，LoRA |
| `beacon_qwen35_whole_searchr1.yaml` | Qwen3.5 整段 information 模式 |
| `beacon_qwen35_searchr1.yaml` | Qwen3.5 整段模式，取消窗口和子文档切分 |
| `qwen3_plain_full_sft.yaml` / `qwen35_plain_full_sft.yaml` | 全参数无压缩 Search-R1 基线 |
| `qwen3_plain_sft.yaml` / `qwen35_plain_sft.yaml` | 两种模型的无压缩 Search-R1 基线 |
| `native_qwen35.yaml` | 原生 Trainer 的 Search-R1 配置 |

Qwen3.5 的窗口、子文档、问题记忆实现继续保留；本次暂停策略主要针对 Qwen3。需要恢复旧窗口实验时，显式设置 `beacon_window_enabled=true` 和 `beacon_information_mode=window`，并提供合法窗口参数。

## 训练入口与参数

以下命令供后续使用，本次修改没有启动训练。

```bash
# 默认 Qwen3、Search-R1、append、无 information 续写。
bash scripts/24_beacon_train_searchr1.sh

# 22 保留为兼容入口，同样只运行 Search-R1，不再构建旧轨迹。
bash scripts/22_beacon_train.sh

# 交错插入并启用整段 information 续写 CE。
BEACON_POS=intersect INFORMATION_LOSS=true CONTINUATION_WEIGHT=0.1 \
  bash scripts/24_beacon_train_searchr1.sh \
  --set train_data_path=/path/to/qwen3-4b-instruct-sft.jsonl

# Qwen3.5 整段模式。
bash scripts/24_beacon_train_searchr1.sh \
  configs/train/beacon_qwen35_whole_searchr1.yaml

# 无压缩 Qwen3 基线。
bash scripts/20_plain_train.sh
```

24 支持 `DISTILL_TEACHER_MODEL_NAME_OR_PATH`、`OUTPUT_DISTILL_WEIGHT`、`OUTPUT_DISTILL_STRATEGY`、`MODEL_NAME_OR_PATH`、`INFORMATION_MODE`、`BEACON_POS`、`INFORMATION_LOSS`、`CONTINUATION_WEIGHT`、`DOC_REGION_SPLIT`、`BEACON_RATIO`、`WINDOW_ENABLED`、`BEACON_WINDOW` 和 `BEACON_STRIDE`。未设置时沿用 YAML；额外 `--set` 的优先级最高。22 将全部参数透传给 24。

数据仅支持 batch size 1，使用 `grad_accum_steps` 扩大有效 batch。

## 合并、推理与工具

完整 Beacon checkpoint 保存布局和暂停参数；加载时由模型类型自动分派到独立的 Qwen3 / Qwen3.5 modeling 文件。LoRA、独立 `beacon_state.pt`、自动合并及 `resolved_config.json` 继续保留。

```bash
bash scripts/30_merge_adapter.sh outputs/models/beacon_qwen3_searchr1
bash scripts/23_beacon_eval.sh outputs/models/beacon_qwen3_searchr1/final
bash scripts/21_plain_eval.sh /path/to/plain-checkpoint
```

原生探针、直接回答基线、在线检索、Search-R1 语料构建、文档规范化、统计和 Excel 导出继续保留。文本截断辅助函数已迁移到 `search_comp.data.trajectory`，不再依赖已删除的旧数据构建器。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  python -m pytest tests -q
python -m compileall -q search_comp tests
for script in scripts/*.sh; do bash -n "$script"; done
```

CPU 测试使用小模型；不启动训练，不下载真实模型权重。具体实现和检查范围见 [implementation.md](implementation.md)。
