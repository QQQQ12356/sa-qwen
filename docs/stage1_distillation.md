# 第一阶段：完整轨迹 Teacher + Beacon 输出 KL

本次实现对应 `sa_v2.md` 第 1–4 节。默认按实际训练要求覆盖所有参与 assistant CE 的 token；文档的前缀/分层采样作为可选消融。没有启动正式训练，也没有效果评测结论。

## 损失与对齐

冻结的原生 teacher 接收原始完整轨迹（包括所有历史 information），每个 micro-batch 一次因果 teacher-forcing forward，`no_grad`、`eval`、`use_cache=false`。`logits_to_keep` 只减少 LM head 读出行数，不删减 teacher 上下文，也不增加 forward 次数。Student 仍沿原有 whole information + append + ratio=16 路径运行，保留 Beacon KV 和 Qwen3.5 的持久线性状态。

```text
L = L_original + lambda * T² * mean_token KL(p_teacher(T) || p_student(T))
```

默认 `lambda=0.1`、`T=1.0`，仅作为可配置初值。主 CE/continuation 的原归一化保持不变，KL 独立相加；主配置继续关闭 information loss、question-memory 和 readout 蒸馏。

目标 token j 对齐两边的预测行 j-1。`all` 包括首次检索前和所有检索后的有效 assistant token，以及标签；`labels=-100`、被分段配置关闭或权重为零的类别不参与。information 正文不是 KL 读出位置；Search-R1 的 information 前后标签也保持原来的 mask。Student 的物理缓存长度变化不改变原轨迹坐标，按每段 keep 的原坐标读出；运行时检查两边读出 token 数相等。教师词表的 token ID 映射也必须与 student 一致。

Teacher 不注册到 student，不进入优化器或 checkpoint；推理无需 teacher。评估仍只返回原有 SFT loss，`kd_loss=0`。训练日志中的 `kd_loss` 已乘 lambda，与其他分量相加等于总 loss。

## 两种架构和全参数 plain 基线

| 模型 | 默认模型 | 全参数无压缩配置 | Beacon 配置 |
| --- | --- | --- | --- |
| Qwen3 | `Qwen/Qwen3-4B-Thinking-2507` | `configs/train/qwen3_plain_full_sft.yaml` | `configs/train/beacon_qwen3_searchr1.yaml` |
| Qwen3.5 | `Qwen/Qwen3.5-4B` | `configs/train/qwen35_plain_full_sft.yaml` | `configs/train/beacon_qwen35_searchr1.yaml` |

新增 full 配置设置 `use_lora=false`、`optim=adamw_torch`、开启 gradient checkpointing。原来的 `*_plain_sft.yaml` 保留 LoRA 选项。轨迹均不截断，`max_length` 为兼容字段。全参数 plain 的输出是完整 `final/` 模型；LoRA plain 则使用合并后的 `final_merged/`。不要将仅有 adapter 的目录当作完整模型传入。

以下是后续执行命令，本次未执行：

```bash
# 先运行 Qwen3 的全参数 plain 基线。
bash scripts/20_plain_train.sh configs/train/qwen3_plain_full_sft.yaml

# 从训练后的完整 plain 模型初始化 Beacon；teacher 默认使用同一个路径。
MODEL_NAME_OR_PATH=/abs/path/to/outputs/models/qwen3_plain_full_sft/final \
  bash scripts/24_beacon_train_searchr1.sh configs/train/beacon_qwen3_searchr1.yaml

# Qwen3.5 同样流程。
bash scripts/20_plain_train.sh configs/train/qwen35_plain_full_sft.yaml
MODEL_NAME_OR_PATH=/abs/path/to/outputs/models/qwen35_plain_full_sft/final \
  bash scripts/24_beacon_train_searchr1.sh configs/train/beacon_qwen35_searchr1.yaml

# 可单独指定冻结 teacher，或用 CLI 覆盖；两种架构都支持。
bash scripts/24_beacon_train_searchr1.sh configs/train/beacon_qwen3_searchr1.yaml \
  --set model_name_or_path=/abs/path/to/plain/final \
  --set distill_teacher_model_name_or_path=/abs/path/to/teacher/final
```

`distill_teacher_model_name_or_path: null` 自动跟随最终生效的 `model_name_or_path`，解析后的路径写入 `resolved_config.json`。Shell 也支持 `DISTILL_TEACHER_MODEL_NAME_OR_PATH`、`OUTPUT_DISTILL_WEIGHT`、`OUTPUT_DISTILL_STRATEGY`；CLI `--set` 优先级更高。原生模型从其 `config.json` 分派到对应架构，本地路径无需包含 Qwen 名字。

为了比较蒸馏本身的收益，从同一 plain checkpoint 分别运行 `beacon_output_distill_weight=0.0` 与 `0.1` 的 Beacon 训练，固定数据、顺序、步数、ratio、布局、可训练参数范围等。Beacon 默认仍为 LoRA + Beacon 参数；若实验要求第二段也做全参数训练，可设置 `--set use_lora=false --set optim=adamw_torch`，并在所有 Beacon 组保持一致。Plain 与 Beacon 配置的默认累积步数及停止方式不同，对照时需显式统一实验预算。

## 位置策略消融

- `beacon_output_distill_strategy: all`：本次默认，全有效 assistant token，忽略采样预算。
- `prefix`：旧行为，每个压缩后 keep 段前 k 个有效预测位置，不含首次检索前输出。
- `stratified`：每个压缩后 keep 段最多 k 个位置，轮流覆盖 thinking 开头、均匀 thinking、search 正文、answer 正文；缺失类别的预算重分配，正文不足时用标签/other 补齐。结果去重并按原坐标排序。

后两种均通过 `beacon_output_distill_tokens` 设置每段预算，默认 8；同预算做比较。选择 stratified 时 Dataset 即使 CE 全部等权也显式提供类别。分段权重仅决定 KL 位置是否参与，正权重的大小不改变 KL 的逐 token 平均。没有有效读出位置时不调用 teacher。

`all` 相比采样会增加保存的 `[1, N_loss, vocab_size]` teacher logits 和计算开销。Student LM head 的 KL 仍分块并支持反向重算；这不等于 teacher logits 也被分块卸载。`distill_teacher_device=cpu` 可减少 teacher 参数占用的 GPU 显存，但已选 logits 仍需送往 student。全参数 plain 与全覆盖 KL 在本机是否能容纳长轨迹尚未实测，不能保证单卡显存足够。

第二阶段的文档移除/依赖加权以及片段重建没有实现。

## 验证记录

CPU 小模型回归共 120 项通过，覆盖 Qwen3/Qwen3.5 的 whole 压缩、全 token 多轮 KL 对齐、teacher 单次前向/冻结、Beacon 梯度、CE 不变、LoRA 与保存恢复，以及本地完整 plain checkpoint 分派加载。还验证了默认等权时的分层类别输出、8 份训练 YAML 与三个训练 Shell 入口的语法。未运行正式训练；实际 4B 模型的训练显存与效果仍待后续实验验证。
