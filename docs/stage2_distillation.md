# 第二阶段：文档依赖加权蒸馏

Qwen3 / Qwen3.5 共用实现，默认关闭。它只替换输出 KL 的逐 token 归一化方式，不增加新的可训练模型或 JS 损失，不修改主 CE、Beacon 压缩布局和推理流程。本次没有启动训练。

## 配置

三份 Beacon 训练 YAML 均包含：

```yaml
beacon:
  beacon_output_distill_weight: 0.1
  beacon_output_distill_strategy: all
  beacon_output_distill_temperature: 1.0
  beacon_output_distill_doc_weighting: false  # true 开启第二阶段
  beacon_output_distill_doc_epsilon: 0.1     # 有限正数，最低位置权重
```

后续可用以下命令开启（本次未执行）：

```bash
bash scripts/24_beacon_train_searchr1.sh configs/train/beacon_qwen3_searchr1.yaml \
  --set model_name_or_path=/abs/path/to/plain/final \
  --set beacon.beacon_output_distill_doc_weighting=true

# Qwen3.5 替换配置为 configs/train/beacon_qwen35_searchr1.yaml。
```

必须同时有正的 `beacon_output_distill_weight` 才计算 KD。Teacher 路径仍默认跟随 student 初始化路径，也可用 `distill_teacher_model_name_or_path` 指定独立的完整原生模型。关闭第二阶段时保持第一阶段行为，不执行文档删除前向。`all`、`prefix` 和 `stratified` 都支持第二阶段，权重仅计算于各策略实际选择的位置。配置保存到 Beacon checkpoint；推理加载无需 teacher。

## 一条轨迹的计算

以 `prompt + a0 + d1 + a1 + d2 + a2` 为例，`a` 是真实 assistant 输出，`d` 是检索正文。

1. 冻结 teacher 对完整原始轨迹前向一次，在选中的目标位置得到 `p+`。
2. 对 `a1` 的监督位置，额外让同一 teacher 对删除 `d1` 的轨迹前向；对 `a2` 的位置，额外对删除 `d2`、仍保留 `d1` 的轨迹前向。每个位置只分析最近一次已提交的 region。删除仅针对 `regions` 指定的正文，保留 information 标签及全部真实 assistant 输出。非默认子文档配置下，region 指一个子文档。
3. 同一目标 token 使用同一个真实历史输出前缀；所有 teacher 前向均因果、`eval`、`no_grad`、`use_cache=false`，未来文档/输出不可见。原始预测行 `i` 在删除正文 `[start,end)` 后映射为 `i-(end-start)`，不能直接沿用旧坐标。
4. 从完整分布 `p+` 与删除分布 `p-` 计算 JS 和权重。Student 的分布不参与权重计算。首次检索前的位置权重为 epsilon，没有相应监督位置的 region 不调用删除版 teacher。
5. Student 按原有 Beacon 路径处理轨迹，计算原有损失以及加权 KL，反向只更新 student 的可训练参数。Teacher 和权重都停止梯度；Qwen3.5 的线性 Writer 梯度路径保持不变。

有 m 个 region 实际需要计算权重时，teacher 共前向 `1+m` 次；不是整条轨迹固定两次。Student 不为第二阶段增加额外前向。没有有效 KD token 时不调用 teacher，eval 仍只计算原有损失。这里的 m 取决于被选择的监督位置，而非所有 region 的数量。

## 损失定义

同一个蒸馏温度 T 用于 JS 和 KD：

```text
p+ = softmax(full_teacher_logits / T)
p- = softmax(document_removed_teacher_logits / T)
pS = softmax(student_logits / T)
m  = (p+ + p-) / 2
JS_j = (KL(p+_j || m_j) + KL(p-_j || m_j)) / 2
w_j = epsilon + clip(JS_j / ln(2), 0, 1)

KD_weighted = T² × Σ_j[w_j × KL(p+_j || pS_j)] / Σ_j[w_j]
L = L_original + lambda × KD_weighted
```

JS 基于整个词表分布，使用自然对数。权重在 `[epsilon, epsilon+1]` 内，只改变各位置的相对 KD 份额；所有位置权重相同时退化为普通 KL。Student 始终学习完整上下文 teacher，删除文档的 teacher 只用于计算权重。JS 不作为独立损失加入总损失。

KD 在 LM-head chunk 内加权，跨 keep 窗口按各窗口的权重和聚合，最终等价于整条轨迹一次加权平均。按窗口或按 token 数重新等权平均会得到不同的损失，因此不能混用。日志 `kd_loss` 已乘 lambda。

## 成本、限制与验证

额外开销主要来自删除版 teacher 前向。各版本顺序执行，JS 按 `beacon_loss_chunk_size` 分块计算，不保留所有删除版本的 logits；完整 teacher 的选中 logits 和当前删除版本的选中 logits 仍占显存。CPU teacher 的选项继续可用。

JS 表示 teacher 对文档移除的敏感程度，不保证是正确证据；重复文档、位置偏移、teacher 错误，以及真实 thinking 中已经包含的事实都会影响该信号。epsilon 和 lambda 都需要后续实验验证，未声称当前默认值最优。

CPU 小模型测试覆盖删除坐标/attention mask、JS 参考公式、温度、权重停止梯度、两种架构和两种 Beacon 布局、多轮全局归一化、CE 保持不变、checkpoint loss 重算与梯度、关闭开关及 eval/无监督时跳过分支；不执行正式训练。
