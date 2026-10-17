分析日期：2026-10-17。代码基准：`8a5d68b3803c5e3d4abafd66f898a61c341a28ec`；检查对象为当前工作区。实验来源为 [codex-data-sa-qwen.xlsx](../material/codex-data-sa-qwen.xlsx)。本文是代码审计、实验解释与待验证方案，没有运行正式训练或修改模型实现。其他服务器的实际 resolved config、逐题输出和 checkpoint 不在本地，不能假定当前默认配置就是表中实验配置。

**核心判断：先分离线性状态的原生遗忘与 Beacon 状态重建造成的额外损失。** 当前 Qwen3.5 在压缩边界丢弃原生循环状态，改用独立 writer 从 Beacon 激活重新构造同样尺寸的矩阵，并清空短卷积状态。这个设计保证文档只能通过 Beacon 载体进入后续记忆，但并没有进一步缩小线性状态。对于以任务质量、实际缓存和延迟为目标的系统，应优先测试保留原生线性状态；对于必须满足 Beacon-only 信息通路的方法，应优先改进状态可观测性、写入坐标系、边界监督和门控，而非只增加输出 KL 权重。

**1. 实验表能支持哪些结论。**

Qwen3.5 的记录如下。EM/F1/格式列均为百分比；表中“训练集”仅表示结果文件名包含 train，不能据此确认它与优化器所见样本的重叠情况。

| 模型 | 集合 / 样本数 | EM | F1 | 格式正确率 | 格式正确子集 EM | 平均检索轮数 |
|---|---|---:|---:|---:|---:|---:|
| Plain SFT | train / 100 | 79.0 | 84.30 | 93.0 | 84.95 | 1.53 |
| Plain SFT | 1000qa / 1000 | 43.0 | 52.36 | 67.7 | 63.52 | 1.427 |
| 原模型起点 + 压缩 | train / 100 | 36.0 | 47.31 | 97.0 | 37.11 | 1.35 |
| 原模型起点 + 压缩 | test / 100 | 23.0 | 31.27 | 91.0 | 25.27 | 1.50 |
| LoRA-SFT 起点 + 压缩 | train / 100 | 63.0 | 74.36 | 94.0 | 67.02 | 1.51 |
| LoRA-SFT 起点 + 压缩 | test / 100 | 28.0 | 36.31 | 70.0 | 40.00 | 2.08 |

两种压缩模型都备注为 LoRA-FT teacher、stage2 KL、保留 CE；文档压缩比均约 15.6–15.7，压缩载体数量的差异很小。

- 从已训练模型初始化，train EM 提高 27 个百分点，test EM 提高 5 个百分点；test F1 提高约 5.05 个百分点。不能把训练集的巨大提升直接解释成泛化提升。
- 测试格式正确率下降 21 个百分点，平均检索轮数增加 0.58。当前评测的“格式正确”主要指能够解析到闭合的 `<answer>...</answer>`，不是对所有搜索协议步骤的完整校验，见 [statistics.py](../search_comp/evaluation/statistics.py#L42)。必须查看逐题 stop reason，才能区分重复检索、用尽生成预算、未闭合标签或其他原因。
- 两行数据满足 `0.23 = 0.91 × 0.252747…`、`0.28 = 0.70 × 0.40`。按新模型的条件准确率作纯算术分解，条件准确率变化贡献约 +13.4 个百分点，格式覆盖下降贡献 −8.4 个百分点，净增 +5。两个条件子集不是同一批题，不能把这个分解解释为因果效应。
- Qwen3.5 plain 测试是 1000 题，压缩测试是 100 题；未确认 sample IDs 之前，不能用 43%→28% 作为配对压缩损失。两种压缩实验是否使用完全相同的 100 题也需要核对。
- `qwen3` sheet 第 4–7 行的所有结果、路径和备注，与 `qwen3.5` sheet 对应行相同，路径也写着 `qwen35`。因此它们不能作为独立 Qwen3 压缩实验的证据。Qwen3 plain 的 train/test EM 为 84%/47%，目前无法可靠比较两种架构对初始化的敏感程度。
- 100 题上的 5 个百分点差异需要逐题配对检验与多随机种子结果。仅凭聚合表不能判断其统计显著性，也不能归因到线性层。

合理的待检验解释是：原模型起点同时学习搜索任务、压缩编码和状态重建，已 SFT 起点已有任务行为基础。但后者对无压缩上下文形成的依赖，可能在状态被替换后表现为更长搜索轨迹或未能终止。其额外的 SFT 训练预算本身也是混杂因素；增加一个“原模型起点、匹配总训练 token 数”的对照，才能区分初始化路径与训练量。

**2. 当前实际执行的压缩流程。**

[beacon_trainer.py](../search_comp/trainer/beacon_trainer.py#L154) 在读取配置后强制设置 `whole`、关闭 window、关闭子文档拆分、`keep_window=0`。一条轨迹仍按 keep / information / keep 依次计算，只是每个 information 完整处理。单纯覆盖旧 window 参数不会产生分窗实验。

正文长 N 时生成 `ceil(N / beacon_ratio)` 个 Beacon，默认 ratio=16、append。全注意力层让 Beacon 读取整个正文，然后只提交 Beacon KV，已有 keep 和 Beacon KV 继续保留。位置按压缩后缓存重新编号；并非始终保留原始 token 位置。

线性层的路径位于 [beacon_qwen35.py](../search_comp/models/beacon_qwen35.py#L354)：

```text
上轮持久状态 (S_old, C_old)
        │ 克隆后作为 reader 初态
        ▼
扫描 [原文 D, Beacon B] → 原生临时状态 (S_reader, C_reader)
        │
        ├─ 本层 attention + MLP 后的 Beacon 激活 H_B
        │       └─ 独立 Writer(H_B, S_old) → S_new
        │
        └─ S_reader、C_reader 在压缩提交时被丢弃

后续 keep 从 (S_new, 空卷积状态) 开始；keep 的状态正常持续更新。
```

`clone()` 保留 autograd 连接，并不等于 detach。已有测试确认后续 token 的梯度能回传到较早 writer。现在的核心问题是学习信号和表示是否足够，而不是已确认的梯度断开。

**3. Qwen3.5-4B 的线性层到底保存了什么。**

官方 [4B config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json) 显示：32 层，24 个 GatedDeltaNet 线性层和 8 个 full-attention 层，按 3:1 交替；线性层有 16 个 Q/K heads、32 个 V heads，key/value head dimension 均为 128，卷积核长度 4。文件顶部关于“24 层、18+6”的注释说的是 2B，不能套到当前默认 4B。

沿用本地实现的矩阵方向，每个 value head 保存 `S ∈ R^(d_k × d_v)`。省略 batch/head，原生递推可写为：

\[
\alpha_t=\exp[-\exp(A_{\log})\,\mathrm{softplus}(a_t+dt_{bias})],\quad
\beta_t=\sigma(b_t),
\]
\[
\widetilde S_t=\alpha_t S_{t-1},\qquad
S_t=\widetilde S_t+k_t\left[\beta_t(v_t-\widetilde S_t^\top k_t)\right]^\top,
\qquad o_t=S_t^\top q_t/\sqrt{d_k}.
\]

Q/K/V 经过短卷积和 SiLU，Q/K 还做 L2 归一化；读出再经过 gated RMSNorm 和 out projection。公式对应本地 Transformers 5.9.0 的 `torch_recurrent_gated_delta_rule` 以及项目 `_linear_attention_forward`；可对照 [Transformers 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。

信息损耗来源有三个层次：

1. 固定尺寸关联记忆的容量限制。相近 key 会相互干扰，后续写入会修改已有方向；不能像完整 KV 一样让未来 query 逐个检索全部历史 token。
2. 动态遗忘。alpha 对一个 head 的整个矩阵实施衰减，delta 更新只沿当前 key 修正关联。长串无关文档和后续生成都可能减弱早先证据。
3. 当前项目增加的二次压缩。原文先进入原生状态，再经少量 Beacon 的读出、残差/MLP、低维投影，最后被写入另一个矩阵；可观测信息和读写坐标系均可能丢失。

前两个是原生架构特性，第三个是本项目可直接优化的设计。[Gated DeltaNet 原论文](https://arxiv.org/abs/2412.06464) 讨论了门控和定向 delta 更新的互补性，以及有限维关联记忆的检索限制；不能由此推出本项目下降全部是模型先天缺陷。

**4. 当前实现中最值得优先处理的具体瓶颈。**

**4.1 独立 writer 需要重新学习原生状态的坐标系。**

[BeaconLinearStateWriter](../search_comp/models/beacon_qwen35.py#L39) 使用 `LayerNorm → down(2560→128) → SiLU → up`，为全部 heads 产生 key、value、decay、beta。writer 接收的是本层 MLP 之后的 Beacon hidden；原生 recurrence 使用的是本层 input norm 后的投影和卷积结果，两者不是同一种表示。

full-attention 的 Beacon Q/K/V 会从已加载的基础模型复制初始化，但 writer 的 down/up 没有对应预训练权重。后续原生 query 必须能读懂 writer 写入的矩阵，这种匹配当前主要靠远端 CE/KL 间接学习。`rank=128` 是各 heads 共享的投影瓶颈宽度，不是把持久矩阵分解为秩 128，也不是每个 head 有独立的 128 维编码器。

**4.2 同一个 Beacon embedding 在首个线性层形成重复读取方向。**

所有 Beacon 共享一个 embedding。首层是线性层，不在 Q/K 上施加 full-attention 的 RoPE；append 又把相同 embedding 连续放在末尾。卷积核长度为 4，因此第 4 个及之后的 Beacon，在首层的卷积窗口都由同样的 Beacon 输入组成，产生相同的 Q/K/V 和门控。

这不意味着它们输出完全相同：状态仍在变化。但大量 Beacon 并没有提供大量独立 query。固定 Q/K/V/门控区间可写成：

\[
S_j=A S_{j-1}+U,\qquad A=\alpha(I-\beta kk^\top).
\]

`A` 只沿 k 方向和其正交补采取两种缩放。该区间对进入状态的线性观测是 `qᵀ A^j S_0`，其 query 方向位于 `span(q,k)` 中。因此，不断增加重复 Beacon 的数量不等于持续扩大对这个 head 的进入状态的可观测子空间。

这个结论严格限制在首层、卷积预热后的恒定输入区间；最前几个 Beacon 的卷积过渡有不同 query，更深层的输入会变化，后面的 full-attention 也能读取原文。不能据此宣称整个模型只能保存两个方向。它仍然揭示了浅层 writer 的严重结构性限制：非线性 MLP 无法恢复读出中从未观察到的状态分量。

本地小模型探针验证了首层该区间 Q 的最大差异为 0。优先比较 `whole+intersect`、多 slot Beacon embedding、或从文档状态做多 query 的只读提取；后三者中的新模块需要实现。intersect 可以更早采样文档中间状态，但在整段模式下不产生 chunk 隔离，也不保证最后一个 Beacon 覆盖之外的局部信息自动保留。

**4.3 writer 会随 Beacon 数量累计衰减旧记忆。**

当前每个 Beacon 先执行 `S *= sigmoid(decay)`，decay bias 初始化为 4。若暂忽略输入相关投影、只看这个标量衰减项，则 `sigmoid(4)≈0.9820`：

| Beacon 数 m | 对应 ratio=16 的正文长度 | 标量累计保留因子 |
|---:|---:|---:|
| 64 | 1024 | 0.3130 |
| 128 | 2048 | 0.0980 |
| 256 | 4096 | 0.00960 |
| 512 | 8192 | 0.000092 |

这些是初始化附近的机制示意，不是训练后实际测量。delta 的定向擦写与重新写入还会进一步改变旧状态。它提示：降低 ratio、增加 Beacon 数可能反而加重遗忘，不能预设“更多 Beacon 必然更好”。

可把衰减改成 `alpha_j = exp(-softplus(a_j) / m)`，或者让各 Beacon 的写入预算归一化、单独保留旧证据槽。归一化会改变时间语义；若希望模拟完整原文的 native transition，应由 teacher 学习按源 token 跨度累计的衰减，不能强制所有长度都具有相同遗忘量。现有 question writer 已含 `/count` 衰减，但同时改变了相关性、novelty 和更新幅度，打开它不是纯门控消融。

**4.4 压缩边界清空了短卷积上下文。**

代码直接 `self._mem._linear_conv.pop(idx, None)`。因此下一段最前几个 token 的 Q/K/V 从零左上下文计算；长度 4 的卷积直接影响开头至多 3 个位置，但被扰动的递推状态会继续影响后文。不能仅通过匹配 S 就认为边界行为一致。

严格 Beacon-only 路径可以从压缩 Beacon 预测或重放得到 C；使用 native 输入投影时应对齐输入层语义，不能直接把 post-MLP hidden 当作原投影输入。另一种方案是单独的 conv-state decoder。append 下若尾部至少 4 个 Beacon，reader 的卷积缓存最后 4 个投影位置本来都是 Beacon；intersect 或少于 4 个 Beacon 时则可能包含原文位置。任何保留尾部卷积的方案都应按实际位置判断信息来源，并单独验证。

**4.5 默认 LoRA 没有适配线性层的读写投影。**

[训练配置](../configs/train/beacon_qwen35_searchr1.yaml) 只选了 `q_proj/k_proj/v_proj/o_proj` 和 MLP 的 `gate/up/down_proj`。因此 linear attention 的 `in_proj_qkv/in_proj_a/in_proj_b/in_proj_z/out_proj` 没有 LoRA；`out_proj` 与 `o_proj` 不同。MLP 和前层变化仍能间接影响线性输入，不能说线性输出完全不可适配。

可在现有列表基础上添加 `in_proj_qkv/in_proj_z/out_proj`，再独立消融 `in_proj_a/in_proj_b`；后者控制遗忘和写入，建议较小学习率并监测状态范数。A_log、dt_bias、conv1d 并不会因这份 LoRA 列表自动训练。为了公平区分“适配位置”与“更多参数”，应报告或匹配额外 LoRA 参数量。

此外，LoRA 模式会把所有名称含 beacon 的参数设为全量可训练。按官方 4B 维度、q/k/v 三个独立 Beacon 投影和 writer rank=128 推算，新增 Beacon 参数约 **2.433 亿**，其中 full-attention Beacon 投影约 2.097 亿，writer 等约 3355 万；这还不含 LoRA。不能把训练困难简单归因于“只有少量 LoRA 参数”。原生坐标对齐与监督质量可能比继续增大 writer 更重要。

**4.6 当前信息续写损失不等于压缩重建。**

[beacon_common.py](../search_comp/models/beacon_common.py#L215) 的 whole 模式用原文行预测下一原文 token。append 时这些原文行位于 Beacon 之前，所以同一段的损失无法监督当前尾部 Beacon 或其 writer。它能训练普通 backbone/LoRA，也可能在多轮情形监督之前已经提交的压缩记忆，但不能被当作当前正文的压缩后重建监督。

两份 Qwen3.5 配置不同：`beacon_qwen35_searchr1.yaml` 未显式关闭此项、权重 0.1；`beacon_qwen35_whole_searchr1.yaml` 显式设 `beacon_information_loss=false`。实际实验应检查 resolved config。

本地单 information 小模型，仅开启此损失时，loss=4.837116，writer 非零梯度参数数为 0，Beacon embedding 最大梯度为 0。这是该因果路径的验证，不能外推为当前总训练损失对 writer 没有梯度。

**4.7 现有蒸馏约束对任务证据仍不充分。**

输出 KL 已正确使用提交压缩后的 keep 段，但 `all` 包括首次检索前的 token，长 thinking 占据大量 token。stage2 删除最近 information 的 JS 加权改善了位置选择，却不等同于全历史证据归因：较早文档、重复证据，以及已在真实 thinking 中复述的事实，可能让删除对照不敏感。teacher forcing 下，真实推理文本也可能替压缩记忆补回事实，导致 teacher-forced loss 好而自由生成仍差。

[question readout](../search_comp/models/beacon_question_memory.py#L48) 默认未启用，且仅在 question writer 分支计算。其 probes 来源于 writer 对问题 token embedding 的投影，不是后续 native GDN 实际使用的 query；target 是当前 student 的临时 reader，且已处理插入的 Beacon，不是独立 full-context SFT teacher。可以作为局部辅助约束，但不能宣称已有完备的原生状态蒸馏。

**5. 两条优化路线及优先级。**

**路线 A：以实际任务质量和缓存效率为目标，先保留原生线性状态。**

压缩 full-attention 的原文 KV，同时让 linear attention 保留原生 S/C。先用最小改动提交现有 reader 末态定位 writer 的代价；注意这个末态已经扫描 Beacon，不完全等于无 Beacon 的 plain 状态。再比较“在正文结束处取 native state、Beacon 仅做只读提取”的版本，隔离 Beacon 自身对状态的扰动。

这种策略不满足当前严格的“文档信息必须经 Beacon”约束，应作为明确独立的 hybrid 方法和对照，不能冒充同一种压缩方法。已有 reader-poison 隔离测试也不能用在这个策略上；它们应只约束 strict 模式。

它对持久内存的意义是：writer 的 S 与 native S 大小相同。按 batch=1、线性状态 FP32、KV BF16 推算，24 层的 S 合计 **48 MiB**；8 层 full attention 的 KV 每 token **32 KiB**；所有线性层 BF16 卷积缓存仅约 **1.5 MiB**。在 keep 段内当前实现也本来维护卷积缓存。保留 native S 不增加循环矩阵的尺寸。

例如保留的 keep 为 2048 token、正文总量 8192 token、ratio=16，忽略卷积和小对象：

\[
M_{plain}=48+32\mathrm{KiB}\times10240=368\mathrm{MiB},
\quad M_{compressed}=48+32\mathrm{KiB}\times2560=128\mathrm{MiB}.
\]

持久缓存约节省 65.2%，而不是整机内存缩小 16 倍。保留 native S 的路线也基本保留这一收益。表中的 15.6–15.7 是正文 token / Beacon token，见 [statistics.py](../search_comp/evaluation/statistics.py#L115)，不包含 keep、固定状态、权重、训练激活或 teacher。

**路线 B：保留严格 Beacon-only 约束，改造状态编码与重建。**

按以下顺序推进，每一步独立消融：

1. **先增加有用的读取方向。** 比较 whole+intersect；新设计可使用不同 slot embedding、问题与当前 search query 条件化的多 query 读出，或对文档 hidden 做交叉注意力提取 Beacon。让 Beacon query 提取状态时不再同时擦写同一个 reader state。对新增 cross-attention 的编码成本单独计量。
2. **让 writer 使用 native 坐标。** 比较原生投影/递推的 Beacon replay 加小残差 adapter，与当前独立 writer。replay 从上轮已提交的压缩 S/C 开始，输入只能来自 Beacon；严禁把当前原文 reader state 当作持久初态。适配器负责将 post-layer Beacon 表示变换为原生层输入所需的表示。简单复制不相容权重不能解决坐标问题。
3. **同时重建 S 与 C。** 避免每次压缩结束都让卷积冷启动。可对 native 坐标的 C 做小权重归一化 MSE，更重要的是比较真实后续 token 的读出。
4. **独立修正 writer 门控。** 测试按 Beacon 数归一化的保留预算、按源跨度蒸馏的 transition、以及保留旧证据的残差/分槽写入。监测每层累计 log-alpha、beta、状态范数和旧事实召回。
5. **提供边界上的直接监督。** 从同一个压缩前状态 fork 两条局部分支：参考分支处理原文并保留 native 状态；学生分支经 Beacon 提交重建状态。再给两条分支相同的后续 8–32 个 token，比较线性层读出、层输出和 logits。这能定位替换状态造成的误差，避免把前面所有轮次累计误差都算在当前 writer 上。训练参考分支 stop-gradient。
6. **最后再增加参数量或更强结构。** writer rank 从 128 到 256/512、按 head group 分解 writer、多文档状态 bank、稀疏保留证据 KV 都值得测试，但不能绕过前面的可观测性和读写对齐问题。状态 bank 每完整增加一组所有线性层状态约需 48 MiB；它也改变 strict 信息通路，应单独定义预算与方法。

适合 native 坐标一致时的读出蒸馏是：

\[
\mathcal L_{read}=\frac{1}{|\mathcal P|}\sum_{q\in\mathcal P}
\frac{\|\hat S^\top q-\mathrm{sg}(S_T^\top q)\|_2^2}
{\|\mathrm{sg}(S_T^\top q)\|_2^2+\epsilon}.
\]

P 优先包含后续真实 native query，辅以固定随机/正交 probes 和问题相关 probes。若 probes 只覆盖低维子空间，则只能约束该子空间，不能声称恢复完整矩阵。若 teacher/student 的投影因 LoRA 已处于不同坐标系，应比较变换后的层输出或 logits，或显式学习对齐；不能直接要求两个任意坐标的 S 做 MSE。

完整 teacher 与局部参考有不同用途：冻结 SFT teacher 提供任务目标，局部 fork 参考隔离当前压缩边界的机制误差。两者应分别记录损失。先将新监督强度调到有可见梯度但不压倒任务 CE，再按验证集选择，避免一次叠加大量固定权重。

若增加证据重建辅助任务，应在原文 KV 和 reader state 已删除后，用独立查询或短 span 起始提示从压缩状态预测实体、日期、关系、支持句；每条目标 token 使用正确的自回归前缀。一个相同 hidden 同时预测多个不同未来 token 只会拟合目标词频混合，不能称为多 token 自回归重建。辅助分支的状态应 fork 并丢弃，不污染主轨迹；输入不得含尚未预测的答案。

**6. 训练组织与可执行的实验顺序。**

继续优先以同一 plain-SFT merged checkpoint 初始化 student 和 teacher。先冻结已有任务主干，预热新的 Beacon/writer；再用较小 backbone LoRA 学习率联合适配，必要时逐步加入线性投影。实现参数组后，可以把 backbone 学习率设为压缩模块的约 1/5–1/10 作为起始搜索范围；这是候选超参，不是已验证最优值，当前 trainer 不提供这一独立配置。

ratio 课程可先比较 4/8 再到 16，但必须同时观察 Beacon 数增加带来的门控累积和尾部重复读取。对于不同实验，固定训练样本顺序、总监督 token 数、teacher、检索环境和生成预算；匹配总训练量的原模型对照也必须保留。

| 优先级 / 实验 | 相对控制组只改变什么 | 想回答的问题 |
|---|---|---|
| P0：统一 plain / strict baseline | 同一组题、同一检索与生成配置 | 真实压缩质量损失多大？ |
| P0：提交 native reader S/C | 取消 writer 替换及 conv 清空 | 二次重建是否是主要损失？ |
| P0：只替换 S / 只清空 C 的因子实验 | 独立控制两个边界操作 | 矩阵重建与卷积断点各占多少？ |
| P1：whole+intersect | 只改插入布局 | 更早读取和打散重复 Beacon 是否有效？ |
| P1：加 GDN 投影 LoRA | 保留现有 targets 并增加线性投影 | 读写适配能力是否受限？ |
| P1：局部后续读出蒸馏 | strict baseline 上只加边界监督 | writer 是否缺直接训练信号？ |
| P2：归一化门控 | 只改 writer 保留预算 | 长文/多轮的旧证据遗忘是否改善？ |
| P2：native replay + conv 重建 | 明确标为结构组合实验 | 能否改善 strict 方法的状态兼容性？ |
| P2：ratio 4/8/16、writer rank | 独立扫描，保持其他项固定 | 质量、真实缓存和时延如何权衡？ |

先在固定检索轨迹下做离线 continuation / QA，对比相同文档输入，排除不同搜索行为造成的文档分布变化；再跑完整自由交互评测。重点统计总体 EM/F1、格式成功率、相同题目上的配对成功/失败、检索轮数、重复 query、停止原因、耗尽 token/轮数预算比例。条件子集分数只作为辅助指标。

按文档长度、搜索轮数、证据处于早/中/晚的位置分桶；增加实体/数字保持、干扰文档、第一轮证据在后续多轮仍需使用的压力测试。训练损失之外记录每层 state/readout 误差、Beacon query 多样性与有效秩、累计门控和旧事实保留。更长远的收益用至少 3 个训练种子和逐题 paired bootstrap / McNemar 检验确认。

检索格式稳定性要单独优化：适度提高答案、搜索和结束标签的 CE 份额，并混合短的高质量任务轨迹；实际无答案原因若是循环检索，单纯调标签权重不一定有效。当前正的 segment 权重大小只改变 CE，不自动改变 KL 权重；要分段归一化 KL，需修改实现。使用学生自由生成的轨迹进行后续蒸馏可减轻 teacher-forcing 偏差，但应先完成上述机制消融。

另外，whole 路径显式构造 full-attention score 矩阵，仍处理完整正文加 Beacon；压缩的是提交后的缓存，不代表预填充 FLOPs 或训练峰值显存同比下降。优化质量之后，可单独测试 SDPA/FlashAttention 等算子替换与 writer 的循环开销，使用一致的数值和缓存语义验证。

**7. 本次验证及结论边界。**

本地环境为 PyTorch 2.6.0+cu124、Transformers 5.9.0；CPU 小模型使用原生 PyTorch fallback。执行：

```bash
python -m pytest -q tests/test_beacon_qwen35_linear_memory.py \
  tests/test_document_dependency_distillation.py \
  tests/test_beacon_continuation.py
```

结果 **77 passed**，约 11 秒；两个 SWIG deprecation warning。覆盖线性状态隔离、跨窗口 writer 梯度、增量缓存一致性、JS 加权和续写相关路径。另用同一 tiny-model fixture 验证了单段 whole+append 续写不向当前 writer 提供梯度，以及首层重复 Beacon 在卷积预热后 Q 相同。参数量、内存和门控表均为按明确假设计算的解析值。

这些验证支持本文的代码机制判断，不验证任何方案的 4B 任务质量、GPU fused-kernel 行为或训练后状态分布。最终优先建议是：**先补齐相同题目的对照，执行 native-state 保留消融；若必须 strict Beacon-only，则先改读取多样性与边界状态蒸馏，再做 writer 门控、卷积状态和线性 LoRA 的独立实验。**
