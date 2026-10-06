# 实现与验证

## 数据模式清理

所有训练入口仅接受 Search-R1 messages。已删除 interactive Dataset、构建器、对应原生 Dataset/collator、轨迹序列构建函数及相关 YAML；旧 single RAG 训练数据管线也已清理。Search-R1、检索协议、文档解析、BM25 和在线多轮推理仍保留。

删除旧构建器前，将在线推理仍使用的 `truncate_docs_by_tokens` 迁移至 `data/trajectory.py`，并更新评测与探针 imports。脚本 22 保留为 Search-R1 脚本 24 的兼容入口。

保留的 Accelerate Trainer 也只走 Search-R1 分支，其训练前向明确读取 `ModelOutput.loss`，不再按过时的二元组解包。

## 参数暂停与默认值

新增 `beacon_window_enabled`，默认 true 兼容已有配置；当前 Qwen3 Search-R1 YAML 显式设 false。暂停时状态机不读取 window/stride；keep 的独立长度参数为 null 时整段处理，不隐式回落到旧 window。

Qwen3 的问题记忆开关为 false，readout 权重为 0，子文档切分为 false，information 模式为 whole。原生 Qwen3 checkpoint 没有 Beacon 配置时也采用这一策略；已有显式配置保持原样。

两种模型构造 Beacon 注意力层前，先将解析后的 Beacon 配置写回模型 config，保证原生 checkpoint 首次加载时也创建默认 beacon 投影，避免默认配置与实际参数结构不一致。

## 缓存语义

Qwen3 每轮 information 在一次前向中插入全部 beacon。各全注意力层只提交 beacon 行的 K/V；原文 K/V 仅存在于当前 information 的临时前向中。decode 的历史缓存为 keep KV + beacon KV，不含原文 KV。

Qwen3 modeling 不包含线性 Reader/Writer 或线性状态缓存。Qwen3.5 的混合层隔离实现独立保留。窗口/子文档/线性问题记忆代码未从 Qwen3.5 建模中删除，可供显式配置使用。

## 验证范围

本次最终回归结果：**279 passed / 2 skipped**。Python 编译、全部 shell 语法与模块导入检查通过。

回归测试覆盖 Search-R1 数据与分段损失、模型缓存、生成复用、续写监督、蒸馏、LoRA 合并、保存重载、统计及文档格式。

新增或调整的检查包括：

- 所有训练入口拒绝已移除的数据模式。
- 新建原生 Qwen3 模型采用暂停策略并正常创建 beacon 投影。
- 窗口暂停后，即使 window/stride 为无效旧值，也不影响 information 或 keep 的前向计划。
- append/intersect 下，逐层持久缓存精确等于 prefix KV 加当前 beacon 行 KV。
- information 完成后的 decode 仅收到保留的 keep + beacon KV，不收到原文 KV。
- 所有剩余 YAML 均为 Search-R1，Qwen3 配置显式暂停不用的功能。
- Python 编译、所有 shell 语法及已删除模块的引用清理。

使用 CPU 小模型测试，未启动训练；真实模型训练的显存、收敛和任务效果未进行验证。GPU BF16 与显式启用的真实预训练模型测试按条件跳过。
