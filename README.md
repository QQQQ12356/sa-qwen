# Search-R1 SearchAgent Beacon：Qwen3 / Qwen3.5

默认模型为 `Qwen/Qwen3-4B-Thinking-2507`，训练数据仅支持 Search-R1 `messages` 格式。

Qwen3 默认将每轮 information 整段插入 beacon 后一次前向，仅提交 beacon KV，后续 token 无法直接访问原始 information KV。问题记忆、线性 readout 蒸馏、子文档切分和旧窗口参数均通过配置暂停。Qwen3.5 保留独立的混合注意力实现。

文档：[使用说明](docs/README.md) · [实现与验证](docs/implementation.md)。

未启动训练，未修改参考项目或其他项目目录。
