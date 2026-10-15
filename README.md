# Search-R1 SearchAgent Beacon：Qwen3 / Qwen3.5

Qwen3 默认模型为 `Qwen/Qwen3-4B-Thinking-2507`，Qwen3.5 默认为 `Qwen/Qwen3.5-4B`，训练数据仅支持 Search-R1 `messages` 格式。

Qwen3 默认将每轮 information 整段插入 beacon 后一次前向，仅提交 beacon KV，后续 token 无法直接访问原始 information KV。问题记忆、线性 readout 蒸馏、子文档切分和旧窗口参数均通过配置暂停。Qwen3.5 保留独立的混合注意力实现。

已实现第一阶段完整轨迹 teacher 输出 KL，默认监督所有有效 assistant loss token；新增 plain 全参数配置，Beacon 可从本地 plain checkpoint 初始化。

文档：[第二阶段文档依赖加权](docs/stage2_distillation.md) · [第一阶段训练与对照实验](docs/stage1_distillation.md) · [使用说明](docs/README.md) · [实现与验证](docs/implementation.md)。

未启动训练，未修改参考项目或其他项目目录。
