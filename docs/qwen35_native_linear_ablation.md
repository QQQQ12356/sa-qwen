路线 A 的最小消融：全注意力层仍只提交 Beacon KV，线性层通过一个开关选择状态提交方式。

```yaml
beacon:
  beacon_preserve_native_linear_state: true
```

- `false`（默认）：原有 strict 路径，writer 从 Beacon 激活重建循环状态 S，清空卷积状态 C。
- `true`：直接保留 reader 的原生 S/C，跳过 writer 及其 readout 蒸馏。S/C 已扫描原文和插入的 Beacon，属于 hybrid 方法，不等于无 Beacon 的 plain 状态。

开关自动写入模型 `config.json`，加载 checkpoint 后评测沿用；启动日志的 `native S/C` 显示实际配置。保留 writer 模块以兼容已有 checkpoint 结构，但开启此开关时它们不参与前向或获得梯度。主 CE、information 续写和输出 KL 的配置保持原有语义。

对照使用 `configs/train/beacon_qwen35_whole_searchr1.yaml`，新增实验使用 `configs/train/beacon_qwen35_whole_native_linear_searchr1.yaml`。两份 YAML 仅实验名和此开关不同。实际实验应为两组指定相同的 student/teacher 起点、数据、随机种子和训练预算；若用 plain-SFT merged checkpoint，两组均覆盖 `model_name_or_path`（teacher 默认跟随）。

在训练服务器执行：

```bash
python -m search_comp.trainer.beacon_trainer \
  --config configs/train/beacon_qwen35_whole_native_linear_searchr1.yaml \
  --set model_name_or_path=/path/to/plain-sft-merged
```

同一 YAML 可加 `--set beacon.beacon_preserve_native_linear_state=false --set exp_name=beacon_qwen35_whole_strict_searchr1` 跑 strict 对照。两组如需第二阶段文档依赖 KL，均加 `--set beacon.beacon_output_distill_doc_weighting=true`。

本次只实现路线 A 的 reader 末态保留对照；“正文结束处取状态、Beacon 只读提取”是另一项实验，尚未实现。没有启动正式训练或任务效果评测。
