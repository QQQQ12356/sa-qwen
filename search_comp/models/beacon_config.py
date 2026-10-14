"""Beacon 压缩机制的配置数据类。

本模块定义了 Activate Beacon 长上下文压缩机制的全部超参数，
作为独立的数据类，方便从 YAML / dict 初始化并与 HuggingFace
模型 config 互相转换。

参考论文: Soaring from 4K to 400K: Extending LLM's Context with
Activation Beacon (https://arxiv.org/abs/2401.03462)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

from ..loss_segments import LossSegmentConfig, field_default


@dataclass
class BeaconConfig:
    """Beacon 压缩机制的超参数配置。

    核心思路：把检索到的文档按窗口切分，每 ``beacon_ratio`` 个 token
    生成 1 个 beacon token。beacon token 通过自注意力聚合所在窗口的
    信息（软提示压缩），后续解码只关注 beacon 的 K/V cache，原始文档
    的 K/V 在生成 beacon 后即被丢弃。
    """

    #: 是否启用 beacon 压缩机制（False 时退化为普通 causal LM）
    enable_beacon: bool = True
    #: 滑动窗口大小（token 数）。文档区按此大小切分窗口
    beacon_window: int = 1024
    #: 窗口步长（token 数）。窗口无重叠，append 和 intersect 均按 window 分 chunk
    beacon_stride: int = 1024
    #: 暂停旧窗口参数；需使用 whole，keep_window=None 时整段 keep 前向。
    beacon_window_enabled: bool = True
    #: 压缩率：每多少个原始 token 生成 1 个 beacon。例：64 表示每 64 token -> 1 beacon
    beacon_ratio: int = 64
    #: beacon 的注意力模式。full-coverage 因果地覆盖当前 chunk 中此前的 token
    beacon_attn: str = "full-coverage"
    #: 为 beacon 引入哪些独立投影矩阵，取值 "q"/"k"/"v"/"o" 的任意组合（空格分隔）
    beacon_param: str = "q k v"
    #: beacon 嵌入向量的初始化来源："eos" / "bos"
    beacon_embed_init: str = "eos"
    #: 序列开头始终保留的 attention sink token 数（本实现用 skip_first 保留 question 区，置 0）
    beacon_sink_size: int = 0
    #: beacon 是否能关注历史 beacon（跨窗口记忆）
    beacon_attend_prev: bool = True
    #: beacon 放置方式：append 在 chunk 末尾追加；intersect 在 chunk 内每 ratio 个 token 插入一个
    beacon_pos: str = "append"
    #: 保留兼容字段；当前 Qwen3.5 窗口状态机统一使用 beacon_ratio
    eval_beacon_ratio: Optional[int] = None
    #: 线性注意力层的 Beacon-only 状态写入器秩
    beacon_linear_writer_rank: int = 128
    beacon_question_memory_v1: bool = False
    beacon_question_max_tokens: int = 128
    beacon_readout_distill_weight: float = 0.1
    #: 提交压缩记忆后的输出 KL 蒸馏；0 保持旧训练行为。
    beacon_output_distill_weight: float = 0.0
    beacon_output_distill_temperature: float = 1.0
    #: all 覆盖全部 CE token；prefix / stratified 使用每段预算。
    beacon_output_distill_strategy: str = "all"
    #: 仅 prefix / stratified 使用，每个压缩后 keep 段的采样预算。
    beacon_output_distill_tokens: int = 8
    #: 把 ``<information>`` 块按子文档切成多个压缩区（每篇文档独立一个窗口序列）。
    #: 开启后「同一压缩段的非首窗」等价于「同一篇文档的续写窗」，供续写损失使用。
    beacon_doc_region_split: bool = False
    #: window 保留历史分窗行为；whole 对整个 information 插入 beacon 后仅前向一次。
    beacon_information_mode: str = "window"
    #: None 沿用续写权重开关；False 禁用；True 启用（需正的续写权重）。
    beacon_information_loss: Optional[bool] = None
    #: 主 CE 的权重。与 ``beacon_continuation_loss_weight`` 一起按**份额**归一：
    #: ``L = (w_ce·ce + w_cont·cont) / (w_ce + w_cont)``，分母只统计本次前向实际出现的项。
    #: 因此只有 CE 时 ``L = ce``（改 w_ce 不动尺度），两项都在时才按比例加权；
    #: 权重为 0 时跳过主 CE；CE 与续写均关闭时仅使用独立的蒸馏加项。
    #: ``beacon_readout_distill_weight`` / ``beacon_output_distill_weight`` 不进这个分母。
    beacon_ce_loss_weight: float = 1.0
    #: 续写损失的权重份额。0 表示关闭（默认，逐位等价于历史行为）；开启后与
    #: ``beacon_ce_loss_weight`` 按上式归一，不再是绝对加项。
    beacon_continuation_loss_weight: float = 0.0
    #: 每个读出点预测的目标 token 数 k（两种模式都生效）。window 模式 = 每窗窗头前 k 个
    #: token，由窗头第 0..k-1 行分别预测；beacon 模式 = 每 chunk 的 beacon 预测其后 k 个
    #: token，k 个目标共用 beacon 那一行（chunk 隔离下唯一的纯压缩读出点）。k <=
    #: beacon_ratio 时目标正好是下一个 chunk 的前 k 个，超出则伸进更后面的 chunk。
    beacon_continuation_tokens: int = 4
    #: 续写监督的粒度：``False`` = 每个压缩段的**非首窗窗头** ``beacon_continuation_tokens``
    #: 个 token（历史行为）；``True`` = 窗内**每个 chunk** 的 beacon 监督紧随其后的
    #: ``beacon_continuation_tokens`` 个 token（目标钳制在压缩段内，不跨子文档）。
    #:
    #: ``beacon_window == beacon_ratio`` 时窗内只有一个 chunk，读出点即窗末唯一 beacon，
    #: 隔离来自**窗口边界**（窗末只提交 beacon、丢弃原始 K/V），后续 chunk 因此只能看见已
    #: 提交的 beacon。``beacon_window > beacon_ratio`` 时窗内有多个 chunk，必须配
    #: ``beacon_pos="intersect"``（只有 intersect 把 beacon 放在各自 chunk 之后）；此时窗内
    #: chunk **不互相隔离** —— 后一个 chunk 的 token 同时读得到更早 chunk 的原始 token 与其
    #: beacon（full-attention 层靠整窗因果掩码，linear-attention 层是无掩码的循环扫描，
    #: 掩码挡不住，见 ``tests/test_beacon_window_visibility.py``）。隔离只发生在窗口边界，
    #: 因此「仅凭压缩记忆预测」只在 ``window == ratio`` 下成立。
    beacon_continuation_per_beacon: bool = False
    #: keep 段（提示词 / 模型生成内容）的切窗长度：``None`` = 跟随 ``beacon_window``
    #: （历史行为）；``0`` = 整段不切；``>0`` = 上限，超出才切（防长序列 O(L²) 注意力
    #: 显存）。keep 段的切窗粒度**不影响结果**（该窗全部 K/V 都会提交、位置全局单调、
    #: 卷积/循环状态跨窗携带），只影响前向次数与峰值显存。
    beacon_keep_window: Optional[int] = None
    #: 训练 loss 的有效 token 分块大小；避免一次生成超大词表 logits
    beacon_loss_chunk_size: int = 64
    #: loss 分块是否使用 activation checkpoint，在反向时重算 LM head
    beacon_checkpoint_loss: bool = True
    #: 训练时把 autograd 保存的激活卸载到 CPU，显著省显存但降低速度
    beacon_cpu_offload_activations: bool = False
    #: 启用 CPU offload 时的最小输入长度；0 表示所有样本都启用
    beacon_cpu_offload_threshold: int = 0
    #: 分段损失：按片段类别（think/search/answer 的标签与正文）控制开关与权重。
    #: 默认全启用、权重 1.0，此时损失与不加权逐位一致。
    loss_segments: LossSegmentConfig = field(default_factory=LossSegmentConfig)

    def __post_init__(self) -> None:
        """校验参数合法性。"""
        if not isinstance(self.beacon_window_enabled, bool):
            raise ValueError("beacon_window_enabled 必须为 bool")
        if not self.beacon_window_enabled and self.beacon_information_mode != "whole":
            raise ValueError("暂停窗口时必须使用 beacon_information_mode=whole")
        if self.beacon_information_mode not in {"window", "whole"}:
            raise ValueError("beacon_information_mode 必须为 window 或 whole")
        if self.beacon_information_loss is not None and not isinstance(self.beacon_information_loss, bool):
            raise ValueError("beacon_information_loss 必须为 bool 或 None")
        if self.beacon_information_loss and self.beacon_continuation_loss_weight <= 0:
            raise ValueError("启用 information 续写损失需要正的 beacon_continuation_loss_weight")
        if self.beacon_information_mode == "whole" and self.beacon_doc_region_split:
            raise ValueError("whole 模式不区分子文档，必须关闭 beacon_doc_region_split")
        if not 0 <= self.beacon_output_distill_weight < float("inf"):
            raise ValueError("beacon_output_distill_weight 必须为有限非负数")
        if not 0 < self.beacon_output_distill_temperature < float("inf"):
            raise ValueError("beacon_output_distill_temperature 必须为有限正数")
        if self.beacon_output_distill_strategy not in {"all", "prefix", "stratified"}:
            raise ValueError("beacon_output_distill_strategy 必须为 all、prefix 或 stratified")
        if self.beacon_output_distill_tokens <= 0:
            raise ValueError("beacon_output_distill_tokens 必须为正数")
        if self.beacon_question_max_tokens <= 0:
            raise ValueError("beacon_question_max_tokens 必须为正数")
        if not 0 <= self.beacon_readout_distill_weight < float("inf"):
            raise ValueError("beacon_readout_distill_weight 必须为有限非负数")
        if not 0 <= self.beacon_ce_loss_weight < float("inf"):
            raise ValueError("beacon_ce_loss_weight 必须为有限非负数")
        if not 0 <= self.beacon_continuation_loss_weight < float("inf"):
            raise ValueError("beacon_continuation_loss_weight 必须为有限非负数")
        if self.beacon_continuation_tokens <= 0:
            raise ValueError("beacon_continuation_tokens 必须为正数")
        if self.beacon_information_mode == "window" and self.beacon_continuation_per_beacon and self.beacon_window != self.beacon_ratio:
            # beacon_ratio 的合法性断言在 ``enable_beacon`` 块内，关掉 beacon 就一并跳过，
            # 所以这里必须自己把 ratio 挡住，否则 window//ratio 会在 describe_layout 里炸。
            if self.beacon_ratio <= 0 or self.beacon_window < self.beacon_ratio:
                raise ValueError(
                    "beacon_continuation_per_beacon 需要 0 < beacon_ratio <= beacon_window"
                    f"（当前 window={self.beacon_window} ratio={self.beacon_ratio}）："
                    "一个窗口连一个 chunk 都装不下，没有可供 beacon 监督的 chunk 边界"
                )
            if self.beacon_pos != "intersect":
                raise ValueError(
                    "beacon_continuation_per_beacon 在 beacon_window != beacon_ratio 时需要 "
                    f"beacon_pos='intersect'（当前 window={self.beacon_window} "
                    f"ratio={self.beacon_ratio} pos={self.beacon_pos}）："
                    "append 布局下 beacon 全贴在窗口末尾、位置不对应任何 chunk 边界，"
                    "无法按 chunk 逐个读出"
                )
        if self.beacon_keep_window is not None and self.beacon_keep_window < 0:
            raise ValueError("beacon_keep_window 必须为 None（跟随 beacon_window）、0（不切）或正整数")
        if self.enable_beacon:
            assert not self.beacon_window_enabled or self.beacon_window > 0, "beacon_window 必须为正数"
            assert (
                self.beacon_information_mode == "whole" or self.beacon_window >= self.beacon_stride
            ), f"beacon_window({self.beacon_window}) 必须 >= beacon_stride({self.beacon_stride})"
            assert (
                self.beacon_ratio > 0
            ), f"beacon_ratio 必须为正数，当前为 {self.beacon_ratio}"
            assert self.beacon_linear_writer_rank > 0, (
                "beacon_linear_writer_rank 必须为正数，当前为 "
                f"{self.beacon_linear_writer_rank}"
            )
            assert self.beacon_loss_chunk_size > 0, (
                "beacon_loss_chunk_size 必须为正数，当前为 "
                f"{self.beacon_loss_chunk_size}"
            )
            assert self.beacon_cpu_offload_threshold >= 0, (
                "beacon_cpu_offload_threshold 不能为负数，当前为 "
                f"{self.beacon_cpu_offload_threshold}"
            )
            assert (
                self.beacon_information_mode == "whole" or self.beacon_window % self.beacon_ratio == 0
            ), f"beacon_window({self.beacon_window}) 必须能被 beacon_ratio({self.beacon_ratio}) 整除"
            assert (
                self.beacon_attn == "full-coverage"
            ), f"当前实现仅支持 full-coverage 注意力模式，收到 {self.beacon_attn}"
            assert (
                self.beacon_pos in {"append", "intersect"}
            ), f"beacon_pos 必须为 append 或 intersect，收到 {self.beacon_pos}"
            valid_params = {"q", "k", "v", "o"}
            for p in self.beacon_param.split():
                assert p in valid_params, f"beacon_param 含非法投影类型: {p}"
        # 允许直接传 dict（如从 YAML 构造）时自动转成配置对象
        if not isinstance(self.loss_segments, LossSegmentConfig):
            self.loss_segments = LossSegmentConfig.from_dict(self.loss_segments)

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转换为 dict（用于写入模型 config 与超参数记录）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeaconConfig":
        """从 dict 构造，忽略未知字段。"""
        valid = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in valid}
        # 嵌套的 loss_segments 在 YAML/JSON 里是 dict，需要还原成配置对象
        if "loss_segments" in payload:
            payload["loss_segments"] = LossSegmentConfig.from_dict(payload["loss_segments"])
        return cls(**payload)

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    @property
    def beacon_size_per_window(self) -> int:
        """每个满窗口生成的 beacon 数量 = window // ratio。"""
        return self.beacon_window // self.beacon_ratio

    def describe_layout(self) -> str:
        """一行描述实际生效的压缩区布局（训练/评测启动时打印，便于确认与训练一致）。"""
        blocks = (
            "每个 <information> 子文档独立成段"
            if self.beacon_doc_region_split
            else "每个 <information> 块整块成段"
        )
        if self.beacon_keep_window is None:
            keep = "跟随 window" if self.beacon_window_enabled else "整段不切（窗口已暂停）"
        elif self.beacon_keep_window == 0:
            keep = "整段不切"
        else:
            keep = str(self.beacon_keep_window)
        if self.beacon_information_mode == "whole":
            loss = self.beacon_information_loss is not False and self.beacon_continuation_loss_weight > 0
            return (f"每个 <information> 整段一次前向；pos={self.beacon_pos}；"
                    f"ratio={self.beacon_ratio}；beacon数=ceil(段长/ratio)；"
                    f"窗口参数={'启用' if self.beacon_window_enabled else '暂停'}；keep_window={keep}；"
                    f"information 因果续写损失={'开启' if loss else '关闭'}")
        if self.beacon_information_loss is False or self.beacon_continuation_loss_weight <= 0:
            continuation = "续写监督=关闭"
        elif self.beacon_continuation_per_beacon:
            continuation = f"续写监督=每 beacon {self.beacon_continuation_tokens} 个"
            if self.beacon_window != self.beacon_ratio:
                # 如实上报实际布局（不写死 "intersect"：配置被构造后改动时，
                # 这里报的必须是真实生效的 beacon_pos），并说明这种布局下读出点看得见
                # 本窗更早 chunk 的原文，「仅凭压缩记忆预测」不成立。
                continuation += f"（{self.beacon_pos}，窗内 chunk 间不隔离）"
        elif self.beacon_pos == "append":
            continuation = f"续写监督=每窗头 {self.beacon_continuation_tokens} 个"
        else:
            # 窗头监督只在 append 布局产生（见 beacon_qwen3._build_cont_labels）；
            # 这里必须如实报告「权重设了但不会生效」，否则打印反而掩盖 train/eval 漂移。
            continuation = "续写监督=关闭（intersect 不产生窗头监督）"
        return (
            f"{blocks}；window={self.beacon_window} stride={self.beacon_stride} "
            f"ratio={self.beacon_ratio} → 每窗 {self.beacon_size_per_window} 个 beacon；"
            f"keep_window={keep}；{continuation}"
        )

    def merge_into_config(self, model_config: Any) -> "BeaconConfig":
        """把 beacon 超参写入 HuggingFace model.config。

        字段名保持原样（如 ``beacon_window``），与基础模型配置字段无冲突，
        保存权重时会一并写入 config.json。

        Args:
            model_config: HuggingFace PretrainedConfig 对象。

        Returns:
            写入后的自身引用。
        """
        for name, value in self.to_dict().items():
            setattr(model_config, name, value)
        return self

    @classmethod
    def from_model_config(cls, model_config: Any) -> "BeaconConfig":
        """从 HuggingFace model.config 读取 beacon 字段。

        ``field_default`` 兼容 ``default_factory`` 字段（如 ``loss_segments``）：
        旧 checkpoint 的 config.json 里没有该字段时回落到默认配置。
        """
        data = {
            f.name: getattr(model_config, f.name, field_default(f)) for f in fields(cls)
        }
        return cls.from_dict(data)
