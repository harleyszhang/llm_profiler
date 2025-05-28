# -*- coding  : utf-8 -*-
# Description : gpu, model, Parallelism, data, train and inference config definition

import math, json
from .constants import *
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering
from transformers import AutoConfig
import os


class ActivationRecomputation(Enum):
    NONE = 0
    """No activation recomputation; requires the most amount of memory."""

    SELECTIVE = 1
    """Selectively checkpoints and recomputes only parts of each transformer
    layer that take up a considerable amount of memory but are not
    computationally expensive to recompute, i.e. Q K V matrix multiplies, 
    QK^T matrix multiply, softmax, softmax dropout, and attention over V."""

    FULL = 2
    """Full activation recomputation stores the input to EVERY transformer
    layer, which is sharded across the tensor parallel group, thus requiring an
    extra all-gather (ignored for now) per layer and add communication
    overhead; requires the lease amount of memory; requires an extra forward
    pass."""


@total_ordering
class DSZeRO(Enum):
    NONE = 0
    """No DeepSPeed ZeRO; requires the most amount of memory."""

    STAGE_1 = 1
    """ZeRO stage 1 shards the optimizer states across the data parallel
    group."""

    STAGE_2 = 2
    """ZeRO stage 2 shards the optimizer states and gradients across the data
    parallel group."""

    STAGE_3 = 3
    """ZeRO stage 3 shards the optimizer states, gradients, and model weights
    across the data parallel group."""

    def __lt__(self, other):
        # 炫技写法
        if other.__class__ is self.__class__:
            return self.value < other.value  # Enum 枚举类自动赋值
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, DSZeRO):
            return self.value == other.value
        return NotImplemented


@dataclass
class GPUEfficiencyConfig:
    flops_efficiency: float = 1.0
    hbm_memory_efficiency: float = 1.0
    intra_node_memory_efficiency: float = 1.0
    inter_node_memory_efficiency: float = 1.0


@dataclass
class InferenceConfig:
    """Inference configuration dataclass."""

    bs: int = None  # batch size
    seq_len: int = 522  # input sequence length
    generate_len: int = 1526  # number of tokens to generate
    context_len: int = None  # context length
    bytes_per_param: int = BYTES_FP16  # model weight bytes
    act_dtype_bytes: int = BYTES_FP16  # activation data type bytes
    kv_cache_bytes: int = BYTES_FP16  # key/value cache data type bytes

    def __post_init__(self):
        if self.context_len is None:
            self.context_len = self.seq_len + self.generate_len


@dataclass
class ParallelismConfig:
    """dataclass module provides a decorator and functions for automatically adding generated special methods
    such as __init__() and __repr__() to user-defined classes
    """

    tp_size: int = (
        1  # tensor parallelism size, Megatron-LM tensor parallelism implementation
    )
    pp_size: int = (
        1  # pipeline parallelism size, Megatron-LM pipeline parallelism implementation
    )
    dp_size: int = 1  # data parallelism size, DeepSpeed Zero parallelism implementation
    sp_size: int = (
        1  # sequence parallelism size, Megatron-LM sequence parallelism implementation
    )


@dataclass
class ModelConfig:
    num_layers: Optional[int] = None  # number of transformer layers (blocks)
    num_heads: Optional[int] = None  # number of attention heads
    head_dim: Optional[int] = None          # <— 新增：允许显式传入
    hidden_size: Optional[int] = None  # hidden dimension
    vocab_size: Optional[int] = None  # vocabulary size
    num_kv_heads: Optional[int] = None
    max_seq_len: Optional[int] = None  # max sequence length
    intermediate_size: Optional[int] = None  # hidden dimension of FFN, default to 4 * hidden_size
    model_type: str = (
        None  # model type as tagged on Hugging Face (e.g., gpt2, opt, llama.)
    )
    model_name: str = (
        None  # model name as tagged on Hugging Face (e.g., gpt2-xl, opt, llama-13b.)
    )

    # -------- post-init 逻辑 -------- #
    def __post_init__(self) -> None:
        # ① KV-heads 默认 = Q-heads
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        # ② FFN 维度默认 = 4×hidden_size
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4

        # ③ **核心：head_dim 计算**  
        #    若用户 / HF config 已提供，则直接用；否则按经典公式推断
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads

            # ④ 一致性检查（可选：遇到 MoE/GQA 可放宽）
            assert (
                self.hidden_size == self.head_dim * self.num_heads
            ), (
                "hidden_size 与 num_heads×head_dim 不一致；"
                "若模型采用变体架构，请显式指定 head_dim"
            )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, trust_remote_code: bool = True
    ):
        """
        Load a Hugging Face model configuration and map it to ModelConfig.

        Args:
            pretrained_model_name_or_path (str): Path or name of the pretrained model.
            trust_remote_code (bool): Whether to trust remote code for custom models.

        Returns:
            ModelConfig: An instance of the custom ModelConfig class.
        """
        # Load the Hugging Face configuration
        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )

        # Create a ModelConfig instance by mapping the fields
        return cls(
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attentionum_headss,
            hidden_size=hf_config.hidden_size,
            vocab_size=hf_config.vocab_size,
            num_kv_heads=getattr(hf_config, "num_kv_heads", None),
            max_seq_len=hf_config.max_position_embeddings,
            intermediate_size=hf_config.intermediate_size,
            model_type=hf_config.model_type,
            model_name=hf_config.name_or_path,
        )


@dataclass
class GPUConfig:
    # 1, gpu 型号和显存大小
    name: str  # GPU config name
    memory_GPU_in_GB: float  # memory per GPU in GB
    onchip_buffer: float = None  # on-chip buffer size in bytes, e.g., register file size

    # 2, gpu 显存带宽、节点内带宽、节点间带宽
    hbm_bandwidth_in_GB_per_sec: float=None  # GPU HBM bandwidth in GB/s
    intra_node_bandwidth_in_GB_per_sec: float=None # intra node GPU bandwidth in GB/s.(PCIE/NVLINK)
    intra_node_min_message_latency: float=None # minimum intra node message latency in seconds
    inter_node_bandwidth_in_GB_per_sec: float = 200  # inter node bandwidth in GB/s, assuming Mellanox 200Gbps HDR Infiniband

    # 3, 不同精度的 Tensor core 的计算性能
    peak_fp32_TFLOPS: float = None  # peak Tensor TFLOPS for FP32
    peak_fp16_TFLOPS: float = None  # peak Tensor TFLOPS for FP16
    peak_int8_TFLOPS: float = None  # peak Tensor TFLOPS for INT8
    peak_int4_TFLOPS: float = None  # peak Tensor TFLOPS for INT4

    FLOPS_EFFICIENCY = 0.9
    HBM_MEMORY_EFFICIENCY = 0.9
    INTRA_NODE_BANDWIDTH_EFFICIENCY = 0.9

    def __post_init__(self):
        """
        Post-initialization processing to compute missing values and apply efficiencies.
        """
        # Ensure FP32 TFLOPS is calculated if missing
        if self.peak_fp32_TFLOPS is None and self.peak_fp16_TFLOPS is not None:
            self.peak_fp32_TFLOPS = self.peak_fp16_TFLOPS / 2

        # Ensure INT8 and INT4 TFLOPS are calculated if missing
        if self.peak_int8_TFLOPS is None and self.peak_fp16_TFLOPS is not None:
            self.peak_int8_TFLOPS = 2 * self.peak_fp16_TFLOPS
        if self.peak_int4_TFLOPS is None and self.peak_fp16_TFLOPS is not None:
            self.peak_int4_TFLOPS = 4 * self.peak_fp16_TFLOPS

        # Apply FLOPS efficiency and round to nearest integer
        if self.FLOPS_EFFICIENCY:
            self.actual_peak_fp32_TFLOPS = math.ceil(
                self.peak_fp32_TFLOPS * self.FLOPS_EFFICIENCY
            )
            self.actual_peak_fp16_TFLOPS = math.ceil(
                self.peak_fp16_TFLOPS * self.FLOPS_EFFICIENCY
            )
            self.actual_peak_int8_TFLOPS = math.ceil(
                self.peak_int8_TFLOPS * self.FLOPS_EFFICIENCY
            )
            self.actual_peak_int4_TFLOPS = math.ceil(
                self.peak_int4_TFLOPS * self.FLOPS_EFFICIENCY
            )


class LLMConfigs(object):
    """LLMConfigs is a dataclass that contains all the configurations for the LLM model."""

    def __init__(
        self,
        gpu_config: GPUConfig,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig = ParallelismConfig(),
        inference_config: InferenceConfig = InferenceConfig(),
        gpu_efficiency_config: GPUEfficiencyConfig = GPUEfficiencyConfig(),
    ) -> None:
        self.model_config = model_config
        self.gpu_config = gpu_config
        self.parallelism_config = parallelism_config
        self.inference_config = inference_config  # 用户自行指定配置
        self.gpu_efficiency_config = gpu_efficiency_config  # 用户自行指定配置


def get_model_and_gpu_config_by_name(
    model_name="llama-13b", gpu_name="v100-pcie-32gb"
) -> dict:
    """Read model and gpu configs from a json file."""
    current_dir = os.path.dirname(__file__)
    model_config_path = os.path.join(current_dir, "../configs/model_configs.json")
    gpu_config_path = os.path.join(current_dir, "../configs/gpu_configs.json")

    with open(model_config_path, "r") as f:
        config_json = json.load(f)  # 类似于 dict 类型
        if model_name in config_json:
            print(f"model name {model_name} is found in {model_config_path}")
            config_dict = config_json[model_name]
            model_config = ModelConfig(**config_dict)
        else:
            print(
                f"model name {model_name} is not found in {model_config_path} so need to apply transformers AutoConfig"
            )
            # 加载模型配置
            model_config = ModelConfig.from_pretrained(model_name, trust_remote_code=True)

    with open(gpu_config_path, "r") as f:
        config_json = json.load(f)  # 类似于 dict 类型
        config_dict = config_json[gpu_name]
        assert gpu_name in config_json, (
            f"gpu name {gpu_name} not found in {gpu_config_path}"
        )
        gpu_config = GPUConfig(**config_dict)

    return model_config, gpu_config


def get_TFLOPS_per_gpu(
    gpu_config: GPUConfig, data_type="fp16", flops_efficiency=FLOPS_EFFICIENCY
) -> float:
    """Get the expected TFLOPS per GPU for the specified data type
    configuration/GPU (adjusted by flops_efficiency)

    Returns:
        float: TFLOPS per GPU and unit is T.
    """
    if data_type == "int8":
        gemm_TFOPS = gpu_config.peak_int8_TFLOPS
    elif data_type == "fp16":
        gemm_TFOPS = gpu_config.peak_fp16_TFLOPS
    else:
        print("weight_bits and activation_bits must be 8, or 16!")

    return gemm_TFOPS * flops_efficiency


def get_gpu_hbm_bandwidth(
    gpu_config: GPUConfig, hbm_memory_efficiency=HBM_MEMORY_EFFICIENCY
) -> list:
    return gpu_config.hbm_bandwidth_in_GB_per_sec * hbm_memory_efficiency, gpu_config.onchip_buffer


def get_intra_node_bandwidth(
    gpu_config: GPUConfig, intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY
) -> float:
    return gpu_config.intra_node_bandwidth_in_GB_per_sec * intra_node_memory_efficiency


def get_inter_node_bandwidth(
    gpu_config: GPUConfig, inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY
) -> float:
    return gpu_config.inter_node_bandwidth_in_GB_per_sec * inter_node_memory_efficiency
