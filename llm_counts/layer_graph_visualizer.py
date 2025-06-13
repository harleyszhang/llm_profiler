"""
cli entry point for LayerAnalyzer, which analyzes the memory access and FLOPs of a model.
Usage:
    ```bash
    python -m llm_counts.layer_graph_visualizer \
    --result-json results.json \
    --model-type qwen3 \
    --output my_layer_graph
```
"""
from __future__ import annotations
from .utils.constants import BYTES_FP16
from .utils.config import *
from .utils.utils import num_to_string
from .roofline_model import roofline_analysis
from copy import deepcopy
from pathlib import Path
from graphviz import Digraph
import argparse
from typing import Union

import json

class LayerAnalyzer(object):
    """Count memory access of the model and layers."""

    def __init__(self, model_config: 'ModelConfig', gpu_config: 'GPUConfig', tp_size: int) -> None:
        self.tp_size = tp_size
        self.bandwidth, self.onchip_buffer = get_gpu_hbm_bandwidth(gpu_config) # GB/s
        self.bandwidth *= 10**9 
        self.gpu_max_ops = get_TFLOPS_per_gpu(gpu_config, data_type="fp16") * 10**12  # TFLOPs
    
        self.model_type = model_config.model_type
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        self.num_heads = model_config.num_heads
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = model_config.head_dim

        self.is_moe = (getattr(model_config, "num_experts", None) is not None)
        self.is_qwen3moe = "qwen3_moe" == self.model_type
        
        if self.is_moe:
            # Default to 1 expert if not specified
            self.num_experts = getattr(model_config, "num_experts", 1)  
            self.num_experts_per_tok = getattr(model_config, "num_experts_per_tok", 1)

        # attention linear layers
        if self.is_qwen3moe:
            self.linear_layers = {
                "q_proj": [self.hidden_size, self.num_heads * self.head_dim],
                "k_proj": [self.hidden_size, self.num_kv_heads * self.head_dim],
                "v_proj": [self.hidden_size, self.num_kv_heads * self.head_dim],
                "out_proj": [self.num_heads * self.head_dim, self.hidden_size],
                # MoE结构
                "router_gate": [self.hidden_size, self.num_experts],
                "expert_gate_proj": [self.hidden_size, self.intermediate_size],
                "expert_up_proj": [self.hidden_size, self.intermediate_size],
                "expert_down_proj": [self.intermediate_size, self.hidden_size],
            }
        else:
            self.linear_layers = {
                "q_proj": [self.hidden_size, self.num_heads * self.head_dim],
                "k_proj": [self.hidden_size, self.num_kv_heads * self.head_dim],
                "v_proj": [self.hidden_size, self.num_kv_heads * self.head_dim],
                "out_proj": [self.num_heads * self.head_dim, self.hidden_size],
                "gate_proj": [self.hidden_size, self.intermediate_size],
                "up_proj": [self.hidden_size, self.intermediate_size],
                "down_proj": [self.intermediate_size, self.hidden_size],
            }

        self.results = {"decode": {}, "prefill": {}}

    def _analyze_to_results(
        self,
        stage,
        kernel_name,
        flops,
        load_weight,
        load_act,
        store_act,
        load_kv_cache,
        store_kv_cache,
    ):
        memory_access = (load_weight + load_act + store_act  + load_kv_cache + store_kv_cache)
        a_intensity, att_flops, bound = roofline_analysis(self.gpu_max_ops, 
                                                          self.bandwidth, 
                                                          flops, memory_access) # Arithmetic Intensity

        self.results[stage][kernel_name] = {
            "flops": num_to_string(flops),
            "memory_access": f"{num_to_string(memory_access)}B",
            "arithmetic_intensity": int(a_intensity),
            "att_flops": num_to_string(att_flops),
            "bound": bound,
            "load_weight": f"{num_to_string(load_weight)}B",
            "load_act": num_to_string(load_act),
            "store_act": num_to_string(store_act),
            "load_kv_cache": num_to_string(load_kv_cache),
            "store_kv_cache": num_to_string(store_kv_cache),
        }

        return self.results

    def analyze_linear_layers(
        self, 
        bs: int,
        seq_len: int,
        linear_weight_bytes: int = BYTES_FP16,
        act_byte: int = BYTES_FP16,
        kv_byte: int = BYTES_FP16,
    ):
        """
        Count and save the FLOPs and memory access of self-attention layers.
        This function is used to analyze the self-attention layers in the model.
        """
        # 1. attention linear layers analysis
        for name, (in_ch, out_ch) in self.linear_layers.items():
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj

            self._analyze_to_results(
                "prefill",
                name,
                flops=2 * bs * seq_len * in_ch * out_ch // self.tp_size,
                load_weight=in_ch * out_ch * linear_weight_bytes // self.tp_size,
                load_act=in_ch * bs * seq_len * act_byte // self.tp_size,
                store_act=0 if is_kv_proj else  bs * seq_len * out_ch * act_byte // self.tp_size,
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else out_ch * bs * seq_len * kv_byte) // self.tp_size
            )  
            self._analyze_to_results(
                "decode",
                name,
                flops=2 * bs * in_ch * out_ch // self.tp_size,
                load_weight=in_ch * out_ch * linear_weight_bytes // self.tp_size,
                load_act=in_ch * bs * act_byte // self.tp_size,
                store_act=0 if is_kv_proj else out_ch * bs * act_byte // self.tp_size,
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else out_ch * bs * kv_byte) // self.tp_size,
            )
    
    def analyze_self_atten_kernel(
        self, 
        bs: int,
        seq_len: int,
        generate_len: int,
        num_kv_heads: int,
        num_heads: int,
        head_dim: int,
        flash_attn: bool = False,
        act_byte: int = BYTES_FP16,
        kv_byte: int = BYTES_FP16,
    ):
        """
        Count and save the FLOPs and memory access of self-attention kernels.
        This function is used to analyze the self-attention kernels in the model.
        """
        hidden_size = num_heads * head_dim
        if not flash_attn:
            ##########################prefill stage##########################
            # 1, qkt kernel analysis
            name = "qk_matmul"
            load_q_mem = bs * self.num_heads * seq_len * self.head_dim
            load_k_mem = bs * self.num_kv_heads * seq_len * self.head_dim
            qk_store_mem = bs * self.num_heads * seq_len * seq_len
            self._analyze_to_results(
                "prefill",
                name,
                flops=2 * seq_len * seq_len * self.head_dim * bs * self.num_heads,
                load_weight=0,
                load_act=(load_q_mem + load_k_mem) * act_byte, # load q and k act, shape is [s, h]
                store_act=qk_store_mem * act_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
            # 2, softmax kernel analysis
            name = f"softmax"
            load_softmax_mem = qk_store_mem
            softmax_store_mem = bs * self.num_heads * seq_len * seq_len
            self._analyze_to_results(
                "prefill",
                name,
                flops= (bs * num_heads * seq_len * seq_len * 1 * 5),
                load_weight=0,
                load_act=load_softmax_mem * act_byte,
                store_act=softmax_store_mem * act_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
            # 3, sv kernel analysis
            name = "sv_matmul"
            load_s_mem = softmax_store_mem
            load_v_mem = bs * self.num_kv_heads * seq_len * self.head_dim
            sv_store_mem = bs * self.num_heads * seq_len * self.head_dim
            self._analyze_to_results(
                "prefill",
                name,
                flops=bs * 2 * seq_len * seq_len * head_dim * num_heads,
                load_weight=0,
                load_act=load_s_mem * act_byte, # load score(qkt) act, shape is [s, s]
                store_act=sv_store_mem * act_byte,
                load_kv_cache=load_v_mem,
                store_kv_cache=0,
            )
            ##########################decode stage##########################
            name = "qk_matmul"
            # load q and k, k is form kv cache
            qk_matmul_flops = 2 * self.num_heads * self.head_dim * (seq_len + generate_len)
            load_q_mem = bs * self.num_heads * 1  * self.head_dim
            load_k_mem = bs * self.num_kv_heads * (seq_len + generate_len) * self.head_dim
            qk_store_mem = bs * self.num_heads * (seq_len + generate_len) * (seq_len + generate_len)
            self._analyze_to_results(
                "decode",
                name,
                flops=qk_matmul_flops,
                load_weight=0,
                load_act=load_q_mem * act_byte,
                store_act=qk_store_mem * act_byte,
                load_kv_cache=load_k_mem * kv_byte,
                store_kv_cache=0,
            )
            # 2, softmax kernel analysis
            name = f"softmax"
            load_softmax_mem = qk_store_mem
            softmax_store_mem = bs * self.num_heads * (seq_len + generate_len) * (seq_len + generate_len)
            self._analyze_to_results(
                "decode",
                name,
                flops= (bs * num_heads * seq_len * seq_len * 1 * 5),
                load_weight=0,
                load_act=load_softmax_mem * act_byte,
                store_act=softmax_store_mem * act_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
            # 3, sv kernel analysis
            name = "sv_matmul"
            load_s_mem = softmax_store_mem
            load_v_mem = bs * self.num_kv_heads * (seq_len + generate_len) * self.head_dim
            sv_store_mem = bs * self.num_heads * (seq_len + generate_len) * self.head_dim
            self._analyze_to_results(
                "decode",
                name,
                flops=qk_matmul_flops,
                load_weight=0,
                load_act=load_s_mem * act_byte, # load score(qkt) act, shape is [s, s]
                store_act=sv_store_mem * act_byte,
                load_kv_cache=load_v_mem,
                store_kv_cache=0,
            )
        else:
            name = f"fused_attention" # flash_attn2
            qk_matmul_OPs = seq_len * seq_len * head_dim * num_heads * bs * 2
            sv_matmul_OPs = seq_len * head_dim * seq_len * num_heads * bs * 2
            softmax_OPs = bs * num_heads * seq_len * seq_len * 5

            block_size_r = min(math.ceil(self.onchip_buffer / (kv_byte * head_dim)), head_dim)
            n_blocks_r = math.ceil(seq_len / block_size_r)
            q_numel = seq_len * head_dim * bs * num_heads * act_byte
            o_numel = seq_len * seq_len * bs * num_heads * act_byte

            self._analyze_to_results(
                "prefill",
                name,
                flops=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seq_len) * head_dim * bs * num_kv_heads * kv_byte * 2,
                store_kv_cache=0,
            )

            qk_matmul_OPs = seq_len * head_dim * num_heads * bs * 2
            sv_matmul_OPs = 1 * head_dim * seq_len * num_heads * bs * 2
            softmax_OPs = bs * num_heads * seq_len * 1 * 5

            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_dim * bs * num_heads * act_byte
            o_numel = 1 * seq_len * bs * num_heads * act_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seq_len) * head_dim * bs * num_kv_heads * kv_byte * 2,
                store_kv_cache=0,
            )

        if self.model_type == "qwen3":
            kernel_names = ["q_norm", "k_norm"]
            # qwen3 模型中 rms_norm 计算中使用了一个额外的线性变换
            q_norm_flops = bs * 4 * seq_len * self.head_dim
            q_norm_load_weight = self.head_dim * BYTES_FP16
            q_norm_load_act = bs * seq_len * self.head_dim * BYTES_FP16 # equal k_norm_load_act
            q_norm_store_act = bs * seq_len * self.head_dim * BYTES_FP16

            # prefill/decode 阶段
            for stage in ["prefill", "decode"]:
                if stage == "decode":
                    q_norm_flops = int(q_norm_flops // seq_len)
                    q_norm_load_act = int(q_norm_load_act // seq_len)
                    q_norm_store_act = int(q_norm_store_act // seq_len)

                for _, kernel_name in enumerate(kernel_names):
                    self._analyze_to_results(
                        stage,
                        kernel_name,
                        flops=q_norm_flops // self.tp_size,
                        load_weight=q_norm_load_weight // self.tp_size,
                        load_act=q_norm_load_act // self.tp_size,
                        store_act=q_norm_store_act // self.tp_size,
                        load_kv_cache=0,
                        store_kv_cache=0,
                    )
    
    def analyze_other_kernels(
        self,
        bs: int,
        seq_len: int,
        act_byte: int = BYTES_FP16,
    ):
        norm_flops = bs * seq_len * 4 * self.hidden_size  # mlp_norm, attn_norm
        norm_load_weight = self.hidden_size * BYTES_FP16
        norm_load_act = bs * seq_len * self.hidden_size * BYTES_FP16
        norm_store_act = bs * seq_len * self.hidden_size * BYTES_FP16

        # silu 和 dot * 都是纯逐元素操作算子
        silu_dot_flops = (bs * 4 * seq_len * self.intermediate_size)  # 每个张量元素执行 4 次操作
        silu_dot_load_act = bs * 2 * seq_len * self.intermediate_size * act_byte
        silu_dot_store_act = (bs * 2 * seq_len * self.intermediate_size * act_byte)

        mlp_add_flops = bs * seq_len * self.hidden_size
        mlp_add_load_act = bs * seq_len * self.hidden_size * act_byte
        mlp_add_store_act = bs * seq_len * self.hidden_size * act_byte

        # other kernels (memory bound)
        kernel_names = ["attn_norm", "mlp_norm", "mlp_silu_dot", "attn_add", "mlp_add"]
        flops_list = [norm_flops, norm_flops, silu_dot_flops, mlp_add_flops, mlp_add_flops]
        
        load_act_list = [norm_load_act, norm_load_act, silu_dot_load_act, mlp_add_load_act, mlp_add_load_act,]
        store_act_list = [norm_store_act, norm_store_act, silu_dot_store_act, mlp_add_store_act, mlp_add_store_act,]

        # prefill/decode 阶段
        for stage in ["prefill", "decode"]:
            for i, kernel_name in enumerate(kernel_names):
                load_weight = (0 if (kernel_name not in ["attn_norm", "mlp_norm"]) else norm_load_weight)

                load_act = load_act_list[i]
                store_act = store_act_list[i]
                flops = flops_list[i]

                if stage == "decode":
                    flops = int(flops // seq_len)
                    load_act = int(load_act // seq_len)
                    store_act = int(store_act // seq_len)
                    
                self._analyze_to_results(
                    stage,
                    kernel_name,
                    flops=flops // self.tp_size,
                    load_weight=load_weight // self.tp_size,
                    load_act=load_act // self.tp_size,
                    store_act=store_act // self.tp_size,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
    
    def analyze_model(
        self,
        bs: int,
        seq_len: int,
        generate_len: int = 0,
        flash_attn: bool = False,
        act_byte: int = BYTES_FP16,
        kv_byte: int = BYTES_FP16,
    ):
        """
        Analyze the model and save the results.
        This function is used to analyze the model and save the results.
        """
        # 1. analyze linear layers
        self.analyze_linear_layers(bs, seq_len, act_byte=act_byte, kv_byte=kv_byte)

        # 2. analyze self attention kernels
        self.analyze_self_atten_kernel(
            bs, seq_len, generate_len, 
            num_kv_heads=self.num_kv_heads, 
            num_heads=self.num_heads, 
            head_dim=self.head_dim, 
            flash_attn=flash_attn, 
            act_byte=act_byte, 
            kv_byte=kv_byte
        )

        # 3. analyze other kernels
        self.analyze_other_kernels(
            bs, seq_len,
        )

        return self.results
    
BASE__DEPENDENCIES = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "qk_matmul": ["q_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "out_proj": ["sv_matmul"],
    "attn_add": ["input", "out_proj"],
    "mlp_norm": ["attn_add"],
    "gate_proj": ["mlp_norm"],
    "up_proj": ["mlp_norm"],
    "mlp_silu_dot": ["up_proj", "gate_proj"],
    "down_proj": ["mlp_silu_dot"],
    "mlp_add": ["attn_add", "down_proj"],
    "output": ["mlp_add"],
}

# 基础的 Transformer 结构（不包含 MoE 部分）
QWEN3_MOE_DEPS = {
    "input": [],
    "embedding": ["input"],
    "input_layernorm": ["embedding"],
    "q_proj": ["input_layernorm"],
    "k_proj": ["input_layernorm"],
    "v_proj": ["input_layernorm"],

    "qk_matmul": ["q_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "out_proj": ["sv_matmul"],
    "attn_add": ["input", "out_proj"],
    "post_attention_layernorm": ["attn_add"],

    "router_gate": ["post_attention_layernorm"],
    "router_softmax": ["router_gate"],
    "router_top-k": ["router_softmax"],
    "router_norm_topk_prob": ["router_top-k"],

    "mlp_add": ["attn_add", "index_add_"],  # 修正顺序：attn_add 和 concat
    "qwen3_moe_norm": ["mlp_add"],  # qwen3 模型中的额外归一化层
    "lm_head": ["qwen3_moe_norm"],
    "output": ["lm_head"],
}


def _expand_moe_deps_with_moe_mlp(num_experts: int) -> dict[str, list[str]]:
    """为 MoE 模型展开依赖关系，使用 moe_mlp_x 作为专家节点"""
    deps = deepcopy(QWEN3_MOE_DEPS)
    
    # 收集所有 moe_mlp 节点名称
    moe_mlp_nodes = []
    
    # 为每个 moe_mlp 专家节点建立依赖关系
    for i in range(num_experts):
        moe_mlp_node = f"moe_mlp_{i}"
        moe_mlp_nodes.append(moe_mlp_node)
        # 每个 moe_mlp 节点从 router_gate 获取输入
        deps[moe_mlp_node] = ["router_norm_topk_prob"]
    
    # concat 节点汇聚所有专家的输出
    deps["index_add_"] = moe_mlp_nodes
    from pprint import pprint
    pprint(deps)

    return deps



class LayerGraphVisualizer:
    """Render a transformer layer's roofline-analysis graph (Graphviz PNG)."""

    def __init__(self, model_cfg: Union[str, object], results: dict, num_experts: int = None):
        self.results = results
        self.model_type = getattr(model_cfg, "model_type", model_cfg)            

        if self.model_type == "qwen3_moe":
            num_experts = getattr(model_cfg, "num_experts_per_tok", 8)
            self.num_experts = num_experts
            self.deps = _expand_moe_deps_with_moe_mlp(num_experts)
            self.deps.update({"q_norm": ["q_proj"], "k_norm": ["k_proj"]})
            self.deps["qk_matmul"] = ["q_norm", "k_norm"]
            print(f"[Debug] Detected {num_experts} experts")
        else:
            self.deps = BASE__DEPENDENCIES.copy()

    def _get_node_color(self, node: str) -> str:
        """根据节点类型返回颜色，使用更精确的匹配规则"""
        # 精确匹配优先
        exact_matches = {
            "router_gate": "gold",
            "mlp_add": "lightcoral",
            "attn_add": "lightcoral",
            "concat": "orange",
        }
        
        if node in exact_matches:
            return exact_matches[node]
        
        # 前缀匹配
        prefix_matches = {
            "moe_mlp_": "lightgreen",
            "expert": "lightgreen",
        }
        
        for prefix, color in prefix_matches.items():
            if node.startswith(prefix):
                return color
        
        # 包含匹配
        contains_matches = {
            "proj": "lightblue",
            "matmul": "plum",
            "router": "gold",
            "norm": "lightyellow",
        }
        
        for keyword, color in contains_matches.items():
            if keyword in node:
                return color
                
        return "lightcyan"  # 默认颜色

    def _format_node_label(self, node: str, data: dict = None) -> str:
        """格式化节点标签"""
        if data is None:
            return f"{node}\\nFlops:0  Access:0\\nParams:0  Bound:N/A"
        
        flops = data.get('flops', 0)
        memory_access = data.get('memory_access', 0)
        load_weight = data.get('load_weight', 0)
        bound = data.get('bound', 'N/A')
        
        # 为 moe_mlp 节点使用更紧凑的标签
        node_display = node
        if node.startswith("moe_mlp_"):
            expert_id = node.split("_")[-1]
            node_display = f"Expert {expert_id}"
        
        return (f"{node_display}\\n"
                f"Flops:{flops}  Access:{memory_access}\\n"
                f"Params:{load_weight}  Bound:{bound}")

    def render(self, out_prefix: str = "layer_graph") -> None:
        """渲染图形"""
        out_dir = Path("figures")
        out_dir.mkdir(exist_ok=True)

        for stage, res in self.results.items():
            print(f"\n[Debug] Processing stage: {stage}")
            print(f"[Debug] Available keys: {sorted(res.keys())}")
            
            # 只使用在结果中实际存在的节点
            use_nodes = self.deps.keys()
            print(f"[Debug] Using nodes: {sorted(use_nodes)}")


            # 创建图
            dot = Digraph(
                format="png",
                node_attr={
                    "style": "filled",
                    "shape": "box",
                    "fontname": "Arial",
                    "fontsize": "9",
                },
                graph_attr={
                    "rankdir": "TB",
                    "splines": "ortho",
                    "ranksep": "0.8",
                    "nodesep": "0.5",
                }
            )
            # 添加节点
            if self.model_type == "qwen3_moe":
                self._add_moe_nodes_with_subgraph(dot, self.deps, res, use_nodes)
            else:
                self._add_regular_nodes(dot, self.deps, res, use_nodes)

            # 添加边
            self._add_edges(dot, self.deps, use_nodes)

            # 渲染
            png_path = out_dir / f"graph_{stage}_{out_prefix}"
            dot.render(str(png_path), cleanup=True)
            print(f"[LayerGraphVisualizer] Saved to {png_path}.png")

    def _add_edges(self, dot: Digraph, deps: dict, use_nodes: set) -> None:
        """添加边，支持不同样式"""
        for node, parents in deps.items():
            for parent in parents:
                if parent in use_nodes:
                    # 为专家节点的边使用不同的样式
                    if node.startswith("moe_mlp_") or parent.startswith("moe_mlp_"):
                        dot.edge(parent, node, color="green", style="dashed")
                    elif node == "concat" or parent == "concat":
                        dot.edge(parent, node, color="orange", style="bold")
                    else:
                        dot.edge(parent, node)

    def _add_moe_nodes_with_subgraph(self, dot: Digraph, deps: dict, res: dict, use_nodes: set) -> None:
        """为 MoE 模型添加节点，将专家节点组织在子图中"""
        # 添加非专家节点
        for node in use_nodes:
            if not node.startswith("moe_mlp_"):
                label = self._format_node_label(node, res.get(node))
                color = self._get_node_color(node)
                dot.node(node, label=label, fillcolor=color)
        # 创建专家子图
        moe_nodes_in_use = [n for n in use_nodes if n.startswith("moe_mlp_")]
        if moe_nodes_in_use:
            with dot.subgraph(name="cluster_experts") as experts_subgraph:
                experts_subgraph.attr(
                    label="MoE Experts",
                    style="dashed",
                    color="green",
                    fontsize="12",
                    fontname="Arial Bold"
                )
                
                for node in sorted(moe_nodes_in_use):
                    label = self._format_node_label(node, res.get(node))
                    color = self._get_node_color(node)
                    experts_subgraph.node(node, label=label, fillcolor=color)

    def _add_regular_nodes(self, dot: Digraph, deps: dict, res: dict, use_nodes: set):
        """为常规模型添加节点"""
        for node in use_nodes:
            if node in deps:
                label = self._format_node_label(node, res.get(node))
                color = self._get_node_color(node)
                dot.node(node, label=label, fillcolor=color)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate transformer layer graph")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--output", default="layer_graph")
    args = parser.parse_args()

    with args.result_json.open() as fp:
        results = json.load(fp)

    LayerGraphVisualizer(
        model_cfg=args.model_type,
        results=results,
        num_experts=args.num_experts,
    ).render(args.output)


if __name__ == "__main__":
    _main()
