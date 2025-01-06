# -*- coding  : utf-8 -*-
# author      : honggao.zhang
# Create      : 2024-7-19
# Update      : 2025-01-05
# Version     : 0.2.0
# Description : transformer model(llm) profiling tools,
#               with latency, memory, flops, and params distribution analysis.

import logging
import pprint
import matplotlib.pyplot as plt
import numpy as np

from math import floor
from .config import *
from .utils import *

from .model_analyzer.count_flops import CountCausalLMFlops
from .model_analyzer.count_params import CountCausalLMParams
from .model_analyzer.count_memory import CountCausalLMMemory
from .model_analyzer.count_latency import CountCausalLMLatency
from .model_analyzer.count_memory_access import CountCausalLMMemoryAccess

logger = logging.getLogger()

class LLMProfiler(object):
    """Measures the latency, memory, number of estimated floating-point operations, and parameters of each module in a PyTorch model."""
    def __init__(self, llm_configs: LLMConfigs) -> None:
        self.llm_configs = llm_configs
        self.model_config = llm_configs.model_config
        self.gpu_config = llm_configs.gpu_config
        self.inference_config = llm_configs.inference_config
        self.parallelism_config = llm_configs.parallelism_config
        self.gpu_efficiency_config = llm_configs.gpu_efficiency_config
        
        self.h = self.model_config.hidden_size
        self.l = self.model_config.num_layers
        self.V = self.model_config.vocab_size
        
        self.b = llm_configs.inference_config.bs
        self.s = llm_configs.inference_config.seq_len
        self.o = llm_configs.inference_config.generate_len
        self.bytes_per_param = llm_configs.inference_config.bytes_per_param
        
        self.tp_size = self.parallelism_config.tp_size
        self.pp_size = self.parallelism_config.pp_size
        self.num_layers_per_gpu = int(self.l / self.parallelism_config.pp_size)
        
        self.gpu_memory_in_GB = llm_configs.gpu_config.memory_GPU_in_GB * 10**9  # 单位: B
        
        # 初始化各种计数器
        self.llm_params = CountCausalLMParams(self.model_config)
        self.llm_flops = CountCausalLMFlops(self.model_config, self.b, self.s)
        self.llm_memory = CountCausalLMMemory(llm_configs)
        self.llm_latency = CountCausalLMLatency(llm_configs)
    
    def infer_profile(
        self, 
        bs: int = 1, 
        seq_len: int = 522, 
        generate_len: int = 1526,
        is_inference: bool = True,
        use_kv_cache: bool = True,
        act_recomputation: ActivationRecomputation = ActivationRecomputation.NONE,
        act_dtype_bytes: int = BYTES_FP16,
        kv_cache_dtype_bytes: int = BYTES_FP16,
        qkvo_proj_dtype_bytes: int = BYTES_FP16,
        mlp_act_dtype_bytes = BYTES_FP16,
        flops_efficiency: float = None,
        hbm_memory_efficiency: float = HBM_MEMORY_EFFICIENCY,
        intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY,
        inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY,
        print_flag=True
    ) -> dict:
        """LLM inference analysis given the llm configs and inputs."""

        if self.model_config.max_seq_len is not None:
            assert (
                seq_len + generate_len <= self.model_config.max_seq_len
            ), f"seq_len {seq_len} + generate_len {generate_len} 超过了模型的 max_seq_len {self.model_config.max_seq_len}"
        
        if self.l % self.pp_size != 0:
            logger.warning(
                "Warning: the number of layers is not divisible by pp_size, please taking the floor!"
            )
        
        infer_config_dict = {
            "inference_config":{
                "model_name": self.model_config.model_name,
                "bs": bs,
                "seq_len": seq_len,
                "tp_size": self.tp_size,
                "pp_size": self.pp_size,
                "generate_len": generate_len,
                "use_kv_cache": use_kv_cache,
            },
            "gpu_config": {
                "name": self.gpu_config.name,
                "memory_GPU_in_GB": f"{self.gpu_config.memory_GPU_in_GB} GB",
                "gpu_hbm_bandwidth": f"{self.gpu_config.hbm_bandwidth_in_GB_per_sec} GB/s",
                "gpu_intra_node_bandwidth": f"{self.gpu_config.intra_node_bandwidth_in_GB_per_sec} GB/s",
                "gpu_fp16_TFLOPS": f"{self.gpu_config.peak_fp16_TFLOPS} TFLOPS",
            }
        }

        # -------------------------- 1. Params --------------------------
        params_per_layer, dict_params_per_layer = self.llm_params.count_params_per_layer()
        num_params_model = self.llm_params.count_params_model()
        
        # -------------------------- 2. FLOPs ---------------------------
        flops_per_layer, dict_flops_per_layer = self.llm_flops.count_flops_per_layer(bs, seq_len)
        num_flops_model = self.llm_flops.count_flops_model(bs, seq_len)
        
        # -------------------------- 3. Memory --------------------------
        memory_prefill_summary_dict, memory_decode_summary_dict = self.llm_memory.count_memory_per_gpu(
            bs,
            seq_len,
            generate_len,
            is_inference=True,
            flash_attn=False,
            use_kv_cache=use_kv_cache,
            act_recomputation=ActivationRecomputation.NONE,
            qkvo_proj_dtype_bytes=qkvo_proj_dtype_bytes,
            mlp_act_dtype_bytes=mlp_act_dtype_bytes,
            kv_cache_dtype_bytes=kv_cache_dtype_bytes
        )

        # -------------------------- 4. Latency -------------------------
        prefill_latency_per_layer, prefill_dict_latency_per_layer = self.llm_latency.count_latency_per_layer(bs, seq_len)
        decode_latency_per_layer, decode_dict_latency_per_layer = self.llm_latency.count_latency_per_layer(bs, 1)
        prefill_latency_breakdown, decode_latency_breakdown = self.llm_latency.count_latency(
            bs,
            seq_len,
            generate_len,
            is_inference=is_inference,
            use_kv_cache=use_kv_cache,
            kv_cache_dtype_bytes=kv_cache_dtype_bytes,
            rmsnorm_dtype_bytes=act_dtype_bytes,
            act_recomputation=act_recomputation,
        )

        infer_result_dict = {
            "model_params": num_params_model,
            "model_flops": num_flops_model,
            "prefill_first_token_latency": prefill_latency_breakdown["prefill_latency"],
            "decode_per_token_latency": decode_latency_breakdown["decode_avg_latency"],
            "kv_cache_latency": decode_latency_breakdown["kv_cache_avg_latency"],
            "total_infer_latency": prefill_latency_breakdown["prefill_latency"] 
                                   + decode_latency_breakdown["decode_avg_latency"] * generate_len,
        }

        # --------------------------- 5. Memory Access ----------------------
        llm_count_mac = CountCausalLMMemoryAccess(self.llm_configs)
        results = llm_count_mac.count_memory_access(bs=bs, seq_len=seq_len, generate_len=generate_len)
        self.create_layer_graph(results)
        self.print_format_summary_dict(results, get_dict_depth(results))

        # -------------------------- 绘图：Pie 图示例 --------------------------
        prefill_latency_pie_save_path = f"./figures/{self.model_config.model_name}_{self.gpu_config.name}_prefill_latency_pie.png"
        decode_latency_pie_save_path = f"./figures/{self.model_config.model_name}_{self.gpu_config.name}_decode_latency_pie.png"
        flops_pie_save_path = f"./figures/{self.model_config.model_name}_flops_pie.png"
        params_pie_save_path = f"./figures/{self.model_config.model_name}_params_pie.png"

        self.plot_distribution_pie(dict_params_per_layer, "Params Distribution: MHA, MLP, RMSNorm", params_pie_save_path)
        self.plot_distribution_pie(dict_flops_per_layer, "FLOPS Distribution: QKVO_Proj, MLP, RMSNorm", flops_pie_save_path)
        self.plot_distribution_pie(prefill_dict_latency_per_layer, "Prefill Latency Distribution: QKVO_Proj, MLP, RMSNorm, TP_Comm", prefill_latency_pie_save_path)
        self.plot_distribution_pie(decode_dict_latency_per_layer, "Decode Latency Distribution: QKVO_Proj, MLP, RMSNorm, TP_Comm", decode_latency_pie_save_path)
        
        # ------------------- 打印 & 可视化 --------------------
        if print_flag:
            print("\n-------------------------- LLM main infer config --------------------------")
            pprint.pprint(infer_config_dict, indent=4, sort_dicts=False)
            
            print("\n-------------------------- LLM infer performance analysis --------------------------")
            self.print_format_summary_dict(infer_result_dict, get_dict_depth(infer_result_dict))

            print("\n---------------------------- LLM Params analysis ----------------------------")
            self.print_format_summary_dict(dict_params_per_layer, get_dict_depth(dict_params_per_layer))
            pprint.pprint({"params_model": num_to_string(num_params_model)}, indent=4, sort_dicts=False)
            
            print("\n---------------------------- LLM Flops analysis -----------------------------")
            self.print_format_summary_dict(dict_flops_per_layer, get_dict_depth(dict_flops_per_layer))
            pprint.pprint({"prefill flops_model": num_to_string(num_flops_model)}, indent=4, sort_dicts=False)
            
            print("\n---------------------------- LLM Memory analysis -----------------------------")
            self.print_format_summary_dict(memory_prefill_summary_dict, get_dict_depth(memory_prefill_summary_dict))
            self.print_format_summary_dict(memory_decode_summary_dict, get_dict_depth(memory_decode_summary_dict))
            
            print("\n-------------------------- LLM Latency analysis --------------------------")
            self.print_format_summary_dict(prefill_latency_breakdown, get_dict_depth(prefill_latency_breakdown))
            self.print_format_summary_dict(decode_latency_breakdown, get_dict_depth(decode_latency_breakdown))
        
        return memory_decode_summary_dict["max_batch_total_tokens"]
            
    def print_format_summary_dict(self, summary_dict: dict, depth: int) -> str:
        """打印时对 params / flops / latency / memory 等进行统一转换显示。"""
        for key, value in summary_dict.items():
            if "params" in key or "flops" in key:
                if not isinstance(value, dict):
                    summary_dict.update({key: num_to_string(value)})
                else:
                    self.print_format_summary_dict(value, get_dict_depth(value)-1)  # 递归
            if "latency" in key:
                if not isinstance(value, dict):
                    summary_dict.update({key: latency_to_string(value)})
                else:
                    self.print_format_summary_dict(value, get_dict_depth(value)-1)
            if "memory" in key:
                if not isinstance(value, dict):
                    summary_dict.update({key: f"{num_to_string(value)}B"})
                else:
                    self.print_format_summary_dict(value, get_dict_depth(value)-1)
        if depth >= 1:
            pprint.pprint(summary_dict, indent=4, sort_dicts=False)


    def plot_distribution_pie(self, data, title, save_path):
        labels = data.keys()
        sizes = data.values()
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(title)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.savefig(save_path)

    def create_layer_graph(self, results):
        from graphviz import Digraph
        stage_list = ["prefill", "decode"]
        for stage in stage_list:
            res = results[stage]
            # Define the transformer layer graph with operation and memory info
            transformer_layer_graph = {
                "input": {"dependencies": [], "ops": "0", "access": "0"},
                "attn_norm": {"dependencies": ["input"], "ops": res["attn_norm"]["flops"], "access": res["attn_norm"]["memory_access"]},
                "q_proj": {"dependencies": ["attn_norm"], "ops": res["q_proj"]["flops"], "access": res["q_proj"]["memory_access"]},
                "k_proj": {"dependencies": ["attn_norm"], "ops": res["k_proj"]["flops"], "access": res["k_proj"]["memory_access"]},
                "v_proj": {"dependencies": ["attn_norm"], "ops": res["v_proj"]["flops"], "access": res["v_proj"]["memory_access"]},
                "qk_matmul": {"dependencies": ["q_proj", "k_proj"], "ops": res["qk_matmul"]["flops"], "access": res["qk_matmul"]["memory_access"]},
                "softmax": {"dependencies": ["qk_matmul"], "ops": res["softmax"]["flops"], "access": res["softmax"]["memory_access"]},
                "sv_matmul": {"dependencies": ["softmax", "v_proj"], "ops": res["sv_matmul"]["flops"], "access": res["sv_matmul"]["memory_access"]},
                
                "out_proj": {"dependencies": ["sv_matmul"], "ops": res["out_proj"]["flops"], "access": res["out_proj"]["memory_access"]},
                "attn_add": {"dependencies": ["input", "out_proj"], "ops": res["attn_add"]["flops"], "access": res["attn_add"]["memory_access"]},
                
                "mlp_norm": {"dependencies": ["attn_add"], "ops": res["mlp_norm"]["flops"], "access": res["mlp_norm"]["memory_access"]},
                "gate_proj": {"dependencies": ["mlp_norm"], "ops": res["gate_proj"]["flops"], "access": res["gate_proj"]["memory_access"]},
                "up_proj": {"dependencies": ["mlp_norm"], "ops": res["up_proj"]["flops"], "access": res["up_proj"]["memory_access"]},
                "mlp_silu_dot": {"dependencies": ["up_proj", "gate_proj"], "ops": res["mlp_silu_dot"]["flops"], "access": res["mlp_silu_dot"]["memory_access"]},
                "down_proj": {"dependencies": ["mlp_silu_dot"], "ops": res["down_proj"]["flops"], "access": res["down_proj"]["memory_access"]},
                
                "mlp_add": {"dependencies": ["attn_add", "down_proj"], "ops": res["mlp_add"]["flops"], "access": res["mlp_add"]["memory_access"]},
                
                "output": {"dependencies": ["mlp_add"], "ops": "0", "access": "0"},
            }

            # Initialize the Digraph
            dot = Digraph(format="png", node_attr={"style": "filled", "shape": "box", "fontname": "Arial"})

            # Add nodes and edges
            for node, details in transformer_layer_graph.items():
                # Add the node with operation and access details
                label = f"{node}\nOPs: {details['ops']}, Access: {details['access']}"
                dot.node(node, label=label, fillcolor="lightblue" if "proj" in node else "lightcyan")

                # Add edges based on dependencies
                for dep in details["dependencies"]:
                    dot.edge(dep, node)

            # Save and render the graph
            output_path = f"./figures/{stage}_{self.model_config.model_name}_tp{self.tp_size}_bs{self.b}_seqlen{self.s}_genlen{self.o}_graph_visual"
            dot.render(output_path, cleanup=True)
