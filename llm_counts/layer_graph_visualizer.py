"""
cli entry point for LayerAnalyzer, which analyzes the memory access and FLOPs of a model.
Usage:
    ```bash
    python -m llm_counts.llm_analyzer \
    --result-json path/to/results.json \
    --model-type qwen3 \
    --output my_layer_graph
```
"""
from .utils.constants import BYTES_FP16
from .utils.config import *
from .utils.utils import num_to_string
from .roofline_model import roofline_analysis


class LayerAnalyzer(object):
    """Count memory access of the model and layers."""

    def __init__(self, model_config,  gpu_config, tp_size) -> None:
        self.tp_size = tp_size
        self.bandwidth, self.onchip_buffer = get_gpu_hbm_bandwidth(gpu_config) # GB/s
        self.bandwidth *= 10**9 
        self.gpu_max_ops = get_TFLOPS_per_gpu(gpu_config, data_type="fp16") * 10**12  # TFLOPs
    
        self.model_type = model_config.model_type
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        self.num_heads = model_config.num_heads
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads

        # attention linear layers
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
        data_type="fp16"
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
    

# ---------------------------------------------------------------------------
# Transformer‑layer graph visualisation
# ---------------------------------------------------------------------------

_DEPENDENCIES = {
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

class LayerGraphVisualizer:
    """Render a transformer layer’s roofline‑analysis graph as a PNG."""

    def __init__(self, model_type: str, results: dict, shapes: dict = None) -> None:
        self.model_type = model_type
        self.results = results
        if model_type == "qwen3":
            # qwen3 模型中有额外的 q_norm 和 k_norm 层
            _DEPENDENCIES["q_norm"] = ["q_proj"]
            _DEPENDENCIES["k_norm"] = ["k_proj"]
        # self.shapes = shapes or {}          # optional {kernel: "B×S×C"} mapping

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #
    def _label(self, node: str, kernel_stats: dict) -> str:
        """Build a neat multi‑line Graphviz label, optionally with shape info."""
        label = f"{node}\nFlops: {kernel_stats['flops']}, Access: {kernel_stats['memory_access']}, \nParams: {kernel_stats.get('load_weight', 0)}, Bound: {kernel_stats.get('bound', 'N/A')}"
        return label

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def render(self, base_path: str = "layer_graph") -> None:
        """Generate one PNG per stage (prefill / decode) under ./figures/."""
        from graphviz import Digraph

        for stage, stage_res in self.results.items():
            dot = Digraph(
                format="png",
                node_attr={"style": "filled", "shape": "box", "fontname": "Arial"},
            )

            # Only include nodes and deps relevant for this stage, but always include "input" and "output"
            pruned_deps = {
                n: [d for d in deps if d in stage_res or d in ("input","output")]
                for n, deps in _DEPENDENCIES.items()
                if n in stage_res or n in ("input","output")
            }

            for node, deps in pruned_deps.items():
                color = (
                    "lightblue" if "proj" in node
                    else "plum" if "matmul" in node
                    else "lightcyan"
                )
                if node in stage_res:
                    label = self._label(node, stage_res[node])
                else:
                    # default zero stats for input/output
                    label = (
                        f"{node}\n"
                        "Flops: 0, Access: 0\n"
                        "Params: 0, Bound: N/A"
                    )
                dot.node(node, label=label, fillcolor=color)
                for dep in deps:
                    if dep in pruned_deps:
                        dot.edge(dep, node)
            graph_path = f"./figures/grpah_{stage}_{base_path}"
            dot.render(graph_path, cleanup=True)

# ---------------------------------------------------------------------------
# Command‑line entry‑point
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse, json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate a transformer layer graph (Graphviz PNG) from "
                    "an LayerAnalyzer result JSON."
    )
    parser.add_argument("--result-json", type=Path, required=True,
                        help="Path to the analysis‑result JSON produced by LayerAnalyzer")
    parser.add_argument("--model-type", required=True,
                        help="Model type tag, e.g. 'llama' or 'qwen3'")
    parser.add_argument("--output", default="layer_graph",
                        help="Base filename for the generated PNG(s)")
    args = parser.parse_args()

    with args.result_json.open() as fp:
        results = json.load(fp)

    LayerGraphVisualizer(args.model_type, results).render(args.output)

if __name__ == "__main__":  # pragma: no cover
    _main()