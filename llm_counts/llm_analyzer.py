from .utils.config import LLMConfigs
from .utils.constants import BYTES_FP16
from .utils.config import *
from .roofline_model import roofline_analysis
from .utils.utils import num_to_string


class LLMAnalyzer(object):
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
            "q_proj": [self.hidden_size, self.hidden_size],
            "k_proj": [self.hidden_size, self.hidden_size * self.num_kv_heads / self.num_heads],
            "v_proj": [self.hidden_size, self.hidden_size * self.num_kv_heads / self.num_heads],
            "out_proj": [self.hidden_size, self.hidden_size],
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
            # 1, qkt kernel analysis
            name = "qk_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                flops=2 * seq_len * seq_len * self.head_dim * bs * self.num_heads,
                load_weight=0,
                load_act= 2 * bs * seq_len * hidden_size* act_byte, # load q and k act, shape is [s, h]
                store_act=seq_len * seq_len * bs * num_heads * act_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
            # load q and k, k is form kv cache
            self._analyze_to_results(
                "decode",
                name,
                flops=2 * bs * num_heads * head_dim * (seq_len + generate_len),
                load_weight=0,
                load_act=bs * seq_len * hidden_size * act_byte,
                store_act=bs * (seq_len + generate_len) * num_kv_heads * head_dim * act_byte,
                load_kv_cache=bs * (seq_len + generate_len) * num_kv_heads * head_dim * kv_byte,
                store_kv_cache=0,
            )
            # 2, sv kernel analysis
            name = "sv_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                flops=bs * 2 * seq_len * seq_len * head_dim * num_heads,
                load_weight=0,
                load_act=2 * bs * seq_len * seq_len * act_byte, # load score(qkt) act, shape is [s, s]
                store_act=bs * seq_len * hidden_size * act_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

            self._analyze_to_results(
                "decode",
                name,
                flops=2 * bs * num_heads * head_dim * (seq_len + generate_len),
                load_weight=0,
                load_act=bs * (seq_len + generate_len) * act_byte, # load score(qkt) act, shape is [1, s+o]
                store_act=bs * seq_len * num_heads * head_dim * act_byte,
                load_kv_cache=bs * (seq_len + generate_len) * num_kv_heads * head_dim * kv_byte,
                store_kv_cache=0,
            )

            # 3, softmax kernel analysis
            name = f"softmax"
            self._analyze_to_results(
                "prefill",
                name,
                flops= (bs * num_heads * seq_len * seq_len * 1 * 5),
                load_weight=0,
                load_act=bs * self.num_heads * seq_len * seq_len * kv_byte,
                store_act=bs * self.num_heads * seq_len * seq_len * kv_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

            self._analyze_to_results(
                "decode",
                name,
                flops= (bs * num_heads * (seq_len + generate_len) * 1 * 5) ,
                load_weight=0,
                load_act=bs * self.num_heads * (seq_len + generate_len)  * 1 * kv_byte,
                store_act=bs * self.num_heads * (seq_len + generate_len)  * 1 * kv_byte,
                load_kv_cache=0,
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
    
    @staticmethod
    def create_layer_graph(model_type, results, base_path):
        from graphviz import Digraph

        stage_list = ["prefill", "decode"]
        for stage in stage_list:
            res = results[stage]
            # Define the transformer layer graph with operation and memory info
            llama_layer_graph = {
                "input": {
                    "dependencies": [],
                    "ops": "0",
                    "access": "0",
                    "bound": "N/A",
                },
                "attn_norm": {
                    "dependencies": ["input"],
                    "ops": res["attn_norm"]["flops"],
                    "access": res["attn_norm"]["memory_access"],
                    "params": res["attn_norm"]["load_weight"],
                    "bound": res["attn_norm"]["bound"],
                },
                "q_proj": {
                    "dependencies": ["attn_norm"],
                    "ops": res["q_proj"]["flops"],
                    "access": res["q_proj"]["memory_access"],
                    "params": res["q_proj"]["load_weight"],
                    "bound": res["q_proj"]["bound"],
                },
                "k_proj": {
                    "dependencies": ["attn_norm"],
                    "ops": res["k_proj"]["flops"],
                    "access": res["k_proj"]["memory_access"],
                    "params": res["k_proj"]["load_weight"],
                    "bound": res["k_proj"]["bound"],
                },
                "v_proj": {
                    "dependencies": ["attn_norm"],
                    "ops": res["v_proj"]["flops"],
                    "access": res["v_proj"]["memory_access"],
                    "params": res["v_proj"]["load_weight"],
                    "bound": res["v_proj"]["bound"],
                },
                "qk_matmul": {
                    "dependencies": ["q_proj", "k_proj"],
                    "ops": res["qk_matmul"]["flops"],
                    "access": res["qk_matmul"]["memory_access"],
                    "params": res["qk_matmul"]["load_weight"],
                    "bound": res["qk_matmul"]["bound"],
                },
                "softmax": {
                    "dependencies": ["qk_matmul"],
                    "ops": res["softmax"]["flops"],
                    "access": res["softmax"]["memory_access"],
                    "params": res["softmax"]["load_weight"],
                    "bound": res["softmax"]["bound"],
                },
                "sv_matmul": {
                    "dependencies": ["softmax", "v_proj"],
                    "ops": res["sv_matmul"]["flops"],
                    "access": res["sv_matmul"]["memory_access"],
                    "params": res["sv_matmul"]["load_weight"],
                    "bound": res["sv_matmul"]["bound"],
                },
                "out_proj": {
                    "dependencies": ["sv_matmul"],
                    "ops": res["out_proj"]["flops"],
                    "access": res["out_proj"]["memory_access"],
                    "params": res["out_proj"]["load_weight"],
                    "bound": res["out_proj"]["bound"],
                },
                "attn_add": {
                    "dependencies": ["input", "out_proj"],
                    "ops": res["attn_add"]["flops"],
                    "access": res["attn_add"]["memory_access"],
                    "params": res["attn_add"]["load_weight"],
                    "bound": res["attn_add"]["bound"],
                },
                "mlp_norm": {
                    "dependencies": ["attn_add"],
                    "ops": res["mlp_norm"]["flops"],
                    "access": res["mlp_norm"]["memory_access"],
                    "params": res["mlp_norm"]["load_weight"],
                    "bound": res["mlp_norm"]["bound"],
                },
                "gate_proj": {
                    "dependencies": ["mlp_norm"],
                    "ops": res["gate_proj"]["flops"],
                    "access": res["gate_proj"]["memory_access"],
                    "params": res["gate_proj"]["load_weight"],
                    "bound": res["gate_proj"]["bound"],
                },
                "up_proj": {
                    "dependencies": ["mlp_norm"],
                    "ops": res["up_proj"]["flops"],
                    "access": res["up_proj"]["memory_access"],
                    "params": res["up_proj"]["load_weight"],
                    "bound": res["up_proj"]["bound"],
                },
                "mlp_silu_dot": {
                    "dependencies": ["up_proj", "gate_proj"],
                    "ops": res["mlp_silu_dot"]["flops"],
                    "access": res["mlp_silu_dot"]["memory_access"],
                    "params": res["mlp_silu_dot"]["load_weight"],
                    "bound": res["mlp_silu_dot"]["bound"],
                },
                "down_proj": {
                    "dependencies": ["mlp_silu_dot"],
                    "ops": res["down_proj"]["flops"],
                    "access": res["down_proj"]["memory_access"],
                    "params": res["down_proj"]["load_weight"],
                    "bound": res["down_proj"]["bound"],
                },
                "mlp_add": {
                    "dependencies": ["attn_add", "down_proj"],
                    "ops": res["mlp_add"]["flops"],
                    "access": res["mlp_add"]["memory_access"],
                    "params": res["mlp_add"]["load_weight"],
                    "bound": res["mlp_add"]["bound"],
                },
                "output": {"dependencies": ["mlp_add"], "ops": "0", "access": "0"},
            }

            if model_type == "qwen3":
                llama_layer_graph["q_norm"] = {
                    "dependencies": ["q_proj"],
                    "ops": res["q_norm"]["flops"],
                    "access": res["q_norm"]["memory_access"],
                    "params": res["q_norm"]["load_weight"],
                    "bound": res["q_norm"]["bound"],
                }
                llama_layer_graph["k_norm"] = {
                    "dependencies": ["k_proj"],
                    "ops": res["k_norm"]["flops"],
                    "access": res["k_norm"]["memory_access"],
                    "params": res["k_norm"]["load_weight"],
                    "bound": res["k_norm"]["bound"],
                }
                llama_layer_graph["qk_matmul"] = {
                    "dependencies": ["q_norm", "k_norm"],
                    "ops": res["qk_matmul"]["flops"],
                    "access": res["qk_matmul"]["memory_access"],
                    "params": res["qk_matmul"]["load_weight"],
                    "bound": res["qk_matmul"]["bound"],
                }

            # Initialize the Digraph
            dot = Digraph(
                format="png",
                node_attr={"style": "filled", "shape": "box", "fontname": "Arial"},
            )

            # Add nodes and edges
            for node, details in llama_layer_graph.items():
                # Add the node with operation and access details
                label = f"{node}\nOPs: {details['ops']}, Access: {details['access']}, \nParams: {details.get('params', 0)}, Bound: {details.get('bound', 'N/A')}"
                dot.node(
                    node,
                    label=label,
                    fillcolor="lightblue" if "proj" in node else "lightcyan",
                )

                # Add edges based on dependencies
                for dep in details["dependencies"]:
                    dot.edge(dep, node)

            graph_path = f"./figures/grpah_{stage}" + base_path
            dot.render(graph_path, cleanup=True)
