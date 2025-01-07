from .utils.config import LLMConfigs
from .utils.constants import BYTES_FP16
from .utils.config import *
from .roofline_model import roofline_analysis
from .utils.utils import num_to_string

class LLMAnalyzer(object):
    """Count memory access of the model and layers."""
    def __init__(self, llm_configs: LLMConfigs) -> None:
        self.model_config = llm_configs.model_config
        self.gpu_config = llm_configs.gpu_config

        self.hidden_size = self.model_config.hidden_size
        self.intermediate_size = self.model_config.intermediate_size
        self.num_heads = self.model_config.num_heads
        self.num_key_value_heads = self.model_config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.tp_size = llm_configs.parallelism_config.tp_size

        self.results = {"decode": {}, "prefill": {}}
        self.llama_layers = {
            "q_proj":[self.hidden_size, self.hidden_size],
            "k_proj":[self.hidden_size, self.hidden_size*self.num_key_value_heads/self.num_heads],
            "v_proj":[self.hidden_size, self.hidden_size*self.num_key_value_heads/self.num_heads],
            "out_proj":[self.hidden_size, self.hidden_size],
            "gate_proj":[self.hidden_size, self.intermediate_size],
            "up_proj":[self.hidden_size, self.intermediate_size],
            "down_proj":[self.intermediate_size, self.hidden_size],
        }

    def get_hardware_info(self, data_type="fp16"):
        gpu_hbm_bandwidth = get_gpu_hbm_bandwidth(self.gpu_config) * 10**9   # 单位 GB/s
        gpu_max_ops = get_TFLOPS_per_gpu(self.gpu_config, data_type=data_type) * 10**12 # 单位 TFLOPS

        return gpu_hbm_bandwidth, gpu_max_ops
    
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

        bandwidth, gpu_max_ops = self.get_hardware_info()
        memory_access = (load_weight + load_act + store_act + load_kv_cache + store_kv_cache)

        arithmetic_intensity, performance, bound = roofline_analysis(
            gpu_max_ops, bandwidth, flops, memory_access
        )

        self.results[stage][kernel_name] = {
            "flops": num_to_string(flops),
            "memory_access": f"{num_to_string(memory_access)}B",
            "arithmetic_intensity": int(arithmetic_intensity),
            "performance": num_to_string(performance),
            "bound": bound,
            "load_weight": f"{num_to_string(load_weight)}B",
            "load_act": num_to_string(load_act),
            "store_act": num_to_string(store_act),
            "load_kv_cache": num_to_string(load_kv_cache),
            "store_kv_cache": num_to_string(store_kv_cache), 
        }
        
        return self.results
    
    def count_memory_access(
        self, 
        bs: int,
        seq_len: int,
        generate_len: int,
        flash_attn: bool = False,
        linear_weight_bytes: int = BYTES_FP16,
        linear_act_bytes: int = BYTES_FP16,
        kv_cache_bytes: int = BYTES_FP16,
    ) -> tuple:
        # stage = "prefill" if seq_len > 1 else
        for name, (in_ch, out_ch) in self.llama_layers.items():
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            self._analyze_to_results(
                "decode",
                name,
                flops=2*bs*in_ch*out_ch//self.tp_size,
                load_weight=in_ch * out_ch * linear_weight_bytes//self.tp_size,
                load_act=in_ch * bs * linear_act_bytes//self.tp_size,
                store_act=0 if is_kv_proj else out_ch * bs * linear_act_bytes//self.tp_size,
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else out_ch * bs * kv_cache_bytes)//self.tp_size,
            )

            self._analyze_to_results(
                "prefill",
                name,
                flops=2*bs*seq_len * in_ch *out_ch//self.tp_size,
                load_weight=in_ch * out_ch * linear_weight_bytes//self.tp_size,
                load_act=in_ch * bs * seq_len * linear_act_bytes//self.tp_size,
                store_act=0 if is_kv_proj else out_ch * bs * seq_len * linear_act_bytes//self.tp_size,
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else out_ch * bs * seq_len*kv_cache_bytes)//self.tp_size,
            )
 
        # 标准 self-attention 会执行 kv repeat 操作
        qk_matmul_flops = bs * 2 * seq_len * seq_len * self.hidden_size
        qk_matmul_load_act = bs * seq_len * self.num_key_value_heads * kv_cache_bytes # 加载 q 和 k 张量激活
        qk_matmul_store_act = bs * seq_len * seq_len * self.num_heads * kv_cache_bytes # 存储 qk^t 结果
        qk_matmul_load_kv_cache=(seq_len+generate_len) * self.head_dim * bs * self.num_key_value_heads * kv_cache_bytes

        sv_matmul_flops = bs * 2 * seq_len * seq_len * self.hidden_size
        sv_matmul_load_act = bs * seq_len * seq_len * self.num_heads * kv_cache_bytes
        sv_matmul_store_act = bs * seq_len * self.head_dim * self.num_heads * kv_cache_bytes
        sv_matmul_load_kv_cache=(seq_len+generate_len) * self.head_dim * bs * self.num_key_value_heads * kv_cache_bytes
        
        softmax_flops =  bs * 3 * seq_len * self.hidden_size # e^x / sum(e^x); bs = 1 和 seq_len = 1 时 flops 为 3d-1, 张量中每个元素约执行 3 次操作
        softmax_loat_act = bs * self.num_heads * seq_len * seq_len * kv_cache_bytes
        softmax_store_act = bs * self.num_heads * seq_len * seq_len * kv_cache_bytes

        # rms_norm = \gamma * (x/(rms(x))), rms(x) = (\sumx^2) /d + eps. 一种结合了逐元素操作和归一化操作的算子
        rmsnorm_flops = bs * 4 * seq_len * self.hidden_size # mlp_norm, attn_norm
        rmsnorm_load_weight = self.hidden_size * BYTES_FP16
        rmsnorm_load_act = bs * seq_len * self.hidden_size * BYTES_FP16
        rmsnorm_store_act = bs * seq_len * self.hidden_size * BYTES_FP16

        # silu 和 dot * 都是纯逐元素操作算子
        silu_dot_flops = bs * 4 * seq_len * self.intermediate_size # 每个张量元素执行 4 次操作
        silu_dot_load_act = bs * 2 * seq_len * self.intermediate_size * linear_act_bytes
        silu_dot_store_act = bs * 2 * seq_len * self.intermediate_size * linear_act_bytes

        mlp_add_flops = bs * seq_len * self.hidden_size
        mlp_add_load_act =  bs * seq_len * self.hidden_size * linear_act_bytes
        mlp_add_store_act = bs * seq_len * self.hidden_size * linear_act_bytes

        # other kernels (memory bound)
        kernels = ["qk_matmul", "sv_matmul", "softmax", "attn_norm", "mlp_norm","mlp_silu_dot", "attn_add", "mlp_add"]
        flops_list = [qk_matmul_flops, sv_matmul_flops, softmax_flops, rmsnorm_flops, rmsnorm_flops, silu_dot_flops, mlp_add_flops, mlp_add_flops]
        load_act_list = [qk_matmul_load_act, sv_matmul_load_act, softmax_loat_act, rmsnorm_load_act, rmsnorm_load_act, silu_dot_load_act, mlp_add_load_act, mlp_add_load_act]
        store_act_list = [qk_matmul_store_act, sv_matmul_store_act, softmax_store_act, rmsnorm_store_act, rmsnorm_store_act, silu_dot_store_act, mlp_add_store_act, mlp_add_store_act]
        
        # prefill 阶段
        for stage in ["prefill", "decode"]:
            for i, kernel_name in enumerate(kernels):
                load_weight = 0 if (kernel_name not in ["attn_norm", "mlp_norm"]) else rmsnorm_load_weight

                load_act = load_act_list[i]
                store_act = store_act_list[i]
                flops = flops_list[i]

                if stage == "decode":
                    load_act = int(load_act // seq_len)
                    store_act = int(store_act // seq_len)
                    flops = int(flops // seq_len)
                
                load_kv_cache = sv_matmul_load_kv_cache if (kernel_name in ["qk_matmul", "sv_matmul"] and stage == "decode") else stage == "prefill"
        
                self._analyze_to_results(
                    stage,
                    kernel_name,
                    flops=flops//self.tp_size ,
                    load_weight=load_weight//self.tp_size,
                    load_act=load_act//self.tp_size ,
                    store_act=store_act//self.tp_size ,
                    load_kv_cache=load_kv_cache//self.tp_size,
                    store_kv_cache=0,
                )
        # 返回累积分析结果
        return self.results
        # total_results = {"decode": {}, "prefill": {}}