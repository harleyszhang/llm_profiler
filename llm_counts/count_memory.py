from .utils.config import LLMConfigs
from .utils.constants import BYTES_FP16, ActivationRecomputation
from .utils.utils import num_to_string

from .count_params import CountCausalLMParams


class CountCausalLMMemory(object):
    """Count memory of the model and layers."""

    def __init__(self, llm_configs: LLMConfigs) -> None:
        self.model_config = llm_configs.model_config
        self.hidden_size = self.model_config.hidden_size
        self.intermediate_size = self.model_config.intermediate_size

        self.num_layers = self.model_config.num_layers
        self.V = self.model_config.vocab_size
        self.num_heads = self.model_config.num_heads
        self.num_kv_heads = self.model_config.num_kv_heads
        self.head_dim = self.hidden_size / self.num_heads

        self.b = llm_configs.inference_config.bs
        self.s = llm_configs.inference_config.seq_len
        self.o = llm_configs.inference_config.generate_len

        self.bytes_per_param = llm_configs.inference_config.bytes_per_param
        self.act_dtype_bytes = BYTES_FP16

        self.tp_size = llm_configs.parallelism_config.tp_size
        self.pp_size = llm_configs.parallelism_config.pp_size
        self.num_layers_per_gpu = int(self.num_layers / self.pp_size)

        self.gpu_memory_in_GB = llm_configs.gpu_config.memory_GPU_in_GB * 10**9
        self.llm_params = CountCausalLMParams(self.model_config)

    def count_mac_per_layer_attn_kernel(
        self,
        bs: int,
        seq_len,
        flash_attn: bool = False,
        kv_cache_bytes: int = BYTES_FP16,
    ):
        "self-attention kernle 可应用张量并行操作"
        q_norm_load_weight = self.head_dim * BYTES_FP16
        q_norm_load_act = bs * seq_len * self.head_dim * BYTES_FP16 # equal k_norm_load_act
        q_norm_store_act = bs * seq_len * self.head_dim * BYTES_FP16

        qk_matmul_load_act = (
            bs * seq_len * self.num_kv_heads * kv_cache_bytes
        )  # 加载 q 和 k 张量激活
        qk_matmul_store_act = (
            bs * seq_len * seq_len * self.num_heads * kv_cache_bytes
        )  # 存储 qk^t 结果
        sv_matmul_load_act = bs * seq_len * seq_len * self.num_heads * kv_cache_bytes
        sv_matmul_store_act = (
            bs * seq_len * self.head_dim * self.num_heads * kv_cache_bytes
        )
        softmax_loat_act = bs * self.num_heads * seq_len * seq_len * kv_cache_bytes
        softmax_store_act = bs * self.num_heads * seq_len * seq_len * kv_cache_bytes

        self_attn_kernel_mac = (
            q_norm_load_weight*2
            + q_norm_load_act*2
            + q_norm_store_act*2
            + qk_matmul_load_act
            + qk_matmul_store_act
            + sv_matmul_load_act
            + sv_matmul_store_act
            + softmax_loat_act
            + softmax_store_act
        )

        return self_attn_kernel_mac

    def count_mac_per_layer_kv_cache(
        self,
        bs,
        seq_len,
        generate_len,
        flash_attn: bool = False,
        kv_cache_bytes: int = BYTES_FP16,
    ):
        if not flash_attn:
            qk_matmul_load_k_cache = (
                (self.s + generate_len)
                * self.head_dim
                * bs
                * self.num_kv_heads
                * kv_cache_bytes
            )
            sv_matmul_load_v_cache = (
                (self.s + generate_len)
                * self.head_dim
                * bs
                * self.num_kv_heads
                * kv_cache_bytes
            )
            store_k_cache_k_linear = (
                self.num_kv_heads * self.head_dim * bs * seq_len * kv_cache_bytes
            )
            store_v_cache_v_linear = (
                self.num_kv_heads * self.head_dim * bs * seq_len * kv_cache_bytes
            )

            if seq_len == 1:
                kv_cache_mac = (
                    qk_matmul_load_k_cache
                    + sv_matmul_load_v_cache
                    + store_k_cache_k_linear
                    + store_v_cache_v_linear
                )
            else:
                kv_cache_mac = store_k_cache_k_linear + store_v_cache_v_linear
        return kv_cache_mac

    def count_mac_per_layer_qkvo_proj(
        self,
        bs: int,
        seq_len: int,
        qkvo_proj_dtype_bytes=BYTES_FP16,
    ) -> float:
        """Count the memory (in bytes) required  to store the acts of the
        attention in a transformer layer, given the batch size, sequence length,
        whether it is inference or training, the act recomputation strategy,
        and the act data type.

        The `attn` acts include the input to Q/K/V gemm, QK^T matrix multiply,
        softmax, softmax dropout attention over V, the input to the attention output Gemm.
        """
        # qkv_proj_load_act = 3bsh = 完整激活存储需求：即用于存储输入到注意力层的激活值。
        qkvo_proj_load_act = (
            4 * bs * seq_len * self.hidden_size * qkvo_proj_dtype_bytes
        )
        qkvo_proj_store_act = (
            4 * bs * seq_len * self.hidden_size * qkvo_proj_dtype_bytes
        )
        qkvo_proj_mac = qkvo_proj_load_act + qkvo_proj_store_act
        return qkvo_proj_mac

    def count_mac_per_layer_mlp(
        self,
        bs: int,
        seq_len: int,
        mlp_act_dtype_bytes=BYTES_FP16,
    ) -> float:
        """The `mlp` acts include the input to the two linear layers.
        Refer to https://arxiv.org/abs/2205.05198 for details.
        The two linear layers store their inputs with size 2bsh and 8bsh
        """
        load_act_gate_proj = bs * seq_len * self.hidden_size * mlp_act_dtype_bytes
        store_act_gate_proj = bs * seq_len * self.hidden_size * mlp_act_dtype_bytes

        load_act_gelu = bs * seq_len * self.intermediate_size * mlp_act_dtype_bytes
        store_act_gelu = bs * seq_len * self.intermediate_size * mlp_act_dtype_bytes

        load_act_down_proj = bs * seq_len * self.intermediate_size * mlp_act_dtype_bytes
        store_act_down_proj = bs * seq_len * self.hidden_size * mlp_act_dtype_bytes

        mlp_act_memory_per_gpu = max(
            load_act_gate_proj + store_act_gate_proj,
            load_act_gelu + store_act_gelu,
            load_act_down_proj + store_act_down_proj,
        )

        return mlp_act_memory_per_gpu

    def count_mac_per_layer_norm(
        self,
        bs: int,
        seq_len: int,
    ) -> float:
        """Get the memory (in bytes) required to store the acts of a single layernorm in a transformer layer."""
        rmsnorm_load_weight = self.hidden_size * self.act_dtype_bytes
        rmsnorm_load_act = bs * seq_len * self.hidden_size * self.act_dtype_bytes
        rmsnorm_store_act = bs * seq_len * self.hidden_size * self.act_dtype_bytes

        rn_mac_per_gpu = (
            rmsnorm_load_weight + rmsnorm_load_act + rmsnorm_store_act
        )

        return rn_mac_per_gpu

    def count_mac_input_embedding(self, bs: int, seq_len: int) -> float:
        input_embedding_load_act = bs * seq_len * self.act_dtype_bytes
        input_embedding_store_act = (
            bs * seq_len * self.hidden_size * self.act_dtype_bytes
        )
        input_embedding_mac_per_gpu = (
            input_embedding_load_act + input_embedding_store_act
        )

        return input_embedding_mac_per_gpu

    def count_memory_weight_per_gpu(self, ):
        """Get the memory of the model weights"""
        params_model = self.llm_params.count_params_model()
        memory_weight_per_gpu = round(params_model * self.bytes_per_param / self.tp_size, 3)

        return memory_weight_per_gpu

    def count_memory_kv_cache_per_layer(
        self,
        bs: int,
        seq_len: int,
        generate_len: int,
        kv_cache_bytes: int = BYTES_FP16,
    ) -> float:
        """Get the memory (in bytes) required to store the key and value cache
        for a transformer layer in inference, given the batch size, sequence
        length, act data type, and tensor parallelism size.

        memory_kv_cache = 4blh(s+o) unit is byte
        Args:
            bs (int): batch size
            context_len (int): seq_len + generate_len

        Returns:
            float: the memory (in bytes) required  to store the key and value cache for a transformer layer in inference
        """

        # At least on attention head on each tensor-parallel GPU
        num_kv_heads_per_gpu = max(self.num_kv_heads, 1)
        memory_kv_cache_per_layer = (
            bs
            * (seq_len + generate_len)
            * num_kv_heads_per_gpu
            * self.head_dim
            * 2
            * kv_cache_bytes
        )

        return memory_kv_cache_per_layer

    def count_memory_act_per_layer(
        self,
        bs: int,
        seq_len: int,
        is_inference: bool = True,
        flash_attn=True,
        act_recomputation=ActivationRecomputation.NONE,
        qkvo_proj_dtype_bytes: int = BYTES_FP16,
        mlp_act_dtype_bytes=BYTES_FP16,
    ) -> float:
 
        assert act_recomputation == ActivationRecomputation.NONE, (
            f"Inference does not need act recomputation, \
        but got act_recomputation = {act_recomputation}"
        )

        act_mem_per_layer_qkvo_proj = (
            self.count_mac_per_layer_qkvo_proj(
                bs,
                seq_len,
                qkvo_proj_dtype_bytes=qkvo_proj_dtype_bytes,
            )
            / 2
        )

        act_mem_per_layer_mlp = (
            self.count_mac_per_layer_mlp(
                bs,
                seq_len,
                mlp_act_dtype_bytes=mlp_act_dtype_bytes,
            )
            / 2
        )

        act_mem_per_layer_rn = (self.count_mac_per_layer_norm(bs, seq_len) / 2)

        act_memory_per_layer = max(
            act_mem_per_layer_qkvo_proj, act_mem_per_layer_mlp, act_mem_per_layer_rn
        )

        return act_memory_per_layer
    
    def count_memory_per_gpu(
        self,
        bs: int,
        seq_len: int,
        generate_len: int,
        is_inference: bool = True,
        flash_attn: bool = True,
        act_recomputation=ActivationRecomputation.NONE,
        qkvo_proj_dtype_bytes: int = BYTES_FP16,
        mlp_act_dtype_bytes=BYTES_FP16,
        kv_cache_bytes: int = BYTES_FP16,
    ) -> tuple:
        # 1, prefill stage count memory and max_bs
        weight_memory_per_gpu = self.count_memory_weight_per_gpu()  # count model weights memory
        memory_left_per_gpu = self.gpu_memory_in_GB - weight_memory_per_gpu

        prefill_act_memory_bs_1 = self.count_memory_act_per_layer(
            1,
            seq_len,
            is_inference,
            flash_attn,
            act_recomputation=act_recomputation,
            qkvo_proj_dtype_bytes=qkvo_proj_dtype_bytes,
            mlp_act_dtype_bytes=mlp_act_dtype_bytes,
        )

        prefill_max_bs = int(memory_left_per_gpu / prefill_act_memory_bs_1)
        prefill_act_memory_per_gpu = bs * prefill_act_memory_bs_1

        # 2, decode stage count memory and max_bs
        kv_cache_memory_bs_1 = (self.count_memory_kv_cache_per_layer(1, seq_len + generate_len, kv_cache_bytes) * self.num_layers_per_gpu) / self.tp_size

        kv_cache_memory_per_gpu = bs * kv_cache_memory_bs_1
        decode_act_memory_bs_1 = prefill_act_memory_bs_1 / seq_len
        decode_act_memory_per_gpu = decode_act_memory_bs_1 * bs

        decode_max_bs = int(memory_left_per_gpu / (decode_act_memory_bs_1 + kv_cache_memory_bs_1))
        max_batch_total_tokens = decode_max_bs * (seq_len + generate_len)

        assert bs <= decode_max_bs, (
            f"bs {bs} is too large to fit"
            " in GPU memory, decode_max_bs:"
            f" {decode_max_bs}"
        )

        assert memory_left_per_gpu > (
            kv_cache_memory_per_gpu + decode_act_memory_per_gpu
        ), (
            "kv_cache and act memory with bs ="
            f" {bs} is too large to fit in GPU memory"
        )
        
        consume_memory_per_gpu = (
            weight_memory_per_gpu + decode_act_memory_per_gpu + kv_cache_memory_per_gpu
        )

        # memory summary
        memory_prefill_summary_dict = {
            "weight_memory_per_gpu": weight_memory_per_gpu,
            "prefill_max_bs": prefill_max_bs,
            "prefill_act_memory_per_gpu": prefill_act_memory_per_gpu,
        }

        memory_decode_summary_dict = {
            "decode_act_memory_per_gpu": decode_act_memory_per_gpu,
            "kv_cache_memory_per_gpu": kv_cache_memory_per_gpu,
            "consume_memory_per_gpu": consume_memory_per_gpu,
            "decode_max_bs": decode_max_bs,
            "max_batch_total_tokens": int(max_batch_total_tokens * 0.97),
        }

        return memory_prefill_summary_dict, memory_decode_summary_dict
