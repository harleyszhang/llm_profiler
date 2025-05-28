from .utils.config import LLMConfigs
from .utils.constants import BYTES_FP16, ActivationRecomputation


from .count_params import CountCausalLMParams

# --- Helper for byte-count products --- #
from functools import reduce
import operator as _op

def _B(*dims):
    """Utility: multiply arbitrary dimensions to get a byte count."""
    return reduce(_op.mul, dims, 1)


class CountCausalLMMemory(object):
    """Count memory of the model and layers."""

    def __init__(self, llm_configs: LLMConfigs) -> None:
        self.model_config = llm_configs.model_config
        self.model_type = self.model_config.model_type
        self.hidden_size = self.model_config.hidden_size
        self.intermediate_size = self.model_config.intermediate_size

        self.num_layers = self.model_config.num_layers
        self.V = self.model_config.vocab_size
        self.num_heads = self.model_config.num_heads
        self.num_kv_heads = self.model_config.num_kv_heads
        # use integer division to avoid float head dimensions
        self.head_dim = self.hidden_size // self.num_heads

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
        generate_len: int = 0,
        flash_attn: bool = False,
        kv_cache_bytes: int = BYTES_FP16,
    ):
        "self-attention kernle 可应用张量并行操作"
        if self.model_type == "qwen3":
        # --- LayerNorm on Q & K (×2) ------------------------------------ #
            norm_bytes = 2 * (
                _B(self.head_dim, BYTES_FP16)                                  # load γ & β
                + 2 * _B(bs, seq_len, self.head_dim, BYTES_FP16)               # load + store acts
            )
        else:
            norm_bytes = 0

        if not flash_attn:
            if seq_len != 1:
                # dim changge: (bs, seq_len, hidden_size) -> (bs, seq_len, num_heads, head_dim)
                # (bs, seq_len, num_heads, head_dim) -> (bs, num_heads, seq_len, head_dim)
                # qk^t: (bs, num_heads, seq_len, head_dim) *  (bs, num_kv_heads, seq_len, head_sim) -> (bs, num_heads, seq_len, seq_len)
                # sv: (bs, num_heads, seq_len, seq_len) * (bs, num_kv_heads, seq_len, head_dim) -> (bs, num_heads, seq_len, head_dim)

                load_q_mem = bs * self.num_heads * seq_len * self.head_dim
                load_k_mem = bs * self.num_kv_heads * seq_len * self.head_dim
                qk_store_mem = bs * self.num_heads * seq_len * seq_len

                load_softmax_mem = qk_store_mem
                softmax_store_mem = bs * self.num_heads * seq_len * seq_len

                load_s_mem = softmax_store_mem
                load_v_mem = bs * self.num_kv_heads * seq_len * self.head_dim
                sv_store_mem = bs * self.num_heads * seq_len * self.head_dim

                self_atten_mac = (load_q_mem + load_k_mem + qk_store_mem
                                  + load_softmax_mem + softmax_store_mem
                                  + load_s_mem + load_v_mem + sv_store_mem)
                return self_atten_mac * kv_cache_bytes + norm_bytes

            else:
                # dim changge: (bs, 1, hidden_size) -> (bs, 1, num_heads, head_dim)
                # (bs, 1, num_heads, head_dim) -> (bs, num_heads, 1, head_dim)
                # qk^t: (bs, num_heads, seq_len + generate_len, head_dim) *  (bs, num_kv_heads, seq_len + generate_len, head_sim) -> (bs, num_heads, seq_len + generate_len, seq_len + generate_len)
                # sv: (bs, num_heads, seq_len + generate_len, seq_len + generate_len) * (bs, num_kv_heads, seq_len + generate_len, head_dim) -> (bs, num_heads, seq_len + generate_len, head_dim)

                load_q_mem = bs * self.num_heads * 1  * self.head_dim
                load_k_mem = bs * self.num_kv_heads * (seq_len + generate_len) * self.head_dim
                qk_store_mem = bs * self.num_heads * (seq_len + generate_len) * (seq_len + generate_len)

                load_softmax_mem = qk_store_mem
                softmax_store_mem = bs * self.num_heads * (seq_len + generate_len) * (seq_len + generate_len)

                load_s_mem = softmax_store_mem
                load_v_mem = bs * self.num_kv_heads * (seq_len + generate_len) * self.head_dim
                sv_store_mem = bs * self.num_heads * (seq_len + generate_len) * self.head_dim

                self_atten_mac = (load_q_mem + load_k_mem + qk_store_mem
                                  + load_softmax_mem + softmax_store_mem
                                  + load_s_mem + load_v_mem + sv_store_mem)
                
                return self_atten_mac * kv_cache_bytes + norm_bytes
        
    def count_mac_per_layer_kv_cache(
        self,
        bs,
        seq_len,
        generate_len: int = 0,
        flash_attn: bool = False,
        kv_cache_bytes: int = BYTES_FP16,
    ):
        if not flash_attn:
            store_k_cache = (
                self.num_kv_heads * self.head_dim * bs * seq_len * kv_cache_bytes
            )
            store_v_cache = (
                self.num_kv_heads * self.head_dim * bs * seq_len * kv_cache_bytes
            )
            if seq_len != 1:
                return store_k_cache + store_v_cache
            else:
                qk_matmul_load_k_cache = (
                    (seq_len + generate_len)
                    * self.head_dim
                    * bs
                    * self.num_kv_heads
                    * kv_cache_bytes
                )
                sv_matmul_load_v_cache = (
                    (seq_len + generate_len)
                    * self.head_dim
                    * bs
                    * self.num_kv_heads
                    * kv_cache_bytes
                )
            
                kv_cache_mac = (
                    qk_matmul_load_k_cache
                    + sv_matmul_load_v_cache
                    + store_k_cache
                    + store_v_cache
                )
                return kv_cache_mac
        else:
            # FlashAttention path: compute attention on‑the‑fly; only new K/V cache entries are stored
            kv_cache_mac = (
                self.num_kv_heads
                * self.head_dim
                * bs
                * seq_len
                * 2  # K + V
                * kv_cache_bytes
            )

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
        # 4 projections, each load + store  (4 * 2 = 8)
        return 8 * _B(bs, seq_len, self.hidden_size, qkvo_proj_dtype_bytes)

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
        load_hidden = _B(bs, seq_len, self.hidden_size, mlp_act_dtype_bytes)
        load_inter  = _B(bs, seq_len, self.intermediate_size, mlp_act_dtype_bytes)
        # Peak bytes across gate‑proj, GELU and down‑proj
        return max(2 * load_hidden, 2 * load_inter, load_hidden + load_inter)

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
        memory_weight_per_gpu = params_model * self.bytes_per_param / self.tp_size

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
            float: the memory (in bytes) required  to store the key and value cache 
            for a transformer layer in inference.
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

    def count_mac_per_layer(
        self,
        bs: int,
        seq_len_ctx: int,
        generate_len: int = 0,  # used only for decode stage
        *,
        stage: str = "prefill",        # "prefill" | "decode"
        flash_attn: bool = False,
        qkvo_proj_dtype_bytes: int = BYTES_FP16,
        mlp_act_dtype_bytes: int = BYTES_FP16,
    ) -> float:
        # --- sanity ---------------------------------------------------- #
        assert stage in {"prefill", "decode"}

        # For decode stage each step handles just **one token**.
        tokens = 1 if stage == "decode" else seq_len_ctx

        act_mem_per_layer_qkvo_proj = self.count_mac_per_layer_qkvo_proj(
                bs,
                tokens,
                qkvo_proj_dtype_bytes=qkvo_proj_dtype_bytes,
            )

        act_mem_per_layer_mlp = self.count_mac_per_layer_mlp(
                bs,
                tokens,
                mlp_act_dtype_bytes=mlp_act_dtype_bytes,
            )

        act_mem_per_layer_rn = self.count_mac_per_layer_norm(bs, tokens) 

        act_memory_per_layer = max(act_mem_per_layer_qkvo_proj, act_mem_per_layer_mlp, act_mem_per_layer_rn)

        return act_memory_per_layer
    
    def count_memory_per_gpu(
        self,
        bs: int,
        seq_len: int,
        generate_len: int,
        flash_attn: bool = True,
        qkvo_proj_dtype_bytes: int = BYTES_FP16,
        mlp_act_dtype_bytes=BYTES_FP16,
        kv_cache_bytes: int = BYTES_FP16,
    ) -> tuple:
        # 1, prefill stage count memory and max_bs
        weight_memory_per_gpu = self.count_memory_weight_per_gpu()  # count model weights memory
        memory_left_per_gpu = self.gpu_memory_in_GB - weight_memory_per_gpu

        # --- 1) PREFILL stage ----------------------------------------- #
        prefill_act_memory_bs_1 = self.count_mac_per_layer(
            1,
            seq_len,
            generate_len=generate_len,
            stage="prefill",
            flash_attn=flash_attn,
            qkvo_proj_dtype_bytes=qkvo_proj_dtype_bytes,
            mlp_act_dtype_bytes=mlp_act_dtype_bytes,
        )
        prefill_max_bs = int(memory_left_per_gpu / prefill_act_memory_bs_1)
        prefill_act_memory_per_gpu = bs * prefill_act_memory_bs_1

        # --- 2) DECODE stage ------------------------------------------ #
        kv_cache_memory_bs_1 = (self.count_memory_kv_cache_per_layer(1, seq_len, generate_len, kv_cache_bytes) * self.num_layers_per_gpu) / self.tp_size
        kv_cache_memory_per_gpu = bs * kv_cache_memory_bs_1
        
        decode_act_memory_bs_1 = self.count_mac_per_layer(
            1,
            seq_len,
            generate_len=generate_len,
            stage="decode",
            flash_attn=flash_attn,
            qkvo_proj_dtype_bytes=qkvo_proj_dtype_bytes,
            mlp_act_dtype_bytes=mlp_act_dtype_bytes,
        )
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
