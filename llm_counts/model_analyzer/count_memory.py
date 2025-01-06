from ..config import LLMConfigs
from ..constants import BYTES_FP16, ActivationRecomputation
from ..utils import num_to_string

from .count_params import CountCausalLMParams

class CountCausalLMMemory(object):
    """Count memory of the model and layers."""
    def __init__(self, llm_configs: LLMConfigs) -> None:
        self.model_config = llm_configs.model_config
        self.hidden_size = self.model_config.hidden_size
        self.intermediate_size = self.model_config.intermediate_size

        self.num_layers = self.model_config.num_layers
        self.Vocab_size = self.model_config.vocab_size
        self.num_heads = self.model_config.num_heads
        self.num_key_value_heads = self.model_config.num_key_value_heads
        self.head_dim = self.hidden_size / self.num_heads

        self.b = llm_configs.inference_config.bs
        self.s = llm_configs.inference_config.seq_len
        self.o = llm_configs.inference_config.generate_len

        self.bytes_per_param = llm_configs.inference_config.bytes_per_param
        self.act_dtype_bytes = BYTES_FP16

        self.tp_size = llm_configs.parallelism_config.tp_size
        self.pp_size = llm_configs.parallelism_config.pp_size
        self.num_layers_per_gpu = int(self.num_layers / self.pp_size)
        
        self.gpu_memory_in_GB = llm_configs.gpu_config.memory_GPU_in_GB * 10**9  # 单位 GB
        
        self.num_layerslm_params = CountCausalLMParams(self.model_config)
        

    def count_memory_weights(self, embedding_dtype_bytes: int = BYTES_FP16):
        """Get the memory of the model weights"""
        params_per_layer, dict_params_per_layer = self.num_layerslm_params.count_params_per_layer()
        params_embedding = self.num_layerslm_params.count_params_embedding()
        
        memory_weight_per_layer = (
            (params_per_layer / self.tp_size) * self.bytes_per_param
        )
        memory_weight_per_gpu = memory_weight_per_layer *  self.num_layers_per_gpu
        
        memory_embedding = (params_embedding / self.tp_size) * embedding_dtype_bytes
        memory_weight_per_gpu = memory_weight_per_gpu + memory_embedding
        
        return memory_weight_per_gpu
    
    def count_memory_act_per_layer_attn(
        self,
        bs: int,
        seq_len: int,
        is_inference: bool = True,
        qkvo_proj_dtype_bytes = BYTES_FP16,
        flash_attn: bool = False,
        act_recomputation = ActivationRecomputation.NONE
    ) -> float:
        """Count the memory (in bytes) required  to store the acts of the
        attention in a transformer layer, given the batch size, sequence length, 
        whether it is inference or training, the act recomputation strategy, 
        and the act data type.
        
        The `attn` acts include the input to Q/K/V gemm, QK^T matrix multiply,
        softmax, softmax dropout attention over V, the input to the attention output Gemm.
        """
        if act_recomputation == ActivationRecomputation.ATTN_COMPUTE:
            return 0
        if act_recomputation == ActivationRecomputation.NONE:
            if flash_attn:
                # 减少内存开销：
                memory_attn_compute = (2 * seq_len * bs * self.hidden_size +
                                       4 * self.num_heads * seq_len * bs
                                       ) * self.act_dtype_bytes / self.tp_size
            else:
                # 注意力机制需要计算 Query 和 Key 的点积：
                memory_attn_compute = 2 * self.num_heads * seq_len**2 * bs * self.act_dtype_bytes / self.tp_size
        else:
            raise ValueError(
                f'Invalid act_recomputation: {act_recomputation}'
            )
        
        if is_inference:
            return max(
                # 3bsh = 完整激活存储需求：即用于存储输入到注意力层的激活值。 # memory_attn_compute 中间激活计算需求：即用于计算 Query 和 Key 的点积以及 Softmax 等操作的临时激活值存储。
                3 * bs * seq_len * self.hidden_size * qkvo_proj_dtype_bytes /
                self.tp_size, memory_attn_compute)

    def count_memory_act_per_layer_mlp(
        self,
        bs: int,
        seq_len: int,
        is_inference: bool = True,
        mlp_act_dtype_bytes = BYTES_FP16,
    ) -> float:
        """ The `mlp` acts include the input to the two linear layers.
        Refer to https://arxiv.org/abs/2205.05198 for details.
        The two linear layers store their inputs with size 2bsh and 8bsh
        """
        act_gate_proj = bs * seq_len * self.hidden_size * mlp_act_dtype_bytes / self.tp_size
        act_gelu = bs * seq_len * self.intermediate_size * mlp_act_dtype_bytes / self.tp_size
        act_down_proj = bs * seq_len * self.intermediate_size * mlp_act_dtype_bytes / self.tp_size
        
        if is_inference:
            mlp_act_memory_per_gpu = max(act_gate_proj, act_gelu, act_down_proj)
        else:
            mlp_act_memory_per_gpu = act_gate_proj + act_gelu + act_down_proj

        return mlp_act_memory_per_gpu
    
    def count_memory_act_per_layer_rmsnorm(
        self,
        bs: int,
        seq_len: int,
    ) -> float:
        """Get the memory (in bytes) required to store the acts of a single layernorm in a transformer layer.
        """
        rn_act_memory_per_gpu = bs * seq_len * self.hidden_size * self.act_dtype_bytes / self.tp_size
        return rn_act_memory_per_gpu
    
    def count_memory_act_input_embedding(self, bs: int,
                                              seq_len: int) -> float:
        input_embedding_act_memory_per_gpu = bs * seq_len * self.hidden_size * self.act_dtype_bytes / self.tp_size
        return input_embedding_act_memory_per_gpu
    
    def count_memory_act_output_embedding(self, bs: int,
                                               seq_len: int) -> float:
        output_embedding_act_memory_per_gpu = self.Vocab_size * bs * seq_len * self.act_dtype_bytes / self.tp_size
        return output_embedding_act_memory_per_gpu
    
    def count_memory_act_per_layer(
        self,
        bs: int,
        seq_len: int,
        is_inference: bool = True,
        flash_attn = True,
        act_recomputation = ActivationRecomputation.NONE,
        qkvo_proj_dtype_bytes: int = BYTES_FP16,
        mlp_act_dtype_bytes = BYTES_FP16,
    ) -> float:
        
        if is_inference:
            assert act_recomputation == ActivationRecomputation.NONE, f'Inference does not need act recomputation, \
            but got act_recomputation = {act_recomputation}'
        
        act_mem_per_layer_attn = self.count_memory_act_per_layer_attn(
                bs,
                seq_len,
                is_inference,
                qkvo_proj_dtype_bytes=qkvo_proj_dtype_bytes,
                flash_attn=flash_attn,
                act_recomputation=act_recomputation
            )
    
        act_mem_per_layer_mlp = self.count_memory_act_per_layer_mlp(
                bs,
                seq_len,
                is_inference,
                mlp_act_dtype_bytes=mlp_act_dtype_bytes,
            )

        act_mem_per_layer_rn = self.count_memory_act_per_layer_rmsnorm(
                bs,
                seq_len,
            )
        
        if is_inference:
            act_memory_per_layer = max(act_mem_per_layer_attn,
                                       act_mem_per_layer_mlp,
                                       act_mem_per_layer_rn)
        
        return act_memory_per_layer
    
        # act_mem_per_layer_in_embed = self.count_memory_act_input_embedding(
        #         bs,
        #         seq_len,
        #         is_inference,
        #         act_recomputation,
        #     )
        
        # act_mem_per_layer_out_embed = self.count_memory_act_output_embedding(
        #         bs,
        #         seq_len,
        #         is_inference,
        #         act_recomputation,
        #     )
        
    def count_memory_kv_cache_per_layer(
        self,
        bs: int,
        seq_len: int,
        generate_len: int,
        kv_cache_dtype_bytes: int = BYTES_FP16,
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
        num_kv_heads_per_gpu = max(self.num_key_value_heads / self.tp_size, 1)  
        memory_kv_cache_per_layer = bs * (seq_len + generate_len) * num_kv_heads_per_gpu * self.head_dim * 2 * kv_cache_dtype_bytes
        
        return memory_kv_cache_per_layer
    
    def count_memory_per_gpu(
        self, 
        bs: int,
        seq_len: int,
        generate_len: int,
        is_inference: bool = True,
        flash_attn: bool = True,
        use_kv_cache: bool = True,
        act_recomputation = ActivationRecomputation.NONE,
        qkvo_proj_dtype_bytes: int = BYTES_FP16,
        mlp_act_dtype_bytes = BYTES_FP16,
        kv_cache_dtype_bytes: int = BYTES_FP16,
    ) -> tuple:
        
        # 1, prefill stage count memory and max_bs
        weight_memory_per_gpu = self.count_memory_weights() # count model weights memory
        memory_left = self.gpu_memory_in_GB - weight_memory_per_gpu

        # count model acts memory of prefill stage
        prefill_act_memory_bs_1 = self.count_memory_act_per_layer(
            1, seq_len, is_inference, flash_attn,
            act_recomputation=act_recomputation, 
            qkvo_proj_dtype_bytes=qkvo_proj_dtype_bytes,
            mlp_act_dtype_bytes=mlp_act_dtype_bytes,
        )
        
        prefill_max_bs = int(
            memory_left / prefill_act_memory_bs_1
        )

        prefill_act_memory_per_gpu = bs * prefill_act_memory_bs_1
        
        assert memory_left > prefill_act_memory_per_gpu, (
            f"weight_memory_per_gpu {num_to_string(weight_memory_per_gpu)}, \
            prefill stage activations memory {num_to_string(prefill_act_memory_per_gpu)} \
            is too large can't fit in GPU memory! memory_left is {num_to_string(memory_left)}!"
        )
        
        # 2, decode stage count memory and max_bs
        if use_kv_cache:
            kv_cache_memory_bs_1 = (
                self.count_memory_kv_cache_per_layer(
                    1,
                    seq_len + generate_len,
                    kv_cache_dtype_bytes
                )
                * self.num_layers_per_gpu
            )
            
            kv_cache_memory_per_gpu = bs * kv_cache_memory_bs_1
            decode_act_memory_bs_1 = prefill_act_memory_bs_1 / seq_len           
            decode_act_memory_per_gpu = decode_act_memory_bs_1 * bs
 
            decode_max_bs = int(memory_left / (decode_act_memory_bs_1 + kv_cache_memory_bs_1))    
            max_batch_total_tokens = decode_max_bs * (seq_len + generate_len)
                
            assert bs <= decode_max_bs, (
                f"bs {bs} is too large to fit"
                " in GPU memory, decode_max_bs:"
                f" {decode_max_bs}"
            )
            
            assert memory_left > (
                kv_cache_memory_per_gpu + decode_act_memory_per_gpu
            ), ("kv_cache and act memory with bs ="
                f" {bs} is too large to fit in GPU memory"
            )
        else:
            # 上下文长度不再是新生成的那个 token，而是 seq_len + generate_len
            decode_act_memory_bs_1 = (
                self.count_memory_act_per_layer(
                    1, seq_len + generate_len, is_inference, flash_attn,
                    act_recomputation=act_recomputation, 
                    qkvo_proj_dtype_bytes=qkvo_proj_dtype_bytes,
                    mlp_act_dtype_bytes=mlp_act_dtype_bytes,
                )
                * self.num_layers_per_gpu
            )
            decode_max_bs = int(
                memory_left / decode_act_memory_bs_1
            )
            assert bs <= decode_max_bs, (
                f"bs {bs} is too large to fit"
                " in GPU memory, decode_max_bs:"
                f" {decode_max_bs}"
            )
            
            decode_act_memory_per_gpu = bs * decode_act_memory_bs_1
            kv_cache_memory_per_gpu = 0
        
        decode_memory_total = (weight_memory_per_gpu + decode_act_memory_per_gpu + kv_cache_memory_per_gpu)
        
        # memory summary
        memory_prefill_summary_dict = {
            "weight_memory_per_gpu": weight_memory_per_gpu,
            "prefill_act_memory_bs_1": prefill_act_memory_bs_1,
            "prefill_max_bs": prefill_max_bs,
            "prefill_act_memory_per_gpu": prefill_act_memory_per_gpu, 
        }
        
        memory_decode_summary_dict = {
            "weight_memory_per_gpu": weight_memory_per_gpu,
            "decode_act_memory_per_gpu": decode_act_memory_per_gpu,
            "kv_cache_memory_per_gpu": kv_cache_memory_per_gpu,
            "decode_memory_total": decode_memory_total,
            "decode_max_bs": decode_max_bs,
            "max_batch_total_tokens": int(max_batch_total_tokens * 0.97),
        }
        
        return memory_prefill_summary_dict, memory_decode_summary_dict
    
        