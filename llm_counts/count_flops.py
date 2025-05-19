from .utils.config import ModelConfig
from .utils.constants import TOLERANCE
from .utils.utils import within_range
from .count_params import CountCausalLMParams


class CountCausalLMFlops(object):
    """The count is model-specific and does not depend on the parallelism strategy.
    And ignore layer normalization and other element-wise operations."""

    def __init__(
        self,
        model_config: ModelConfig,
        bs: int,
        seq_len: int,
        tp_size: int,
        simp_count=False,
    ) -> None:
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        self.l = model_config.num_layers
        self.V = model_config.vocab_size

        self.b = bs
        self.s = seq_len
        self.tp_size = tp_size

        if not simp_count:
            llm_params = CountCausalLMParams(model_config)
            self.model_flops = llm_params(self.hidden_size, self.l, self.V) * 2

    def count_flops_per_layer_qkvo_proj(self, bs: int, seq_len: int) -> int:
        """Get the number of floating point operations (flops) for the forward
        pass of the attention module in a transformer layer, given the batch
        size and sequence length.

        mainly including four linear calculations: query/key/value projection and output
        matrices multiplication、self-attention internal operation, and element-wise operations are ignored.

        flops_qkvo_proj = flops_q + flops_k + flops_v + flops_output + flops_self_attention
              = 4(bsh^2) + 2(2bs^2h)
        Args:
            bs (int): batch size
            seq_len (int): sequence length

        Returns:
            int: flops for the forward pass of the attention module in a transformer layer
        """
        q_proj_flops = 2 * bs * seq_len * self.hidden_size * self.hidden_size
        k_proj_flops = 2 * bs * seq_len * self.hidden_size * self.hidden_size
        v_proj_flops = 2 * bs * seq_len * self.hidden_size * self.hidden_size
        o_proj_flops = 2 * bs * seq_len * self.hidden_size * self.hidden_size
        qkvo_proj_flops = q_proj_flops + k_proj_flops + v_proj_flops + o_proj_flops

        return qkvo_proj_flops

    def count_flops_per_layer_attn_kernel(self, bs: int, seq_len: int) -> int:
        qk_matmul_flops = bs * 2 * seq_len * seq_len * self.hidden_size
        sv_matmul_flops = bs * 2 * seq_len * seq_len * self.hidden_size
        softmax_flops = (
            bs * 3 * seq_len * self.hidden_size
        )  # e^x / sum(e^x); bs = 1 和 seq_len = 1 时 flops 为 3d-1, 张量中每个元素约执行 3 次操作

        flops_self_attention_kernel = qk_matmul_flops + sv_matmul_flops + softmax_flops

        return flops_self_attention_kernel

    def count_flops_per_layer_mlp(self, bs: int, seq_len: int) -> int:
        """Count two flops of matrices multiplication(two linear layers in the MLP module.)
        eg. llama3.2-1B: self.intermediate_size = 4 * self.hidden_size
        eg. flops_mlp(llama3.2-1B) = flops_fc1 + flops_fc2 + flops_fc3 = 2bs(4h^2) + 2bs(4h^2) + 2bs(4h^2) = 24bsh^2
        """
        flops_gate_proj = 2 * bs * seq_len * self.hidden_size * self.intermediate_size
        flops_up_proj = 2 * bs * seq_len * self.hidden_size * self.intermediate_size
        flops_down_proj = 2 * bs * seq_len * self.intermediate_size * self.hidden_size

        return flops_gate_proj + flops_up_proj + flops_down_proj

    def count_flops_per_layer_rmsnorm(self, bs: int, seq_len: int) -> int:
        """flops of 2 rmsnrom per layer"""
        return bs * 4 * seq_len * self.hidden_size

    def count_flops_per_layer(self, bs: int, seq_len: int, ln_ignore=True) -> tuple:
        flops_per_layer_qkvo_proj = self.count_flops_per_layer_qkvo_proj(bs, seq_len)
        flops_per_layer_mlp = self.count_flops_per_layer_mlp(bs, seq_len)
        flops_per_layer_attention_kernel = self.count_flops_per_layer_attn_kernel(
            bs, seq_len
        )
        flops_per_layer_rmsnorm = (
            self.count_flops_per_layer_rmsnorm(bs, seq_len) * 2
        )  # atten_rmsnorm mlp_rmsnorm
        flops_positional_embedding = self.count_flops_positional_embedding()

        flops_per_layer = (
            flops_per_layer_qkvo_proj
            + flops_per_layer_mlp
            + flops_per_layer_rmsnorm
            + flops_per_layer_attention_kernel
        )

        # 默认计算 prefill 阶段的计算量
        dict_flops_per_layer = {
            "attention_kernel": flops_per_layer_attention_kernel,
            "qkvo_proj": flops_per_layer_qkvo_proj,
            "mlp": flops_per_layer_mlp,
            "rmsnorm": flops_per_layer_rmsnorm * 2,
            "positional_embedding": flops_positional_embedding,
            "input_embedding": 0,
        }

        return flops_per_layer, dict_flops_per_layer

    def count_flops_positional_embedding(
        self,
    ) -> int:
        """flops of output token logits layer"""
        return 2 * self.b * self.s * self.hidden_size

    def count_flops_model(self, bs: int, seq_len: int) -> int:
        """Count flops of the forward pass of the transformer model, given the batch size and sequence length."""
        num_flops_model = (
            self.count_flops_per_layer(bs, seq_len)[0] * self.l
            + self.count_flops_positional_embedding()
        )

        return num_flops_model

    def count_flops_bwd_model(self, bs: int, seq_len: int) -> int:
        """Get the number of floating point operations (flops) for the backward
        pass of the entire transformer model, given the batch size and sequence"""
        return 2 * self.count_flops_model(bs, seq_len)
