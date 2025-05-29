from .utils.config import ModelConfig
from .utils.constants import *


class CountCausalLMParams(object):
    def __init__(self, model_config: ModelConfig) -> None:
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        self.num_layers = model_config.num_layers
        self.V = model_config.vocab_size

        self.num_heads = model_config.num_heads
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = model_config.head_dim
        self.model_type = model_config.model_type

    def count_params_embedding(self, shared_embedding: bool = True) -> int:
        """Get the number of parameters in the embedding layer. 
        params_te = vocab_size * d_model
        Args:
            shared_embedding (bool, optional):  whether the output embedding 
            shares weights with the input embedding. Defaults to True.

        Returns: 
            int: the number of parameters in the embedding layer.
        """
        num_params_input_embedding = self.V * self.hidden_size
        num_params_output_embedding = (
            self.V * self.hidden_size if not shared_embedding else 0
        )

        return num_params_input_embedding + num_params_output_embedding

    def count_params_per_layer_mha(self) -> int:
        """Get the number of parameters per layer in the attention module
        which include 4 linear layer: q/k/v/o linear layers.

        Returns:
            int: the number of parameters per layer in the attention module(mha)
        """
        params_qo_proj = 2 * self.hidden_size * self.num_heads * self.head_dim
        params_kv_proj = 2 * self.hidden_size * self.num_kv_heads * self.head_dim
        return params_qo_proj + params_kv_proj

    def count_params_per_layer_mlp(self) -> int:
        """Get the number of parameters in the MLP linear layers, including the
        intermediate and output matrices.
        params_mlp = params_gate_proj + params_up_proj + params_down_proj
        Returns:
            int: the number of parameters in the two MLP linear layers
        """
        params_gate_proj = self.hidden_size * self.intermediate_size
        params_up_proj = self.hidden_size * self.intermediate_size
        params_down_proj = self.intermediate_size * self.hidden_size
        params_mlp = params_gate_proj + params_up_proj + params_down_proj

        return params_mlp

    def count_params_per_layer_norm(self) -> int:
        """Get the number of atten_norm and mlp_norm parameters per layer.
        """
        # q_norm、k_norm、atten_norm、mlp_norm
        if self.model_type == "qwen3":
            return 2 * self.hidden_size + 2 * self.head_dim
        else:
            return 2 * self.hidden_size

    def count_params_per_layer(self, norm_ignore=False) -> tuple:
        """Get the number of params per layer mainly including the attention and MLP layers.

        params_per_layer = params_mha + params_mlp + params_norm

        """
        params_per_layer_mha = self.count_params_per_layer_mha()
        params_per_layer_mlp = self.count_params_per_layer_mlp()
        params_per_layer_norm = 0 if norm_ignore else self.count_params_per_layer_norm()
        params_input_embedding = self.count_params_embedding()

        params_per_layer = (
            params_per_layer_mha + params_per_layer_mlp + params_per_layer_norm
        )

        dict_params_per_layer = {
            "qkvo_proj": params_per_layer_mha,
            "mlp": params_per_layer_mlp,
            "rmsnorm": params_per_layer_norm,
            "input_embedding": params_input_embedding,
            "output_embedding": params_input_embedding,
        }

        return params_per_layer, dict_params_per_layer

    def count_params_model(self) -> int:
        """Get the total number of parameters in the model 
        including all layers and token embedding layer.
        params_model = params_embedding + params_per_layer * num_layers
                    = V * d_model + 12 * d_model**2 * num_layers
        Returns:
            int: the total number of parameters in the model
        """
        params_per_layer, _ = self.count_params_per_layer()
        params_model = (
            params_per_layer * self.num_layers + self.count_params_embedding()
        )

        return params_model