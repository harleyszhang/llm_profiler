from .utils.config import ModelConfig
from .utils.constants import *

class CountCausalLMParams(object):
    def __init__(self, model_config: ModelConfig) -> None:
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        self.num_layers = model_config.num_layers
        self.Vocab_size = model_config.vocab_size
        
        self.num_key_value_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim
        self.model_config = model_config
        
    def count_params_embedding(self, shared_embedding: bool = True) -> int:
        """Get the number of parameters in the embedding layer. params_te = vocab_size * d_model
        Args:
            shared_embedding (bool, optional):  whether the output embedding \
                shares weights with the input embedding. Defaults to True.

        Returns: 
            int: the number of parameters in the embedding layer
        """
        num_params_input_embedding = self.Vocab_size * self.hidden_size
        num_params_output_embedding = self.Vocab_size * self.hidden_size if not shared_embedding else 0
        
        return num_params_input_embedding + num_params_output_embedding
    
    def count_params_per_layer_mha(self) -> int:
        """Get the number of parameters per layer in the attention module 
        which include 4 linear layer: q_proj/k_proj/v_proj/o_proj shape is [h, h]
        params_mha(mha) = params_q + params_k + params_v + params_o = 4 * d_model**2

        Returns:
            int: the number of parameters per layer in the attention module(mha)
        """
        params_qo_proj = 2 * self.hidden_size * self.hidden_size
        params_kv_proj = 2 * self.hidden_size * self.num_key_value_heads * self.head_dim
        return (params_qo_proj + params_kv_proj)
    
    def count_params_per_layer_mlp(self) -> int:
        """Get the number of parameters in the MLP linear layers, including the
        intermediate and output matrices.
        params_mlp = params_gate_proj + params_up_proj + params_down_proj
        Returns:
            int: the number of parameters in the two MLP linear layers
        """
        params_gate_proj= self.hidden_size * self.intermediate_size
        params_up_proj = self.hidden_size * self.intermediate_size
        params_down_proj = self.intermediate_size * self.hidden_size
        params_mlp = params_gate_proj + params_up_proj + params_down_proj
        
        return params_mlp
    
    def count_params_per_layer_rn(self, dtype=BYTES_FP16) -> int:
        """Get the number of parameters per layer in the two layer normalization module.
        params_rn = 4 * d_model
        
        Returns:
            int: the number of parameters per layer in the two layer normalization module
        """
        return 2 * self.hidden_size
    
    def count_params_per_layer(self, ln_ignore=False) -> tuple:
        """Get the number of params per layer in the transformer decoder blocks,
        mainly including the attention and MLP layers
        
        params_per_layer = params_mha + params_mlp + params_rn 
                         = 4d_model^2 + 8d_model^2 + 2*4d_model = 12d_model^2 + 8d_model
        
        Return:
            int: the number of params per layer in the transformer decoder blocks
        """
        params_per_layer_mha = self.count_params_per_layer_mha()
        params_per_layer_mlp = self.count_params_per_layer_mlp()
        params_per_layer_rn = 0 if ln_ignore else self.count_params_per_layer_rn()
        params_input_embedding = self.count_params_embedding()

        params_per_layer = (
            params_per_layer_mha
            + params_per_layer_mlp
            + params_per_layer_rn
        )
                
        dict_params_per_layer = {
            "qkvo_proj": params_per_layer_mha,
            "mlp": params_per_layer_mlp,
            "rmsnorm": params_per_layer_rn,
            "input_embedding": params_input_embedding,
            "output_embedding": params_input_embedding,
        }
        
        return params_per_layer, dict_params_per_layer
                
    def count_params_model(self) -> int:
        """Get the total number of parameters in the model including all layers and token embedding layer.
        params_model = params_embedding + params_per_layer * num_layers 
                    = V * d_model + 12 * d_model**2 * num_layers
        Returns:
            int: the total number of parameters in the model
        """
        params_per_layer, dict_params_per_layer = self.count_params_per_layer()
        params_model = params_per_layer * self.num_layers + self.count_params_embedding()
        
        return params_model
        
    def __call__(self, hidden_size, num_layers, vocab_size) -> int:
        """Simplified estimation of model parameters"""
        return (vocab_size * hidden_size 
                + 12 * hidden_size ** 2 * num_layers
            )

