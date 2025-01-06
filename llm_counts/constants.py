from enum import Enum
from functools import total_ordering

#########################################
#######     llm profiler    ############
#########################################

FLOPS_EFFICIENCY = 0.9  # FLOPS efficiency achieved by Megatron-LM is ~0.5 for LLM training
HBM_MEMORY_EFFICIENCY = 0.9  # GPU HBM memory efficiency
INTRA_NODE_MEMORY_EFFICIENCY = 0.9  # intra-node (nvlink) memory efficiency
INTER_NODE_MEMORY_EFFICIENCY = 0.9  # inter-node memory efficiency

NUM_GPUS_PER_NODE = 8  # number of GPUs per node

TOLERANCE = 0.01  # tolerance for floating point comparisons

BITS_PER_BYTE = 8  # number of bits in a byte

BITS_FP32 = 32  # number of bits in FP32 data type
BITS_FP16 = 16  # number of bits in FP16 data type
BITS_INT8 = 8  # number of bits in INT8 data type
BITS_INT4 = 4  # number of bits in INT4 data type

BYTES_FP32 = BITS_FP32 // BITS_PER_BYTE  # number of bytes in FP32 data type
BYTES_FP16 = BITS_FP16 // BITS_PER_BYTE  # number of bytes in FP16 data type
BYTES_INT8 = BITS_INT8 // BITS_PER_BYTE  # number of bytes in INT8 data type
BYTES_INT4 = BITS_INT4 // BITS_PER_BYTE  # number of bytes in INT4 data type

PRINT_LINE_WIDTH = 100

GPUS = [1, 2, 4, 8]

@total_ordering
class ActivationRecomputation(Enum):
    NONE = 0
    """No activation recomputation; requires the most amount of memory."""
    ATTN_COMPUTE = 1
    """Selectively checkpoints the attention computation (QK^T matrix multiply, softmax, softmax dropout, and attention over
    V.) in the attention module of a transformer layer;
    this part takes up a considerable amount of memory but are not computationally expensive to
    recompute"""
    ATTN = 2
    """Selectively checkpoints the input to the attention module in a transformer layer; requires an extra forward pass on attention."""
    NORM_ATTN_NORM = 3
    """Selectively checkpoints the input to the sequence of modules (layernom-attention-layernom) in a transformer layer; requires an extra forward pass on (layernom-attention-layernom)."""
    FULL = 4
    """Full activation recomputation stores the input to the transformer layer; requires the least
    amount of memory; requires an extra forward pass of the layer."""

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented