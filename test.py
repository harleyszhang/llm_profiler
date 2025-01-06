from llm_counts.model_analyzer.count_memory_access import CountCausalLMMemoryAccess

# 假设与 CountCausalLMMemoryAccess 位于同一个包下测试

class ModelConfig:
    def __init__(self, hidden_size, intermediate_size, num_heads, num_key_value_heads):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads

class GPUConfig:
    # 这里可放置一些 GPU 配置相关的信息，比如 SM 数量 / 主频 / HBM 带宽等
    def __init__(self, name="A100"):
        self.name = name

class LLMConfigs:
    def __init__(self, hidden_size, intermediate_size, num_heads, num_key_value_heads):
        self.model_config = ModelConfig(hidden_size, intermediate_size, num_heads, num_key_value_heads)
        self.gpu_config = GPUConfig()

def test_count_memory_access():
    # 构造一个简易的 LLM 配置
    llm_configs = LLMConfigs(
        hidden_size=512,
        intermediate_size=2048,
        num_heads=8,
        num_key_value_heads=8
    )
    cma = CountCausalLMMemoryAccess(llm_configs)

    # batch_size=1, seq_len=16, generate_len=8 进行测试
    results = cma.count_memory_access(bs=1, seq_len=16, generate_len=8)

    # 简单打印一下结果
    print("===== Decode 阶段结果 =====")
    for k, v in results["decode"].items():
        print(k, v)

    print("===== Prefill 阶段结果 =====")
    for k, v in results["prefill"].items():
        print(k, v)

if __name__ == "__main__":
    test_count_memory_access()
