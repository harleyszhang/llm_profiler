from llm_counts.llm_analyzer import LLMAnalyzer
from llm_counts.utils.utils import *
from llm_counts.utils.config import get_model_and_gpu_config_by_name
import pprint

def print_format_summary_dict(summary_dict: dict, depth: int) -> str:
    """打印时对 params / flops / latency / memory 等进行统一转换显示。"""
    for key, value in summary_dict.items():
        if "params" in key or "flops" in key:
            if not isinstance(value, dict):
                summary_dict.update({key: num_to_string(value)})
            else:
                print_format_summary_dict(
                    value, get_dict_depth(value) - 1
                )  # 递归
        if "latency" in key:
            if not isinstance(value, dict):
                summary_dict.update({key: latency_to_string(value)})
            else:
                print_format_summary_dict(value, get_dict_depth(value) - 1)
        if "memory" in key:
            if not isinstance(value, dict):
                summary_dict.update({key: f"{num_to_string(value)}B"})
            else:
                print_format_summary_dict(value, get_dict_depth(value) - 1)
    if depth >= 1:
        pprint.pprint(summary_dict, indent=4, sort_dicts=False)

def test_llm_analyzer(
        model_name: str = "Qwen/Qwen3-8B",
        gpu_name="a100-sxm-80gb",
        bs: int = 1,
        seq_len: int = 522,
        generate_len: int = 1526,
        tp_size: int = 1,
    ):
    model_config, gpu_config = get_model_and_gpu_config_by_name(model_name, gpu_name)
    model_type = model_config.model_type
    llm_analyzer = LLMAnalyzer(model_config, gpu_config, tp_size=tp_size)
    results = llm_analyzer.analyze_model(bs=bs, seq_len=seq_len, generate_len=generate_len)

    # -------------------------- 绘图：模型 graph 图示例 --------------------------
    base_path = f"_{model_name}_tp{tp_size}_bs{bs}_seqlen{seq_len}_genlen{generate_len}.png"
    llm_analyzer.create_layer_graph(model_type, results, base_path)
    depth = get_dict_depth(results)
    print_format_summary_dict(results, depth)


if __name__ == "__main__":
    test_llm_analyzer(
        model_name="Qwen3-32B",
        gpu_name="a100-sxm-80gb",
        bs=1,
        seq_len=522,
        generate_len=1526,
        tp_size=1,
    )
    # test_llm_analyzer(
    #     model_name="Qwen/Qwen3-14B",
    #     gpu_name="a100-sxm-80gb",
    #     bs=1,
    #     seq_len=522,
    #     generate_len=1526,
    #     tp_size=1,
    # )