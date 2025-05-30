from llm_counts.layer_graph_visualizer import LayerAnalyzer, LayerGraphVisualizer
from llm_counts.utils.utils import *
from llm_counts.utils.config import get_model_and_gpu_config_by_name
import pprint
import argparse


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
    llm_analyzer = LayerAnalyzer(model_config, gpu_config, tp_size=tp_size)
    results = llm_analyzer.analyze_model(bs=bs, seq_len=seq_len, generate_len=generate_len)

    # -------------------------- 绘图：模型 graph 图示例 --------------------------
    base_filename = f"{model_name.replace('/', '_')}_tp{tp_size}_bs{bs}_seqlen{seq_len}_genlen{generate_len}"
    print("base_filename", base_filename)
    LayerGraphVisualizer(model_type, results).render(base_filename)
    depth = get_dict_depth(results)
    # print_format_summary_dict(results, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LayerAnalyzer, print a formatted summary, "
                    "and generate per‑stage layer‑graph PNGs."
    )
    parser.add_argument("--model-name", default="Qwen3-32B")
    parser.add_argument("--gpu-name", default="a100-sxm-80gb")
    parser.add_argument("--bs",          type=int, default=16)
    parser.add_argument("--seq-len",     type=int, default=1024)
    parser.add_argument("--generate-len",type=int, default=128)
    parser.add_argument("--tp-size",     type=int, default=4)
    args = parser.parse_args()

    test_llm_analyzer(
        model_name=args.model_name,
        gpu_name=args.gpu_name,
        bs=args.bs,
        seq_len=args.seq_len,
        generate_len=args.generate_len,
        tp_size=args.tp_size,
    )

""""
python cli_structure_analyzer.py \
  --model-name llama2-70B \
  --gpu-name  a100-sxm-80gb \
  --bs 16 \
  --seq-len 1024 \
  --generate-len 128 \
  --tp-size 4
"""