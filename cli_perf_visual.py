from llm_counts.utils.config import *
from llm_counts.benchmark_analyzer import llm_profile
import math


####################################################################################################################
def print_list(list):
    """print one-dimensional list

    :param list: List[int]
    :return: None
    """
    for _, x in enumerate(list):
        print(x, end="\n")

####################################################################################################################
def print_all_llm_analyzer():
    model_name_list = [
        "llama-7b",
        "llama-13b",
        "llama-65b",
        "llama2-70b",
        "internlm-20b",
    ]
    gpu_name_list = [
        "a30-sxm-24gb",
        "a40-pcie-48gb",
        "a100-sxm-40gb",
        "a100-sxm-80gb",
        "910b-64gb",
        "v100-sxm-32gb",
        "t4-pcie-15gb",
    ]
    tp_nums_list = [1, 2, 4, 8]
    tgi_service_dict_list = []
    seq_len, generate_len = 1024, 1024

    for model_name in model_name_list:
        if model_name in ["llama2-70b", "internlm-20b"]:
            seq_len, generate_len = 1024, 1024

        for gpu_name in gpu_name_list:
            for tp_size in tp_nums_list:
                try:
                    res1, _ = llm_profile(
                        model_name=model_name,
                        gpu_name=gpu_name,
                        tp_size=tp_size,
                        seq_len=seq_len,
                        generate_len=generate_len,
                        print_flag=False,
                        visual_flag=False,
                    )
                    max_batch_total_tokens = int(res1["max_batch_total_tokens"])
                except Exception as e:
                    print(
                        f"model_name: {model_name}, gpu_name: {gpu_name}, tp_size: {tp_size}, error: {e}"
                    )
                    continue

                tgi_service_dict = {
                    "model_name": model_name,
                    "gpu_name": gpu_name,
                    "tp_size": tp_size,
                    "max_batch_total_tokens": max_batch_total_tokens,
                    "max_bs": math.floor(
                        max_batch_total_tokens / (seq_len + generate_len)
                    ),
                }
                tgi_service_dict_list.append(tgi_service_dict)

    print(
        "============================ TGI+LightLLM service max_batch_total_tokens params list ======================"
    )
    print_list(tgi_service_dict_list)

if __name__ == "__main__":
    # llm_profile(model_name="llama-7b", tp_size=1, print_flag=True, visual_flag=True)
    llm_profile(model_name="Qwen3-30B-A3B", gpu_name = "a100-sxm-40gb", tp_size=8, 
                batch_size = 32, seq_len = 1024, generate_len=128,
                print_flag=True, visual_flag=True)
