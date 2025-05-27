from llm_counts.utils.config import *
from llm_counts.utils.utils import num_to_string
from llm_counts.visual_analyzer import LLMAnalyzerVisual
import time
import pandas as pd
import matplotlib.pyplot as plt
import math


####################################################################################################################
def runTime(func):
    """decorator: print the cost time of run function"""

    def wapper(arg, *args, **kwargs):
        start = time.time()
        res = func(arg, *args, **kwargs)
        end = time.time()
        print("=" * 80)
        print("function name: %s" % func.__name__)
        print("run time: %.4fs" % (end - start))
        print("=" * 80)
        return res

    return wapper
####################################################################################################################
def print_list(list):
    """print one-dimensional list

    :param list: List[int]
    :return: None
    """
    for i, x in enumerate(list):
        print(x, end="\n")
####################################################################################################################
def llm_profile(
    model_name,
    gpu_name: str = "a100-sxm-40gb",
    bytes_per_param: int = BYTES_FP16,
    bs: int = 20,
    seq_len: int = 1024,
    generate_len=1024,
    dp_size: int = 1,
    tp_size: int = 8,
    pp_size: int = 1,
    sp_size: int = 1,
    use_kv_cache: bool = True,
    act_dtype_bytes: int = BYTES_FP16,
    kv_cache_bytes: int = BYTES_FP16,
    flops_efficiency: float = FLOPS_EFFICIENCY,
    hbm_memory_efficiency: float = HBM_MEMORY_EFFICIENCY,
    intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY,
    inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY,
    print_flag: bool = False,
    visual_flag: bool = False,
) -> dict:
    """Returns dict of the total floating-point operations, MACs, parameters and latency of a llm.
    It now returns a dictionary containing FLOPs, latency, HBM memory usage and max_batch_total_tokens.

    Args:
        model_name (str, optional): model name to query the pre-defined `model_configs.json`. Defaults to "llama-13b".
        gpu_name (str, optional): gpu name to query the pre-defined `model_configs.json`. Defaults to "v100-sxm2-32gb".
        bs (int, optional): _description_. Defaults to 1.
        seq_len (int, optional): batch size per GPU.. Defaults to 522.
        generate_len (int, optional): The maximum numbers of tokens to generate, 
            ignoring the number of tokens in the prompt. Defaults to 1526.
        dp_size (int, optional): data parallelism size. Defaults to 1.
        tp_size (int, optional): tensor parallelism size. Defaults to 1.
        pp_size (int, optional): pipeline parallelism size. Defaults to 1.
        sp_size (int, optional): sequence parallelism size. Defaults to 1.
        use_kv_cache (bool, optional): Whether or not the model should use the 
            past last key/values attentions (if applicable to the model) to speed up decoding. Defaults to True.
        layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations..
            Defaults to BYTES_FP16.
        kv_cache_bytes (int, optional): number of bytes in the data type for the kv_cache. Defaults to None.
        flops_efficiency (float, optional): flops efficiency, ranging from 0 to 1. Defaults to None.
        hbm_memory_efficiency (float, optional): GPU HBM memory efficiency, ranging from 0 to 1. Defaults to HBM_MEMORY_EFFICIENCY.
        intra_node_memory_efficiency (_type_, optional): intra-node memory efficiency, ranging from 0 to 1.. Defaults to INTRA_NODE_MEMORY_EFFICIENCY.
        inter_node_memory_efficiency (_type_, optional): inter-node memory efficiency, ranging from 0 to 1.. Defaults to INTER_NODE_MEMORY_EFFICIENCY.

    Returns:
        dict: a summary dictionary of the inference analysis
    """
    model_config, gpu_config = get_model_and_gpu_config_by_name(model_name, gpu_name)

    parallelism_config = ParallelismConfig(
        tp_size=tp_size, pp_size=pp_size, dp_size=dp_size, sp_size=sp_size
    )

    inference_config = InferenceConfig(
        bs=bs,
        seq_len=seq_len,
        generate_len=generate_len,
        use_kv_cache=use_kv_cache,
        bytes_per_param=bytes_per_param,
        act_dtype_bytes=act_dtype_bytes,
        kv_cache_bytes=kv_cache_bytes,
    )

    gpu_efficiency_config = GPUEfficiencyConfig(
        flops_efficiency=flops_efficiency,
        hbm_memory_efficiency=hbm_memory_efficiency,
        intra_node_memory_efficiency=intra_node_memory_efficiency,
        inter_node_memory_efficiency=inter_node_memory_efficiency,
    )

    llm_configs = LLMConfigs(
        model_config=model_config,
        gpu_config=gpu_config,
        parallelism_config=parallelism_config,
        inference_config=inference_config,
        gpu_efficiency_config=gpu_efficiency_config,
    )

    profiler = LLMAnalyzerVisual(llm_configs)

    infer_result_dict = profiler.infer_profile(
        bs=bs,
        seq_len=seq_len,
        generate_len=generate_len,
        use_kv_cache=use_kv_cache,
        act_dtype_bytes=act_dtype_bytes,
        flops_efficiency=flops_efficiency,
        hbm_memory_efficiency=hbm_memory_efficiency,
        print_flag=print_flag,
    )

    # ---------------------------------------------------------------------
    # Collect summary metrics (keep raw numbers for downstream maths)      #
    # ---------------------------------------------------------------------
    weight_memory_per_gpu = infer_result_dict.get("weight_memory_per_gpu", None)
    consume_memory_per_gpu = infer_result_dict.get("consume_memory_per_gpu", None)

    prefill_flops = infer_result_dict.get("prefill_flops", None)
    
    table_results = {
        "seq_len": seq_len,
        "generate_len": generate_len,
        "prefill_flops": num_to_string(prefill_flops),           
        "weight_memory_per_gpu": num_to_string(weight_memory_per_gpu),
        "consume_memory_per_gpu": num_to_string(consume_memory_per_gpu),       
        "TTFT": infer_result_dict.get("prefill_first_token_latency", None),
        "TTOT": infer_result_dict.get("decode_per_token_latency", None),
        "Total_latency": infer_result_dict.get("total_infer_latency", None),
    }
    visual_results = {
        "seq_len": seq_len,
        "generate_len": generate_len,
        "prefill_flops": prefill_flops,            # raw number
        "weight_memory_per_gpu": weight_memory_per_gpu,
        "consume_memory_per_gpu": consume_memory_per_gpu,           # raw bytes
        "TTFT": infer_result_dict.get("prefill_first_token_latency", None),
        "TTOT": infer_result_dict.get("decode_per_token_latency", None),
        "Total_latency": infer_result_dict.get("total_infer_latency", None),
    }
    return table_results, visual_results


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
        "================================== TGI+LightLLM service max_batch_total_tokens params list ============================="
    )
    print_list(tgi_service_dict_list)


# ========================================================================== #
# New utilities: sweep sequence‑lengths & visualise                          #
# ========================================================================== #
def plot_seq_len_sweep(
    df: pd.DataFrame,
    model_name: str,
    gpu_name: str,
    *,
    output_dir: str = "figures",
    flops_unit: str = "TFLOPs",
    mem_unit: str = "GiB",
    dpi: int = 300,
    show: bool = False,
) -> None:
    """Visualise how sequence length affects compute, latency and memory.

    The function produces **two** publication‑quality PNG files and (optionally)
    shows them interactively:

    ├─ {model_name}_{gpu_name}_flops_latency_vs_seq_len.png
    └─ {model_name}_{gpu_name}_memory_vs_seq_len.png

    Args
    ----
    df : DataFrame
        Must contain the numeric columns:
        ``seq_len``, ``prefill_flops``, ``TTOT``, ``consume_memory_per_gpu``.
    model_name : str
    gpu_name   : str
    output_dir : str, optional
        Directory to save the figures (created if missing).
    flops_unit : {"GFLOPs","TFLOPs","PFLOPs"}, optional
        Preferred engineering unit for FLOPs axis.
    mem_unit   : {"MiB","GiB"}, optional
        Preferred engineering unit for memory axis.
    dpi        : int, optional
        Resolution for saved PNGs.
    show       : bool, optional
        If *True*, call ``plt.show()`` after saving.

    Notes
    -----
    *   Uses ``constrained_layout=True`` for automatic spacing.
    *   Combines legends from both y‑axes into a single, neat entry.
    *   Applies Matplotlib 3.6+ built‑in seaborn theme (``seaborn-v0_8-paper``).
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np

    # ------------------------------------------------------------------ #
    # Prepare output directory & style                                   #
    # ------------------------------------------------------------------ #
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-paper")  # modern seaborn style
    plt.rcParams.update({"axes.spines.right": False})  # cleaner look

    # Ensure df sorted by seq_len for consistent line ordering
    df_sorted = df.sort_values("seq_len")

    # Engineering‑unit scaling helpers --------------------------------- #
    _scale_map = {
        "GFLOPs": 1e9,
        "TFLOPs": 1e12,
        "PFLOPs": 1e15,
        "MiB": 2**20,
        "GiB": 2**30,
    }
    flops_div = _scale_map.get(flops_unit, 1.0)
    mem_div = _scale_map.get(mem_unit, 1.0)

    # ------------------------------------------------------------------ #
    # 1) FLOPs + latency dual‑axis plot                                  #
    # ------------------------------------------------------------------ #
    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)

    ax1.set_xlabel("Sequence length (tokens)")
    ax1.set_ylabel(f"Prefill {flops_unit}", color="tab:blue")
    ax1.plot(
        df_sorted["seq_len"],
        df_sorted["prefill_flops"] / flops_div,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label=f"Prefill {flops_unit}",
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Total_latency Latency (s)", color="tab:red")
    ax2.plot(
        df_sorted["seq_len"],
        df_sorted["Total_latency"],
        marker="s",
        linestyle="-.",
        linewidth=2,
        color="tab:red",
        label="Total_latency Latency",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Combined legend --------------------------------------------------- #
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.suptitle(
        f"{model_name} on {gpu_name}\nFLOPs & Latency vs Sequence Length",
        fontsize=12,
    )
    fig.savefig(
        Path(output_dir)
        / f"{model_name}_{gpu_name}_flops_latency_vs_seq_len.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2) Memory usage plot                                               #
    # ------------------------------------------------------------------ #
    fig_mem, ax_mem = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax_mem.plot(
        df_sorted["seq_len"],
        df_sorted["consume_memory_per_gpu"] / mem_div,
        marker="d",
        linewidth=2,
        color="tab:green",
        label=f"Total HBM ({mem_unit})",
    )

    ax_mem.set_xlabel("Sequence length (tokens)")
    ax_mem.set_ylabel(f"Total consume HBM memory per gpu ({mem_unit})")
    ax_mem.set_title(
        f"{model_name} on {gpu_name}\nMemory vs Sequence Length", fontsize=12
    )
    ax_mem.grid(True, linestyle="--", alpha=0.3)
    ax_mem.legend(loc="upper left")

    fig_mem.savefig(
        Path(output_dir)
        / f"{model_name}_{gpu_name}_memory_vs_seq_len.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close(fig_mem)

    # ------------------------------------------------------------------ #
    # 3) Nicely formatted console table                                  #
    # ------------------------------------------------------------------ #
    summary_df = df_sorted.copy()
    summary_df["prefill_flops"] = (
        summary_df["prefill_flops"] / flops_div
    ).map("{:,.2f}".format)
    summary_df["consume_memory_per_gpu"] = (
        summary_df["consume_memory_per_gpu"] / mem_div
    ).map("{:,.2f}".format)

    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)


def sweep_seq_len(model_name, gpu_name="h100-sxm-80gb", bs=16, generate_len=1024, tp_size=2, seq_len_list=None, **kwargs):
    """Profile a model over several sequence lengths and print / plot a table.

    Args:
        model_name (str): name of the LLM
        gpu_name (str): target GPU
        seq_len_list (List[int]): list of sequence lengths to test
        **kwargs: forwarded to llm_profile
    Returns:
        pandas.DataFrame: one row per sequence length with metrics
    """
    if seq_len_list is None:
        seq_len_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 35456, 37960]

    records1 = []
    records2 = []
    for seq in seq_len_list:
        res1, res2 = llm_profile(
            model_name=model_name,
            gpu_name=gpu_name,
            bs=bs,
            seq_len=seq,
            generate_len=generate_len,
            tp_size=tp_size,
            print_flag=False,
            visual_flag=False,
        )
        print("=" * 80)
        print(f"model_name: {model_name}, gpu_name: {gpu_name}, tp_size: {tp_size}, batch_size: {bs}, seq_len: {seq}, generate_len: {generate_len}")

        records1.append(res1)
        records2.append(res2)

    df1 = pd.DataFrame(records1)
    print("=" * 80)
    print(df1.to_string(index=False))
    print("=" * 80)

    df2 = pd.DataFrame(records2)
    # Visualise the results using *plot_seq_len_sweep*
    if kwargs.get("visual_flag", True):
        plot_seq_len_sweep(df2, model_name, gpu_name)

    return df1


if __name__ == "__main__":
    # llm_profile(model_name="llama-7b", tp_size=1, print_flag=True, visual_flag=True)
    # llm_profile(model_name="llama2-70b", tp_size=8, print_flag=True, visual_flag=False)

    # Example: sweep different sequence lengths and visualise
    sweep_seq_len(model_name="Qwen3-32B", gpu_name="a100-sxm-80gb", 
                  generate_len=2048, bs=32, tp_size=4, visual_flag=True)
