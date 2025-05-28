from llm_counts.utils.config import *
from llm_counts.visual_analyzer import llm_profile
import pandas as pd

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
    """
    Visualise how sequence length affects compute, latency and memory.

    Generated PNGs:

        ├─ {model_name}_{gpu_name}_flops_vs_seq_len.png
        ├─ {model_name}_{gpu_name}_latency_vs_seq_len.png
        └─ {model_name}_{gpu_name}_memory_vs_seq_len.png
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------ #
    # Preparation & styling                                              #
    # ------------------------------------------------------------------ #
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"axes.spines.right": False})

    df = df.sort_values("seq_len")

    _scale = {
        "GFLOPs": 1e9,
        "TFLOPs": 1e12,
        "PFLOPs": 1e15,
        "MiB": 2**20,
        "GiB": 2**30,
    }
    flops_div = _scale.get(flops_unit, 1.0)
    mem_div = _scale.get(mem_unit, 1.0)

    # ------------------------------------------------------------------ #
    # 1) FLOPs figure                                                    #
    # ------------------------------------------------------------------ #
    fig_flops, ax_flops = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax_flops.plot(
        df["seq_len"],
        df["prefill_flops"] / flops_div,
        marker="o",
        linewidth=2,
        label=f"Prefill {flops_unit}",
    )
    ax_flops.set_xlabel("Sequence length (tokens)")
    ax_flops.set_ylabel(f"Prefill {flops_unit}")
    ax_flops.set_title(f"{model_name} on {gpu_name}\nFLOPs vs Sequence Length")
    ax_flops.grid(True, linestyle="--", alpha=0.3)
    ax_flops.legend(loc="upper left")

    fig_flops.savefig(
        Path(output_dir) / f"{model_name}_{gpu_name}_flops_vs_seq_len.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close(fig_flops)

    # ------------------------------------------------------------------ #
    # 2) Latency figure: TTFT (s) & TTOT (ms) on twin y‑axes             #
    # ------------------------------------------------------------------ #
    fig_lat, ax_lat = plt.subplots(figsize=(7, 4), constrained_layout=True)

    # Left axis ‑‑ TTFT in seconds (no conversion)
    if "TTFT" in df.columns:
        line_ttft, = ax_lat.plot(
            df["seq_len"],
            df["TTFT"],
            marker="s",
            linestyle="-.",
            linewidth=2,
            label="TTFT (s)",
        )
        ax_lat.set_ylabel("TTFT (s)")

    # Right axis ‑‑ TTOT in milliseconds (convert from s → ms if needed)
    ax_lat2 = ax_lat.twinx()
    if "TTOT" in df.columns:
        _ttot_ms = df["TTOT"] * 1000.0  # convert to ms
        line_ttot, = ax_lat2.plot(
            df["seq_len"],
            _ttot_ms,
            marker="^",
            linestyle=":",
            linewidth=2,
            color="tab:red",
            label="TTOT (ms)",
        )
        ax_lat2.set_ylabel("TTOT (ms)")

    # Common X‑axis & styling
    ax_lat.set_xlabel("Sequence length (tokens)")
    ax_lat.set_title(f"{model_name} on {gpu_name}\nTTFT (s) & TTOT (ms) vs Sequence Length")
    ax_lat.grid(True, linestyle="--", alpha=0.3)

    # Build a combined legend that includes lines from both axes
    handles, labels = [], []
    for ax in (ax_lat, ax_lat2):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    ax_lat2.legend(handles, labels, loc="upper left")

    # Save & optionally show
    fig_lat.savefig(
        Path(output_dir) / f"{model_name}_{gpu_name}_latency_vs_seq_len.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close(fig_lat)

    # ------------------------------------------------------------------ #
    # 3) Memory figure                                                   #
    # ------------------------------------------------------------------ #
    fig_mem, ax_mem = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax_mem.plot(
        df["seq_len"],
        df["consume_memory_per_gpu"] / mem_div,
        marker="d",
        linewidth=2,
        color="tab:green",
        label=f"Total HBM ({mem_unit})",
    )
    ax_mem.set_xlabel("Sequence length (tokens)")
    ax_mem.set_ylabel(f"Total HBM per GPU ({mem_unit})")
    ax_mem.set_title(f"{model_name} on {gpu_name}\nMemory vs Sequence Length")
    ax_mem.grid(True, linestyle="--", alpha=0.3)
    ax_mem.legend(loc="upper left")

    fig_mem.savefig(
        Path(output_dir) / f"{model_name}_{gpu_name}_memory_vs_seq_len.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close(fig_mem)

    # ------------------------------------------------------------------ #
    # 4) Console table                                                   #
    # ------------------------------------------------------------------ #
    summary = df.copy()
    summary["prefill_flops"] = (summary["prefill_flops"] / flops_div).map(
        "{:,.2f}".format
    )
    summary["consume_memory_per_gpu"] = (
        summary["consume_memory_per_gpu"] / mem_div
    ).map("{:,.2f}".format)

    print("=" * 80)
    print(summary.to_string(index=False))
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
    # Example: sweep different sequence lengths and visualise
    sweep_seq_len(model_name="Qwen3-32B", gpu_name="a100-sxm-80gb", 
                  generate_len=2048, bs=32, tp_size=4, visual_flag=True)
