from llm_counts.utils.config import *
from llm_counts.visual_analyzer import LLMAnalyzerVisual
import time

####################################################################################################################
def runTime(func):
    """decorator: print the cost time of run function"""
    def wapper(arg, *args, **kwargs):
        start = time.time()
        res = func(arg, *args, **kwargs)
        end = time.time()
        print("="*80)
        print("function name: %s" %func.__name__)
        print("run time: %.4fs" %(end - start))
        print("="*80)
        return res
    return wapper

####################################################################################################################
def print_list(list):
    """print one-dimensional list

    :param list: List[int]
    :return: None
    """
    for i, x in enumerate(list):
        print(x, end='\n')
    
####################################################################################################################

def llm_profile(model_name,
                gpu_name: str = "a100-sxm-40gb",
                bytes_per_param: int = BYTES_FP16,
                bs: int = 20,
                seq_len: int = 1024,
                generate_len=1024,
                ds_zero: int = 0,
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
                mode: str = "inference",
                print_flag: bool = True,
                visual_flag: bool = True,
            ) -> dict:
    """Returns dict of the total floating-point operations, MACs, parameters and latency of a llm.

    Args:
        model_name (str, optional): model name to query the pre-defined `model_configs.json`. Defaults to "llama-13b".
        gpu_name (str, optional): gpu name to query the pre-defined `model_configs.json`. Defaults to "v100-sxm2-32gb".
        bs (int, optional): _description_. Defaults to 1.
        seq_len (int, optional): batch size per GPU.. Defaults to 522.
        generate_len (int, optional): The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. Defaults to 1526.
        ds_zero (int, optional): which DeepSpeed ZeRO stage to use.. Defaults to 0.
        dp_size (int, optional): data parallelism size. Defaults to 1.
        tp_size (int, optional): tensor parallelism size. Defaults to 1.
        pp_size (int, optional): pipeline parallelism size. Defaults to 1.
        sp_size (int, optional): sequence parallelism size. Defaults to 1.
        use_kv_cache (bool, optional): Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding. Defaults to True.
        layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations.. Defaults to BYTES_FP16.
        kv_cache_bytes (int, optional): number of bytes in the data type for the kv_cache. Defaults to None.
        flops_efficiency (float, optional): flops efficiency, ranging from 0 to 1. Defaults to None.
        hbm_memory_efficiency (float, optional): GPU HBM memory efficiency, ranging from 0 to 1. Defaults to HBM_MEMORY_EFFICIENCY.
        intra_node_memory_efficiency (_type_, optional): intra-node memory efficiency, ranging from 0 to 1.. Defaults to INTRA_NODE_MEMORY_EFFICIENCY.
        inter_node_memory_efficiency (_type_, optional): inter-node memory efficiency, ranging from 0 to 1.. Defaults to INTER_NODE_MEMORY_EFFICIENCY.
        mode (str, optional): model training or inference. Defaults to "inference".

    Returns:
        dict: a summary dictionary of the inference analysis
    """
    model_config, gpu_config = get_model_and_gpu_config_by_name(model_name, gpu_name)
    
    parallelism_config = ParallelismConfig(tp_size=tp_size, pp_size=pp_size, 
                                        dp_size=dp_size, sp_size=sp_size
                                        )
    
    inference_config = InferenceConfig(bs=bs, seq_len=seq_len, 
                                       generate_len=generate_len, use_kv_cache=use_kv_cache,
                                       bytes_per_param=bytes_per_param,
                                       act_dtype_bytes=act_dtype_bytes,
                                       kv_cache_bytes=kv_cache_bytes
                                       )
    
    gpu_efficiency_config = GPUEfficiencyConfig(flops_efficiency=flops_efficiency,
                                                hbm_memory_efficiency=hbm_memory_efficiency,
                                                intra_node_memory_efficiency=intra_node_memory_efficiency,
                                                inter_node_memory_efficiency=inter_node_memory_efficiency
    )
    
    llm_configs = LLMConfigs(model_config=model_config, gpu_config=gpu_config,
                             parallelism_config=parallelism_config, inference_config=inference_config,
                             gpu_efficiency_config=gpu_efficiency_config
                            )


    profiler = LLMAnalyzerVisual(llm_configs)
    
    max_batch_total_tokens = profiler.infer_profile(bs=bs, seq_len=seq_len, 
                        generate_len=generate_len, use_kv_cache=use_kv_cache,
                        act_dtype_bytes=act_dtype_bytes,
                        flops_efficiency=flops_efficiency,
                        hbm_memory_efficiency=hbm_memory_efficiency,
                        print_flag=print_flag)
    
    return max_batch_total_tokens 

def print_all_llm_analyzer():
    model_name_list = ["llama-7b", "llama-13b", "llama-65b", "llama2-70b", "internlm-20b"]
    gpu_name_list = ["a30-sxm-24gb", "a40-pcie-48gb", "a100-sxm-40gb", "a100-sxm-80gb", "910b-64gb", "v100-sxm-32gb", "t4-pcie-15gb"]
    tp_nums_list = [1, 2, 4, 8]
    tgi_service_dict_list = []
    seq_len, generate_len = 1024, 1024
    
    for model_name in model_name_list:
        if model_name in ["llama2-70b", "internlm-20b"]:
            seq_len, generate_len = 1024, 1024
            
        for gpu_name in gpu_name_list:
            for tp_size in tp_nums_list:
                try:
                    max_batch_total_tokens = int(llm_profile(model_name=model_name, gpu_name=gpu_name, tp_size=tp_size,
                                                         seq_len=seq_len, generate_len=generate_len, print_flag=False))
                except Exception as e:
                    print(f"model_name: {model_name}, gpu_name: {gpu_name}, tp_size: {tp_size}, error: {e}")
                    continue
                
                tgi_service_dict = {"model_name": model_name, "gpu_name": gpu_name, "tp_size": tp_size, "max_batch_total_tokens": max_batch_total_tokens, "max_bs": floor(max_batch_total_tokens / (seq_len + generate_len))}
                tgi_service_dict_list.append(tgi_service_dict)
    
    print("================================== TGI+LightLLM service max_batch_total_tokens params list =============================")
    print_list(tgi_service_dict_list)


if __name__ == "__main__": 
    # llm_profile(model_name="llama-7b", tp_size=1, print_flag=True, visual_flag=True)
    llm_profile(model_name="llama2-70b", tp_size=8, print_flag=True, visual_flag=True)
