import pandas as pd
from huggingface_hub import snapshot_download
import subprocess
import re
import os
import GPUtil
from transformers import AutoConfig

try:
    from src.display.utils import GPU_TEMP, GPU_Mem, GPU_Power, GPU_Util, GPU_Name
except:
    print("local debug: from display.utils")
    from display.utils import GPU_TEMP, GPU_Mem, GPU_Power, GPU_Util, GPU_Name
    
MEM_BW_DICT ={
    "NVIDIA-A100-PCIe-80GB": 1935e9,
    "NVIDIA-A100-SXM4-80GB": 2039e9,
    "NVIDIA-H100-PCIe-80GB": 2039e9,
    "NVIDIA-RTX-A5000-24GB": 768e9,
    "NVIDIA-RTX-A6000-48GB": 768e9,
}

PEAK_FLOPS_DICT = {
    "float32":{
        "NVIDIA-A100-PCIe-80GB": 312e12,
        "NVIDIA-A100-SXM4-80GB": 312e12,
        "NVIDIA-H100-PCIe-80GB": 756e12,
        "NVIDIA-RTX-A5000-24GB": 222.2e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12
    },
    "float16":{
        "NVIDIA-A100-PCIe-80GB": 624e12,
        "NVIDIA-A100-SXM4-80GB": 624e12,
        "NVIDIA-H100-PCIe-80GB": 1513e12,
        "NVIDIA-RTX-A5000-24GB": 444.4e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12
    },
    "bfloat16":{
        "NVIDIA-A100-PCIe-80GB": 624e12,
        "NVIDIA-A100-SXM4-80GB": 624e12,
        "NVIDIA-H100-PCIe-80GB": 1513e12,
        "NVIDIA-RTX-A5000-24GB": 444.4e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12
    },
    "int8":{
        "NVIDIA-A100-PCIe-80GB": 1248e12,
        "NVIDIA-A100-SXM4-80GB": 1248e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 889e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12
    },
    "fp8":{
        "NVIDIA-A100-PCIe-80GB": 1248e12,
        "NVIDIA-A100-SXM4-80GB": 1248e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 889e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12
    },
    "fp4": {
        "NVIDIA-A100-PCIe-80GB": 2496e12,
        "NVIDIA-A100-SXM4-80GB": 2496e12,
        "NVIDIA-H100-PCIe-80GB": 6052e12,
        "NVIDIA-RTX-A5000-24GB": 1778e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12
    },
    "int4": {
        "NVIDIA-A100-PCIe-80GB": 2496e12,
        "NVIDIA-A100-SXM4-80GB": 2496e12,
        "NVIDIA-H100-PCIe-80GB": 6052e12,
        "NVIDIA-RTX-A5000-24GB": 1778e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12
    }
}

def my_snapshot_download(repo_id, revision, local_dir, repo_type, max_workers):
    for i in range(10):
        try:
            snapshot_download(
                repo_id=repo_id, revision=revision, local_dir=local_dir, repo_type=repo_type, max_workers=max_workers
            )
            return
        except Exception as e:
            print(f"Failed to download {repo_id} at {revision} with error: {e}. Retrying...")
            import time

            time.sleep(60)
    return


def get_dataset_url(row):
    dataset_name = row["Benchmark"]
    dataset_url = row["Dataset Link"]
    benchmark = f'<a target="_blank" href="{dataset_url}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{dataset_name}</a>'
    return benchmark


def get_dataset_summary_table(file_path):
    df = pd.read_csv(file_path)

    df["Benchmark"] = df.apply(lambda x: get_dataset_url(x), axis=1)

    df = df[["Category", "Benchmark", "Data Split", "Data Size", "Language"]]

    return df

def parse_nvidia_smi():
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if visible_devices is not None:
        gpu_indices = visible_devices.split(',')
    else:
        # Query all GPU indices if CUDA_VISIBLE_DEVICES is not set
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Failed to query GPU indices.")
            return []
        gpu_indices = result.stdout.strip().split('\n')
    # print(f"gpu_indices: {gpu_indices}")
    gpu_stats = []

    gpu_info_pattern = re.compile(r'(\d+)C\s+P\d+\s+(\d+)W\s*/\s*\d+W\s*\|\s*(\d+)MiB\s*/\s*\d+MiB\s*\|\s*(\d+)%')
    # gpu_name_pattern = re.compile(r'NVIDIA\s+([\w\s]+\d+(?:\s*GB)?)')
    gpu_name_pattern = re.compile(r'NVIDIA\s+(RTX\s+)?([A-Z0-9]+)')

    gpu_name = ""
    for index in gpu_indices:
        result = subprocess.run(['nvidia-smi', '-i', index], capture_output=True, text=True)
        output = result.stdout.strip()
        lines = output.split("\n")
        for line in lines:
            match = gpu_info_pattern.search(line)
            name_match = gpu_name_pattern.search(line)
            gpu_info = {}
            if name_match:
                gpu_name = ''.join(filter(None, name_match.groups())).strip()
            if match:
                temp, power_usage, mem_usage, gpu_util = map(int, match.groups())
                gpu_info.update({
                    GPU_TEMP: temp,
                    GPU_Power: power_usage,
                    GPU_Mem: round(mem_usage / 1024, 2),
                    GPU_Util: gpu_util
                })

            if len(gpu_info) >= 4:
                gpu_stats.append(gpu_info)
    # print(f"gpu_stats: {gpu_stats}")
    gpu_name = f"{len(gpu_stats)}x{gpu_name}"
    gpu_stats_total = {
                        GPU_TEMP: 0,
                        GPU_Power: 0,
                        GPU_Mem: 0,
                        GPU_Util: 0,
                        GPU_Name: gpu_name
                    }
    for gpu_stat in gpu_stats:
        gpu_stats_total[GPU_TEMP] += gpu_stat[GPU_TEMP]
        gpu_stats_total[GPU_Power] += gpu_stat[GPU_Power]
        gpu_stats_total[GPU_Mem] += gpu_stat[GPU_Mem]
        gpu_stats_total[GPU_Util] += gpu_stat[GPU_Util]
    gpu_stats_total[GPU_Mem] = gpu_stats_total[GPU_Mem] # G
    gpu_stats_total[GPU_TEMP] /= len(gpu_stats)
    gpu_stats_total[GPU_Power] /= len(gpu_stats)
    gpu_stats_total[GPU_Util] /= len(gpu_stats)
    return [gpu_stats_total]

def monitor_gpus(stop_event, interval, stats_list):
    while not stop_event.is_set():
        gpu_stats = parse_nvidia_smi()
        if gpu_stats:
            stats_list.extend(gpu_stats)
        stop_event.wait(interval)

def analyze_gpu_stats(stats_list):
    # Check if the stats_list is empty, and return None if it is
    if not stats_list:
        return None

    # Initialize dictionaries to store the stats
    avg_stats = {}
    max_stats = {}

    # Calculate average stats, excluding 'GPU_Mem'
    for key in stats_list[0].keys():
        if key != GPU_Mem and key != GPU_Name:
            total = sum(d[key] for d in stats_list)
            avg_stats[key] = total / len(stats_list)

    # Calculate max stats for 'GPU_Mem'
    max_stats[GPU_Mem] = max(d[GPU_Mem] for d in stats_list)
    if GPU_Name in stats_list[0]:
        avg_stats[GPU_Name] = stats_list[0][GPU_Name]
    # Update average stats with max GPU memory usage
    avg_stats.update(max_stats)

    return avg_stats

def get_gpu_details():
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]
    name = gpu.name.replace(" ", "-")
    memory_gb = round(gpu.memoryTotal / 1024)
    memory = f"{memory_gb}GB"

    for part in name.split('-'):
        if part.endswith("GB") and part[:-2].isdigit():
            name = name.replace(f"-{part}", "").replace(part, "")

    formatted_name = f"{name}-{memory}"
    
    return formatted_name

def get_peak_bw(gpu_name):
    return MEM_BW_DICT[gpu_name]

def get_peak_flops(gpu_name, precision):
    return PEAK_FLOPS_DICT[precision][gpu_name]

def _calculate_batch_metrics(outputs, decoding_tp, n_layers, d_model, 
                                n_attn_heads, d_head, n_kv_heads, n_experts_per_tok, d_ff, 
                                avg_activated_experts, hf_config, num_gpus, model_name, 
                                used_dtype, batch_size, precision):
    """Calculate metrics for a batch of outputs"""
    gpu_type = get_gpu_details()
    hardware_specs = {
        "peak_bandwidth_tb": get_peak_bw(gpu_type) / 1e12,
        "peak_flops_tf": get_peak_flops(gpu_type, precision=used_dtype) / 1e12,
    }
    kvs = []
    true_kvs = []
    attn_score = []
    
    # Calculate KV sizes
    per_token_kv_size = 2 * n_layers * d_head * n_kv_heads  # Default calculation
    
    if "DeepSeek" in model_name:
        if hasattr(hf_config, "kv_lora_rank") and hasattr(hf_config, "qk_rope_head_dim"):
            per_token_kv_size = n_layers * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)
    
    # Process each output
    for x in outputs:
        output_len = len(x.outputs[0].token_ids)
        context_prefill_size = len(x.prompt_token_ids)
        
        # Calculate attention scores
        if "DeepSeek" in model_name and hasattr(hf_config, "qk_rope_head_dim") and hasattr(hf_config, "qk_nope_head_dim") and hasattr(hf_config, "v_head_dim"):
            q_head_dim = hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim
            origin_per_token_k_state_size = n_layers * n_attn_heads * q_head_dim
            origin_per_token_v_state_size = n_layers * n_attn_heads * hf_config.v_head_dim
            attention_score = context_prefill_size * origin_per_token_k_state_size + (output_len - 1) * origin_per_token_k_state_size / 2
            attention_score += context_prefill_size * origin_per_token_v_state_size + (output_len - 1) * origin_per_token_v_state_size / 2
            attention_score = attention_score / 1e12
        else:
            origin_per_token_kv_states_size = n_layers * n_attn_heads * d_head
            attention_score = context_prefill_size * origin_per_token_kv_states_size + (output_len - 1) * origin_per_token_kv_states_size / 2
            attention_score = attention_score * 2 / 1e12
        
        # Store attention scores and KV sizes
        attn_score.append(attention_score)
        kv_size = context_prefill_size * per_token_kv_size + (output_len - 1) * per_token_kv_size / 2
        kv_size = kv_size / 1e12
        true_kv = (context_prefill_size * per_token_kv_size + output_len * per_token_kv_size) / 1e12 * 1e3
        kvs.append(kv_size)
        true_kvs.append(true_kv)
    
    # Calculate aggregate values
    kv_size = sum(kvs)
    true_kv_size = sum(true_kvs) * 1e3
    attention_score = sum(attn_score) / len(attn_score)
    
    # Calculate attention size per token
    if "DeepSeek" in model_name and hasattr(hf_config, "qk_rope_head_dim") and hasattr(hf_config, "qk_nope_head_dim") and hasattr(hf_config, "v_head_dim") and hasattr(hf_config, "kv_lora_rank"):
        q_head_dim = hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim
        if not hasattr(hf_config, "q_lora_rank") or not hf_config.q_lora_rank:
            attention_size_per_token = (d_model * n_attn_heads * q_head_dim) + \
                (d_model * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)) + \
                    (hf_config.kv_lora_rank * n_attn_heads * (q_head_dim - hf_config.qk_rope_head_dim + hf_config.v_head_dim)) + \
                        (hf_config.v_head_dim * n_attn_heads * d_model)
            attention_size_per_token = attention_size_per_token / 1e12
        else:
            attention_size_per_token = (d_model * hf_config.q_lora_rank) + \
                (hf_config.q_lora_rank * n_attn_heads * q_head_dim) + \
                    (d_model * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)) + \
                        (hf_config.kv_lora_rank * n_attn_heads * (q_head_dim - hf_config.qk_rope_head_dim + hf_config.v_head_dim)) + \
                            (hf_config.v_head_dim * n_attn_heads * d_model)
            attention_size_per_token = attention_size_per_token / 1e12
    else:
        attention_size_per_token = d_model * (n_attn_heads * d_head + n_kv_heads * d_head * 2) + n_attn_heads * d_head * d_model
        attention_size_per_token = attention_size_per_token / 1e12
    
    # Calculate expert sizes
    expert_size = d_ff * 3 * d_model / 1e12
    shared_experts_size_total = 0
    deepseek_dense_ffn_size = 0
    deepseek_sparse_layer_num = 0
    
    if "Qwen" in model_name and hasattr(hf_config, "moe_intermediate_size") and hasattr(hf_config, "shared_expert_intermediate_size"):
        d_ff = hf_config.moe_intermediate_size
        d_ff_share = hf_config.shared_expert_intermediate_size
        shared_experts_size = d_ff_share * 3 * d_model
        expert_size = d_ff * 3 * d_model
        shared_experts_size_total = shared_experts_size / 1e12
        expert_size = expert_size / 1e12
    elif "DeepSeek" in model_name and hasattr(hf_config, "moe_intermediate_size") and hasattr(hf_config, "intermediate_size") and hasattr(hf_config, "first_k_dense_replace"):
        d_ff = hf_config.moe_intermediate_size
        d_ff_dense = hf_config.intermediate_size
        deepseek_num_dense_layer = hf_config.first_k_dense_replace
        shared_experts_size = d_ff * 3 * d_model
        expert_size = d_ff * 3 * d_model
        shared_experts = 2
        shared_experts_size_total = shared_experts_size * shared_experts / 1e12
        expert_size = expert_size / 1e12
        deepseek_sparse_layer_num = n_layers - deepseek_num_dense_layer
        deepseek_dense_ffn_size = d_ff_dense * 3 * d_model / 1e12
    
    # Calculate S-MBU and S-MFU
    if "Qwen" in model_name:
        smbu = ((n_layers*(avg_activated_experts * expert_size + shared_experts_size_total + attention_size_per_token) + 
                kv_size) * precision/(batch_size / decoding_tp) ) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * (attention_size_per_token + n_experts_per_tok * expert_size + shared_experts_size_total) + attention_score) \
            * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'])
    elif "DeepSeek" in model_name:
        smbu = ((n_layers * attention_size_per_token + deepseek_sparse_layer_num * \
                (avg_activated_experts * expert_size + shared_experts_size_total) + \
                deepseek_num_dense_layer * deepseek_dense_ffn_size + \
                kv_size) * precision/(batch_size / decoding_tp) ) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * attention_size_per_token + deepseek_sparse_layer_num * \
                (n_experts_per_tok * expert_size + shared_experts_size_total) + \
                deepseek_num_dense_layer * deepseek_dense_ffn_size + attention_score) \
                * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'])
    else:
        smbu = ((n_layers*(avg_activated_experts * expert_size + attention_size_per_token) + 
                kv_size) * precision/(batch_size / decoding_tp) ) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * (attention_size_per_token + n_experts_per_tok * expert_size) + attention_score) \
            * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'])
    
    return {
        'smbu': smbu,
        'smfu': smfu,
        'kv_size': true_kv_size,
        'decoding_throughput': decoding_tp
    }


def _calculate_batch_metrics_hflm(output_len, context_prefill_size, decoding_tp, n_layers, d_model, 
                                n_attn_heads, d_head, n_kv_heads, n_experts_per_tok, d_ff, 
                                avg_activated_experts, hf_config, num_gpus, model_name, 
                                used_dtype, batch_size, precision):
    """Calculate metrics for a batch of outputs"""
    gpu_type = get_gpu_details()
    hardware_specs = {
        "peak_bandwidth_tb": get_peak_bw(gpu_type) / 1e12,
        "peak_flops_tf": get_peak_flops(gpu_type, precision=used_dtype) / 1e12,
    }
    
    # Calculate KV sizes
    per_token_kv_size = 2 * n_layers * d_head * n_kv_heads  # Default calculation
    
    if "DeepSeek" in model_name:
        if hasattr(hf_config, "kv_lora_rank") and hasattr(hf_config, "qk_rope_head_dim"):
            per_token_kv_size = n_layers * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)
    
        
    # Calculate attention scores
    if "DeepSeek" in model_name and hasattr(hf_config, "qk_rope_head_dim") and hasattr(hf_config, "qk_nope_head_dim") and hasattr(hf_config, "v_head_dim"):
        q_head_dim = hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim
        origin_per_token_k_state_size = n_layers * n_attn_heads * q_head_dim
        origin_per_token_v_state_size = n_layers * n_attn_heads * hf_config.v_head_dim
        attention_score = context_prefill_size * origin_per_token_k_state_size + (output_len - 1) * origin_per_token_k_state_size / 2
        attention_score += context_prefill_size * origin_per_token_v_state_size + (output_len - 1) * origin_per_token_v_state_size / 2
        attention_score = attention_score / 1e12
    else:
        origin_per_token_kv_states_size = n_layers * n_attn_heads * d_head
        attention_score = context_prefill_size * origin_per_token_kv_states_size + (output_len - 1) * origin_per_token_kv_states_size / 2
        attention_score = attention_score * 2 / 1e12
    
    # Store attention scores and KV sizes
    kv_size = context_prefill_size * per_token_kv_size + (output_len - 1) * per_token_kv_size / 2
    kv_size = kv_size / 1e12
    true_kv = (context_prefill_size * per_token_kv_size + output_len * per_token_kv_size) / 1e12 * 1e3
    
    # Calculate aggregate values
    kv_size = kv_size * batch_size
    true_kv_size = true_kv * batch_size * 1e3    
    # Calculate attention size per token
    if "DeepSeek" in model_name and hasattr(hf_config, "qk_rope_head_dim") and hasattr(hf_config, "qk_nope_head_dim") and hasattr(hf_config, "v_head_dim") and hasattr(hf_config, "kv_lora_rank"):
        q_head_dim = hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim
        if not hasattr(hf_config, "q_lora_rank") or not hf_config.q_lora_rank:
            attention_size_per_token = (d_model * n_attn_heads * q_head_dim) + \
                (d_model * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)) + \
                    (hf_config.kv_lora_rank * n_attn_heads * (q_head_dim - hf_config.qk_rope_head_dim + hf_config.v_head_dim)) + \
                        (hf_config.v_head_dim * n_attn_heads * d_model)
            attention_size_per_token = attention_size_per_token / 1e12
        else:
            attention_size_per_token = (d_model * hf_config.q_lora_rank) + \
                (hf_config.q_lora_rank * n_attn_heads * q_head_dim) + \
                    (d_model * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)) + \
                        (hf_config.kv_lora_rank * n_attn_heads * (q_head_dim - hf_config.qk_rope_head_dim + hf_config.v_head_dim)) + \
                            (hf_config.v_head_dim * n_attn_heads * d_model)
            attention_size_per_token = attention_size_per_token / 1e12
    else:
        attention_size_per_token = d_model * (n_attn_heads * d_head + n_kv_heads * d_head * 2) + n_attn_heads * d_head * d_model
        attention_size_per_token = attention_size_per_token / 1e12
    
    # Calculate expert sizes
    expert_size = d_ff * 3 * d_model / 1e12
    shared_experts_size_total = 0
    deepseek_dense_ffn_size = 0
    deepseek_sparse_layer_num = 0
    
    if "Qwen" in model_name and hasattr(hf_config, "moe_intermediate_size") and hasattr(hf_config, "shared_expert_intermediate_size"):
        d_ff = hf_config.moe_intermediate_size
        d_ff_share = hf_config.shared_expert_intermediate_size
        shared_experts_size = d_ff_share * 3 * d_model
        expert_size = d_ff * 3 * d_model
        shared_experts_size_total = shared_experts_size / 1e12
        expert_size = expert_size / 1e12
    elif "DeepSeek" in model_name and hasattr(hf_config, "moe_intermediate_size") and hasattr(hf_config, "intermediate_size") and hasattr(hf_config, "first_k_dense_replace"):
        d_ff = hf_config.moe_intermediate_size
        d_ff_dense = hf_config.intermediate_size
        deepseek_num_dense_layer = hf_config.first_k_dense_replace
        shared_experts_size = d_ff * 3 * d_model
        expert_size = d_ff * 3 * d_model
        shared_experts = 2
        shared_experts_size_total = shared_experts_size * shared_experts / 1e12
        expert_size = expert_size / 1e12
        deepseek_sparse_layer_num = n_layers - deepseek_num_dense_layer
        deepseek_dense_ffn_size = d_ff_dense * 3 * d_model / 1e12
    
    # Calculate S-MBU and S-MFU
    if "Qwen" in model_name:
        smbu = ((n_layers*(avg_activated_experts * expert_size + shared_experts_size_total + attention_size_per_token) + 
                kv_size) * precision/(batch_size / decoding_tp) ) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * (attention_size_per_token + n_experts_per_tok * expert_size + shared_experts_size_total) + attention_score) \
            * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'])
    elif "DeepSeek" in model_name:
        smbu = ((n_layers * attention_size_per_token + deepseek_sparse_layer_num * \
                (avg_activated_experts * expert_size + shared_experts_size_total) + \
                deepseek_num_dense_layer * deepseek_dense_ffn_size + \
                kv_size) * precision/(batch_size / decoding_tp) ) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * attention_size_per_token + deepseek_sparse_layer_num * \
                (n_experts_per_tok * expert_size + shared_experts_size_total) + \
                deepseek_num_dense_layer * deepseek_dense_ffn_size + attention_score) \
                * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'])
    else:
        smbu = ((n_layers*(avg_activated_experts * expert_size + attention_size_per_token) + 
                kv_size) * precision/(batch_size / decoding_tp) ) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * (attention_size_per_token + n_experts_per_tok * expert_size) + attention_score) \
            * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'])
    
    return {
        'smbu': smbu,
        'smfu': smfu,
        'kv_size': true_kv_size,
        'decoding_throughput': decoding_tp
    }
class ModelInfoRetriever:
    def __init__(self, model_name: str, precision: str = 'float16'):
        if precision not in ['float32', 'float16', 'bfloat16', 'int8', 'int4', 'awq', 'gptq', 'fp8', 'fp4']:
            raise ValueError("Precision must be one of ['float32', 'float16', 'bfloat16', 'int8', 'int4', 'awq', 'gptq', 'fp8', 'fp4']")
        self.model_name = model_name
        self.precision = precision
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model_type = self.config.model_type

    def get_model_precision_bits(self):
        """Returns bit width used by the given quantization format."""
        if self.precision == 'float32':
            return 4
        if self.precision in ['float16', 'bfloat16']:
            return 2
        if self.precision in ['int8', 'fp8']:
            return 1
        if self.precision in ['int4', 'fp4', 'gptq', 'awq']:
            return 0.5
        raise ValueError(f"Unsupported precision: {self.precision}")

    def get_attention_info(self):
        """Returns attention-related info"""
        return {
            'num_attention_heads': getattr(self.config, "num_attention_heads", None),
            'num_key_value_heads': getattr(self.config, "num_key_value_heads", getattr(self.config, "num_kv_heads", None)),
            'head_dim': getattr(self.config, "head_dim", getattr(self.config, "hidden_size", None) // getattr(self.config, "num_attention_heads", 1))
        }

    def get_rope_info(self):
        """Returns RoPE (rotary embedding) info if available"""
        if hasattr(self.config, "rope_scaling"):
            return {
                "type": self.config.rope_scaling.get("type"),
                "factor": self.config.rope_scaling.get("factor")
            }
        elif hasattr(self.config, "use_alibi"):
            return {"type": "alibi", "enabled": self.config.use_alibi}
        else:
            return {"type": "none"}

    def get_moe_info(self, d_model=None):
        """Returns MoE configuration such as number of experts and FFN dim"""
        if d_model is None:
            d_model = getattr(self.config, "hidden_size", None)

        num_experts = (
            getattr(self.config, "num_local_experts", None) or
            getattr(self.config, "num_experts", None) or
            getattr(self.config, "n_routed_experts", None) or
            getattr(getattr(self.config, "ffn_config", {}), "moe_num_experts", None) or
            1
        )
        n_experts_per_tok = (
            getattr(self.config, "num_experts_per_tok", None) or
            getattr(self.config, "num_selected_experts", None) or
            getattr(getattr(self.config, "ffn_config", {}), "moe_top_k", None) or
            1
        )
        d_ff = (
            getattr(self.config, "ffn_dim", None) or
            getattr(self.config, "intermediate_size", None) or
            getattr(self.config, "d_ff", None) or
            (d_model * getattr(self.config, "ff_ratio", 4)) or
            getattr(getattr(self.config, "ffn_config", {}), "ffn_hidden_size", None) or
            (4 * d_model)
        )

        return {
            "num_experts": num_experts,
            "experts_per_token": n_experts_per_tok,
            "ffn_dim": d_ff
        }

    def get_architecture_info(self):
        """Returns model-wide architecture info"""
        return {
            "model_type": self.model_type,
            "hidden_size": getattr(self.config, "hidden_size", None),
            "num_hidden_layers": getattr(self.config, "num_hidden_layers", None),
            "max_position_embeddings": getattr(self.config, "max_position_embeddings", None),
            "vocab_size": getattr(self.config, "vocab_size", None),
            "architectures": getattr(self.config, "architectures", []),
        }

    def summarize(self):
        """Aggregate all extracted info in a dictionary"""
        d_model = getattr(self.config, "hidden_size", None)
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "precision_bits": self.get_model_precision_bits(),
            "architecture": self.get_architecture_info(),
            "attention": self.get_attention_info(),
            "rope": self.get_rope_info(),
            "moe": self.get_moe_info(d_model)
        }
    


# if __name__ == "__main__":
#     print(analyze_gpu_stats(parse_nvidia_smi()))
#     print(get_gpu_details())