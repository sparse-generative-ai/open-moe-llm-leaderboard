import pandas as pd
from huggingface_hub import snapshot_download
import subprocess
import re
import os
import GPUtil
from transformers import AutoConfig
from typing import List

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
        "NVIDIA-RTX-A5000-24GB": 222.2e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12
    },
    "bfloat16":{
        "NVIDIA-A100-PCIe-80GB": 624e12,
        "NVIDIA-A100-SXM4-80GB": 624e12,
        "NVIDIA-H100-PCIe-80GB": 1513e12,
        "NVIDIA-RTX-A5000-24GB": 222.2e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12
    },
    "int8":{
        "NVIDIA-A100-PCIe-80GB": 1248e12,
        "NVIDIA-A100-SXM4-80GB": 1248e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 222.2e12,
        "NVIDIA-RTX-A6000-48GB": 309.7e12
    },
    "fp8":{
        "NVIDIA-A100-PCIe-80GB": 1248e12,
        "NVIDIA-A100-SXM4-80GB": 1248e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 0,
        "NVIDIA-RTX-A6000-48GB": 0
    },
    "fp4": {
        "NVIDIA-A100-PCIe-80GB": 1248e12,
        "NVIDIA-A100-SXM4-80GB": 1248e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 0,
        "NVIDIA-RTX-A6000-48GB": 0
    },
    "int4": {
        "NVIDIA-A100-PCIe-80GB": 1248e12,
        "NVIDIA-A100-SXM4-80GB": 1248e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 222.2e12,
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
    elif "Qwen3" in model_name and hasattr(hf_config, "moe_intermediate_size"):
        d_ff = hf_config.moe_intermediate_size
        expert_size = d_ff * 3 * d_model
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
    if "Qwen" in model_name and not "Qwen3" in model_name:
        smbu = ((n_layers*(avg_activated_experts * expert_size + shared_experts_size_total + attention_size_per_token) + 
                kv_size) * precision/ (batch_size / decoding_tp)) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * (attention_size_per_token + n_experts_per_tok * expert_size + shared_experts_size_total) + attention_score) \
            * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    elif "Qwen3" in model_name:
        smbu = ((n_layers * (avg_activated_experts * expert_size + attention_size_per_token) + 
                kv_size) * precision/ (batch_size / decoding_tp)) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * (attention_size_per_token + n_experts_per_tok * expert_size) + attention_score) \
            * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    elif "DeepSeek" in model_name:
        smbu = ((n_layers * attention_size_per_token + deepseek_sparse_layer_num * \
                (avg_activated_experts * expert_size + shared_experts_size_total) + \
                deepseek_num_dense_layer * deepseek_dense_ffn_size + \
                kv_size) * precision/ (batch_size / decoding_tp)) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * attention_size_per_token + deepseek_sparse_layer_num * \
                (n_experts_per_tok * expert_size + shared_experts_size_total) + \
                deepseek_num_dense_layer * deepseek_dense_ffn_size + attention_score) \
                * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    else:
        smbu = ((n_layers*(avg_activated_experts * expert_size + attention_size_per_token) + 
                kv_size) * precision/ (batch_size / decoding_tp) ) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * (attention_size_per_token + n_experts_per_tok * expert_size) + attention_score) \
            * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return {
        'smbu': smbu,
        'smfu': smfu,
        'kv_size': true_kv_size,
        'decoding_throughput': decoding_tp
    }

def only_prefill_metrics(outputs, batch_size):
    output_data = _extract_output_data(outputs)
    ttft, prefill_tp = _calculate_throughput_metrics(batch_size, output_data['prefill_lengths'],
                                                       output_data['max_duration'])
    return {
        'prefill_smbu': 0,
        'prefill_smfu': 0,
        'decoding_smbu': 0,
        'decoding_smfu': 0,
        'kv_size': 0,
        'decoding_throughput': 0,
        'prefill_tp': prefill_tp,
        'ttft': ttft
    }
   

def _calculate_batch_metrics_sglang(outputs, decoding_tp, n_layers, d_model, 
                                n_attn_heads, d_head, n_kv_heads, n_experts_per_tok, d_ff, 
                                avg_activated_experts, hf_config, num_gpus, model_name, 
                                used_dtype, batch_size, precision, ttft=None, prefill_tp=None):
    """Calculate metrics for a batch of outputs"""
    # Initialize hardware specs and output lists
    hardware_specs = _get_hardware_specs(used_dtype)
    output_data = _extract_output_data(outputs)
    
    # Calculate model-specific sizes
    per_token_kv_size = _calculate_kv_size(model_name, hf_config, n_layers, d_head, n_kv_heads)
    attention_size_per_token = _calculate_attention_size(model_name, hf_config, d_model, n_attn_heads, d_head, n_kv_heads)
    expert_config = _calculate_expert_config(model_name, hf_config, d_ff, d_model, n_layers)
    
    # Process outputs and calculate metrics
    metrics_data = _process_outputs(output_data, per_token_kv_size, attention_size_per_token, 
                                  model_name, hf_config, n_layers, n_attn_heads, d_head)

    # Calculate throughput metrics
    if ttft is None or prefill_tp is None:
        ttft, prefill_tp = _calculate_throughput_metrics(batch_size, output_data['prefill_lengths'],
                                                       output_data['max_duration'])

    
    # Calculate S-MBU and S-MFU
    smbu_smfu_metrics = _calculate_smbu_smfu(model_name, n_layers, attention_size_per_token,
                                           expert_config, avg_activated_experts, metrics_data,
                                           hardware_specs, num_gpus, precision, ttft, prefill_tp,
                                           batch_size, decoding_tp)
    
    return {
        'prefill_smbu': smbu_smfu_metrics['prefill_smbu'],
        'prefill_smfu': smbu_smfu_metrics['prefill_smfu'],
        'decoding_smbu': smbu_smfu_metrics['decoding_smbu'],
        'decoding_smfu': smbu_smfu_metrics['decoding_smfu'],
        'kv_size': metrics_data['true_kv_size'],
        'decoding_throughput': decoding_tp,
        'prefill_tp': prefill_tp,
        'ttft': ttft
    }

def _calculate_continuous_metrics_sglang(n_layers, d_model, 
                                n_attn_heads, d_head, n_kv_heads, d_ff, hf_config, num_gpus, model_name, 
                                used_dtype, precision, output_data):
    """Calculate metrics for a batch of outputs"""
    # Initialize hardware specs and output lists
    hardware_specs = _get_hardware_specs(used_dtype)
    
    # Calculate model-specific sizes
    per_token_kv_size = _calculate_kv_size(model_name, hf_config, n_layers, d_head, n_kv_heads)
    attention_size_per_token = _calculate_attention_size(model_name, hf_config, d_model, n_attn_heads, d_head, n_kv_heads)
    expert_config = _calculate_expert_config(model_name, hf_config, d_ff, d_model, n_layers)
    
    # Process outputs and calculate metrics
    ttfts = []
    tpots = []
    prefill_tps = []
    decoding_tps = []
    true_kvs = []
    prefill_smbus = []
    prefill_smfus = []
    decoding_smbus = []
    decoding_smfus = []

    for out in output_data:
        if out['expert_activation'] == 0:
            continue
        metrics_data = _process_outputs_continuous(out, per_token_kv_size, attention_size_per_token, 
                                    model_name, hf_config, n_layers, n_attn_heads, d_head)

        true_kvs.append(metrics_data['true_kv_size'])

        # Calculate throughput metrics
        if out['forward_mode'] == 'prefill':
            prefill_activation = out['expert_activation']
            ttft = out['latency']
            prefill_tp = out['seq_lens_sum'] / ttft
            ttfts.append(ttft)
            prefill_tps.append(prefill_tp)
            prefill_smbu, prefill_smfu = _calculate_prefill_metrics(model_name=model_name, n_layers=n_layers, attention_size_per_token=attention_size_per_token,
                                               expert_config=expert_config, hardware_specs=hardware_specs, num_gpus=num_gpus, precision=precision, ttft=ttft, 
                                               prefill_tp=prefill_tp, prefill_activation=prefill_activation, metrics_data=metrics_data)
            prefill_smbus.append(prefill_smbu)
            prefill_smfus.append(prefill_smfu)

        else:
            decoding_activation = out['expert_activation']
            tpot = out['latency']
            batch_size = out['batch_size']
            decoding_tp = batch_size / tpot
            tpots.append(tpot)
            decoding_tps.append(decoding_tp)

            decoding_smbu, decoding_smfu = _calculate_decoding_metrics(model_name=model_name, n_layers=n_layers, attention_size_per_token=attention_size_per_token,
                                               expert_config=expert_config, decode_steps_activation=decoding_activation, metrics_data=metrics_data,
                                               hardware_specs=hardware_specs, num_gpus=num_gpus, precision=precision, batch_size=batch_size, decoding_tp=decoding_tp, tpot=tpot)
            decoding_smbus.append(decoding_smbu)
            decoding_smfus.append(decoding_smfu)


        # Calculate S-MBU and S-MFU
        # smbu_smfu_metrics = _calculate_smbu_smfu(model_name, n_layers, attention_size_per_token,
        #                                     expert_config, avg_activated_experts, metrics_data,
        #                                     hardware_specs, num_gpus, precision, ttft, prefill_tp,
        #                                     batch_size, decoding_tp)
    
    # Aggregate metrics
    prefill_smbu = sum(prefill_smbus) / len(prefill_smbus) if prefill_smbus else 0
    prefill_smfu = sum(prefill_smfus) / len(prefill_smfus) if prefill_smfus else 0
    decoding_smbu = sum(decoding_smbus) / len(decoding_smbus) if decoding_smbus else 0
    decoding_smfu = sum(decoding_smfus) / len(decoding_smfus) if decoding_smfus else 0
    decoding_tp = sum(decoding_tps) / len(decoding_tps) if decoding_tps else 0
    tpot = sum(tpots) / len(tpots) if tpots else 0
    ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    prefill_tp = sum(prefill_tps) / len(prefill_tps) if prefill_tps else 0
    kv_size = sum(true_kvs) / len(true_kvs) if true_kvs else 0


    return {
        'prefill_smbu': prefill_smbu,
        'prefill_smfu': prefill_smfu,
        'decoding_smbu': decoding_smbu,
        'decoding_smfu': decoding_smfu,
        'kv_size': kv_size,
        'decoding_throughput': decoding_tp,
        'prefill_tp': prefill_tp,
        'ttft': ttft,
        'tpot': tpot
    }


def _get_hardware_specs(used_dtype):
    """Get hardware specifications"""
    gpu_type = get_gpu_details()
    return {
        "peak_bandwidth_tb": get_peak_bw(gpu_type) / 1e12,
        "peak_flops_tf": get_peak_flops(gpu_type, precision=used_dtype) / 1e12,
    }


def _extract_output_data(outputs):
    """Extract relevant data from outputs"""
    prefill_lengths = []
    output_lengths = []
    max_duration = 0.0
    
    for x in outputs:
        output_lengths.append(x['meta_info']['completion_tokens'])
        prefill_lengths.append(x['meta_info']['prompt_tokens'])
        max_duration = max(max_duration, x['meta_info']['e2e_latency'])
    
    return {
        'prefill_lengths': prefill_lengths,
        'output_lengths': output_lengths,
        'max_duration': max_duration
    }


def _calculate_kv_size(model_name, hf_config, n_layers, d_head, n_kv_heads):
    """Calculate per-token KV size based on model type"""
    if "DeepSeek" in model_name and hasattr(hf_config, "kv_lora_rank") and hasattr(hf_config, "qk_rope_head_dim"):
        return n_layers * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)
    return 2 * n_layers * d_head * n_kv_heads


def _calculate_attention_size(model_name, hf_config, d_model, n_attn_heads, d_head, n_kv_heads):
    """Calculate attention size per token based on model type"""
    if ("DeepSeek" in model_name and 
        hasattr(hf_config, "qk_rope_head_dim") and 
        hasattr(hf_config, "qk_nope_head_dim") and 
        hasattr(hf_config, "v_head_dim") and 
        hasattr(hf_config, "kv_lora_rank")):
        
        return _calculate_deepseek_attention_size(hf_config, d_model, n_attn_heads)
    
    return (d_model * (n_attn_heads * d_head + n_kv_heads * d_head * 2) + 
            n_attn_heads * d_head * d_model) / 1e12


def _calculate_deepseek_attention_size(hf_config, d_model, n_attn_heads):
    """Calculate DeepSeek-specific attention size"""
    q_head_dim = hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim
    
    base_size = ((d_model * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim)) +
                (hf_config.kv_lora_rank * n_attn_heads * 
                 (q_head_dim - hf_config.qk_rope_head_dim + hf_config.v_head_dim)) +
                (hf_config.v_head_dim * n_attn_heads * d_model))
    
    if hasattr(hf_config, "q_lora_rank") and hf_config.q_lora_rank:
        q_size = (d_model * hf_config.q_lora_rank + 
                 hf_config.q_lora_rank * n_attn_heads * q_head_dim)
    else:
        q_size = d_model * n_attn_heads * q_head_dim
    
    return (base_size + q_size) / 1e12


def _calculate_expert_config(model_name, hf_config, d_ff, d_model, n_layers):
    """Calculate expert configuration based on model type"""
    config = {
        'expert_size': d_ff * 3 * d_model / 1e12,
        'shared_experts_size_total': 0,
        'deepseek_dense_ffn_size': 0,
        'deepseek_sparse_layer_num': 0,
        'deepseek_num_dense_layer': 0
    }
    
    if "Qwen" in model_name and not "Qwen3" in model_name:
        config.update(_get_qwen_expert_config(hf_config, d_model))
    elif "Qwen3" in model_name:
        config.update(_get_qwen3_expert_config(hf_config, d_model))
    elif "DeepSeek" in model_name:
        config.update(_get_deepseek_expert_config(hf_config, d_model, n_layers))
    
    return config


def _get_qwen_expert_config(hf_config, d_model):
    """Get Qwen-specific expert configuration"""
    if (hasattr(hf_config, "moe_intermediate_size") and 
        hasattr(hf_config, "shared_expert_intermediate_size")):
        
        return {
            'expert_size': hf_config.moe_intermediate_size * 3 * d_model / 1e12,
            'shared_experts_size_total': hf_config.shared_expert_intermediate_size * 3 * d_model / 1e12
        }
    return {}


def _get_qwen3_expert_config(hf_config, d_model):
    """Get Qwen3-specific expert configuration"""
    if hasattr(hf_config, "moe_intermediate_size"):
        return {
            'expert_size': hf_config.moe_intermediate_size * 3 * d_model / 1e12
        }
    return {}


def _get_deepseek_expert_config(hf_config, d_model, n_layers):
    """Get DeepSeek-specific expert configuration"""
    if (hasattr(hf_config, "moe_intermediate_size") and 
        hasattr(hf_config, "intermediate_size") and 
        hasattr(hf_config, "first_k_dense_replace")):
        
        deepseek_num_dense_layer = hf_config.first_k_dense_replace
        return {
            'expert_size': hf_config.moe_intermediate_size * 3 * d_model / 1e12,
            'shared_experts_size_total': hf_config.moe_intermediate_size * 3 * d_model * 2 / 1e12,
            'deepseek_dense_ffn_size': hf_config.intermediate_size * 3 * d_model / 1e12,
            'deepseek_sparse_layer_num': n_layers - deepseek_num_dense_layer,
            'deepseek_num_dense_layer': deepseek_num_dense_layer
        }
    return {}


def _process_outputs(output_data, per_token_kv_size, attention_size_per_token, 
                    model_name, hf_config, n_layers, n_attn_heads, d_head):
    """Process outputs to calculate KV sizes and attention scores"""
    kvs = []
    true_kvs = []
    attn_scores = []
    
    for prefill_len, output_len in zip(output_data['prefill_lengths'], output_data['output_lengths']):
        # Calculate attention score
        attn_score = _calculate_attention_score(model_name, hf_config, prefill_len, output_len,
                                              n_layers, n_attn_heads, d_head)
        attn_scores.append(attn_score)
        
        # Calculate KV sizes
        kv_size = (prefill_len * per_token_kv_size + (output_len - 1) * per_token_kv_size / 2) / 1e12
        true_kv = (prefill_len * per_token_kv_size + output_len * per_token_kv_size) / 1e9
        
        kvs.append(kv_size)
        true_kvs.append(true_kv)
    
    return {
        'kv_size': sum(kvs),
        'true_kv_size': sum(true_kvs) * 1e3,
        'attention_score': sum(attn_scores)
    }


def _process_outputs_continuous(out, per_token_kv_size, attention_size_per_token, 
                    model_name, hf_config, n_layers, n_attn_heads, d_head):
    """Process outputs to calculate KV sizes and attention scores"""
    kvs = []
    true_kvs = []
    attn_scores = []
    
    # Calculate attention score
    ctx_len = out['seq_lens_sum']
    attn_score = _calculate_attention_score(model_name, hf_config, ctx_len, 1,
                                          n_layers, n_attn_heads, d_head)
    attn_scores.append(attn_score)
    
    # Calculate KV sizes
    kv_size = (ctx_len * per_token_kv_size) / 1e12
    true_kv = (ctx_len * per_token_kv_size + 1 * per_token_kv_size) / 1e9
    kvs.append(kv_size)
    true_kvs.append(true_kv)
    
    return {
        'kv_size': sum(kvs),
        'true_kv_size': sum(true_kvs) * 1e3,
        'attention_score': sum(attn_scores) / len(attn_scores)
    }


def _calculate_attention_score(model_name, hf_config, prefill_len, output_len, 
                             n_layers, n_attn_heads, d_head):
    """Calculate attention score for a single output"""
    if ("DeepSeek" in model_name and 
        hasattr(hf_config, "qk_rope_head_dim") and 
        hasattr(hf_config, "qk_nope_head_dim") and 
        hasattr(hf_config, "v_head_dim")):
        
        q_head_dim = hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim
        k_size = n_layers * n_attn_heads * q_head_dim
        v_size = n_layers * n_attn_heads * hf_config.v_head_dim
        
        score = (prefill_len * k_size + (output_len - 1) * k_size / 2 +
                prefill_len * v_size + (output_len - 1) * v_size / 2)
    else:
        kv_size = n_layers * n_attn_heads * d_head
        score = (prefill_len * kv_size + (output_len - 1) * kv_size / 2) * 2
    
    return score / 1e12


def _calculate_throughput_metrics(batch_size, prefill_lengths, max_duration):
    """Calculate throughput metrics"""
    total_prefill = sum(prefill_lengths)
    prefill_tp = total_prefill / (max_duration)
    ttft = max_duration / batch_size
    return ttft, prefill_tp


def _calculate_smbu_smfu(model_name, n_layers, attention_size_per_token, expert_config,
                        avg_activated_experts, metrics_data, hardware_specs, num_gpus,
                        precision, ttft, prefill_tp, batch_size, decoding_tp):
    """Calculate S-MBU and S-MFU metrics"""
    prefill_activation = avg_activated_experts[1]
    decode_steps_activation = avg_activated_experts[2:]
    
    # Calculate prefill metrics
    prefill_smbu, prefill_smfu = _calculate_prefill_metrics(
        model_name, n_layers, attention_size_per_token, expert_config,
        prefill_activation, metrics_data['attention_score'], hardware_specs,
        num_gpus, precision, ttft, prefill_tp
    )
    
    # Calculate decoding metrics
    decoding_smbu, decoding_smfu = _calculate_decoding_metrics(
        model_name, n_layers, attention_size_per_token, expert_config,
        decode_steps_activation, metrics_data, hardware_specs,
        num_gpus, precision, batch_size, decoding_tp
    )
    
    return {
        'prefill_smbu': prefill_smbu,
        'prefill_smfu': prefill_smfu,
        'decoding_smbu': decoding_smbu,
        'decoding_smfu': decoding_smfu
    }




def _calculate_prefill_metrics(model_name, n_layers, attention_size_per_token, expert_config,
                             prefill_activation, hardware_specs,
                             num_gpus, precision, ttft, prefill_tp, metrics_data):
    """Calculate prefill S-MBU and S-MFU"""
    model_calculators = {
        'Qwen': _calculate_qwen_prefill,
        'Qwen3': _calculate_qwen3_prefill,
        'DeepSeek': _calculate_deepseek_prefill
    }
    
    for model_type, calculator in model_calculators.items():
        if model_type in model_name and (model_type != 'Qwen' or 'Qwen3' not in model_name):
            return calculator(n_layers, attention_size_per_token, expert_config,
                            prefill_activation, hardware_specs,
                            num_gpus, precision, ttft, prefill_tp, metrics_data)
    
    # Default case
    return _calculate_default_prefill(n_layers, attention_size_per_token, expert_config,
                                    prefill_activation, hardware_specs,
                                    num_gpus, precision, ttft, prefill_tp, metrics_data)


def _calculate_decoding_metrics(model_name, n_layers, attention_size_per_token, expert_config,
                              decode_steps_activation, metrics_data, hardware_specs,
                              num_gpus, precision, batch_size, decoding_tp, tpot=None):
    """Calculate decoding S-MBU and S-MFU"""
    
    if "Qwen" in model_name and "Qwen3" not in model_name:
        smbu, smfu = _calculate_qwen_decoding(n_layers, attention_size_per_token, expert_config,
                                            decode_steps_activation, metrics_data, hardware_specs, num_gpus,
                                            precision, batch_size, decoding_tp, tpot)
    elif "Qwen3" in model_name:
        smbu, smfu = _calculate_qwen3_decoding(n_layers, attention_size_per_token, expert_config,
                                             decode_steps_activation, metrics_data, hardware_specs, num_gpus,
                                             precision, batch_size, decoding_tp, tpot)
    elif "DeepSeek" in model_name:
        smbu, smfu = _calculate_deepseek_decoding(n_layers, attention_size_per_token, expert_config,
                                                decode_steps_activation, metrics_data, hardware_specs, num_gpus,
                                                precision, batch_size, decoding_tp, tpot)
    else:
        smbu, smfu = _calculate_default_decoding(n_layers, attention_size_per_token, expert_config,
                                               decode_steps_activation, metrics_data, hardware_specs, num_gpus,
                                               precision, batch_size, decoding_tp, tpot)

    return smbu, smfu


# Helper functions for specific model calculations
def _calculate_qwen_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation,
                          hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data):
    smbu_numerator = (n_layers * (prefill_activation * expert_config['expert_size'] + 
                                expert_config['shared_experts_size_total'] + 
                                attention_size_per_token) + metrics_data['kv_size']) * precision / ttft
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = (n_layers * (attention_size_per_token + expert_config['expert_size'] + 
                                expert_config['shared_experts_size_total']) + metrics_data['attention_score']) * 2 * prefill_tp
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_qwen3_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation,
                           hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data):
    smbu_numerator = (n_layers * (prefill_activation * expert_config['expert_size'] + 
                                attention_size_per_token) + metrics_data['kv_size']) * precision / ttft
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = (n_layers * (attention_size_per_token + expert_config['expert_size']) + 
                     metrics_data['attention_score']) * 2 * prefill_tp
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_deepseek_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation,
                              hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data):
    smbu_numerator = ((n_layers * attention_size_per_token + 
                      expert_config['deepseek_sparse_layer_num'] * 
                      (prefill_activation * expert_config['expert_size'] + 
                       expert_config['shared_experts_size_total']) + 
                      expert_config['deepseek_num_dense_layer'] * 
                      expert_config['deepseek_dense_ffn_size'] + metrics_data['kv_size']) * precision / ttft)
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = ((n_layers * attention_size_per_token + 
                      expert_config['deepseek_sparse_layer_num'] * 
                      (expert_config['expert_size'] + expert_config['shared_experts_size_total']) + 
                      expert_config['deepseek_num_dense_layer'] * 
                      expert_config['deepseek_dense_ffn_size'] + metrics_data['attention_score']) * 2 * prefill_tp)
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_default_prefill(n_layers, attention_size_per_token, expert_config, prefill_activation,
                             hardware_specs, num_gpus, precision, ttft, prefill_tp, metrics_data):
    # Default implementation
    smbu_numerator = (n_layers * (prefill_activation * expert_config['expert_size'] + 
                                attention_size_per_token) + metrics_data['kv_size']) * precision / ttft
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = (n_layers * (attention_size_per_token + expert_config['expert_size']) + 
                     metrics_data['attention_score']) * 2 * prefill_tp
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_qwen_decoding(n_layers, attention_size_per_token, expert_config, activation,
                           metrics_data, hardware_specs, num_gpus, precision, batch_size=None, decoding_tp=None, tpot=None):

    if tpot is None:
        assert decoding_tp is not None and batch_size is not None, "Either tpot or decoding_tp and batch_size must be provided."
        tpot = batch_size / decoding_tp
    smbu_numerator = ((n_layers * (activation * expert_config['expert_size'] + 
                                 expert_config['shared_experts_size_total'] + 
                                 attention_size_per_token) + 
                      metrics_data['kv_size']) * precision / tpot)
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = ((n_layers * (attention_size_per_token + expert_config['expert_size'] + 
                                 expert_config['shared_experts_size_total']) + 
                      metrics_data['attention_score']) * 2 * decoding_tp)
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_qwen3_decoding(n_layers, attention_size_per_token, expert_config, activation,
                            metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot=None):
    if tpot is None:
        tpot = batch_size / decoding_tp

    smbu_numerator = ((n_layers * (activation * expert_config['expert_size'] + 
                                 attention_size_per_token) + 
                      metrics_data['kv_size']) * precision / tpot)
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = ((n_layers * (attention_size_per_token + expert_config['expert_size']) + 
                      metrics_data['attention_score']) * 2 * decoding_tp)
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_deepseek_decoding(n_layers, attention_size_per_token, expert_config, activation,
                               metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot=None):
    if tpot is None:
        tpot = batch_size / decoding_tp

    smbu_numerator = ((n_layers * attention_size_per_token + 
                      expert_config['deepseek_sparse_layer_num'] * 
                      (activation * expert_config['expert_size'] + 
                       expert_config['shared_experts_size_total']) + 
                      expert_config['deepseek_num_dense_layer'] * 
                      expert_config['deepseek_dense_ffn_size'] + 
                      metrics_data['kv_size']) * precision / tpot)
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = ((n_layers * attention_size_per_token + 
                      expert_config['deepseek_sparse_layer_num'] * 
                      (expert_config['expert_size'] + expert_config['shared_experts_size_total']) + 
                      expert_config['deepseek_num_dense_layer'] * 
                      expert_config['deepseek_dense_ffn_size'] + 
                      metrics_data['attention_score']) * 2 * decoding_tp)
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu


def _calculate_default_decoding(n_layers, attention_size_per_token, expert_config, activation,
                              metrics_data, hardware_specs, num_gpus, precision, batch_size, decoding_tp, tpot=None):
    if tpot is None:
        tpot = batch_size / decoding_tp

    smbu_numerator = ((n_layers * (activation * expert_config['expert_size'] + 
                                 attention_size_per_token) + 
                      metrics_data['kv_size']) * precision / tpot)
    smbu = smbu_numerator / (num_gpus * hardware_specs['peak_bandwidth_tb'])
    
    smfu_numerator = ((n_layers * (attention_size_per_token + expert_config['expert_size']) + 
                      metrics_data['attention_score']) * 2 * decoding_tp)
    smfu = smfu_numerator / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return smbu, smfu

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
    elif "Qwen3" in model_name and hasattr(hf_config, "moe_intermediate_size"):
        d_ff = hf_config.moe_intermediate_size
        expert_size = d_ff * 3 * d_model
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
            * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    elif "Qwen3" in model_name:
        smbu = ((n_layers * (avg_activated_experts * expert_size + attention_size_per_token) + 
                kv_size) * precision/(batch_size / decoding_tp) ) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * (attention_size_per_token + n_experts_per_tok * expert_size) + attention_score) \
            * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    elif "DeepSeek" in model_name:
        smbu = ((n_layers * attention_size_per_token + deepseek_sparse_layer_num * \
                (avg_activated_experts * expert_size + shared_experts_size_total) + \
                deepseek_num_dense_layer * deepseek_dense_ffn_size + \
                kv_size) * precision/(batch_size / decoding_tp) ) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * attention_size_per_token + deepseek_sparse_layer_num * \
                (n_experts_per_tok * expert_size + shared_experts_size_total) + \
                deepseek_num_dense_layer * deepseek_dense_ffn_size + attention_score) \
                * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    else:
        smbu = ((n_layers*(avg_activated_experts * expert_size + attention_size_per_token) + 
                kv_size) * precision/(batch_size / decoding_tp) ) / (num_gpus * hardware_specs['peak_bandwidth_tb'])
        smfu = (n_layers * (attention_size_per_token + n_experts_per_tok * expert_size) + attention_score) \
            * 2 * decoding_tp / (num_gpus * hardware_specs['peak_flops_tf'] / 2)
    
    return {
        'smbu': smbu,
        'smfu': smfu,
        'kv_size': true_kv_size,
        'decoding_throughput': decoding_tp,
        'ttft': 0
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