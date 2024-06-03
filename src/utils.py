import pandas as pd
from huggingface_hub import snapshot_download
import subprocess
import re
import os
import GPUtil

try:
    from src.display.utils import GPU_TEMP, GPU_Mem, GPU_Power, GPU_Util, GPU_Name
except:
    print("local debug: from display.utils")
    from display.utils import GPU_TEMP, GPU_Mem, GPU_Power, GPU_Util, GPU_Name
    
MEM_BW_DICT ={
    "NVIDIA-A100-PCIe-80GB": 1935,
    "NVIDIA-A100-SXM-80GB": 2039,
    "NVIDIA-H100-PCIe-80GB": 2039,
    "NVIDIA-RTX-A5000-24GB": 768
}

PEAK_FLOPS_DICT = {
    "float32":{
        "NVIDIA-A100-PCIe-80GB": 312e12,
        "NVIDIA-A100-SXM-80GB": 312e12,
        "NVIDIA-H100-PCIe-80GB": 756e12,
        "NVIDIA-RTX-A5000-24GB": 222.2e12
    },
    "float16":{
        "NVIDIA-A100-PCIe-80GB": 624e12,
        "NVIDIA-A100-SXM-80GB": 624e12,
        "NVIDIA-H100-PCIe-80GB": 1513e12,
        "NVIDIA-RTX-A5000-24GB": 444.4e12
    },
    "bfloat16":{
        "NVIDIA-A100-PCIe-80GB": 624e12,
        "NVIDIA-A100-SXM-80GB": 624e12,
        "NVIDIA-H100-PCIe-80GB": 1513e12,
        "NVIDIA-RTX-A5000-24GB": 444.4e12
    },
    "8bit":{
        "NVIDIA-A100-PCIe-80GB": 1248e12,
        "NVIDIA-A100-SXM-80GB": 1248e12,
        "NVIDIA-H100-PCIe-80GB": 3026e12,
        "NVIDIA-RTX-A5000-24GB": 889e12
    },
    "4bit": {
        "NVIDIA-A100-PCIe-80GB": 2496e12,
        "NVIDIA-A100-SXM-80GB": 2496e12,
        "NVIDIA-H100-PCIe-80GB": 6052e12,
        "NVIDIA-RTX-A5000-24GB": 1778e12
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
            # print matched groups
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
        # print(f"gpu_stats: {gpu_stats}")
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

def transfer_precision2bytes(precision):
    if precision == "float32":
        return 4
    elif precision in ["float16", "bfloat16"]:
        return 2
    elif precision == "8bit":
        return 1
    elif precision == "4bit":
        return 0.5
    else:
        raise ValueError(f"Unsupported precision: {precision}")

if __name__ == "__main__":
    print(analyze_gpu_stats(parse_nvidia_smi()))
    print(get_gpu_details())