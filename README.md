---
title: OPEN-MOE-LLM-LEADERBOARD
emoji: 🔥
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: true
license: apache-2.0
fullWidth: true
tags:
  - leaderboard
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Current leaderboard is at https://huggingface.co/spaces/sparse-generative-ai/open-moe-llm-leaderboard. It is still under developing! Stay tuned!

# Contributing to Open-MOE-LLM-Leaderboard

Thank you for your interest in contributing to the Open-MOE-LLM-Leaderboard project! We welcome contributions from everyone. Below you'll find guidance on how to set up your development environment, understand our architecture, and contribute effectively. If you have any questions or wish to discuss your contributions, please reach out to Yinsicheng Jiang or Yao Fu via email at [ysc.jiang@ed.ac.uk](mailto:ysc.jiang@ed.ac.uk) or [Y.Fu@ed.ac.uk](mailto:y.fu@ed.ac.uk).

## What We're Looking For in Contributions

We are looking for contributions in several key areas to enhance the Open-MOE-LLM-Leaderboard project:

1. **General Bug Fixes/Reports**: We welcome reports of any bugs found in the frontend interface or backend, as well as fixes for these issues.

2. **Adding New Tasks (Benchmark Datasets)**: If you have ideas for new benchmark datasets that could be added, your contributions would be greatly appreciated.

3. **Supporting New Inference Frameworks**: Expanding our project to support new inference frameworks is crucial for our growth. If you can contribute in this area, please reach out.

4. **Testing More Models**: To make our leaderboard as comprehensive as possible, we need to test a wide range of models. Contributions in this area are highly valuable.

Documentation is currently of lower priority, but if you have thoughts or suggestions, please feel free to raise them.

Your contributions are crucial to the success and improvement of the Open-MOE-LLM-Leaderboard project. We look forward to collaborating with you.


## Development Setup

To start contributing, set up your development environment as follows:

```bash
conda create -n leaderboard python=3.10
conda activate leaderboard
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ moe-infinity
pip install pydantic==2.6.4 # Resolves a dependency conflict with moe-infinity
python -m spacy download en # Required for selfcheckgpt
```

## Architecture Overview

The Open-MOE-LLM-Leaderboard project uses the following architecture:

- **User Interface (Gradio)** ->upload-> **HuggingFace Dataset (Request)** ->download-> **Backend GPU Server** ->upload-> **HuggingFace Dataset (Result)** ->download-> **User Interface (Gradio)**

In brief:
1. Users submit model benchmarking requests through the Gradio interface ([app.py](./app.py)). These requests are then recorded in a HuggingFace dataset ([sparse-generative-ai/requests](https://huggingface.co/datasets/sparse-generative-ai/requests)).
2. The backend ([backend-cli.py](./backend-cli.py)), running on a GPU server, processes these requests, performs the benchmarking tasks, and uploads the results to another HuggingFace dataset ([sparse-generative-ai/results](https://huggingface.co/datasets/sparse-generative-ai/results)).
3. Finally, the Gradio interface retrieves and displays these results to the users.

# Quick Start
## Profiling Expert Activation
Example usage is like in `run_profiling.py`
```python
from profiling.vllm_profiler.vllm_profiling import VLLMMoEProfiler
import argparse

def main():
    """Main function to parse arguments and run the profiler."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='Model name or path')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
    parser.add_argument('--tensor_parallel_size', default=1, type=int, help='Number of GPUs for tensor parallelism')
    parser.add_argument('--quant', default=None, type=str, help='Quantization type')
    parser.add_argument('--dtype', default="bfloat16", type=str, help='Data type for computation')
    parser.add_argument('--load_format', default="auto", type=str, help='Model loading format')
    parser.add_argument('--gpu_memory_utilization', default=0.9, type=float, help='Target GPU memory utilization')
    parser.add_argument('--temperature', default=0.0, type=float, help='Sampling temperature')
    parser.add_argument('--max_new_tokens', default=50, type=int, help='Maximum new tokens to generate')
    parser.add_argument('--output_dir', default='activation_profiling_results/', type=str, help='Output directory')
    parser.add_argument('--trust_remote_code', action='store_true', default=False, help='Trust remote code for model loading')
    parser.add_argument('--max_model_len', type=int, help='Maximum sequence length')
    parser.add_argument('--num_layers', type=int, help='Override number of layers')
    parser.add_argument('--hidden_size', type=int, help='Override hidden size')
    
    args = parser.parse_args()
    
    profiler = VLLMMoEProfiler(args)
    profiler.run()

if __name__ == "__main__":
    main()
```
Quickly run by:
```
python run_profiling.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --batch_size 16 --tensor_parallel_size 4 --trust_remote_code
```

The result will be in `activation_profilling_results/`
```csv
dataset,batch_size,average activated experts
MATH_4000,16,7.7469783834586465
```
## Running Evaluation
```
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1,2,3 python backend-cli.py  --debug \
                                                --task arena_hard \
                                                --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                                                --precision bfloat16 \
                                                --inference-framework vllm_moe \
                                                --gpu-type NVIDIA-RTX-A6000-48GB \
                                                --batch_size 16 \
                                                --activation_profile_path /path/to/activation_profiling_results \
                                                --tensor_parallel_size 4
```

We current only stably support vLLM, Huggingface Transformers, MoE-Infinity and Accelerator. We will add more systems like SGLang and MoE-Gen. Stay tuned!
## Running the Gradio Interface

To launch the Gradio interface, execute:

```bash
python app.py
```

Then, open your browser and navigate to http://127.0.0.1:7860.

## Running the Backend

To start the backend process, use:

```bash
python backend-cli.py --debug
```

For additional details, please consult the [backend-cli.py](./backend-cli.py) script.

## Draw CAP Radar Diagram
Prepare a YAML file:
```python
import yaml

# Generate a template YAML configuration for the radar plot
config = {
    'model_name': 'Qwen3-30B-A3B',
    'axis_labels': ['Performance (s/token)', 'Cost ($)', 'Accuracy'],
    'baseline': 20,
    'ticks': [0, 2000, 4000, 6000, 8000, 10000],
    'data': {
        'SGLang': {'TPOT': 0.058, 'Cost': 8920, 'Accuracy': 0.911},
        'K-Transformers': {'TPOT': 0.073, 'Cost': 3800, 'Accuracy': 0.800},
        'MoE-Infinity': {'TPOT': 0.15, 'Cost': 3800, 'Accuracy': 0.911}
    }
}

# Save the YAML config file
config_path = "/path/to/radar_config.yaml"
with open(config_path, 'w') as file:
    yaml.dump(config, file)
```
Then draw a plot:
```bash
python draw_radar.py radar_config.yaml
```
Radar example: 

![CAP Radar](assets/radar.png)
---

## Current MoE System Benchmarking

![moe-benchmark](assets/moe-benchmark.png)

We look forward to your contributions and are here to help guide you through the process. Thank you for supporting the Open-MOE-LLM-Leaderboard project!