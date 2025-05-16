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