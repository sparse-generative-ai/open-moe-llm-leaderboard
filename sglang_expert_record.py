import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import pickle
import argparse
import os
import json
import csv
from tqdm import tqdm
import gc
import numpy as np
from collections import defaultdict, Counter
from src.utils import ModelInfoRetriever
from src.backend.tasks import MATH, ArenaHard
import requests
from sglang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
import sglang as sgl
import pandas as pd

os.environ["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = os.path.join(os.getcwd(), "activation_profiling_layer_count")

class SGLangMoEActivationAnalyzer:
    def __init__(self, model_name, pt_output_dir, batch_size="auto", task="gsm8k", device="cuda:0"):
        self.device = device
        self.batch_size = batch_size
        self.task = task
        self.pt_output_dir = pt_output_dir
        os.makedirs(self.pt_output_dir, exist_ok=True)
        
        # Map from shorthand name to HF model name
        self.hf_model_name = model_name
        self.model_info = ModelInfoRetriever(self.hf_model_name)
        self.n_layers = self.model_info.config.num_hidden_layers
        moe_info = self.model_info.get_moe_info()
        self.n_experts = moe_info["num_experts"]
        self.num_experts_per_token = moe_info["experts_per_token"]
        
        # Initialize record storage
        self.record = []
        self.prefilling_finished = False
        
        # Expert activation tracking
        self.layer_expert_counts = defaultdict(Counter)
        self.current_layer = None


    def _load_data(self):
        """Load data based on the selected task"""
        if self.task == "gsm8k":
            gsm8k = load_dataset("gsm8k", "main", split="test")
            all_input_raw = gsm8k['question']
            max_new_tokens = 256
        elif self.task == 'arena_hard':
            max_new_tokens = 4096
            dataset = ArenaHard()
            raw_data = dataset.validation_docs()
            all_input_raw = raw_data['content'][:50]
        elif self.task == 'MATH':
            max_new_tokens = 512
            dataset = MATH()
            raw_data = dataset.validation_docs()
            all_input_raw = raw_data['content']
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        return all_input_raw, max_new_tokens

    def _prepare_inputs(self, all_input_raw, max_new_tokens):
        """Prepare inputs for the model"""       
        # Create batches
        if self.batch_size == 'auto':
            arguments = [{"question": q, "max_new_tokens": max_new_tokens} for q in all_input_raw]
            return arguments  # Single batch for auto mode
        else:
            batch_size = int(self.batch_size)
            batched_arguments = []
            for i in range(0, len(all_input_raw), batch_size):
                batch = all_input_raw[i:i + batch_size]
                batched_arguments.append([{"question": q, "max_new_tokens": max_new_tokens} for q in batch])
            return batched_arguments
    
    def calculate_average_activated_experts(self, token_counts_matrix, dataset_name, output_file='expert_activation.csv'):
        """
        Calculate average activated experts per layer for each step.

        Args:
            token_counts_matrix: torch.Tensor of shape [extra_dim, num_layer, num_experts]
            dataset_name: str, name of the dataset
            output_file: str, output CSV filename
        """
        extra_dim, num_layer, num_experts = token_counts_matrix.shape

        # Calculate activated experts for each step and layer
        # An expert is considered activated if it has any tokens (> 0)
        activated_experts = (token_counts_matrix > 0).float()  # Shape: [extra_dim, num_layer, num_experts]

        # Sum across experts dimension to get number of activated experts per layer per step
        activated_per_layer_per_step = activated_experts.sum(dim=2)  # Shape: [extra_dim, num_layer]

        # Average across layers to get one number per step
        avg_activated_per_step = activated_per_layer_per_step.mean(dim=1)  # Shape: [extra_dim]

        # Convert to numpy for easier handling
        avg_activated_per_step = avg_activated_per_step.cpu().numpy()

        # Create column names
        columns = ['dataset', 'prefill'] + [f'decoding_step_{i}' for i in range(1, extra_dim)]

        # Create the data row
        data_row = [dataset_name] + avg_activated_per_step.tolist()

        # Create DataFrame
        df = pd.DataFrame([data_row], columns=columns)

        # Save to CSV (append if file exists, create if not)
        try:
            # Try to read existing file and append
            existing_df = pd.read_csv(output_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            # File doesn't exist, df will be the first entry
            pass
        
        df.to_csv(output_file, index=False)

        print(f"Average activated experts per step saved to {output_file}")
        print(f"Shape of input matrix: {token_counts_matrix.shape}")
        print(f"Average activated experts - Prefill: {avg_activated_per_step[0]:.2f}")
        if len(avg_activated_per_step) > 1:
            print(f"Average activated experts - Decoding steps: {avg_activated_per_step[1:].mean():.2f}")

        return avg_activated_per_step

    # Alternative function if you want to see per-layer statistics as well
    def calculate_detailed_expert_stats(self, token_counts_matrix, dataset_name, output_file='detailed_expert_activation.csv'):
        """
        Calculate detailed statistics including per-layer information.
        """
        extra_dim, num_layer, num_experts = token_counts_matrix.shape

        # Calculate activated experts for each step and layer
        activated_experts = (token_counts_matrix > 0).float()
        activated_per_layer_per_step = activated_experts.sum(dim=2)

        # Create detailed DataFrame
        data_rows = []

        for step in range(extra_dim):
            step_name = 'prefill' if step == 0 else f'decoding_step_{step}'

            # Overall average for this step
            overall_avg = activated_per_layer_per_step[step].mean().item()

            row = {
                'dataset': dataset_name,
                'step': step_name,
                'avg_activated_experts': overall_avg,
                'min_activated_experts': activated_per_layer_per_step[step].min().item(),
                'max_activated_experts': activated_per_layer_per_step[step].max().item(),
                'std_activated_experts': activated_per_layer_per_step[step].std().item()
            }
            data_rows.append(row)

        df = pd.DataFrame(data_rows)

        # Save to CSV
        try:
            existing_df = pd.read_csv(output_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
        
        df.to_csv(output_file, index=False)

        return df

    def analyze_expert_activations(self):
        """Analyze expert activations from the collected records"""
        all_pt_files = [f for f in os.listdir(self.pt_output_dir) if os.path.isfile(os.path.join(self.pt_output_dir, f))]
        sorted_files = sorted(all_pt_files, key=lambda x: float(x.split('_')[-1].split('.pt')[0]))
        last_pt_file = sorted_files[-1] if sorted_files else None
        assert last_pt_file is not None, "No expert activation record found."
        activation_dict = torch.load(os.path.join(self.pt_output_dir, last_pt_file), map_location='cpu')['logical_count']

        avg_activated = self.calculate_average_activated_experts(
            activation_dict, 
            dataset_name=self.task,
            output_file=f"{self.pt_output_dir}/activation_results/{self.task}_expert_records.csv"
        )

        # Also calculate detailed stats if needed
        detailed_stats = self.calculate_detailed_expert_stats(
            activation_dict,
            dataset_name=self.task, 
            output_file=f"{self.pt_output_dir}/activation_results/{self.task}_detailed_expert_activation.csv"
        )

    def get_model_simple_name(self):
        norm_path = os.path.normpath(self.hf_model_name)
        parts = norm_path.split(os.sep)
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        else:
            return self.hf_model_name

    @sgl.function
    def run_sgl(s, question, max_new_tokens):
        s += question
        s += sgl.gen(
            "answer",
            max_tokens=max_new_tokens,
            stop=["Question", "Assistant:", "<|separator|>"],
        ) 

    def run(self, port=30000):
        # Load data
        all_input_raw, max_new_tokens = self._load_data()
        set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
        # Prepare inputs
        # all_input_raw = all_input_raw[:10]  # Limit to 5 samples for testing
        batched_inputs = self._prepare_inputs(all_input_raw, max_new_tokens)
        # Start recording
        response = requests.post(f"http://localhost:{port}/start_expert_distribution_record")
        response.raise_for_status()
        print("Started expert distribution recording.")
        # Run inference
        if self.batch_size != 'auto':
            activation = []
            for batch in tqdm(batched_inputs, desc=f"Processing {self.task} dataset"):
                # sglang server takes a list of prompts for batching
                response = requests.post(f"http://localhost:{port}/start_expert_distribution_record")
                states = self.run_sgl.run_batch(
                    batch,
                    temperature=0,
                    num_threads=128,
                    progress_bar=True)
                response = requests.post(f"http://localhost:{port}/stop_expert_distribution_record")
                response.raise_for_status()
                response = requests.post(f"http://localhost:{port}/dump_expert_distribution_record")
                response.raise_for_status()
                # Analyze the expert distribution
                self.analyze_expert_activations()
        else:
            # For auto mode, we send all prompts in a single request
            states = self.run_sgl.run_batch(
                    batched_inputs,
                    temperature=0,
                    num_threads=128,
                    progress_bar=True)
            
        # Stop recording
            response = requests.post(f"http://localhost:{port}/stop_expert_distribution_record")
            response.raise_for_status()
            print("Stopped expert distribution recording.")
            # Dump expert distribution record
            response = requests.post(f"http://localhost:{port}/dump_expert_distribution_record")
            response.raise_for_status()
            # Analyze the expert distribution
            self.analyze_expert_activations()
            print(f"Results written to CSV for task: {self.task}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "arena_hard", "MATH"])
    parser.add_argument("--port", type=int, default=30000, help="Port for the SGLang server")
    parser.add_argument("--pt_output_dir", type=str)
    args = parser.parse_args()
    
    analyzer = SGLangMoEActivationAnalyzer(
        model_name=args.model_name,
        pt_output_dir=args.pt_output_dir,
        batch_size=args.batch_size,
        task=args.task
    )
    
    analyzer.run(args.port)


if __name__ == "__main__":
    main()
