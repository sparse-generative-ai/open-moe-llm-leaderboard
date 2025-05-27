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

class MoEActivationAnalyzer:
    def __init__(self, model_name, batch_size=1, task="gsm8k", device="cuda:0"):
        self.device = device
        self.batch_size = batch_size
        self.task = task
        
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

    def _wrap_forward_with_logging(self, original_forward, layer_id):
        """Wrap the forward pass to log router logits"""
        def new_forward(*args, **kwargs):
            if not self.prefilling_finished:
                self.prefilling_finished = True
                do_record = False
            else:
                do_record = True
                self.record.append({"layer_id": layer_id})
                self.current_layer = layer_id
                kwargs["output_router_logits"] = True
                
            # Call the original forward method
            output = original_forward(*args, **kwargs)
            
            if do_record:
                # Get router logits and track expert activations
                router_logits = output[-1]
                self.record[-1]["router_logits"] = router_logits.cpu()
                
                # Track expert activations per layer
                activated_experts = torch.topk(router_logits, self.num_experts_per_token, dim=-1)
                experts_list = activated_experts.indices.flatten().tolist()
                
                # Count activations for each expert in this layer
                for expert_id in experts_list:
                    self.layer_expert_counts[layer_id][expert_id] += 1
                
            return output
        
        return new_forward

    def _inject_logging(self, model):
        """Inject logging into the model's forward passes"""
        for layer_id in range(self.n_layers):
            if "dbrx" in self.hf_model_name:
                original_forward = model.transformer.blocks[layer_id].forward
                model.transformer.blocks[layer_id].original_forward = original_forward
                model.transformer.blocks[layer_id].forward = self._wrap_forward_with_logging(original_forward, layer_id)
            else:
                original_forward = model.model.layers[layer_id].forward
                model.model.layers[layer_id].original_forward = original_forward
                model.model.layers[layer_id].forward = self._wrap_forward_with_logging(original_forward, layer_id)
        return model

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
            all_input_raw = raw_data['content']
        elif self.task == 'MATH':
            max_new_tokens = 512
            dataset = MATH()
            raw_data = dataset.validation_docs()
            all_input_raw = raw_data['content']
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        return all_input_raw, max_new_tokens

    def _prepare_inputs(self, all_input_raw, tokenizer):
        """Prepare inputs for the model"""
        if "Qwen" in self.hf_model_name:
            all_input_chat = [[{"role": "system", "content": "You are a helpful assistant."}, 
                           {"role": "user", "content": f"{input_string}"}] for input_string in all_input_raw]
        else:
            all_input_chat = [[{"role": "user", "content": input_string}] for input_string in all_input_raw]
        
        if "Qwen" in self.hf_model_name or "dbrx" in self.hf_model_name:
            all_input_updated = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) 
                                for chat in all_input_chat]
        else:
            all_input_updated = [tokenizer.apply_chat_template(chat, tokenize=False) 
                                for chat in all_input_chat]
            
        # Create batches
        return [all_input_updated[i:i + self.batch_size] for i in range(0, len(all_input_updated), self.batch_size)]

    def analyze_expert_activations(self):
        """Analyze expert activations from the collected records"""
        num_experts = []
        for record_item in self.record:
            router_logits = record_item["router_logits"]
            activated_experts = torch.topk(router_logits, self.num_experts_per_token, dim=-1)
            num_experts.append(len(set(activated_experts.indices.flatten().tolist())))
        
        avg_activated_experts = sum(num_experts) / len(num_experts) if num_experts else 0
        return avg_activated_experts

    def get_model_simple_name(self):
        """Get a simplified name for the model for file naming"""
        return self.hf_model_name.replace("/", "_").replace(".", "_")

    def write_to_csv(self, avg_activated_experts):
        """Write summary results to a CSV file"""
        csv_filename = f"activation_profiling_results/{self.get_model_simple_name()}.csv"
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['dataset', 'batch_size', 'average_activated_experts']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'dataset': self.task, 
                'batch_size': self.batch_size, 
                'average_activated_experts': avg_activated_experts
            })

    def write_expert_counts_to_csv(self):
        """Write detailed expert activation counts to a CSV file"""
        os.makedirs(f"activation_profiling_results/", exist_ok=True)
        os.makedirs(f"activation_profiling_expert_count/", exist_ok=True)
        csv_filename = f"activation_profiling_layer_count/{self.get_model_simple_name()}_dataset_{self.task}_bs{self.batch_size}_expert_counts.csv"
        
        with open(csv_filename, 'w', newline='') as csvfile:
            # Create header with layer and expert IDs
            fieldnames = ['layer_id'] + [f'expert_{i}' for i in range(self.n_experts)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data for each layer
            for layer_id in range(self.n_layers):
                row = {'layer_id': layer_id}
                for expert_id in range(self.n_experts):
                    row[f'expert_{expert_id}'] = self.layer_expert_counts[layer_id][expert_id]
                writer.writerow(row)
        
        print(f"Expert counts saved to {csv_filename}")

    def run(self):
        """Run the complete analysis pipeline"""
        print(f"Starting analysis for {self.hf_model_name} on {self.task} with batch size {self.batch_size}")

        # Check if we already have results for this configuration
        csv_filename = f"{self.get_model_simple_name()}.csv"
        if os.path.exists(csv_filename):
            with open(csv_filename, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if this exact configuration has been evaluated
                    if row['dataset'] == self.task and int(row['batch_size']) == self.batch_size:
                        print(f"Found existing results for {self.task} with batch size {self.batch_size}")
                        print(f"Average activated experts: {row['average_activated_experts']}")
                        return float(row['average_activated_experts']), None

        # If we get here, we need to run the analysis
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        # Inject logging
        model = self._inject_logging(model)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

        # Load and prepare data
        all_input_raw, max_new_tokens = self._load_data()
        batched_input_data = self._prepare_inputs(all_input_raw, tokenizer)

        # Variables to track convergence
        prev_avg_activated_experts = 0
        convergence_threshold = 1.0  # Stop if difference is less than this threshold

        # Process data in batches
        with torch.no_grad():
            for i, batch in tqdm(enumerate(batched_input_data), total=len(batched_input_data)):
                inputs = tokenizer(batch, return_tensors="pt", padding="longest").to(self.device)
                self.prefilling_finished = False
                output = model.generate(**inputs, max_new_tokens=max_new_tokens)

                # Check for convergence after each batch (except the first one)
                if i > 0:
                    current_avg_activated_experts = self.analyze_expert_activations()

                    print(f"Batch {i}: Avg activated experts = {current_avg_activated_experts}")

                    # Check if the difference is less than the threshold
                    if abs(current_avg_activated_experts - prev_avg_activated_experts) < convergence_threshold:
                        print(f"Convergence reached at batch {i}. Stopping early.")
                        break
                    
                    prev_avg_activated_experts = current_avg_activated_experts

        # Analyze expert activations
        avg_activated_experts = self.analyze_expert_activations()
        print(f"Average number of activated experts: {avg_activated_experts}")

        # Write summary results to CSV
        self.write_to_csv(avg_activated_experts)

        # Write detailed expert activation counts to CSV
        self.write_expert_counts_to_csv()

        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return avg_activated_experts, self.layer_expert_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "arena_hard", "MATH"])
    args = parser.parse_args()
    
    analyzer = MoEActivationAnalyzer(
        model_name=args.model_name,
        batch_size=args.batch_size,
        task=args.task
    )
    
    analyzer.run()


if __name__ == "__main__":
    main()