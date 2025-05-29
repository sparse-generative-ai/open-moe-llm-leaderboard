import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import argparse
import os
import csv
from tqdm import tqdm
import gc
from collections import defaultdict, Counter
from src.utils import ModelInfoRetriever # Assuming these are provided
from src.backend.tasks import MATH, ArenaHard # Assuming these are provided
import pandas as pd
import deepspeed

class MoEActivationAnalyzer:
    def __init__(self, model_name, batch_size=1, task="gsm8k"):
        # No deepspeed.init_distributed() here, as init_inference will handle it
        # or we might need it before AutoConfig if model loading needs dist info.
        # Let's keep it for now, init_inference is fine with pre-initialized dist.
        if not dist.is_initialized():
            deepspeed.init_distributed(dist_backend='nccl')

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.getenv('LOCAL_RANK', '0'))

        self.batch_size = batch_size 
        self.task = task
        self.is_distributed = self.world_size > 1 # True if TP size > 1
        self.process_index = self.rank
        
        self.hf_model_name = model_name
        # ModelInfoRetriever might need to be initialized carefully if config comes from DS
        # For now, assume it can work with the HF model name initially
        self.model_info = ModelInfoRetriever(self.hf_model_name) # This uses HF config
        self.n_layers = self.model_info.config.num_hidden_layers # From HF config
        moe_info = self.model_info.get_moe_info() # From HF config
        self.n_experts = moe_info["num_experts"]
        self.num_experts_per_token = moe_info["experts_per_token"]
        
        self.prefilling_finished = False
        self.layer_expert_counts = defaultdict(Counter)
        self._temp_num_unique_list_for_generate = [] 
        
        if self.rank == 0:
            self.main_process_cumulative_num_unique_list = []
            self.prev_avg_activated_experts = 0.0
            self.final_avg_activated_experts = 0.0

    def _wrap_forward_with_logging(self, original_forward, layer_id, hook_type="generic_layer"):
        # (This function remains largely the same as your previous version)
        # Key: Ensure kwargs["output_router_logits"] = True is respected by the (potentially DeepSpeed-injected) module
        def new_forward(*args, **kwargs):
            is_decode_phase_for_this_forward = self.prefilling_finished
            
            if not self.prefilling_finished:
                self.prefilling_finished = True 
                do_record = False 
            else:
                do_record = True 
            
            if do_record:
                kwargs["output_router_logits"] = True
                
            output = original_forward(*args, **kwargs)
            
            if do_record:
                router_logits = None
                # Attempt to extract router_logits (same logic as before)
                # This might need debugging if DeepSpeed kernels change output structure
                if hook_type == "dbrx_block": # DBRX block output structure
                    if hasattr(output, 'router_logits') and output.router_logits is not None:
                        router_logits = output.router_logits
                # elif hook_type == "mixtral_moe_block": # Mixtral MoE block output (usually (hidden_states, router_logits))
                #     if isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1]):
                #         router_logits = output[1]
                elif hook_type == "mixtral_decoder_layer": # Mixtral/Qwen whole decoder layer
                     if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[-1]): # often (hidden_states, ..., router_logits_tuple)
                        # If router_logits is a tuple from the main block, need to check its contents
                        # This part assumes the hooked forward directly gives the tensor
                        router_logits = output[-1] 
                
                if router_logits is None: # Fallbacks
                    if hasattr(output, 'router_logits') and output.router_logits is not None:
                        router_logits = output.router_logits
                    elif isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1]):
                        router_logits = output[1]
                    elif isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[-1]): # Last tensor in output tuple
                        router_logits = output[-1]

                if router_logits is not None:
                    # If router_logits is a tuple (e.g. from a Mixtral *decoder layer* rather than MoE block itself)
                    # we need to handle it. The hook should be on the module producing *one* set of router logits.
                    # Assuming `router_logits` here is the actual tensor of shape [tokens, num_experts]
                    if not isinstance(router_logits, torch.Tensor):
                        if self.rank == 0 and layer_id == 0:
                             print(f"Warning: MoEActivationAnalyzer: router_logits for layer {layer_id} is not a tensor, but {type(router_logits)}. Skipping recording for this call.")
                        return output # or just proceed without recording

                    activated_experts_indices = torch.topk(router_logits, self.num_experts_per_token, dim=-1).indices
                    flat_indices = activated_experts_indices.flatten().tolist()
                    num_unique_for_this_call = len(set(flat_indices))
                    self._temp_num_unique_list_for_generate.append(num_unique_for_this_call)
                    
                    for expert_id in flat_indices:
                        self.layer_expert_counts[layer_id][expert_id] += 1
                else:
                    if self.rank == 0 and layer_id == 0 : 
                        module_name = "Unknown"
                        try: module_name = original_forward.__self__.__class__.__qualname__
                        except: pass
                        print(f"Warning: MoEActivationAnalyzer: Could not extract router_logits for layer {layer_id} (module: {module_name}, hook_type: {hook_type}). Output type: {type(output)}.")
            return output
        return new_forward
    
    def _inject_logging(self, model_engine): # model_engine is now the DeepSpeed engine
        # THIS IS THE TRICKIEST PART.
        # After `deepspeed.init_inference`, `model_engine.module` gives you access to the original model,
        # but some layers might have been replaced by DeepSpeed kernels if `replace_with_kernel_inject` is true.
        # You need to inspect the structure of `model_engine.module` after DeepSpeed initialization.
        
        # Access the underlying HF model (it might be wrapped)
        hf_model = model_engine.module 
        if self.rank == 0: print(f"Attempting to inject logs into model of type: {type(hf_model)}")

        for layer_id in range(self.n_layers):
            target_module = None
            original_forward_attr_name = f"_original_forward_l{layer_id}"
            hook_type_for_layer = "generic_layer" # Default

            # Try to find the MoE layers within the (potentially DeepSpeed-modified) hf_model structure
            # The paths might change slightly if DeepSpeed replaces modules.
            # Example for DBRX (MoE is in FFN block of each transformer block)
            if "dbrx" in self.hf_model_name.lower():
                # DBRX: model.transformer.blocks[idx].ffn (DatabrickMoE)
                # We want to hook the ffn.forward or the block.forward
                # Let's try hooking the block's forward, assuming it passes output_router_logits down
                try:
                    # If DeepSpeed has replaced transformer.blocks[idx] with its own TP version,
                    # this path might still work if the sub-modules are preserved.
                    # Or, DeepSpeed might have a specific way to access the original sub-modules.
                    # For DBRX the MoE is within the block, so hooking the block itself is safer
                    # if the block's forward can return router logits.
                    # model.transformer.blocks[layer_id].ffn (if ffn has router logits output)
                    # model.transformer.blocks[layer_id] (if block itself can output router logits)
                    target_module = hf_model.transformer.blocks[layer_id] # Hook the entire block
                    hook_type_for_layer = "dbrx_block"
                    if self.rank == 0 and layer_id == 0: print(f"DBRX: Targeting transformer.blocks[{layer_id}]")
                except Exception as e:
                    if self.rank == 0: print(f"Could not access DBRX block {layer_id}: {e}")
                    continue
            
            # Example for Mixtral/Qwen MoE (MoE is a specific submodule)
            elif hasattr(hf_model, 'model') and hasattr(hf_model.model, 'layers') and \
                 layer_id < len(hf_model.model.layers):
                layer_module_from_hf = hf_model.model.layers[layer_id]
                # If no specific MoE block, hook the whole decoder layer
                target_module = layer_module_from_hf
                if "mixtral" in self.hf_model_name.lower() or "qwen" in self.hf_model_name.lower():
                    hook_type_for_layer = "mixtral_decoder_layer" # Expects router logits in output tuple
                if self.rank == 0 and layer_id == 0: print(f"Mixtral/Qwen: Targeting entire layer {layer_id}")
            else:
                if self.rank == 0: print(f"Model structure not recognized for layer {layer_id} or layer index out of bounds.")


            if target_module and hasattr(target_module, 'forward'):
                if hasattr(target_module, original_forward_attr_name):
                    if self.rank == 0: print(f"Layer {layer_id} ({target_module.__class__.__name__}) already hooked. Skipping.")
                    continue
                
                setattr(target_module, original_forward_attr_name, target_module.forward)
                original_forward = getattr(target_module, original_forward_attr_name)
                target_module.forward = self._wrap_forward_with_logging(original_forward, layer_id, hook_type_for_layer)
                if self.rank == 0 and layer_id == 0: print(f"Successfully injected logging for layer {layer_id} ({target_module.__class__.__name__})")
            else:
                if self.rank == 0:
                    print(f"Warning: Could not find target module or its forward method for layer {layer_id} of model {self.hf_model_name} within DeepSpeed engine. Skipping layer injection.")
        return model_engine # Return the DeepSpeed engine

    # _load_data, _prepare_prompt_batches, get_model_simple_name, write_to_summary_csv, write_expert_counts_to_csv
    # remain IDENTICAL to your previous version. I'll omit them for brevity.

    def _load_data(self):
        # (Identical to previous version)
        if self.task == "gsm8k":
            gsm8k_dataset = load_dataset("gsm8k", "main", split="test")
            all_input_raw = gsm8k_dataset['question']
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
    
    def _prepare_prompt_batches(self, all_input_raw, tokenizer):
        # (Identical to previous version, with .lower() for robustness)
        if "qwen" in self.hf_model_name.lower():
            all_input_chat = [[{"role": "system", "content": "You are a helpful assistant."}, 
                               {"role": "user", "content": f"{input_string}"}] for input_string in all_input_raw]
            all_input_updated = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) 
                                 for chat in all_input_chat]
        elif "dbrx" in self.hf_model_name.lower():
             all_input_chat = [[{"role": "user", "content": input_string}] for input_string in all_input_raw]
             all_input_updated = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) 
                                 for chat in all_input_chat]
        else: 
            all_input_chat = [[{"role": "user", "content": input_string}] for input_string in all_input_raw]
            all_input_updated = [tokenizer.apply_chat_template(chat, tokenize=False) 
                                 for chat in all_input_chat]
            
        return [all_input_updated[i:i + self.batch_size] for i in range(0, len(all_input_updated), self.batch_size)]

    def get_model_simple_name(self):
        norm_path = os.path.normpath(self.hf_model_name)
        parts = norm_path.split(os.sep)
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        else:
            return self.hf_model_name
    
    def write_to_summary_csv(self, avg_activated_experts_value):
        # (Identical to previous version)
        if self.rank != 0:
            return
        summary_dir = "activation_profiling_results"
        csv_filename = os.path.join(summary_dir, f"{self.get_model_simple_name()}.csv")

        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['dataset', 'batch_size', 'average_activated_experts']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'dataset': self.task, 
                'batch_size': self.batch_size, # This is per-process batch size
                'average_activated_experts': f"{avg_activated_experts_value:.2f}"
            })
        print(f"Summary results saved to {csv_filename}")
    
    def write_expert_counts_to_csv(self, aggregated_layer_expert_counts):
        # (Identical to previous version)
        if self.rank != 0:
            return
        expert_count_dir = "activation_profiling_expert_count"
        csv_filename = os.path.join(expert_count_dir, f"{self.get_model_simple_name()}_dataset_{self.task}_bs{self.batch_size}_tp{self.world_size}_expert_counts.csv") # Added tp_size to filename

        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['layer_id'] + [f'expert_{i}' for i in range(self.n_experts)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for layer_id in range(self.n_layers):
                row = {'layer_id': layer_id}
                counts_for_layer = aggregated_layer_expert_counts.get(layer_id, Counter())
                for expert_id in range(self.n_experts):
                    row[f'expert_{expert_id}'] = counts_for_layer.get(expert_id, 0)
                writer.writerow(row)
        print(f"Detailed expert counts saved to {csv_filename}")

    def run(self):
        if os.path.exists(f'activation_profiling_results/{self.get_model_simple_name()}.csv'):
            df = pd.read_csv(f'activation_profiling_results/{self.get_model_simple_name()}.csv')
            if df[(df['dataset'] == self.task) & (df['batch_size'] == self.batch_size)].shape[0] > 0:
                print(f"Results for {self.hf_model_name} on {self.task} with batch size {self.batch_size} already exist. Skipping.")
                return

        if self.rank == 0:
            print(f"Starting analysis for {self.hf_model_name} on {self.task} with per-process batch size {self.batch_size}")
            print(f"Tensor Parallelism World Size: {self.world_size}, Rank: {self.rank}")

        model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_name,
            torch_dtype=torch.bfloat16, # Target dtype
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ) #.to(self.device) # Don't move to device yet if DS handles it


        if self.rank == 0: print("HF model loaded.")

        # 3. Initialize model with DeepSpeed for Tensor Parallelism
        # This will shard the model according to ds_config and replace layers if configured.
        # It also initializes torch.distributed if not already done.
        model_engine = deepspeed.init_inference(
            model,
            tensor_parallel={"tp_size": self.world_size},
            dtype=torch.bfloat16, 
            # replace_with_kernel_inject=True
        )
        # model_engine now is your DeepSpeed model. model_engine.module is the HF model (possibly modified).
        
        if self.rank == 0: print(f"DeepSpeed Inference Engine initialized. Model type: {type(model_engine)}")
        model_engine.eval() # Ensure eval mode

        # 4. Inject logging hooks into the DeepSpeed-managed model
        model_engine_with_hooks = self._inject_logging(model_engine)
        if self.rank == 0: print("Logging hooks injected into DeepSpeed model engine.")

        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        all_input_raw, max_new_tokens = self._load_data()
        if self.is_distributed: dist.barrier()

        prompt_batches_global = self._prepare_prompt_batches(all_input_raw, tokenizer)
        
        pad_to_multiple_of = 8 
        padding_side_default = tokenizer.padding_side
        tokenizer.padding_side = "left"
        
        tokenized_prompts_global = [
            tokenizer(batch, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
            for batch in prompt_batches_global
        ]
        tokenizer.padding_side = padding_side_default
        
        # Data Sharding: For PURE TP, each rank gets ALL data.
        # The model itself is sharded. Input is replicated.
        # If you had TP + DP, then you'd shard data across DP ranks.
        # For pure TP (tp_size == world_size), no data sharding needed across ranks for prompts.
        # Each rank processes all prompts but on its shard of the model.
        # However, the `generate` call is per batch. We can still iterate through batches.
        # Let's assume each rank processes all batches for now.
        # If a batch is too large for one TP group, it has to be split.
        # The current loop structure with local_batched_prompts is for DP.
        # For pure TP, each rank sees all data:
        local_batched_prompts = tokenized_prompts_global # Each TP rank gets all batches
        # If you want to split the dataset for faster processing even with pure TP (effectively manual DP over TP groups)
        # local_batched_prompts = tokenized_prompts_global[self.rank::self.world_size] # This would be unusual for pure TP measurement on whole dataset

        if self.rank == 0:
            print(f"Total prompt batches (globally): {len(tokenized_prompts_global)}")
            print(f"Prompts for this rank (rank {self.rank}): {len(local_batched_prompts)}")


        completions_per_process = []
        convergence_threshold = 1.0 
        should_stop_signal = [False]

        tqdm_disable = self.rank != 0 
        # self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        for i, batch_on_cpu in enumerate(tqdm(local_batched_prompts, desc=f"Rank {self.rank} Processing", disable=tqdm_disable)):
            # No stop signal broadcast if pure TP and all ranks process all data,
            # as convergence is global based on all data.
            # If data is sharded (manual DP over TP), then broadcast needed.
            # Let's assume data is NOT sharded for pure TP measurement for now.
            # If it IS sharded, uncomment the broadcast logic.

            if i > 0 and self.is_distributed:
                dist.broadcast_object_list(should_stop_signal, src=0)
                if should_stop_signal[0]:
                    if self.rank == 0: print(f"Rank 0: Convergence threshold met. Stopping processing.")
                    break
            
            self.prefilling_finished = False 
            self._temp_num_unique_list_for_generate.clear()
            
            # Move batch to current device. TP handles communication for model ops.
            batch = {k: v.to(self.local_rank) for k, v in batch_on_cpu.items()}
            
            with torch.no_grad():
                 # Use the hooked model_engine_with_hooks for generation
                 outputs = model_engine_with_hooks.generate(**batch, max_new_tokens=max_new_tokens)
            
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            completions_per_process.extend(generated_text) # Each rank will have full list of completions
            
            # For PURE TP:
            # Each rank calculates its portion of expert counts for the *entire dataset*.
            # These partial counts then need to be SUMMED across TP ranks.
            # The `_temp_num_unique_list_for_generate` is local, related to *this rank's shard of experts*.
            # The convergence check logic needs to be re-thought for pure TP.
            # The current convergence logic assumes gathered lists are from *different data subsets*.
            # With pure TP and all ranks processing all data:
            #   - Each rank logs activations for experts *it owns*.
            #   - `_temp_num_unique_list_for_generate` will contain the number of unique experts *seen by this rank's shards*.
            #   - To get total unique experts for a token, you'd need to gather indices and union them.
            # The current metric is "average number of unique experts activated per token *across all layers that this rank sees fragments of*".
            # This metric is still somewhat meaningful per rank.
            # Let's keep the gather for _temp_num_unique_list_for_generate for now and see.
            # The `layer_expert_counts` will be correctly partial per rank.

            # Gather _temp_num_unique_list_for_generate from all TP ranks
            # This list contains, for each generation step on this rank, the number of unique *local* experts activated.
            gathered_num_unique_for_batch_iter = None
            if self.rank == 0:
                gathered_num_unique_for_batch_iter = [None] * self.world_size
            
            dist.gather_object(
                self._temp_num_unique_list_for_generate, # List of num_unique_experts_per_call on this rank for this batch
                gathered_num_unique_for_batch_iter if self.rank == 0 else None,
                dst=0
            )
            
            # On Rank 0, process for convergence (if desired)
            if self.rank == 0:
                if gathered_num_unique_for_batch_iter:
                    # For pure TP, each list in gathered_num_unique_for_batch_iter comes from a different rank
                    # processing the *same data* but for *different expert shards*.
                    # If we sum/average these unique counts, it's "average number of unique local experts activated per call, averaged over TP ranks"
                    # This isn't "total unique experts for the token across all shards".
                    # The original intent of `_temp_num_unique_list_for_generate` for convergence
                    # might need a different metric for pure TP.
                    # For now, let's aggregate them as before to see the trend.
                    for proc_list in gathered_num_unique_for_batch_iter:
                        if proc_list: self.main_process_cumulative_num_unique_list.extend(proc_list)
                
                if self.main_process_cumulative_num_unique_list:
                    current_avg = sum(self.main_process_cumulative_num_unique_list) / len(self.main_process_cumulative_num_unique_list)
                    # Convergence check logic (same as before)
                    # if i * self.world_size > (5 * self.world_size): # Or just i > 5 if each rank processes all data
                    if i > 0: # Check after a few batches
                        if abs(current_avg - self.prev_avg_activated_experts) < convergence_threshold:
                            should_stop_signal[0] = True # This signal isn't used in current pure TP loop
                            print(f"Rank 0: Convergence criteria for unique local experts met at batch {i}. Avg: {current_avg:.2f}.")
                            # if using sharded data, this would trigger broadcast and stop
                            
                    self.prev_avg_activated_experts = current_avg
                    self.final_avg_activated_experts = current_avg # This is avg of unique *local* experts

        if self.is_distributed: dist.barrier()

        # Gather results:
        # `completions_per_process`: if each rank processes all data, these will be identical.
        # We only need completions from rank 0.
        all_completions_gathered_on_main = None
        if self.rank == 0: all_completions_gathered_on_main = [None] * self.world_size
        dist.gather_object(completions_per_process, all_completions_gathered_on_main if self.rank == 0 else None, dst=0)
        
        # `layer_expert_counts`: This is critical. Each rank has counts for its expert shards.
        # These need to be gathered and summed up correctly.
        # defaultdict(Counter) can be gathered.
        all_expert_counts_gathered_on_main = None
        if self.rank == 0: all_expert_counts_gathered_on_main = [None] * self.world_size
        dist.gather_object(self.layer_expert_counts, all_expert_counts_gathered_on_main if self.rank == 0 else None, dst=0)
        
        final_avg_experts_to_return, final_counts_to_return = None, None
        if self.rank == 0:
            # Process completions (take from first rank if all are same)
            completions = []
            if all_completions_gathered_on_main and all_completions_gathered_on_main[0]:
                completions = all_completions_gathered_on_main[0]
            print(f"Rank 0: Total prompts processed (on rank 0): {len(completions)}")
            
            # Aggregate expert counts: Sum counters from all TP ranks
            final_layer_expert_counts = defaultdict(Counter)
            if all_expert_counts_gathered_on_main:
                for proc_counts_dict in all_expert_counts_gathered_on_main: # list of dicts
                    if proc_counts_dict: 
                        for layer_id, counter_obj in proc_counts_dict.items():
                            final_layer_expert_counts[layer_id].update(counter_obj) # Sums counts for same keys
            
            # The `self.final_avg_activated_experts` is currently average of unique *local* experts.
            # A more meaningful "average activated experts per token" for the *whole model* with TP
            # would require a different calculation:
            # For each token: get all activated expert indices (global), count unique ones. Average these counts.
            # This requires gathering raw `activated_experts_indices` per token, which is more involved.
            # The current `final_avg_activated_experts` is a proxy.
            # For now, we'll report it as is, but with the caveat about its meaning in TP.
            print(f"Rank 0: Final average of (unique local experts per call per rank): {self.final_avg_activated_experts:.2f}")
            self.write_to_summary_csv(self.final_avg_activated_experts) # Label clearly what this means
            self.write_expert_counts_to_csv(final_layer_expert_counts) # This should be correct total counts
            
            final_avg_experts_to_return = self.final_avg_activated_experts
            final_counts_to_return = final_layer_expert_counts
        
        del model_engine_with_hooks # Deletes the DeepSpeed engine
        del model # Delete original HF model reference if any
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return final_avg_experts_to_return, final_counts_to_return

def main():
    parser = argparse.ArgumentParser(description="MoE Activation Analyzer with DeepSpeed Tensor Parallelism.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-process batch size.")
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "arena_hard", "MATH"])
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by DeepSpeed.") # Important
    args = parser.parse_args()
    
    analyzer = MoEActivationAnalyzer(
        model_name=args.model_name,
        batch_size=args.batch_size,
        task=args.task,
    )
    analyzer.run()

if __name__ == "__main__":
    main()