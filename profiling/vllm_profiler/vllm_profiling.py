import os
import logging
import pickle
import argparse
from typing import Optional, Callable, List, Dict, Any, Union

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig
from .moe_profiler import MoEProfiler
class VLLMMoEProfiler(MoEProfiler):
    """MoE Profiler for vLLM."""
    
    def patch_moe_functions(self) -> None:
        """Patch vLLM MoE functions to track activated experts."""
        from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
        from vllm.model_executor.layers.fused_moe import FusedMoE
        
        self._old_select_experts = FusedMoE.select_experts
        FusedMoE.select_experts = self._new_select_experts
    
    def restore_moe_functions(self) -> None:
        """Restore original vLLM MoE functions."""
        from vllm.model_executor.layers.fused_moe import FusedMoE
        
        FusedMoE.select_experts = self._old_select_experts
    
    def _new_select_experts(self, hidden_states: torch.Tensor,
                           router_logits: torch.Tensor,
                           top_k: int,
                           use_grouped_topk: bool,
                           renormalize: bool,
                           topk_group: Optional[int] = None,
                           num_expert_group: Optional[int] = None,
                           custom_routing_function: Optional[Callable] = None,
                           scoring_func: str = "softmax",
                           e_score_correction_bias: Optional[torch.Tensor] = None):
        """Replacement for FusedMoE.select_experts that tracks activated experts."""
        from vllm.model_executor.layers.fused_moe.fused_moe import (fused_topk, grouped_topk)
        
        # DeepSeekV2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias)
        elif custom_routing_function is None:
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
            
        # Record activated experts when batch size matches our target
        if topk_ids.shape[0] == self.bs:
            self.logger.info(f"topk_weights shape: {topk_ids}")
            self.logger.info(f"topk_ids shape: {topk_ids.shape}")
            unique_values = torch.unique(topk_ids)
            num_unique = unique_values.numel()
            self.activated_experts_record.append(num_unique)
            
        return topk_weights, topk_ids
    
    def initialize_model(self) -> None:
        """Initialize vLLM model with the specified configuration."""
        from vllm import LLM
        
        # Build model arguments
        model_args = {
            'model': self.args.model,
            'tensor_parallel_size': self.args.tensor_parallel_size,
            'quantization': self.args.quant,
            'load_format': self.args.load_format,
            'gpu_memory_utilization': self.args.gpu_memory_utilization,
            'enforce_eager': True,
            'trust_remote_code': self.args.trust_remote_code,
            'hf_overrides': self.model_config,
            'dtype': self.args.dtype
        }
        
        # Add max_model_len if specified
        if self.args.max_model_len:
            model_args['max_model_len'] = self.args.max_model_len
            
        self.llm = LLM(**model_args)
    
    def run_inference(self) -> None:
        """Run inference with vLLM and collect MoE statistics."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=self.args.temperature, 
            max_tokens=self.args.max_new_tokens
        )
        
        for i in tqdm(range(0, len(self.chat_messages), self.bs), desc='Batch Decoding Process'):
            batch = self.chat_messages[i:i+self.bs]
            self.llm.chat(batch, sampling_params)