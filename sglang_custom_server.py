from sglang.srt.model_executor.model_runner import ModelRunner, logger
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
    ExpertDistributionRecorder,
    _Accumulator,
    _SinglePassGatherer
)
import sglang.srt.eplb.expert_distribution as sglang_eplb_expert_distribution
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable

from typing import Optional, Tuple, Union, List, Literal
import time
import os
import sys
from contextlib import contextmanager
import torch
from pathlib import Path
import json


_OutputMode = Literal["file", "object"]

class _ExpertDistributionRecorderReal2(ExpertDistributionRecorder):
    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata

        self.expert_record_list = []
        self._recording = False
        self._disable_all = False
        self._current_forward_pass_id = Withable()
        self._current_layer_idx = Withable()
        self._current_debug_name = Withable()
        self._accumulator = _Accumulator.init_new(
            server_args, expert_location_metadata, rank
        )
        self._single_pass_gatherers = {
            k: _SinglePassGatherer.init_new(server_args, expert_location_metadata, rank)
            for k in self._accumulator.get_single_pass_gatherer_keys()
        }

        if server_args.enable_expert_distribution_metrics:
            logger.info(
                "ExpertDistributionRecorder auto start record since enable_expert_distribution_metrics"
            )
            self.start_record()

    def with_current_layer(self, layer_idx):
        return self._current_layer_idx.with_value(layer_idx)

    def with_debug_name(self, debug_name):
        return self._current_debug_name.with_value(debug_name)

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        with self._current_forward_pass_id.with_value(forward_pass_id):
            self._on_forward_pass_start(forward_batch)
            try:
                yield
            finally:
                self._on_forward_pass_end(forward_pass_id)

    @contextmanager
    def disable_this_region(self):
        """Context manager to temporarily disable recording."""
        previous_disable_all = self._disable_all
        self._disable_all = True
        try:
            yield
        finally:
            self._disable_all = previous_disable_all

    def _on_forward_pass_start(self, forward_batch: ForwardBatch):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            gatherer.reset()
            gatherer.on_forward_pass_start(forward_batch)

    def _on_forward_pass_end(self, forward_pass_id: int):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            single_pass_data = gatherer.collect()
            self._accumulator.append(forward_pass_id, gatherer_key, single_pass_data)

    def on_select_experts(self, topk_ids: torch.Tensor):
        self._on_hook("on_select_experts", topk_ids=topk_ids)

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        self._on_hook(
            "on_deepep_dispatch_normal",
            local_physical_count_of_layer=local_physical_count_of_layer,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        self._on_hook(
            "on_deepep_dispatch_low_latency",
            local_physical_count_of_layer=local_physical_count_of_layer,
        )

    def _on_hook(self, hook_name: str, **kwargs):
        if self._disable_all:
            return
        if not (self._recording or torch.cuda.is_current_stream_capturing()):
            return
        gatherer = self._single_pass_gatherers[
            self._accumulator.get_single_pass_gatherer_key(
                self._current_debug_name.value
            )
        ]
        getattr(gatherer, hook_name)(layer_idx=self._current_layer_idx.value, **kwargs)

    def _reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting ExpertDistributionRecorder...")
        assert (
            self._current_layer_idx.value is None
        ), f"{self._current_layer_idx.value=}"
        for gatherer in self._single_pass_gatherers.values():
            gatherer.reset()
        self._accumulator.reset()

    def start_record(self):
        """Start recording the expert distribution."""
        if self._recording:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        self._reset()
        self._recording = True

    def stop_record(self):
        """Stop recording the expert distribution."""
        if not self._recording:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._recording = False

    def dump_record(self, output_mode: _OutputMode = "file"):
        """Dump the expert distribution record and reset the recorder after dumping."""
        output = self._accumulator.dump(output_mode=output_mode)
        self._reset()
        if output_mode == "file":
            save_dir = Path(os.environ.get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR", "/tmp"))
            path_output = save_dir / "expert_distribution_record.jsonl"
            logger.info(f"Write expert distribution jsonl to {path_output}")
            with open(path_output, "w", encoding="utf-8") as f:
                for record in self.expert_record_list:
                    f.write(json.dumps(record) + "\n")
        return output

    @property
    def recording(self):
        return self._recording


os.environ["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = os.path.join(os.getcwd(), "activation_profiling_layer_count")

_original_forward = ModelRunner.forward
def forward_expert_record(
    self,
    forward_batch: ForwardBatch,
    skip_attn_backend_init: bool = False,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    reinit_attn_backend: bool = False,
    split_forward_count: int = 1,
    ) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
        self.forward_pass_id += 1

        with get_global_expert_distribution_recorder().with_forward_pass(
            self.forward_pass_id,
            forward_batch,
        ):  
            start_time = time.perf_counter()
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            end_time = time.perf_counter()
            latency = end_time - start_time

        logger.info(f"Batch size: {forward_batch.batch_size}")
        record_output = get_global_expert_distribution_recorder().dump_record(output_mode="object")
        sum_seq_len = forward_batch.seq_lens_sum
        if forward_batch.forward_mode.is_extend():
            forward_mode = "prefill"
        elif forward_batch.forward_mode.is_decode():
            forward_mode = "decode"
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_batch.forward_mode.name}"
            )

        if self.tp_size and self.moe_ep_size == 1:
            record_output['logical_count'] = record_output['logical_count'] / self.tp_size
            # extra_dim, num_layer, num_experts = record_output['logical_count'].shape

        elif self.moe_ep_size > 1 and self.moe_ep_size != self.tp_size:
            record_output['logical_count'] = record_output['logical_count'] / self.moe_ep_size
            # extra_dim, num_layer, num_experts = record_output['logical_count'].shape
        elif self.moe_ep_size == self.tp_size:
            record_output['logical_count'] = record_output['logical_count']
        else:
            raise ValueError(
                f"Invalid tp_size {self.tp_size} and moe_ep_size {self.moe_ep_size} combination."
            )

        activated_experts = (record_output['logical_count'] > 0).float()  # Shape: [extra_dim, num_layer, num_experts]
        # Sum across experts dimension to get number of activated experts per layer per step
        activated_per_layer_per_step = activated_experts.sum(dim=2)  # Shape: [extra_dim, num_layer]
        # Average across layers to get one number per step
        avg_activated_per_step = activated_per_layer_per_step.mean(dim=1)

        if self.tp_rank == 0:
            # logger.info(avg_activated_per_step)
            non_zero_values = avg_activated_per_step[avg_activated_per_step != 0]
            non_zero_value = non_zero_values.item() if non_zero_values.numel() > 0 else 0
            record_dict = {
                "forward_pass_id": self.forward_pass_id,
                "batch_size": forward_batch.batch_size,
                "latency": latency,
                "seq_lens_sum": sum_seq_len,
                "forward_mode": forward_mode,
                "expert_activation": non_zero_value,
            }
            get_global_expert_distribution_recorder().expert_record_list.append(record_dict)
            logger.info(f"Forward pass {self.forward_pass_id} completed with latency {latency:.4f}s, expert activation {non_zero_value}")
        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        return output

ModelRunner.forward = forward_expert_record
sglang_eplb_expert_distribution._ExpertDistributionRecorderReal = _ExpertDistributionRecorderReal2

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)