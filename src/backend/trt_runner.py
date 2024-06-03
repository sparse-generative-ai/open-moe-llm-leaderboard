import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
import argparse
import torch
import os
import io
import transformers
from typing import List
from lm_eval.models.utils import MultiTokenEOSCriteria
from multiprocessing import shared_memory
import pickle
    
from transformers import AutoTokenizer

class BenchmarkProfiler(object):
    cuda_event_dict: dict
    timer_dict: dict
    aux_info: dict
    started: bool

    def __init__(self):
        self.cuda_event_dict = {}
        self.timer_dict = {}
        self.aux_info = {}
        self.started = False

    def clean(self):
        self.cuda_event_dict = {}
        self.timer_dict = {}
        self.aux_info = {}

    def start(self):
        self.started = True

    def stop(self):
        self.started = False
        
    def is_recording_perf_profile(self):
        return self.started

    def get_cuda_event(self, name: str):
        if name not in self.cuda_event_dict.keys():
            event = torch.cuda.Event(enable_timing=True)
            self.cuda_event_dict[name] = event
        return self.cuda_event_dict[name]

    def record_cuda_event(self, name: str):
        if not self.started:
            return
        event = self.get_cuda_event(name)
        event.record()
        self.timer_dict[name] = 0.0

    def get_timer_value(self, timer_name: str):
        # timer is in milliseconds
        return self.timer_dict[timer_name]

    def record_elapsed_time(self, start_event_name: str, end_event_name: str,
                            timer_name: str):
        if timer_name not in self.timer_dict.keys():
            self.timer_dict[timer_name] = 0.0
        if not self.started:
            return
        self.get_cuda_event(start_event_name).synchronize()
        self.get_cuda_event(end_event_name).synchronize()
        # self.timer_dict[timer_name] += self.get_cuda_event(
        #     start_event_name).elapsed_time(self.get_cuda_event(end_event_name))
        self.timer_dict[timer_name] += self.get_cuda_event(
            start_event_name).elapsed_time(self.get_cuda_event(end_event_name))
        
    def get_elapsed_time(self, start_event_name: str, end_event_name: str,
                         timer_name: str):
        self.timer_dict[timer_name] = self.get_cuda_event(
            start_event_name).elapsed_time(self.get_cuda_event(end_event_name))
        return self.timer_dict[timer_name]

    def get_aux_info(self, aux_name):
        print(self.aux_info)
        if aux_name not in self.aux_info.keys():
            return 0
        return self.aux_info[aux_name]

    def add_aux_info(self, aux_name: str, add_value):
        if aux_name not in self.aux_info.keys():
            self.aux_info[aux_name] = 0
        if not self.started:
            return
        self.aux_info[aux_name] += add_value

    
class StoppingCriteriaList(list):
    def __call__(self, step, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        # print(f"{input_ids.shape=}")
        is_done = torch.full((input_ids.shape[0],), False, device=input_ids.device)
        for criteria in self:
            is_done = is_done | criteria(input_ids, scores, **kwargs)
        return False not in is_done

def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> StoppingCriteriaList:
    return StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a TensorRT LLM model")
    parser.add_argument("--ifile", type=str, default="/tmp/trtfile_in", help="file to read input_ids from")
    parser.add_argument("--ofile", type=str, default="/tmp/trtfile_out", help="file to write output_ids to")
    parser.add_argument("--checkpoint", type=str, required=True, help="huggingface model name")
    parser.add_argument("--precision", type=str, default="bfloat16", help="bfloat16, int8, int4")
    parser.add_argument("--mode", type=str, default="gen", help="forward for mmlu task, gen for generation task")
    # parser.add_argument("--end_id", type=int, required=True, help="self.tokenizer.eos_token_id,")
    # parser.add_argument("--pad_id", type=int, required=True, help="self.tokenizer.pad_token_id")

    args = parser.parse_args()
    
    runtime_rank = tensorrt_llm.mpi_rank()
    
    if runtime_rank == 0 and args.mode == "forward":
        # create shm using mp
        shm = shared_memory.SharedMemory(name="trtllm")
    
    model_name = args.checkpoint.split("/")[-1].lower()
    engine_dir = f"{model_name}-engine-{args.precision}"
    print(f"loading model from {engine_dir}")
    runner_kwargs = dict(engine_dir=engine_dir,
                         lora_dir=None,
                         rank=tensorrt_llm.mpi_rank(),
                         debug_mode=False,
                         lora_ckpt_source="hf",
                         )
    runner = ModelRunner.from_dir(**runner_kwargs)
    benchmark_profiler = BenchmarkProfiler()
    benchmark_profiler.start()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    sample_config = SamplingConfig(end_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    eos_token = tokenizer.decode(tokenizer.eos_token_id)
    while True:
        if not os.path.exists(args.ifile):
            continue
        if os.path.getsize(args.ifile) == 0:
            continue
        
        with open(args.ifile, "rb") as f:

            if args.mode == "gen":
            # the first line is stop tokens, comma separated
                line = f.readline()          
                stop = line.decode("utf-8").strip()
                stop = stop.split(",")
                stop_words_list = stop
                # remove eos token
                stop_words_list = [word for word in stop_words_list if word != eos_token]
                
                # the second line is max_new_tokens
                line = f.readline()        
                max_new_tokens = int(line.decode("utf-8").strip())
                data = f.read()
                context = torch.load(io.BytesIO(data))
                print(f"{context.shape=}")
                sample_config.max_new_tokens = max_new_tokens
                sample_config.return_dict = True
                sample_config.gather_context_logits = True
                sample_config.gather_generation_logits = True
                stopping_criteria = stop_sequences_criteria(
                    tokenizer, stop_words_list, context.shape[1], context.shape[0]
                )
            elif args.mode == "forward":
                data = f.read()
                context = torch.load(io.BytesIO(data))
                print(f"{context.shape=}")
                sample_config.max_new_tokens = 1
                sample_config.return_dict = True
                sample_config.gather_context_logits = True
                sample_config.gather_generation_logits = True
                stopping_criteria = None
            # last line is the context
            # if runtime_rank == 0:
            #     context_words = tokenizer.batch_decode(context, skip_special_tokens=True)
            #     print(f"{context_words=}")
            
            tensorrt_llm.mpi_barrier()
            benchmark_profiler.record_cuda_event("start")
            res = runner.generate(batch_input_ids=context, 
                                  sampling_config=sample_config,
                                  stopping_criteria=stopping_criteria,
                                  benchmark_profiler=benchmark_profiler,
                                  return_dict=True,
                                )
            if runtime_rank == 0:
                if args.mode == "gen":
                    output_ids = res["output_ids"]
                    generation_logits = res["generation_logits"]
                    context_logits = res["context_logits"]
                    
                    res_words = tokenizer.batch_decode(output_ids[0], skip_special_tokens=True)
                    res_words_last = tokenizer.batch_decode(output_ids[-1], skip_special_tokens=True)
                    print(f"{res_words=}")
                    print(f"{res_words_last=}")
                    steps = benchmark_profiler.get_aux_info("generation_step_count")
                    benchmark_profiler.record_elapsed_time("start", "first_token", "prefilling")
                    benchmark_profiler.record_elapsed_time("start", "last_token", "generation")
                    
                    first_time = benchmark_profiler.get_elapsed_time("start", "first_token", "prefilling")
                    last_time = benchmark_profiler.get_elapsed_time("start", "last_token", "generation")
                    
                    gen_time = last_time - first_time
                    print(f"Prefilling time: {first_time}ms")
                    print(f"Generation time: {gen_time}ms")
                    print(f"Generation steps: {steps}")
                elif args.mode == "forward":
                    context_logits = res["context_logits"]
                    print(f"{len(context_logits)=}")
                    print(f"{context_logits[0].shape=}")
                
            # exit()
        
        if runtime_rank == 0:
            if args.mode == "gen":
                lines = []
                lines.append(f"{first_time / 1000},{gen_time / 1000}")
                open(args.ifile, 'w').close()
                output_ids = output_ids.squeeze(1)
                with open(args.ofile, "wb") as f:
                    f.write("\n".join(lines).encode("utf-8"))
                    f.write(b"\n")
                    bytes = io.BytesIO()
                    torch.save(output_ids, bytes)
                    f.write(bytes.getvalue())
                print(f"Output written to {args.ofile}")
            if args.mode == "forward":
                # make context_logits a tensor
                open(args.ifile, 'w').close()
                context_logits = torch.stack(context_logits)
                
                # save context_logits to shm
                context_logits = context_logits.cpu()
                context_logits = context_logits.to(torch.float32)
                
                tensor_bytes = pickle.dumps(context_logits)
                n_bytes = len(tensor_bytes)
                shm.buf[:n_bytes] = tensor_bytes
                
                # with open("/tmp/trtfile_context_logits", "wb") as f:
                #     # pickle.dump(context_logits, f)
                #     torch.save(context_logits, f)
                with open("/tmp/trtfile_complete_tag", "w") as f:
                    f.write(str(n_bytes))
                    
            
        tensorrt_llm.mpi_barrier()