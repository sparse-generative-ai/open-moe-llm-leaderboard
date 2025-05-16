import copy
from importlib.metadata import version
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple, Union

from more_itertools import distribute
from packaging.version import parse as parse_version
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, undistribute
from lm_eval.utils import (
    eval_logger,
    get_rolling_token_windows,
    make_disjoint_window,
)
from src.backend.hflm_with_measurement import HFLMWithMeasurement

try:
    import ray
    from vllm import LLM, SamplingParams

    if parse_version(version("vllm")) > parse_version("0.3.0"):
        from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ModuleNotFoundError:
    pass

from vllm.outputs import RequestOutput
import torch
import time
import transformers

from src.utils import get_gpu_details, get_peak_bw, transfer_precision2bytes, get_peak_flops

eval_logger = eval_logger
orig_run_engine = LLM._run_engine

def run_engine_wrapper(func):
    def wrapper(self, use_tqdm, *args, **kwargs):
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests,
                        desc="Processed prompts",
                        dynamic_ncols=True)

        outputs: List[RequestOutput] = []
        start = time.time()
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        end = time.time()
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        decoding_time = end - start
        return outputs, decoding_time
    return wrapper

LLM._run_engine = run_engine_wrapper(orig_run_engine)

@register_model("vllm_moe")
class VLLM_MOE(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tokenizer_revision: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_gen_toks: int = 256,
        swap_space: int = 4,
        batch_size: Union[str, int] = 1,
        max_batch_size=None,
        max_length: int = None,
        max_model_len: int = None,
        seed: int = 1234,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        data_parallel_size: int = 1,
        lora_local_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

        if not find_spec("vllm"):
            raise Exception(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install lm-eval[vllm]` or `pip install -e .[vllm]`"
            )

        assert "cuda" in device or device is None, "vLLM only supports CUDA"
        # assert (
        #     max_length is None or max_model_len is None
        # ), "Either max_length or max_model_len may be provided, but not both"

        self._max_length = max_model_len if max_model_len is not None else max_length
        self.tensor_parallel_size = torch.cuda.device_count()
        self.data_parallel_size = int(data_parallel_size)
        self.model_args = {
            "model": pretrained,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(self.tensor_parallel_size),
            "max_model_len": int(self._max_length) if self._max_length else None,
            "swap_space": int(swap_space),
            "quantization": quantization,
            "seed": int(seed),
        }
        self.model_args.update(kwargs)
        self.model_args.pop("parallelize", None)
        self.model_args.pop("device_map", None)

        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else batch_size
        )
        if self.data_parallel_size <= 1:
            self.model = LLM(**self.model_args)
        else:
            eval_logger.warning(
                "You might experience occasional issues with model weight downloading when data_parallel is in use. To ensure stable performance, run with data_parallel_size=1 until the weights are downloaded and cached."
            )
            self.model_args["worker_use_ray"] = True
            self.batch_size = "auto"
            eval_logger.info("Manual batching is not compatible with data parallelism.")

            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(
                pretrained, trust_remote_code=trust_remote_code, revision=revision
            )
        
        self.batch_size = "auto"
        self.tokenizer = get_tokenizer(
            tokenizer if tokenizer else pretrained,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
        )
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

        self._max_gen_toks = max_gen_toks
        self.pretrained = pretrained

    ##################################### GETTING MODEL INFO #####################################
        model_config = self.model.llm_engine.model_config
        parallel_config = self.model.llm_engine.parallel_config
        hf_config = model_config.hf_text_config
        self.d_model = model_config.get_hidden_size()

        #FIXME Hardcoded, currently do not know how to get them from vLLM. (Yinsicheng)
        if "8x7" in pretrained:
            model_size_param = 46.7e9
            self.element_wise_mul = 0
            self.linear_count = 3
        elif "8x22" in pretrained:
            model_size_param = 141e9
            self.element_wise_mul = 0
            self.linear_count = 3
        elif "dbrx" in pretrained:
            model_size_param = 132e9
            self.linear_count = 3
            self.element_wise_mul = 1
        elif "Qwen1.5" in pretrained:
            model_size_param = 14.3e9
            self.linear_count = 3
            self.element_wise_mul = 1
        elif "Qwen2" in pretrained:
            model_size_param = 57.4e9
            self.linear_count = 3
            self.element_wise_mul = 1
        elif "Llama" in pretrained:
            model_size_param = 8.03e9
            self.element_wise_mul = 1
            self.linear_count = 3
        else:
            raise ValueError("Unknown model")
        #End

        self.n_layers = model_config.get_num_layers(parallel_config)
        self.n_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.d_head = model_config.get_head_size()
        self.precision_bytes = 2

        ## Number of experts
        if hasattr(hf_config, "num_local_experts"):
            num_experts = hf_config.num_local_experts
        elif hasattr(hf_config, "num_experts"):
            num_experts = hf_config.num_experts
        elif hasattr(hf_config, "ffn_config"):
            num_experts = hf_config.ffn_config.moe_num_experts
        else:
            num_experts = 1

        if hasattr(hf_config, "num_experts_per_tok"):
            n_experts_per_tok = hf_config.num_experts_per_tok
        elif hasattr(hf_config, "num_selected_experts"):
            n_experts_per_tok = hf_config.num_selected_experts
        elif hasattr(hf_config, "ffn_config"):
            n_experts_per_tok = hf_config.ffn_config.moe_top_k
        else:
            n_experts_per_tok = 1
        
        if hasattr(hf_config, "ffn_dim"):
            self.d_ff = hf_config.ffn_dim
        elif hasattr(hf_config, "intermediate_size"):
            self.d_ff = hf_config.intermediate_size
        elif hasattr(hf_config, "d_ff"):
            self.d_ff = hf_config.d_ff
        elif hasattr(hf_config, "ff_ratio"):
            self.d_ff = self.d_model * hf_config.ff_ratio
        elif hasattr(hf_config, "ffn_config"):
            self.d_ff = hf_config.ffn_config.ffn_hidden_size
        else:
            raise ValueError("Unknown FFN dimension")
        
        if "Qwen" in pretrained:
            self.d_ff = hf_config.moe_intermediate_size
            self.n_experts_for_ffn = n_experts_per_tok * 2
        else:
            self.n_experts_for_ffn = n_experts_per_tok

        self.ffn_params = self.n_layers * self.d_ff * self.linear_count * self.d_model

        shared_params = model_size_param - num_experts * self.ffn_params

        self.model_size = shared_params + self.n_experts_for_ffn * self.ffn_params

        self.per_token_kv_size = 2 * self.n_layers * self.d_head * self.n_kv_heads * self.precision_bytes

        self.n_vocab = hf_config.vocab_size

        self.total_experts = num_experts

        self.used_experts = n_experts_per_tok

        ##################################### FINISH GETTING MODEL INFO #####################################

        if lora_local_path is not None:
            assert parse_version(version("vllm")) > parse_version(
                "0.3.0"
            ), "lora adapters only compatible with vllm > v0.3.0."
            self.lora_request = LoRARequest("finetuned", 1, lora_local_path)
        else:
            self.lora_request = None

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if self.data_parallel_size <= 1:
            return self.model.llm_engine.model_config.max_model_len
        else:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self._config, attr):
                    return getattr(self._config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def tok_encode(
        self,
        string: str,
        left_truncate_len=None,
        add_special_tokens=None,
        truncation=False,
    ):
        """ """
        if not add_special_tokens:
            add_special_tokens = False or self.add_bos_token
        encoding = self.tokenizer.encode(
            string, add_special_tokens=add_special_tokens, truncation=truncation
        )

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)
            kwargs.pop("max_length", None)
            sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
        else:
            kwargs.pop("max_length", None)
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1
            )
        if self.data_parallel_size > 1:
            # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            # note: this has changed on 0.3.3, and it only works now if num_gpus are set.
            # but then tensor_parallel breaks
            @ray.remote
            def run_inference_one_model(
                model_args: dict, sampling_params, requests: List[List[int]]
            ):
                llm = LLM(**model_args)
                return llm.generate(
                    prompt_token_ids=requests, sampling_params=sampling_params
                )

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            inputs = ((self.model_args, sampling_params, req) for req in requests)
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            return undistribute(results)     

        if self.lora_request is not None:
            start = time.time()
            outputs, decoding_time = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request,
            )
            end = time.time()
        else:
            start = time.time()
            outputs, decoding_time = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
            )
            end = time.time()
        input_length = sum([len(x) for x in requests])
        output_length = sum([len(x.outputs[0].token_ids) for x in outputs])
        token_per_sec = output_length / decoding_time
        tok_per_sec_with_prefill = (output_length + input_length) / (end - start)
        kvs = []
        avg_ctx = []
        if "8x7" in self.pretrained:
            print("yeah")
            if max_tokens==256:
                if self.batch_size == 32:
                    activated_experts = 7.89
            elif max_tokens==4096:
                if self.batch_size == 16:
                    activated_experts = 6.44
                elif self.batch_size == 20:
                    activated_experts = 6.78
            # activated_experts = 7.08
            s_attn = 0.000134
            s_expert = 0.00034
        elif "8x22" in self.pretrained:
            # activated_experts = 7.33
            if max_tokens==256:
                if self.batch_size == 20:
                    activated_experts = 7.68
            elif max_tokens==4096:
                if self.batch_size == 8:
                    activated_experts = 5.68
                elif self.batch_size == 5:
                    activated_experts = 5.34
            s_attn = 0.0003
            s_expert = 0.0006
        elif "dbrx" in self.pretrained:
            # activated_experts = 13.49
            if max_tokens==256:
                if self.batch_size == 8:
                    activated_experts = 12.27
            elif max_tokens==4096:
                if self.batch_size == 8:
                    activated_experts = 12.2
                elif self.batch_size == 5:
                    activated_experts = 10.51
            s_attn = 0.0003
            s_expert = 0.0004
        elif "Qwen" in self.pretrained:
            # activated_experts = 33.32
            if max_tokens==4096:
                if self.batch_size == 16:
                    activated_experts = 27.25
            s_attn = 0.000034
            s_expert = 0.000017
        for x in outputs:
            context_prefill_size = len(x.prompt_token_ids)
            output_len = len(x.outputs[0].token_ids)
            kv_size = context_prefill_size * self.per_token_kv_size + (output_len - 1) * self.per_token_kv_size / 2
            kv_size = kv_size / 1e12
            kvs.append(kv_size)
            ## TODO only support llama-type decoder only models and moe models of switch transformer and mixtrial
            avg_context_length = (context_prefill_size + output_len) / 2
            avg_ctx.append(avg_context_length)
        
        val = 1 #FIXME hardcoded for bf16 (Yinsicheng)
        kv_size = sum(kvs) / len(kvs)
        avg_context_length = sum(avg_ctx) / len(avg_ctx)
        if self.batch_size != "auto":
            e2e_time = (end - start) / self.batch_size
            smfu = (tok_per_sec_with_prefill * self.n_layers * 2 * (self.used_experts * self.linear_count * self.d_ff * self.d_model + 
                                                        avg_context_length * avg_context_length * self.d_model + 
                                                        4 * self.d_model * self.d_model)) / (311 * 1000000000000 * self.tensor_parallel_size) * val
            smbu = ((self.n_layers*(activated_experts * s_expert * val + s_attn * val) + 
                    kv_size)/(self.batch_size / token_per_sec) ) / ( 1 * self.tensor_parallel_size) * val
            print(f"avg_mfu: {smfu}, avg_mbu: {smbu}")
        else:
            # smfu = (token_per_sec * self.n_layers * 2 * (self.used_experts * self.linear_count * self.d_ff * self.d_model + 
            #                                             avg_context_length * avg_context_length * self.d_model + 
            #                                             4 * self.d_model * self.d_model)) / (311 * 1000000000000 * self.tensor_parallel_size) * val
            flops = 2 * self.model_size + ((self.linear_count) * self.n_layers * avg_context_length * self.d_model) + 4 * self.d_model + 2 * self.d_model * self.n_vocab
            smfu = flops * tok_per_sec_with_prefill / (311 * 1000000000000 * self.tensor_parallel_size) * val
            smbu = 0
            e2e_time = 0
        return outputs, e2e_time, 0, token_per_sec, smfu, smbu

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            messages = [{"role": "user", "content": f"{string}"}]
            updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False)
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(updated_string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        # batch tokenize contexts
        updated_strings = []
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        for ctx in context:
            messages = [{"role": "user", "content": f"{ctx}"}]
            if "dbrx" in self.model_args["model"]:
                updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif "Qwen" in self.model_args["model"]:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": ctx}
                ]
                updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False)
                
            updated_strings.append(updated_string)
        
        context = updated_strings[:]
            
        context_encoding = self.tokenizer(context, add_special_tokens=False).input_ids
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests, _collate_gen, group_by="gen_kwargs")
        # self.batch_size = "auto"
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
                )
            # add EOS token to stop sequences
            eos = "<|eot_id|>"
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            # max len for inputs = max length, minus room to generate the max new tokens
            max_ctx_len = self.max_length - max_gen_toks
            context_encoding = [x[-max_ctx_len:] for x in context_encoding]

            # perform batched generation
            cont, end_to_end_time, prefilling_time, token_per_sec, mfu, mbu = self._model_generate(
                requests=context_encoding,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs,
            )

            # cache generations
            for output, context in zip(cont, context):
                generated_text = output.outputs[0].text
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        generated_text = generated_text.split(term)[0]
                res.append((generated_text, end_to_end_time, prefilling_time, token_per_sec, mfu, mbu))
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        return re_ords.get_original(res)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length and batch
        re_ord = Collator(requests, sort_fn=_collate)
        # self.batch_size = "auto"
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs, end_to_end_time, prefilling_time, token_per_sec, mfu, mbu = self._model_generate(requests=inputs, generate=False)

            for output, ctxlen, (cache_key, _, _), inp in zip(
                outputs, ctxlens, chunk, inputs
            ):
                answer = self._parse_logprobs(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                )

                res.append((answer, end_to_end_time, 0, 0, 0, 0))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int) -> Tuple[float, bool]:
        """Process logprobs and tokens.

        :param tokens: list
            Input tokens (potentially left-truncated)
        :param outputs: RequestOutput
            Contains prompt_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        continuation_logprobs_dicts = outputs.prompt_logprobs

        def coerce_logprob_to_num(logprob):
            # vLLM changed the return type of logprobs from float
            # to a Logprob object storing the float value + extra data
            # (https://github.com/vllm-project/vllm/pull/3065).
            # If we are dealing with vllm's Logprob object, return
            # the logprob value stored as an attribute. Otherwise,
            # return the object itself (which should be a float
            # for older versions of vLLM).
            return getattr(logprob, "logprob", logprob)

        continuation_logprobs_dicts = [
            {
                token: coerce_logprob_to_num(logprob)
                for token, logprob in logprob_dict.items()
            }
            if logprob_dict is not None
            else None
            for logprob_dict in continuation_logprobs_dicts
        ]

        # Calculate continuation_logprobs
        # assume ctxlen always >= 1
        continuation_logprobs = sum(
            logprob_dict.get(token)
            for token, logprob_dict in zip(
                tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
            )
        )

        # Determine if is_greedy
        is_greedy = True
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        ):
            # Get the token with the maximum log probability from the logprob_dict
            if logprob_dict:  # Ensure the logprob_dict is not None
                top_token = max(logprob_dict, key=logprob_dict.get)
                if top_token != token:
                    is_greedy = False
                    break

        return continuation_logprobs, is_greedy

    @staticmethod
    def modify_gen_kwargs(kwargs: dict) -> dict:
        # sampling_params
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False or "temperature" not in kwargs:
            kwargs["temperature"] = 0.0
        # hf defaults
        kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
        kwargs["spaces_between_special_tokens"] = kwargs.get(
            "spaces_between_special_tokens", False
        )
        return kwargs

@register_model("vllm_moe_fixbs")
class VLLM_FIX(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tokenizer_revision: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_gen_toks: int = 256,
        swap_space: int = 4,
        batch_size: Union[str, int] = 1,
        max_batch_size=None,
        max_length: int = None,
        max_model_len: int = None,
        seed: int = 1234,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        data_parallel_size: int = 1,
        lora_local_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

        if not find_spec("vllm"):
            raise Exception(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install lm-eval[vllm]` or `pip install -e .[vllm]`"
            )

        assert "cuda" in device or device is None, "vLLM only supports CUDA"
        # assert (
        #     max_length is None or max_model_len is None
        # ), "Either max_length or max_model_len may be provided, but not both"

        self._max_length = max_model_len if max_model_len is not None else max_length
        self.tensor_parallel_size = torch.cuda.device_count()
        self.data_parallel_size = int(data_parallel_size)
        self.model_args = {
            "model": pretrained,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(self.tensor_parallel_size),
            "max_model_len": int(self._max_length) if self._max_length else None,
            "swap_space": int(swap_space),
            "quantization": quantization,
            "seed": int(seed),
        }
        self.model_args.update(kwargs)
        self.model_args.pop("parallelize", None)
        self.model_args.pop("device_map", None)

        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else batch_size
        )
        if self.data_parallel_size <= 1:
            self.model = LLM(**self.model_args)
        else:
            eval_logger.warning(
                "You might experience occasional issues with model weight downloading when data_parallel is in use. To ensure stable performance, run with data_parallel_size=1 until the weights are downloaded and cached."
            )
            self.model_args["worker_use_ray"] = True
            self.batch_size = "auto"
            eval_logger.info("Manual batching is not compatible with data parallelism.")

            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(
                pretrained, trust_remote_code=trust_remote_code, revision=revision
            )
        self.tokenizer = get_tokenizer(
            tokenizer if tokenizer else pretrained,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
        )
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

        self._max_gen_toks = max_gen_toks
        self.pretrained = pretrained

    ##################################### GETTING MODEL INFO #####################################
        model_config = self.model.llm_engine.model_config
        parallel_config = self.model.llm_engine.parallel_config
        hf_config = model_config.hf_text_config
        self.d_model = model_config.get_hidden_size()

        #FIXME Hardcoded, currently do not know how to get them from vLLM. (Yinsicheng)
        if "8x7" in pretrained:
            model_size_param = 46.7e9
            self.element_wise_mul = 0
            self.linear_count = 3
        elif "8x22" in pretrained:
            model_size_param = 141e9
            self.element_wise_mul = 0
            self.linear_count = 3
        elif "dbrx" in pretrained:
            model_size_param = 132e9
            self.linear_count = 3
            self.element_wise_mul = 1
        elif "Qwen" in pretrained:
            model_size_param = 14.3e9
            self.linear_count = 3
            self.element_wise_mul = 1

        self.n_layers = model_config.get_num_layers(parallel_config)
        self.n_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.d_head = model_config.get_head_size()
        self.precision_bytes = 2

        ## Number of experts
        if hasattr(hf_config, "num_local_experts"):
            num_experts = hf_config.num_local_experts
        elif hasattr(hf_config, "num_experts"):
            num_experts = hf_config.num_experts
        elif hasattr(hf_config, "ffn_config"):
            num_experts = hf_config.ffn_config.moe_num_experts
        else:
            num_experts = 1

        if hasattr(hf_config, "num_experts_per_tok"):
            n_experts_per_tok = hf_config.num_experts_per_tok
        elif hasattr(hf_config, "num_selected_experts"):
            n_experts_per_tok = hf_config.num_selected_experts
        elif hasattr(hf_config, "ffn_config"):
            n_experts_per_tok = hf_config.ffn_config.moe_top_k
        else:
            n_experts_per_tok = 1
        
        if hasattr(hf_config, "ffn_dim"):
            self.d_ff = hf_config.ffn_dim
        elif hasattr(hf_config, "intermediate_size"):
            self.d_ff = hf_config.intermediate_size
        elif hasattr(hf_config, "d_ff"):
            self.d_ff = hf_config.d_ff
        elif hasattr(hf_config, "ff_ratio"):
            self.d_ff = self.d_model * hf_config.ff_ratio
        elif hasattr(hf_config, "ffn_config"):
            self.d_ff = hf_config.ffn_config.ffn_hidden_size
        else:
            raise ValueError("Unknown FFN dimension")
        
        if "Qwen" in pretrained:
            self.d_ff = hf_config.moe_intermediate_size
            self.n_experts_for_ffn = 4 + n_experts_per_tok
        else:
            self.n_experts_for_ffn = n_experts_per_tok

        ffn_params = self.n_layers * self.d_ff * self.linear_count * self.d_model

        shared_params = model_size_param - num_experts * ffn_params

        self.model_size = shared_params + n_experts_per_tok * ffn_params

        self.per_token_kv_size = 2 * self.n_layers * self.d_head * self.n_kv_heads * self.precision_bytes

        self.n_vocab = hf_config.vocab_size

        self.total_experts = num_experts

        ##################################### FINISH GETTING MODEL INFO #####################################

        if lora_local_path is not None:
            assert parse_version(version("vllm")) > parse_version(
                "0.3.0"
            ), "lora adapters only compatible with vllm > v0.3.0."
            self.lora_request = LoRARequest("finetuned", 1, lora_local_path)
        else:
            self.lora_request = None

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if self.data_parallel_size <= 1:
            return self.model.llm_engine.model_config.max_model_len
        else:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self._config, attr):
                    return getattr(self._config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def tok_encode(
        self,
        string: str,
        left_truncate_len=None,
        add_special_tokens=None,
        truncation=False,
    ):
        """ """
        if not add_special_tokens:
            add_special_tokens = False or self.add_bos_token
        encoding = self.tokenizer.encode(
            string, add_special_tokens=add_special_tokens, truncation=truncation
        )

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)
            kwargs.pop("max_length", None)
            sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
        else:
            kwargs.pop("max_length", None)
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1
            )
        if self.data_parallel_size > 1:
            # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            # note: this has changed on 0.3.3, and it only works now if num_gpus are set.
            # but then tensor_parallel breaks
            @ray.remote
            def run_inference_one_model(
                model_args: dict, sampling_params, requests: List[List[int]]
            ):
                llm = LLM(**model_args)
                return llm.generate(
                    prompt_token_ids=requests, sampling_params=sampling_params
                )

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            inputs = ((self.model_args, sampling_params, req) for req in requests)
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            return undistribute(results)     

        if self.lora_request is not None:
            start = time.time()
            outputs, decoding_time = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request,
            )
            end = time.time()
        else:
            start = time.time()
            outputs, decoding_time = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
            )
            end = time.time()
        b_size = len(requests)
        output_length = sum([len(x.outputs[0].token_ids) for x in outputs])
        e2e_time = (end - start) / b_size
        token_per_sec = output_length / decoding_time
        kvs = []
        avg_ctx = []
        if "8x7" in self.pretrained:
            print("yeah")
            if max_tokens==256:
                if self.batch_size == 32:
                    activated_experts = 7.89
            elif max_tokens==4096:
                if self.batch_size == 16:
                    activated_experts = 6.44
                elif self.batch_size == 20:
                    activated_experts = 6.78
            # activated_experts = 7.08
            s_attn = 0.000134
            s_expert = 0.00034
        elif "8x22" in self.pretrained:
            # activated_experts = 7.33
            if max_tokens==256:
                if self.batch_size == 20:
                    activated_experts = 7.68
            elif max_tokens==4096:
                if self.batch_size == 8:
                    activated_experts = 5.68
                elif self.batch_size == 5:
                    activated_experts = 5.34
            s_attn = 0.0003
            s_expert = 0.0006
        elif "dbrx" in self.pretrained:
            # activated_experts = 13.49
            if max_tokens==256:
                if self.batch_size == 8:
                    activated_experts = 12.27
            elif max_tokens==4096:
                if self.batch_size == 8:
                    activated_experts = 12.2
                elif self.batch_size == 5:
                    activated_experts = 10.51
            s_attn = 0.0003
            s_expert = 0.0004
        elif "Qwen" in self.pretrained:
            # activated_experts = 33.32
            if max_tokens==4096:
                if self.batch_size == 16:
                    activated_experts = 27.25
            s_attn = 0.000034
            s_expert = 0.000017
        for x in outputs:
            context_prefill_size = len(x.prompt_token_ids)
            output_len = len(x.outputs[0].token_ids)
            kv_size = context_prefill_size * self.per_token_kv_size + (output_len - 1) * self.per_token_kv_size / 2
            kv_size = kv_size / 1e12
            kvs.append(kv_size)
            ## TODO only support llama-type decoder only models and moe models of switch transformer and mixtrial
            avg_context_length = (context_prefill_size + output_len) / 2
            avg_ctx.append(avg_context_length)
            
            val = 1 #FIXME hardcoded for bf16 (Yinsicheng)

            kv_size = sum(kvs) / len(kvs)
            avg_context_length = sum(avg_ctx) / len(avg_ctx)
        if self.batch_size != "auto":
            smfu = (token_per_sec * self.n_layers * 2 * ( self.d_ff * self.d_model * 3 * 2 + 
                                                        self.d_model * self.total_experts + 
                                                        4 * self.d_model * self.d_model)) / (311 * 1000000000000 * self.tensor_parallel_size) * val
            smbu = ((self.n_layers*(activated_experts * s_expert * val + s_attn * val) + 
                    kv_size)/(self.batch_size / token_per_sec) ) / ( 1 * self.tensor_parallel_size) * val

            print(f"avg_mfu: {smfu}, avg_mbu: {smbu}")
        else:
            smfu = (token_per_sec * self.n_layers * 2 * ( self.d_ff * self.d_model * 3 * 2 + 
                                                        self.d_model * self.total_experts + 
                                                        4 * self.d_model * self.d_model)) / (311 * 1000000000000 * self.tensor_parallel_size) * val
            smbu = 0
        return outputs, e2e_time, 0, token_per_sec, smfu, smbu

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            messages = [{"role": "user", "content": f"{string}"}]
            updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False)
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(updated_string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        # batch tokenize contexts
        updated_strings = []
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        for ctx in context:
            messages = [{"role": "user", "content": f"{ctx}"}]
            if "dbrx" in self.model_args["model"]:
                updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif "Qwen" in self.model_args["model"]:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": ctx}
                ]
                updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False)
                
            updated_strings.append(updated_string)
        
        context = updated_strings[:]
            
        context_encoding = self.tokenizer(context, add_special_tokens=False).input_ids
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests, _collate_gen, group_by="gen_kwargs")
        # self.batch_size = "auto"
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
                )
            # add EOS token to stop sequences
            eos = "<|eot_id|>"
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            # max len for inputs = max length, minus room to generate the max new tokens
            max_ctx_len = self.max_length - max_gen_toks
            context_encoding = [x[-max_ctx_len:] for x in context_encoding]

            # perform batched generation
            cont, end_to_end_time, prefilling_time, token_per_sec, mfu, mbu = self._model_generate(
                requests=context_encoding,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs,
            )

            # cache generations
            for output, context in zip(cont, context):
                generated_text = output.outputs[0].text
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        generated_text = generated_text.split(term)[0]
                res.append((generated_text, end_to_end_time, prefilling_time, token_per_sec, mfu, mbu))
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        return re_ords.get_original(res)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length and batch
        re_ord = Collator(requests, sort_fn=_collate)
        # self.batch_size = "auto"
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs, end_to_end_time, prefilling_time, token_per_sec, mfu, mbu = self._model_generate(requests=inputs, generate=False)

            for output, ctxlen, (cache_key, _, _), inp in zip(
                outputs, ctxlens, chunk, inputs
            ):
                answer = self._parse_logprobs(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                )

                res.append((answer, end_to_end_time, 0, 0, 0, 0))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int) -> Tuple[float, bool]:
        """Process logprobs and tokens.

        :param tokens: list
            Input tokens (potentially left-truncated)
        :param outputs: RequestOutput
            Contains prompt_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        continuation_logprobs_dicts = outputs.prompt_logprobs

        def coerce_logprob_to_num(logprob):
            # vLLM changed the return type of logprobs from float
            # to a Logprob object storing the float value + extra data
            # (https://github.com/vllm-project/vllm/pull/3065).
            # If we are dealing with vllm's Logprob object, return
            # the logprob value stored as an attribute. Otherwise,
            # return the object itself (which should be a float
            # for older versions of vLLM).
            return getattr(logprob, "logprob", logprob)

        continuation_logprobs_dicts = [
            {
                token: coerce_logprob_to_num(logprob)
                for token, logprob in logprob_dict.items()
            }
            if logprob_dict is not None
            else None
            for logprob_dict in continuation_logprobs_dicts
        ]

        # Calculate continuation_logprobs
        # assume ctxlen always >= 1
        continuation_logprobs = sum(
            logprob_dict.get(token)
            for token, logprob_dict in zip(
                tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
            )
        )

        # Determine if is_greedy
        is_greedy = True
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        ):
            # Get the token with the maximum log probability from the logprob_dict
            if logprob_dict:  # Ensure the logprob_dict is not None
                top_token = max(logprob_dict, key=logprob_dict.get)
                if top_token != token:
                    is_greedy = False
                    break

        return continuation_logprobs, is_greedy

    @staticmethod
    def modify_gen_kwargs(kwargs: dict) -> dict:
        # sampling_params
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False or "temperature" not in kwargs:
            kwargs["temperature"] = 0.0
        # hf defaults
        kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
        kwargs["spaces_between_special_tokens"] = kwargs.get(
            "spaces_between_special_tokens", False
        )
        return kwargs