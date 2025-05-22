import copy
import os
from datetime import timedelta
import sys
from time import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    DistributedType,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)
from transformers import TextStreamer
from transformers.models.dbrx.modeling_dbrx import DbrxExpertGLU
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)
from lm_eval.models.huggingface import HFLM
from src.utils import ModelInfoRetriever, _calculate_batch_metrics_hflm
from src.submission.check_validity import get_model_size
from src.envs import API
import pandas as pd

class StopWatch(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time() - self.start_prefilling
            self.start_decoding = time()
        self.decoding_iterations += 1
        return

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time() - self.start_decoding
        return


class HFLMWithMeasurement(HFLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pretrained = kwargs.get("pretrained", None)
        self.revision = kwargs.get("revision", None)
        self.used_dtype = kwargs.get("dtype", None)
        self.dataset_name = kwargs.get("dataset_name", None)
        activation_profile = kwargs.get("activation_profile", None)
        
        if self.model.__class__.__name__ == "MoE":
            model_config = self.model.model.config
        else:
            model_config = self.model.config
        
        if not self.used_dtype:
            qconf = getattr(model_config, "quantization_config", None)
            if qconf is None:
                raise ValueError("Missing quantization_config in model config")

            # Map boolean flags or quant_method to precision
            quant_method_map = {
                True: "int4" if getattr(qconf, "_load_in_4bit", False) else "int8" if getattr(qconf, "_load_in_8bit", False) else None,
                "fp8": "fp8",
                "fp4": "fp4",
                "awq": "awq",
                "gptq": "gptq",
            }

            method = getattr(qconf, "quant_method", None)
            self.used_dtype = quant_method_map.get(method, quant_method_map[True] if method is None else None)

            if self.used_dtype is None:
                raise ValueError("Unknown precision")

        self.num_gpus = self._detect_num_gpus_used()
        self.model_info = ModelInfoRetriever(model_name=self.pretrained, precision=self.used_dtype)
        moe_info = self.model_info.get_moe_info()
        attn_info = self.model_info.get_attention_info()

        self.precision = self.model_info.get_model_precision_bits()
        self.d_model = self.model_info.config.hidden_size
        self.d_ff = moe_info["ffn_dim"]
        self.total_experts = moe_info["num_experts"]
        self.used_experts = moe_info["experts_per_token"]
        self.n_layers = self.model_info.config.num_hidden_layers
        self.n_kv_heads = attn_info["num_key_value_heads"]
        self.d_head = attn_info["head_dim"]
        self.n_vocab = self.model_info.config.vocab_size
        self.per_token_kv_size = 2 * self.n_layers * self.d_head * self.n_kv_heads  #GQA, MQA

        assert activation_profile is not None, "Activation profile is required."
        activation_path = f"{activation_profile}/{self.pretrained.split('/')[0]}_{self.pretrained.split('/')[1]}.csv"
        print(f"Loading activation profile from {activation_path}")
        activation_profile = pd.read_csv(activation_path)

        if self.batch_size in activation_profile['batch_size'].values:
            self.avg_activated_experts = activation_profile[
                (activation_profile['dataset'] == self.dataset_name) & 
                (activation_profile['batch_size'] == self.batch_size)
            ]['average activated experts'].values[0]
        else:
            raise ValueError(f"Batch size {self.batch_size} not found in activation profile. Please run activation profiling first.")

    def _detect_num_gpus_used(self):
        gpus = []
        for p in self.model.parameters():
            if p.device.type == "cuda":
                gpus.append(p.device.index)
                
        num_gpus = len(set(gpus))
        return num_gpus

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
            and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            start = time()
            intermediate_res = self._model_call(batched_inps, **call_kwargs)
            end = time()
            multi_logits = F.log_softmax(
                intermediate_res , dim=-1
            )  # [batch, padding_length (inp or cont), vocab]
            per_sample_time = (end - start) / len(multi_logits)

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append((answer, per_sample_time, 0, 0, 0, 0))

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def _model_generate(self, context, max_tokens, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)
        
        # is_gsm8k = generation_kwargs.get("is_gsm8k", False)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        
        # if is_gsm8k:
        #     generation_kwargs.pop("is_gsm8k")
            
        context_length = context.shape[1]

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        stop_watch = StopWatch(self.tokenizer)
        start = time()
        res = self.model.generate(
            input_ids=context,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            streamer=stop_watch,
            **generation_kwargs,
        )
        end = time()

        output_length = stop_watch.decoding_iterations
        context_prefill_size = context_length

        end_to_end_time = (end - start) / self.batch_size
        prefilling_time = stop_watch.prefilling_time / self.batch_size
        decoding_time = stop_watch.decoding_time / self.batch_size
        token_per_sec = output_length / decoding_time
        
        res_dict = _calculate_batch_metrics_hflm(output_len=output_length, context_prefill_size=context_prefill_size, decoding_tp=token_per_sec, 
                                                n_layers=self.n_layers, d_model=self.d_model, 
                                                n_attn_heads=self.n_kv_heads, d_head=self.d_head, 
                                                n_kv_heads=self.n_kv_heads, 
                                                n_experts_per_tok=self.used_experts, d_ff=self.d_ff, 
                                                avg_activated_experts=self.avg_activated_experts, hf_config=self.model_info.config, 
                                                num_gpus=self.num_gpus, used_dtype=self.used_dtype, batch_size=self.batch_size,
                                                model_name=self.pretrained, precision=self.precision)
        
        smfu = res_dict["smfu"]
        smbu = res_dict["smbu"]
        print(f"smfu: {smfu}, smbu: {smbu}")
        
        return res, end_to_end_time, prefilling_time, token_per_sec, smfu, smbu

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
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
                        until = [kwargs]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            eos = "<|eot_id|>"
            if not until:
                until = [eos]
            else:
                until.append(eos)
            
            # is_gsm8k = kwargs.get("is_gsm8k", False)
            # if is_gsm8k:
            #     until = ["Question:", "Question", "</s>"]
            #     eos_ids = [self.tokenizer.eos_token_id, 
            #              self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                
                    
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            
            # print("context: ", self.tok_decode(context_enc[0]))
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = max_gen_toks

            # perform batched generation
            cont, end_to_end_time, prefilling_time, token_per_sec, mfu, mbu = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    # print("After Generation: ", self.tok_decode(cont_toks))
                    cont_toks = cont_toks[context_enc.shape[1] :]
                
                s = self.tok_decode(cont_toks)

                # # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                # if not is_gsm8k:
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]
                
                # print(s)
                res.append((s, end_to_end_time, prefilling_time, token_per_sec, mfu, mbu))

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res
