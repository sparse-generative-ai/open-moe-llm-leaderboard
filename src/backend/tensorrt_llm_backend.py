import torch
import os
import shutil
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM
from typing import List, Tuple, Optional, Union
from multiprocessing import shared_memory
import subprocess
from lm_eval.api.registry import register_model

from src.backend.hflm_with_measurement import HFLMWithMeasurement
import os

@register_model("tensorrt_llm")
class TensorRTHFLM(HFLMWithMeasurement):
    def __init__(
        self,
        pretrained: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        use_chat_template=True,
        *args,
        **kwargs,
    ):
        self.checkpoint = pretrained
        self.model_name = self.checkpoint.split('/')[-1].lower()
        self.model_process = None
        
        self.model_config = AutoConfig.from_pretrained(pretrained)
        self.use_chat_template = use_chat_template
        if "device" in kwargs:
            kwargs.pop("device")
        kwargs["device_map"] = "cuda:0"
        super().__init__(
            *args, **kwargs, pretrained=pretrained
        )  # Assuming HFLM accepts a 'pretrained' arg and handles it

    def __del__(self):
        pass
        # if os.path.exists("{self.model_name}-ckpt"):
        #     shutil.rmtree("{self.model_name}-ckpt") # clean up offload model
            
        # if os.path.exists("{self.model_name}-engine"):
        #     shutil.rmtree("{self.model_name}-engine")

    def _model_call(self, inps, attn_mask=None, labels=None):
        return super()._model_call(inps, attn_mask, labels)

    def _create_model(self, *args, **kwargs):
        """
        Initializes the model in tensorrt
        """
        
        engine_dir = f"{self.model_name}-engine"
        dtype = kwargs.get("dtype", "bfloat16")
        
        
        checkpoint_converter = os.path.join(os.path.expanduser("~"), "TensorRT-LLM", "examples", "llama")
        checkpoint_converter = os.path.join(checkpoint_converter, "convert_checkpoint.py")
        # print(f"converting checkpoint: {self.checkpoint}")
        # if dtype == "bfloat16":
        #     if not os.path.exists(f"{self.model_name}-{dtype}"):
        #         os.system(f"python3 {checkpoint_converter} --tp_size 4 --model_dir {self.checkpoint} --output_dir {self.model_name}-{dtype} --dtype {dtype}")
        #         os.system(f"trtllm-build --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir {self.model_name}-{dtype} --output_dir {engine_dir}-{dtype} --gemm_plugin bfloat16 --max_batch_size 64")
        # elif dtype == "int8":
        #     if not os.path.exists(f"{self.model_name}-{dtype}"):
        #         os.system(f"python3 {checkpoint_converter} --tp_size 4 --model_dir {self.checkpoint} --output_dir {self.model_name}-{dtype} --use_weight_only --weight_only_precision {dtype}")
        #         os.system(f"trtllm-build --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir {self.model_name}-{dtype} --output_dir {engine_dir}-{dtype} --max_batch_size 64")
        # elif dtype == "int4":
        #     if not os.path.exists(f"{self.model_name}-{dtype}"):
        #         os.system(f"python3 {checkpoint_converter} --tp_size 4 --model_dir {self.checkpoint} --output_dir {self.model_name}-{dtype} --use_weight_only --weight_only_precision {dtype}")
        #         os.system(f"trtllm-build --use_custom_all_reduce disable --tp_size 4 --pp_size 1 --checkpoint_dir {self.model_name}-{dtype} --output_dir {engine_dir}-{dtype} --max_batch_size 64")
        # else:
        #     raise ValueError(f"Unsupported dtype: {dtype}")

        # Remove existing files
        if os.path.exists("/tmp/trtfile_in"):
            os.remove("/tmp/trtfile_in")
        if os.path.exists("/tmp/trtfile_out"):
            os.remove("/tmp/trtfile_out")
        if os.path.exists("/tmp/trtfile_context_logits"):
            os.remove("/tmp/trtfile_context_logits")
        if os.path.exists("/tmp/trtfile_complete_tag"):
            os.remove("/tmp/trtfile_complete_tag")
            
        

        # Create new empty files
        open("/tmp/trtfile_in", "wb").close()
        open("/tmp/trtfile_out", "wb").close()
        open("/tmp/trtfile_context_logits", "wb").close()
        open("/tmp/trtfile_complete_tag", "wb").close()
        self.shm = shared_memory.SharedMemory(name="trtllm", create=True, size=16*4096*100352*4)
        
        print(f"Created shared memory: {self.shm.name}")
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        # run a command in the background
        # model_runner = os.path.join(os.path.expanduser("~"), "TensorRT-LLM", "src", "backend", "trt_runner.py")
        model_runner = os.path.join(current_dir, "trt_runner.py")
        # use model runner 
        command = f"mpirun --allow-run-as-root -n 4 /usr/bin/python3 {model_runner} --checkpoint {self.checkpoint}"
        command = command.split(" ")
        # if self.model_process is None:
        #     # self.model_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #     # os.system(f"{command} &> trt.log 2>&1")
        #     with open("/root/open-moe-llm-leaderboard/trt.log", "w") as f:
        #         self.model_process = subprocess.run(command, stdout=f, stderr=f)
        #     self.model_process = 1
        # self.process = subprocess.call(["/usr/bin/mpirun", "--allow-run-as-root", "-n", "4", "/usr/bin/python3", "/root/open-moe-llm-leaderboard/src/backend/trt_runner.py", "--checkpoint", "mistralai/mixtral-8x7b-instruct-v0.1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # ret = os.system(f'"bash -c \"mpirun --allow-run-as-root -n 4 /usr/bin/python3 {model_runner} --checkpoint {self.checkpoint} &> trt.log 2>&1\"')
        # os.system(f"{command} &> trt.log 2>&1 &")
        # print(f"Running command: /usr/bin/mpirun --allow-run-as-root -n 4 /usr/bin/python3 {model_runner} --checkpoint {self.checkpoint} &> trt.log 2>&1")
        # print(f"ret: {ret}")
        # import time
        # time.sleep(10)
        # out, err = self.model_process.communicate()
        # print(f"out: {out}, err: {err}")
                
        self._model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-cased")
        self._model.tensorrt = True
        # self._model.generate = runner.generate
        
        
        # self._model.generate = generation_decorator(runner.generate)
        # self._model.generate = MoEGenDecorator(self._model.generate)

        # with torch.no_grad():
        # outputs = runner.generate(
        #         batch_input_ids,
        #         max_new_tokens=args.max_output_len,
        #         max_attention_window_size=args.max_attention_window_size,
        #         sink_token_length=args.sink_token_length,
        #         end_id=end_id,
        #         pad_id=pad_id,
        #         temperature=args.temperature,
        #         top_k=args.top_k,
        #         top_p=args.top_p,
        #         num_beams=args.num_beams,
        #         length_penalty=args.length_penalty,
        #         early_stopping=args.early_stopping,
        #         repetition_penalty=args.repetition_penalty,
        #         presence_penalty=args.presence_penalty,
        #         frequency_penalty=args.frequency_penalty,
        #         stop_words_list=stop_words_list,
        #         bad_words_list=bad_words_list,
        #         output_cum_log_probs=(args.output_cum_log_probs_npy != None),
        #         output_log_probs=(args.output_log_probs_npy != None),
        #         lora_uids=args.lora_task_uids,
        #         prompt_table=args.prompt_table_path,
        #         prompt_tasks=args.prompt_tasks,
        #         streaming=args.streaming,
        #         output_sequence_lengths=True,
        #         return_dict=True,
        #         medusa_choices=args.medusa_choices)
        #     torch.cuda.synchronize()
    
    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model_config, attr):
                return getattr(self.model_config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.use_chat_template:
            try:
                updated_strings = []
                for input_string in strings:
                    messages = [
                        {"role": "user", "content": f"{input_string}"},
                    ]
                    updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False)
                    updated_strings.append(updated_string)
                strings = updated_strings[:]
            except:
                print(f"failed to update input string with chat template: {self._model}")
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = False

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]
