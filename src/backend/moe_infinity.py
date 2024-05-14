import torch
import os
import shutil
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from moe_infinity import MoE
from typing import List, Tuple, Optional, Union

from lm_eval.api.registry import register_model

from src.backend.hflm_with_measurement import HFLMWithMeasurement


@register_model("moe-infinity")
class MoEHFLM(HFLMWithMeasurement):
    def __init__(
        self,
        pretrained: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        moe_config: dict = None,
        offload_path=os.path.expanduser("~"),
        device_memory_ratio=0.75,
        use_chat_template=True,
        *args,
        **kwargs,
    ):
        # Initialize parent class without calling _create_model in the parent's __init__
        self.checkpoint = pretrained
        self.moe_config = moe_config if moe_config is not None else {}
        self.offload_path = offload_path
        self.device_memory_ratio = device_memory_ratio
        self.use_chat_template = use_chat_template
        if "device" in kwargs:
            kwargs.pop("device")
        if os.path.exists(os.path.join(self.offload_path, "moe-infinity-offloads")):
            shutil.rmtree(os.path.join(self.offload_path, "moe-infinity-offloads"))
        super().__init__(
            *args, **kwargs, pretrained=pretrained, device_map="cuda:0"
    )  # Assuming HFLM accepts a 'pretrained' arg and handles it
        # self._create_model()
        shutil.rmtree(os.path.join(self.offload_path, "moe-infinity-offloads"))

    def __del__(self):
        self._model.engine.clean_up() # clean up hooks
        self._model.engine.archer_engine.clean_up_resources() # clean up resources
        if os.path.exists(os.path.join(self.offload_path, "moe-infinity-offloads")):
            shutil.rmtree(os.path.join(self.offload_path, "moe-infinity-offloads")) # clean up offload model


    def _create_model(self, *args, **kwargs):
        """
        Initializes the MoE model from MoE-infinity with the provided configuration.
        """
        # Ensure default configurations are set if not provided
        default_moe_config = {
            "offload_path": os.path.join(self.offload_path, "moe-infinity-offloads"),
            "device_memory_ratio": self.device_memory_ratio,  # Default value, adjust as necessary
        }
        # Update default config with any user-provided config
        final_moe_config = {**default_moe_config, **self.moe_config}

        # dirty fix, to be removed when MoE-infinity supports move input to correct device
        def MoEGenDecorator(func):
            def wrapper(*args, **kwargs):
                # Ensure all tensor in the input are in the same device as the model
                args = [arg.to("cuda:0") if isinstance(arg, torch.Tensor) else arg for arg in args]
                kwargs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
                return func(*args, **kwargs)
            return wrapper

        self._model = MoE(self.checkpoint, final_moe_config)
        self._model.generate = MoEGenDecorator(self._model.generate)
        # self._model = AutoModelForCausalLM.from_pretrained(
        #     self.checkpoint, torch_dtype=torch.float16, device_map="auto"
        # )

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.model.config, attr):
                return getattr(self.model.model.config, attr)
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
