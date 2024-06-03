from typing import List, Literal, Optional, Tuple, Union
import torch
import transformers

from lm_eval.api.registry import register_model

from src.backend.hflm_with_measurement import HFLMWithMeasurement


@register_model("hf-chat")
class HFLMwithChatTemplate(HFLMWithMeasurement):
    def __init__(self, use_chat_template=True, **kwargs):
        super().__init__(**kwargs)
        self.use_chat_template = use_chat_template

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
                    if "dbrx" in self.model.name_or_path:
                        updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    elif "Qwen" in self.model.name_or_path:
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": input_string}
                        ]
                        updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        updated_string = self.tokenizer.apply_chat_template(messages, tokenize=False)
                    updated_strings.append(updated_string)
                strings = updated_strings[:]
            except:
                print(f"failed to update input string with chat template: {self._model}")
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            add_special_tokens = False
        elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
            add_special_tokens = True

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
