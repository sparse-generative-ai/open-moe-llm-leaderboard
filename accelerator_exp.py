import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import datasets
import random
import pickle
import torch.utils.data.dataloader
from lm_eval.models.utils import stop_sequences_criteria
from tqdm import tqdm
from time import time
from transformers import TextStreamer
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, required=True, help="Model ID")
    parser.add_argument("--use_int8", action="store_true", help="Batch size")
    parser.add_argument("--use_int4", action="store_true", help="Batch size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--precision", type=int, default=1, help="Number of samples")
    return parser.parse_args()

args = get_args()
out_path = args.out_path

if args.precision == 1:
    precision = torch.bfloat16
elif args.precision == 2:
    precision = torch.float16

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
    
model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=precision, load_in_4bit=args.use_int4, load_in_8bit=args.use_int8)

batch_size = args.batch_size
doc = pickle.load(open("sampled_gsm8k.pkl", "rb"))

updated_strings = []
for string in doc:
    messages = [
                    {"role": "user", "content": f"{string}"},
                ]
    updated_string = tokenizer.apply_chat_template(messages, tokenize=False)
    updated_strings.append(updated_string)

if batch_size == 1:
    encodings = []
    attention_masks = []
    for string in updated_strings:
        encoding = tokenizer(string, return_tensors="pt")
        encodings.append(encoding["input_ids"])
        attention_masks.append(encoding["attention_mask"])
    
    dataloader = zip(encodings, attention_masks)
else:
    dataloader = [updated_strings[i:i + batch_size] for i in range(0, len(updated_strings), batch_size)]

until = ["Question:", "Question", "</s>", "<|im_end|>"]
result_dict = {"avg_end2end_time":[], "avg_prefilling_time":[], "avg_decoding_tp":[], "results":[]}

count = 1
it = 0
for batch in tqdm(dataloader):
    streamer = StopWatch(tokenizer)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    encodings = tokenizer(
        batch,
        truncation=False,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=False)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    
    input_id = input_id.to(model.device)
    attn_mask = attention_mask.to(model.device)
    stopping_criteria = stop_sequences_criteria(
            tokenizer, until, input_id.shape[1], input_id.shape[0]
        )
    
    start = time()
    res = model.generate(
        input_id, 
        attention_mask=attn_mask, 
        max_new_tokens=256, 
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopping_criteria,
        use_cache=True,
        streamer=streamer)
    end = time()
    
    batch_size = input_id.shape[0]
    input_length = input_id.shape[1]
    output_length = streamer.decoding_iterations
    end2end_time = (end - start) / batch_size
    result_dict["avg_end2end_time"].append(end2end_time)
    result_dict["avg_prefilling_time"].append(streamer.prefilling_time / batch_size if streamer.prefilling_time is not None else 0)
    decoding_time = streamer.decoding_time / batch_size if streamer.decoding_time is not None else 0
    decoding_tp = output_length / decoding_time if decoding_time != 0 else output_length / end2end_time
    result_dict["avg_decoding_tp"].append(decoding_tp)
    
    for r in res:
        r = r[input_length:]
        res_dict = {"doc_id": count,
                    "Question": updated_strings[count-1], 
                    "Answer": tokenizer.decode(r, skip_special_tokens=True),
                    "end2end_time": end2end_time,
                    "prefilling_time": streamer.prefilling_time if streamer.prefilling_time is not None else 0,
                    "decoding_tp": decoding_tp,
                    "input_length": input_length,
                    "output_length": output_length}
        result_dict["results"].append(res_dict)
    # if it == 2:
    #     break

result_dict["avg_end2end_time"] = sum(result_dict["avg_end2end_time"]) / len(result_dict["avg_end2end_time"])
result_dict["avg_prefilling_time"] = sum(result_dict["avg_prefilling_time"]) / len(result_dict["avg_prefilling_time"])
result_dict["avg_decoding_tp"] = sum(result_dict["avg_decoding_tp"]) / len(result_dict["avg_decoding_tp"])

import json
with open(out_path, "w") as f:
    json.dump(result_dict, f)