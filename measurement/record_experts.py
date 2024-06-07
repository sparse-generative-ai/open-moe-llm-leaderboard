import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from datasets import load_dataset
import pickle
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["mixtral", "dbrx", "qwen", "mixtral22"])
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max-num-batches", type=int, default=10)
    return parser.parse_args()



device = "cuda:0"
prefilling_finished = False
record = []

def wrap_forward_with_logging(original_forward, layer_id):
    def new_forward(*args, **kwargs):
        global prefilling_finished, record
        do_record = False
        if not prefilling_finished:
            prefilling_finished = True
        else:
            do_record = True
            # hidden_states = args[0]
            # attention_mask = kwargs.get("attention_mask", None)
            # position_ids = kwargs.get("position_ids", None)
            # past_key_value = kwargs.get("past_key_value", None)
            # if past_key_value is not None:
            #     # def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
            #     past_key_value = past_key_value.to_legacy_cache()
            #     # move to cpu
            #     past_key_value = tuple(tuple(t.cpu() for t in ten) for ten in past_key_value)
            # cache_position = kwargs.get("cache_position", None)
            # record.append({
            #     "hidden_states": hidden_states.cpu(),
            #     "attention_mask": attention_mask.cpu() if attention_mask is not None else None,
            #     "position_ids": position_ids.cpu() if position_ids is not None else None,
            #     "past_key_value": past_key_value,
            #     "cache_position": cache_position.cpu() if cache_position is not None else None
            # })
            record.append({})
            kwargs["output_router_logits"] = True

        # Call the original forward method
        output = original_forward(*args, **kwargs)

        if do_record:
            # Get router logits
            router_logits = output[-1]
            record[-1]["router_logits"] = router_logits.cpu()

        return output
    
    return new_forward


if __name__ == "__main__":
    args = get_args()
    model_name = args.model_name

    batch_size = args.batch_size

    hf_model_name = {
        "mixtral22": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "dbrx": "databricks/dbrx-instruct",
        "qwen": "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    }[model_name]
    model = AutoModelForCausalLM.from_pretrained(hf_model_name,
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto")
    model.eval()

    # gsm8k = load_dataset("gsm8k", "main")
    all_input_raw = pickle.load(open(os.path.join(os.path.dirname(__file__), "sampled_gsm8k.pkl"), "rb"))

    for layer_id in range(32):
        original_forward = model.model.layers[layer_id].forward
        model.model.layers[layer_id].original_forward = original_forward
        model.model.layers[layer_id].forward = wrap_forward_with_logging(original_forward, layer_id)
        # model.model.layers[0].register_forward_hook(record_input_to_file_hook)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    all_input_chat = [{"role": "user", "content": f"{input_string}"} for input_string in all_input_raw]
    all_input_updated = [tokenizer.apply_chat_template([chat], tokenize=False, add_generation_prompt=True) for chat in all_input_chat]
    batched_input_data = [all_input_updated[i:i + batch_size] for i in range(0, len(all_input_updated), batch_size)]
    with torch.no_grad():
        for i, batch in enumerate(batched_input_data):
            inputs = tokenizer(batch, return_tensors="pt", padding="longest").to(device)
            prefilling_finished = False
            output = model.generate(**inputs, max_new_tokens=256)
            print(len(record))

    # save record to file
    logits = []
    for record_item in record:
        logits.append(record_item["router_logits"])
    with open(f"decoding_trace_{model_name}_bs{batch_size}_layer{layer_id}_tot{len(record)}.pkl", "wb") as f:
        pickle.dump(record, f)
    with open(f"decoding_logits_{model_name}_bs{batch_size}_layer{layer_id}_tot{len(record)}.pkl", "wb") as f:
        pickle.dump(logits, f)
    