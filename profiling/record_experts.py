import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from datasets import load_dataset
import pickle
import argparse
import os
import json
from datasets import Dataset
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["mixtral", "dbrx", "qwen", "mixtral22"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--task", type=str, default="gsm8k")
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

def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions

def transform_data(data):
    transformed_data = []
    for i in range(len(data)):
        transformed_item = {
            "question_id": data[i]["question_id"],
            "content": data[i]["turns"][0]["content"],  # Assuming you want the first turn's content
        }
        transformed_data.append(transformed_item)
    return transformed_data


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
    if args.task == "gsm8k":
        gsm8k = load_dataset("gsm8k", "main", split="test")
        all_input_raw = gsm8k['question']
        all_input_raw = gsm8k['question']
        max_new_tokens = 256
    elif args.task == 'arena':
        max_new_tokens = 4096
        data_path = '/home/a100user/open-moe-llm-leaderboard/src/backend/tasks/arena_hard/question.jsonl'
        data = load_questions(data_path)
        dataset = transform_data(data)
        dataset = Dataset.from_dict({"question_id": [item["question_id"] for item in dataset],
                             "content": [item["content"] for item in dataset]})
        all_input_raw = dataset['content']
        
    # all_input_raw = pickle.load(open(os.path.join(os.path.dirname(__file__), "sampled_gsm8k.pkl"), "rb"))
    if hf_model_name == "databricks/dbrx-instruct":
        n_layers = 40
    elif hf_model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        n_layers = 32
    elif hf_model_name == "mistralai/Mixtral-8x22B-Instruct-v0.1":
        n_layers = 56
    elif hf_model_name == "Qwen/Qwen1.5-MoE-A2.7B-Chat":
        n_layers = 24
        
    for layer_id in range(n_layers):
        if hf_model_name == "databricks/dbrx-instruct":
            original_forward = model.transformer.blocks[layer_id].forward
            model.transformer.blocks[layer_id].original_forward = original_forward
            model.transformer.blocks[layer_id].forward = wrap_forward_with_logging(original_forward, layer_id)
        else:
            original_forward = model.model.layers[layer_id].forward
            model.model.layers[layer_id].original_forward = original_forward
            model.model.layers[layer_id].forward = wrap_forward_with_logging(original_forward, layer_id)
        # model.model.layers[0].register_forward_hook(record_input_to_file_hook)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    if "Qwen" in hf_model_name:
        all_input_chat = [[{"role": "system", "content": "You are a helpful assistant."}, 
                        {"role": "user", "content": f"{input_string}"}] for input_string in all_input_raw]
    else:
        all_input_chat = [[{"role": "user", "content": input_string}] for input_string in all_input_raw]
    
    if "Qwen" in hf_model_name or "dbrx" in hf_model_name:
        all_input_updated = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in all_input_chat]
    else:
        all_input_updated = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in all_input_chat]
        
    batched_input_data = [all_input_updated[i:i + batch_size] for i in range(0, len(all_input_updated), batch_size)]
    with torch.no_grad():
        for i, batch in tqdm(enumerate(batched_input_data), total=len(batched_input_data)):
            inputs = tokenizer(batch, return_tensors="pt", padding="longest").to(device)
            prefilling_finished = False
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # save record to file
    logits = []
    for record_item in record:
        logits.append(record_item["router_logits"])
    with open(f"./trace_res/{args.task}_decoding_trace_{model_name}_bs{batch_size}_tot{len(record)}.pkl", "wb") as f:
        pickle.dump(record, f)
    with open(f"./logits_res/{args.task}_decoding_logits_{model_name}_bs{batch_size}_tot{len(record)}.pkl", "wb") as f:
        pickle.dump(logits, f)
    