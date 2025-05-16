import torch
from tqdm import tqdm

import argparse
import pickle
import gc


device = "cuda:0"

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-name", type=str, required=True, choices=["mixtral22", "dbrx", "qwen", "mixtral"])
#     return parser.parse_args()


if __name__ == "__main__":
    # args = get_args()
    # model_name = args.model_name
    model_name = "mixtral"
    batch_size = 16
    num_experts_per_token = 4
    n_layers = 24
    num_activated_experts = {}
    # get the record from the file
    with open(f"/home/jysc/open-moe-llm-leaderboard/profiling/trace_res/decoding_trace_qwen_bs16_tot1035592.pkl", "rb") as f:
        record = pickle.load(f)
    num_experts = []
    for record_item in record:
        router_logits = record_item["router_logits"]
        activated_experts = torch.topk(router_logits, num_experts_per_token, dim=-1)
        # experts.extend(activated_experts.indices.flatten().tolist())
        num_experts.append(len(set(activated_experts.indices.flatten().tolist())))
    print(f"average number of activated experts: {sum(num_experts) / len(num_experts)}")
    num_activated_experts[(n_layers, batch_size)] = num_experts
    del record
    gc.collect()
    with open(f"/home/jysc/open-moe-llm-leaderboard/profiling/activations/num_activated_experts_{model_name}_tot{batch_size}.pkl", "wb") as f:
        pickle.dump(num_activated_experts, f)
        