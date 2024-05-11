#!/usr/bin/env python

import os
import fnmatch

import json
from huggingface_hub import HfApi


def find_json_files(directory):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, "*.json"):
            matches.append(os.path.join(root, filename))
    return matches


directory_path = "/Users/pasquale/workspace/eval/requests"
json_files = find_json_files(directory_path)

api = HfApi()
model_lst = api.list_models()

model_lst = [m for m in model_lst]

id_to_model = {m.id: m for m in model_lst}

for path in json_files:
    with open(path, "r") as fr:
        data = json.load(fr)

        model_id = data["model"]
        if model_id in id_to_model:
            model = id_to_model[model_id]

            to_overwrite = False

            is_finetuned = any(tag.startswith("base_model:") for tag in id_to_model[data["model"]].tags)

            if is_finetuned:
                data["model_type"] = "fine-tuned"
                to_overwrite = True

            is_instruction_tuned = ("nstruct" in model_id) or ("chat" in model_id)
            if is_instruction_tuned:
                data["model_type"] = "instruction-tuned"
                to_overwrite = True

            if to_overwrite is True:
                with open(path, "w") as fw:
                    json.dump(data, fw)

        else:
            print(f"Model {model_id} not found")
