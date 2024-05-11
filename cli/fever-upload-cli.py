#!/usr/bin/env python3

import glob
import os

import random
from tqdm import tqdm

from datasets import Dataset, DatasetDict, load_dataset


def convert(list_of_dicts):
    res = {}
    for d in list_of_dicts:
        for k, v in d.items():
            res.setdefault(k, []).append(v)
    return res


v10 = load_dataset("fever", "v1.0")
name_lst = ["train", "labelled_dev"]

old_to_new_label_map = {"SUPPORTS": "supported", "REFUTES": "refuted"}

data_map = {}

for name in name_lst:
    instance_lst = []

    for entry in tqdm(v10[name]):
        id_ = entry["id"]
        label = entry["label"]
        claim = entry["claim"]

        evidence_id = entry["evidence_id"]
        evidence_wiki_url = entry["evidence_wiki_url"]

        if evidence_id != -1:
            assert label in {"SUPPORTS", "REFUTES"}

            instance = {"id": id_, "label": old_to_new_label_map[label], "claim": claim}
            instance_lst.append(instance)

    key = "dev" if name in {"labelled_dev"} else name

    instance_lst = sorted([dict(t) for t in {tuple(d.items()) for d in instance_lst}], key=lambda d: d["claim"])

    label_to_instance_lst = {}
    for e in instance_lst:
        if e["label"] not in label_to_instance_lst:
            label_to_instance_lst[e["label"]] = []
        label_to_instance_lst[e["label"]].append(e)

    min_len = min(len(v) for k, v in label_to_instance_lst.items())

    new_instance_lst = []
    for k in sorted(label_to_instance_lst.keys()):
        new_instance_lst += label_to_instance_lst[k][:min_len]

    random.Random(42).shuffle(new_instance_lst)
    data_map[key] = new_instance_lst

ds_path = "pminervini/hl-fever"

task_to_ds_map = {k: Dataset.from_dict(convert(v)) for k, v in data_map.items()}
ds_dict = DatasetDict(task_to_ds_map)

ds_dict.push_to_hub(ds_path, "v1.0")

# breakpoint()
