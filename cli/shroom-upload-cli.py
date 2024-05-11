#!/usr/bin/env python3

import json
from datasets import Dataset, DatasetDict

file_path = "shroom-data/val.model-agnostic.json"
ds_path = "pminervini/shroom"

with open(file_path, "r") as file:
    data = json.load(file)


def convert(list_of_dicts):
    dict_of_lists = {}
    for d in list_of_dicts:
        for key, value in d.items():
            dict_of_lists.setdefault(key, []).append(value)
    return dict_of_lists


task_to_data_map = {}

for entry in data:
    task_name = entry["task"]
    del entry["task"]
    if task_name not in task_to_data_map:
        task_to_data_map[task_name] = []
    task_to_data_map[task_name] += [entry]

task_to_ds_map = {k: Dataset.from_dict(convert(data)) for k, data in task_to_data_map.items()}

ds_dict = DatasetDict(task_to_ds_map)

ds_dict.push_to_hub(ds_path)
