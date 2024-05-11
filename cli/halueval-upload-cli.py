#!/usr/bin/env python3

import random
import requests

from datasets import load_dataset, Dataset, DatasetDict


path = "pminervini/HaluEval"

API_URL = f"https://datasets-server.huggingface.co/splits?dataset={path}"
response = requests.get(API_URL)
res_json = response.json()

gold_splits = {"dialogue", "qa", "summarization", "general"}

available_splits = {split["config"] for split in res_json["splits"]} if "splits" in res_json else set()

name_to_ds = dict()

for name in gold_splits:
    ds = load_dataset("json", data_files={"data": f"data/{name}_data.json"})
    name_to_ds[name] = ds
    # if name not in available_splits:
    ds.push_to_hub(path, config_name=name)


def list_to_dict(lst: list) -> dict:
    res = dict()
    for entry in lst:
        for k, v in entry.items():
            if k not in res:
                res[k] = []
            res[k] += [v]
    return res


for name in gold_splits - {"general"}:
    random.seed(42)
    ds = name_to_ds[name]
    new_entry_lst = []

    for entry in ds["data"]:
        is_hallucinated = random.random() > 0.5
        new_entry = None
        if name in {"qa"}:
            new_entry = {
                "knowledge": entry["knowledge"],
                "question": entry["question"],
                "answer": entry[f'{"hallucinated" if is_hallucinated else "right"}_answer'],
                "hallucination": "yes" if is_hallucinated else "no",
            }
        if name in {"dialogue"}:
            new_entry = {
                "knowledge": entry["knowledge"],
                "dialogue_history": entry["dialogue_history"],
                "response": entry[f'{"hallucinated" if is_hallucinated else "right"}_response'],
                "hallucination": "yes" if is_hallucinated else "no",
            }
        if name in {"summarization"}:
            new_entry = {
                "document": entry["document"],
                "summary": entry[f'{"hallucinated" if is_hallucinated else "right"}_summary'],
                "hallucination": "yes" if is_hallucinated else "no",
            }
        assert new_entry is not None
        new_entry_lst += [new_entry]
    new_ds_map = list_to_dict(new_entry_lst)
    new_ds = Dataset.from_dict(new_ds_map)
    new_dsd = DatasetDict({"data": new_ds})

    new_dsd.push_to_hub(path, config_name=f"{name}_samples")
