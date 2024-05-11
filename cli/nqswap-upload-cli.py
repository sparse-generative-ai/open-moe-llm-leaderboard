#!/usr/bin/env python3

from datasets import load_dataset

path = "pminervini/NQ-Swap"

ds = load_dataset("json", data_files={"original": "nqswap/original.jsonl", "substituted": "nqswap/substituted.jsonl"})
ds.push_to_hub(path)
