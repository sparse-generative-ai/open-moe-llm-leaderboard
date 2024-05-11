#!/usr/bin/env python3

from datasets import load_dataset

path = "pminervini/averitec"

ds = load_dataset(
    "json",
    data_files={
        "train": "/Users/pasquale/workspace/AVeriTeC/data/train.json",
        "dev": "/Users/pasquale/workspace/AVeriTeC/data/dev.json",
    },
)
ds.push_to_hub(path)
