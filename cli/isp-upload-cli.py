#!/usr/bin/env python3

import glob
import os

from datasets import load_dataset

folder_path = "isp-data-json/"  # Replace with your folder path

# Search for all .json files in the folder
json_files = glob.glob(os.path.join(folder_path, "*.jsonl"))

path = "pminervini/inverse-scaling"

for json_path in json_files:
    base_name = os.path.basename(json_path)
    name = base_name.split("_")[0]

    ds = load_dataset("json", data_files={"data": json_path})
    ds.push_to_hub(path, config_name=name)
