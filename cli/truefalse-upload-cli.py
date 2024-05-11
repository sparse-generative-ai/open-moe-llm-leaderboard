#!/usr/bin/env python3

import glob
import os

from datasets import load_dataset

path = "pminervini/true-false"
folder_path = "true-false-data/"  # Replace with your folder path

# Search for all .json files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

ds = load_dataset("csv", data_files={os.path.basename(path).split("_")[0]: path for path in csv_files})
ds.push_to_hub(path)
