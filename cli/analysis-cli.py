#!/usr/bin/env python3

import os
import sys
import json
import pickle

import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage

from src.backend.envs import Tasks, EVAL_REQUESTS_PATH_BACKEND, EVAL_RESULTS_PATH_BACKEND, DEVICE, LIMIT, Task

from src.envs import QUEUE_REPO, RESULTS_REPO, API
from src.utils import my_snapshot_download


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def find_json_files(json_path):
    res = []
    for root, dirs, files in os.walk(json_path):
        for file in files:
            if file.endswith(".json"):
                res.append(os.path.join(root, file))
    return res


def sanitise_metric(name: str) -> str:
    res = name
    res = res.replace("prompt_level_strict_acc", "Prompt-Level Accuracy")
    res = res.replace("acc", "Accuracy")
    res = res.replace("exact_match", "EM")
    res = res.replace("avg-selfcheckgpt", "AVG")
    res = res.replace("max-selfcheckgpt", "MAX")
    res = res.replace("rouge", "ROUGE-")
    res = res.replace("bertscore_precision", "BERT-P")
    res = res.replace("exact", "EM")
    res = res.replace("HasAns_EM", "HasAns")
    res = res.replace("NoAns_EM", "NoAns")
    res = res.replace("em", "EM")
    return res


def sanitise_dataset(name: str) -> str:
    res = name
    res = res.replace("tqa8", "TriviaQA (8-shot)")
    res = res.replace("nq8", "NQ (8-shot)")
    res = res.replace("nq_open", "NQ (64-shot)")
    res = res.replace("triviaqa", "TriviaQA (64-shot)")
    res = res.replace("truthfulqa", "TruthfulQA")
    res = res.replace("ifeval", "IFEval")
    res = res.replace("selfcheckgpt", "SelfCheckGPT")
    res = res.replace("truefalse_cieacf", "True-False")
    res = res.replace("mc", "MC")
    res = res.replace("race", "RACE")
    res = res.replace("squad", "SQuAD")
    res = res.replace("memo-trap", "MemoTrap")
    res = res.replace("cnndm", "CNN/DM")
    res = res.replace("xsum", "XSum")
    res = res.replace("qa", "QA")
    res = res.replace("summarization", "Summarization")
    res = res.replace("dialogue", "Dialog")
    res = res.replace("halueval", "HaluEval")
    res = res.replace("_v2", "")
    res = res.replace("_", " ")
    return res


cache_file = "data_map_cache.pkl"


def load_data_map_from_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        return None


def save_data_map_to_cache(data_map, cache_file):
    with open(cache_file, "wb") as f:
        pickle.dump(data_map, f)


# Try to load the data_map from the cache file
data_map = load_data_map_from_cache(cache_file)


if data_map is None:
    my_snapshot_download(
        repo_id=RESULTS_REPO, revision="main", local_dir=EVAL_RESULTS_PATH_BACKEND, repo_type="dataset", max_workers=60
    )
    my_snapshot_download(
        repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60
    )

    result_path_lst = find_json_files(EVAL_RESULTS_PATH_BACKEND)
    request_path_lst = find_json_files(EVAL_REQUESTS_PATH_BACKEND)

    model_name_to_model_map = {}

    for path in request_path_lst:
        with open(path, "r") as f:
            data = json.load(f)
        model_name_to_model_map[data["model"]] = data

    model_dataset_metric_to_result_map = {}

    # data_map[model_name][(dataset_name, sanitised_metric_name)] = value
    data_map = {}

    for path in result_path_lst:
        with open(path, "r") as f:
            data = json.load(f)
        model_name = data["config"]["model_name"]
        for dataset_name, results_dict in data["results"].items():
            for metric_name, value in results_dict.items():

                if model_name_to_model_map[model_name]["likes"] > 128:

                    to_add = True

                    if "f1" in metric_name:
                        to_add = False

                    if "stderr" in metric_name:
                        to_add = False

                    if "memo-trap_v2" in dataset_name:
                        to_add = False

                    if "faithdial" in dataset_name:
                        to_add = False

                    if "truthfulqa_gen" in dataset_name:
                        to_add = False

                    if "bertscore" in metric_name:
                        if "precision" not in metric_name:
                            to_add = False

                    if "halueval" in dataset_name:
                        if "acc" not in metric_name:
                            to_add = False

                    if "ifeval" in dataset_name:
                        if "prompt_level_strict_acc" not in metric_name:
                            to_add = False

                    if "squad" in dataset_name:
                        # to_add = False
                        if "best_exact" in metric_name:
                            to_add = False

                    if "fever" in dataset_name:
                        to_add = False

                    if ("xsum" in dataset_name or "cnn" in dataset_name) and "v2" not in dataset_name:
                        to_add = False

                    if isinstance(value, str):
                        if is_float(value):
                            value = float(value)
                        else:
                            to_add = False

                    if to_add:
                        if "rouge" in metric_name:
                            value /= 100.0

                        if "squad" in dataset_name:
                            value /= 100.0

                        sanitised_metric_name = metric_name
                        if "," in sanitised_metric_name:
                            sanitised_metric_name = sanitised_metric_name.split(",")[0]
                        sanitised_metric_name = sanitise_metric(sanitised_metric_name)
                        sanitised_dataset_name = sanitise_dataset(dataset_name)

                        model_dataset_metric_to_result_map[
                            (model_name, sanitised_dataset_name, sanitised_metric_name)
                        ] = value

                        if model_name not in data_map:
                            data_map[model_name] = {}
                        data_map[model_name][(sanitised_dataset_name, sanitised_metric_name)] = value

                        print(
                            "model_name",
                            model_name,
                            "dataset_name",
                            sanitised_dataset_name,
                            "metric_name",
                            sanitised_metric_name,
                            "value",
                            value,
                        )

    save_data_map_to_cache(data_map, cache_file)

model_name_lst = [m for m in data_map.keys()]

nb_max_metrics = max(len(data_map[model_name]) for model_name in model_name_lst)

for model_name in model_name_lst:
    if len(data_map[model_name]) < nb_max_metrics - 5:
        del data_map[model_name]

plot_type_lst = ["all", "summ", "qa", "instr", "detect", "rc"]

for plot_type in plot_type_lst:

    data_map_v2 = {}
    for model_name in data_map.keys():
        for dataset_metric in data_map[model_name].keys():
            if dataset_metric not in data_map_v2:
                data_map_v2[dataset_metric] = {}

            if plot_type in {"all"}:
                to_add = True
                if "ROUGE" in dataset_metric[1] and "ROUGE-L" not in dataset_metric[1]:
                    to_add = False
                if "SQuAD" in dataset_metric[0] and "EM" not in dataset_metric[1]:
                    to_add = False
                if "SelfCheckGPT" in dataset_metric[0] and "MAX" not in dataset_metric[1]:
                    to_add = False
                if "64-shot" in dataset_metric[0]:
                    to_add = False
                if to_add is True:
                    data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
            elif plot_type in {"summ"}:
                if "CNN" in dataset_metric[0] or "XSum" in dataset_metric[0]:
                    data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
            elif plot_type in {"qa"}:
                if "TriviaQA" in dataset_metric[0] or "NQ" in dataset_metric[0] or "TruthfulQA" in dataset_metric[0]:
                    data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
            elif plot_type in {"instr"}:
                if "MemoTrap" in dataset_metric[0] or "IFEval" in dataset_metric[0]:
                    data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
            elif plot_type in {"detect"}:
                if "HaluEval" in dataset_metric[0] or "SelfCheck" in dataset_metric[0]:
                    data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
            elif plot_type in {"rc"}:
                if "RACE" in dataset_metric[0] or "SQuAD" in dataset_metric[0]:
                    data_map_v2[dataset_metric][model_name] = data_map[model_name][dataset_metric]
            else:
                assert False, f"Unknown plot type: {plot_type}"

    # df = pd.DataFrame.from_dict(data_map, orient='index')   # Invert the y-axis (rows)
    df = pd.DataFrame.from_dict(data_map_v2, orient="index")  # Invert the y-axis (rows)
    df.index = [", ".join(map(str, idx)) for idx in df.index]

    o_df = df.copy(deep=True)

    # breakpoint()

    print(df)

    # Check for NaN or infinite values and replace them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
    df.fillna(0, inplace=True)  # Replace NaN with 0 (or use another imputation strategy)

    from sklearn.preprocessing import MinMaxScaler

    # scaler = MinMaxScaler()
    # df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    # Calculate dimensions based on the DataFrame size
    cell_height = 1.0  # Height of each cell in inches
    cell_width = 1.0  # Width of each cell in inches

    n_rows = len(df.index)  # Datasets and Metrics
    n_cols = len(df.columns)  # Models

    # Calculate figure size dynamically
    fig_width = cell_width * n_cols + 0
    fig_height = cell_height * n_rows + 0

    col_cluster = True
    row_cluster = True

    sns.set_context("notebook", font_scale=1.3)

    dendrogram_ratio = (0.1, 0.1)

    if plot_type in {"detect"}:
        fig_width = cell_width * n_cols - 2
        fig_height = cell_height * n_rows + 5.2
        dendrogram_ratio = (0.1, 0.2)

    if plot_type in {"instr"}:
        fig_width = cell_width * n_cols - 2
        fig_height = cell_height * n_rows + 5.2
        dendrogram_ratio = (0.1, 0.4)

    if plot_type in {"qa"}:
        fig_width = cell_width * n_cols - 2
        fig_height = cell_height * n_rows + 4
        dendrogram_ratio = (0.1, 0.2)

    if plot_type in {"summ"}:
        fig_width = cell_width * n_cols - 2
        fig_height = cell_height * n_rows + 2.0
        dendrogram_ratio = (0.1, 0.1)
        row_cluster = False

    if plot_type in {"rc"}:
        fig_width = cell_width * n_cols - 2
        fig_height = cell_height * n_rows + 5.2
        dendrogram_ratio = (0.1, 0.4)

    print("figsize", (fig_width, fig_height))

    o_df.to_json(f"plots/clustermap_{plot_type}.json", orient="split")

    print(f"Generating the clustermaps for {plot_type}")

    for cmap in [None, "coolwarm", "viridis"]:
        fig = sns.clustermap(
            df,
            method="ward",
            metric="euclidean",
            cmap=cmap,
            figsize=(fig_width, fig_height),  # figsize=(24, 16),
            annot=True,
            mask=o_df.isnull(),
            dendrogram_ratio=dendrogram_ratio,
            fmt=".2f",
            col_cluster=col_cluster,
            row_cluster=row_cluster,
        )

        # Adjust the size of the cells (less wide)
        plt.setp(fig.ax_heatmap.get_yticklabels(), rotation=0)
        plt.setp(fig.ax_heatmap.get_xticklabels(), rotation=90)

        cmap_suffix = "" if cmap is None else f"_{cmap}"

        # Save the clustermap to file
        fig.savefig(f"blog/figures/clustermap_{plot_type}{cmap_suffix}.pdf")
        fig.savefig(f"blog/figures/clustermap_{plot_type}{cmap_suffix}.png")
        fig.savefig(f"blog/figures/clustermap_{plot_type}{cmap_suffix}_t.png", transparent=True, facecolor="none")
