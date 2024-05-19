import json
import os
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np

from src.display.formatting import has_no_nan_values, make_clickable_model
from src.display.utils import AutoEvalColumn, EvalQueueColumn
from src.leaderboard.filter_models import filter_models
from src.leaderboard.read_evals import get_raw_eval_results, EvalResult, update_model_type_with_open_llm_request_file

from src.backend.envs import Tasks as BackendTasks
from src.display.utils import Tasks
from src.display.utils import system_metrics_to_name_map, gpu_metrics_to_name_map

def get_leaderboard_df(
    results_path: str,
    requests_path: str,
    requests_path_open_llm: str,
    cols: list,
    benchmark_cols: list,
    is_backend: bool = False,
) -> tuple[list[EvalResult], pd.DataFrame]:
    # Returns a list of EvalResult
    raw_data: list[EvalResult] = get_raw_eval_results(results_path, requests_path, requests_path_open_llm)
    if requests_path_open_llm != "":
        for result_idx in tqdm(range(len(raw_data)), desc="updating model type with open llm leaderboard"):
            raw_data[result_idx] = update_model_type_with_open_llm_request_file(
                raw_data[result_idx], requests_path_open_llm
            )

    # all_data_json_ = [v.to_dict() for v in raw_data if v.is_complete()]
    all_data_json_ = [v.to_dict() for v in raw_data] # include incomplete evals

    name_to_bm_map = {}

    task_iterator = Tasks
    if is_backend is True:
        task_iterator = BackendTasks

    for task in task_iterator:
        task = task.value
        name = task.col_name
        bm = (task.benchmark, task.metric)
        name_to_bm_map[name] = bm



    all_data_json = []
    for entry in all_data_json_:
        new_entry = copy.deepcopy(entry)
        for k, v in entry.items():
            if k in name_to_bm_map:
                benchmark, metric = name_to_bm_map[k]
                new_entry[k] = entry[k][metric]
                for sys_metric, metric_namne in system_metrics_to_name_map.items():
                    if sys_metric in entry[k]:
                        new_entry[f"{k} {metric_namne}"] = entry[k][sys_metric]

                for gpu_metric, metric_namne in gpu_metrics_to_name_map.items():
                    if gpu_metric in entry[k]:
                        new_entry[f"{k} {metric_namne}"] = entry[k][gpu_metric]
        all_data_json += [new_entry]

    # all_data_json.append(baseline_row)
    filter_models(all_data_json)

    df = pd.DataFrame.from_records(all_data_json)

    # if AutoEvalColumn.average.name in df:
    #     df = df.sort_values(by=[AutoEvalColumn.average.name], ascending=False)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

    if not df.empty:
        df = df.round(decimals=4)

        # filter out if any of the benchmarks have not been produced
        # df = df[has_no_nan_values(df, benchmark_cols)]

    return raw_data, df


def get_evaluation_queue_df(save_path: str, cols: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entries = [entry for entry in os.listdir(save_path) if not entry.startswith(".")]
    all_evals = []

    for entry in entries:
        if ".json" in entry:
            file_path = os.path.join(save_path, entry)
            with open(file_path) as fp:
                data = json.load(fp)

            data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
            data[EvalQueueColumn.revision.name] = data.get("revision", "main")
            data[EvalQueueColumn.model_framework.name] = data.get("inference_framework", "-")

            all_evals.append(data)
        elif ".md" not in entry:
            # this is a folder
            sub_entries = [e for e in os.listdir(f"{save_path}/{entry}") if not e.startswith(".")]
            for sub_entry in sub_entries:
                file_path = os.path.join(save_path, entry, sub_entry)
                with open(file_path) as fp:
                    data = json.load(fp)

                data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
                data[EvalQueueColumn.revision.name] = data.get("revision", "main")
                data[EvalQueueColumn.model_framework.name] = data.get("inference_framework", "-")
                all_evals.append(data)

    pending_list = [e for e in all_evals if e["status"] in ["PENDING", "RERUN"]]
    running_list = [e for e in all_evals if e["status"] == "RUNNING"]
    finished_list = [e for e in all_evals if e["status"].startswith("FINISHED") or e["status"] == "PENDING_NEW_EVAL"]
    df_pending = pd.DataFrame.from_records(pending_list, columns=cols)
    df_running = pd.DataFrame.from_records(running_list, columns=cols)
    df_finished = pd.DataFrame.from_records(finished_list, columns=cols)
    return df_finished[cols], df_running[cols], df_pending[cols]
