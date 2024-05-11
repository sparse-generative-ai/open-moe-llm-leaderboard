#!/usr/bin/env python

from huggingface_hub import snapshot_download

from src.backend.manage_requests import get_eval_requests
from src.backend.sort_queue import sort_models_by_priority
from src.backend.envs import Tasks, EVAL_REQUESTS_PATH_BACKEND, EVAL_RESULTS_PATH_BACKEND

from src.backend.manage_requests import EvalRequest
from src.leaderboard.read_evals import EvalResult

from src.envs import QUEUE_REPO, RESULTS_REPO, API

import logging
import pprint

logging.getLogger("openai").setLevel(logging.WARNING)

logging.basicConfig(level=logging.ERROR)
pp = pprint.PrettyPrinter(width=80)

PENDING_STATUS = "PENDING"
RUNNING_STATUS = "RUNNING"
FINISHED_STATUS = "FINISHED"
FAILED_STATUS = "FAILED"

TASKS_HARNESS = [task.value for task in Tasks]

snapshot_download(
    repo_id=RESULTS_REPO, revision="main", local_dir=EVAL_RESULTS_PATH_BACKEND, repo_type="dataset", max_workers=60
)
snapshot_download(
    repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60
)


def request_to_result_name(request: EvalRequest) -> str:
    org_and_model = request.model.split("/", 1)
    if len(org_and_model) == 1:
        model = org_and_model[0]
        res = f"{model}_{request.precision}"
    else:
        org = org_and_model[0]
        model = org_and_model[1]
        res = f"{org}_{model}_{request.precision}"
    return res


def process_finished_requests() -> bool:
    current_finished_status = [FINISHED_STATUS]

    if False:
        import os
        import dateutil

        model_result_filepaths = []
        results_path = f"{EVAL_RESULTS_PATH_BACKEND}/EleutherAI/gpt-neo-1.3B"
        requests_path = f"{EVAL_REQUESTS_PATH_BACKEND}/EleutherAI/gpt-neo-1.3B_eval_request_False_False_False.json"

        for root, _, files in os.walk(results_path):
            # We should only have json files in model results
            if len(files) == 0 or any([not f.endswith(".json") for f in files]):
                continue

            # Sort the files by date
            try:
                files.sort(key=lambda x: x.removesuffix(".json").removeprefix("results_")[:-7])
            except dateutil.parser._parser.ParserError:
                files = [files[-1]]

            for file in files:
                model_result_filepaths.append(os.path.join(root, file))

        eval_results = {}
        for model_result_filepath in model_result_filepaths:
            # Creation of result
            eval_result = EvalResult.init_from_json_file(model_result_filepath)
            eval_result.update_with_request_file(requests_path)

            print("XXX", eval_result)

            # Store results of same eval together
            eval_name = eval_result.eval_name
            if eval_name in eval_results.keys():
                eval_results[eval_name].results.update({k: v for k, v in eval_result.results.items() if v is not None})
            else:
                eval_results[eval_name] = eval_result

        print(eval_results)

        return True

    # Get all eval request that are FINISHED, if you want to run other evals, change this parameter
    eval_requests: list[EvalRequest] = get_eval_requests(
        job_status=current_finished_status, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND
    )
    # Sort the evals by priority (first submitted first run)
    eval_requests: list[EvalRequest] = sort_models_by_priority(api=API, models=eval_requests)

    # XXX
    # eval_requests = [r for r in eval_requests if 'neo-1.3B' in r.model]

    import random

    random.shuffle(eval_requests)

    from src.leaderboard.read_evals import get_raw_eval_results

    eval_results: list[EvalResult] = get_raw_eval_results(EVAL_RESULTS_PATH_BACKEND, EVAL_REQUESTS_PATH_BACKEND)

    result_name_to_request = {request_to_result_name(r): r for r in eval_requests}
    result_name_to_result = {r.eval_name: r for r in eval_results}

    for eval_request in eval_requests:
        result_name: str = request_to_result_name(eval_request)

        # Check the corresponding result
        from typing import Optional

        eval_result: Optional[EvalResult] = (
            result_name_to_result[result_name] if result_name in result_name_to_result else None
        )

        # Iterate over tasks and, if we do not have results for a task, run the relevant evaluations
        for task in TASKS_HARNESS:
            task_name = task.benchmark

            if eval_result is None or task_name not in eval_result.results:
                eval_request: EvalRequest = result_name_to_request[result_name]

                # print(eval_result)
                print(result_name, "is incomplete -- missing task:", task_name, eval_result, eval_request.likes)


if __name__ == "__main__":
    res = process_finished_requests()
