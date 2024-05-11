#!/usr/bin/env python

from huggingface_hub import snapshot_download
from src.leaderboard.read_evals import get_raw_eval_results
from src.envs import API, EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, QUEUE_REPO, RESULTS_REPO

from src.backend.run_eval_suite import run_evaluation
from src.backend.manage_requests import check_completed_evals, get_eval_requests, set_eval_request
from src.backend.sort_queue import sort_models_by_priority
from src.backend.envs import Tasks, EVAL_REQUESTS_PATH_BACKEND, EVAL_RESULTS_PATH_BACKEND, DEVICE, LIMIT, Task

from src.leaderboard.read_evals import get_raw_eval_results

from src.backend.manage_requests import EvalRequest
from src.leaderboard.read_evals import EvalResult

snapshot_download(
    repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30
)
snapshot_download(
    repo_id=RESULTS_REPO, local_dir=EVAL_RESULTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30
)

PENDING_STATUS = "PENDING"
RUNNING_STATUS = "RUNNING"
FINISHED_STATUS = "FINISHED"
FAILED_STATUS = "FAILED"

TASKS_HARNESS = [task.value for task in Tasks]

current_finished_status = [FINISHED_STATUS]


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


# Get all eval request that are FINISHED, if you want to run other evals, change this parameter
eval_requests: list[EvalRequest] = get_eval_requests(
    job_status=current_finished_status, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND
)
# Sort the evals by priority (first submitted first run)
eval_requests: list[EvalRequest] = sort_models_by_priority(api=API, models=eval_requests)

eval_results: list[EvalResult] = get_raw_eval_results(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)

result_name_to_request = {request_to_result_name(r): r for r in eval_requests}
result_name_to_result = {r.eval_name: r for r in eval_results}

print("Requests", sorted(result_name_to_request.keys()))
print("Results", sorted(result_name_to_result.keys()))

for eval_request in eval_requests:
    result_name: str = request_to_result_name(eval_request)

    # Check the corresponding result
    eval_result: EvalResult = result_name_to_result[result_name]

    # Iterate over tasks and, if we do not have results for a task, run the relevant evaluations
    for task in TASKS_HARNESS:
        task_name = task.benchmark

        if task_name not in eval_result.results:
            print("RUN THIS ONE!", result_name, task_name)

raw_data = get_raw_eval_results(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
all_data_json = [v.to_dict() for v in raw_data if v.is_complete()]

breakpoint()
