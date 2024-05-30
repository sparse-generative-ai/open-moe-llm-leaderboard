#!/usr/bin/env python

import os
import json
import argparse

import socket
import random
import threading
from datetime import datetime

from src.backend.run_eval_suite import run_evaluation
from src.backend.manage_requests import check_completed_evals, get_eval_requests, set_eval_request
from src.backend.sort_queue import sort_models_by_priority
from src.backend.envs import Tasks, EVAL_REQUESTS_PATH_BACKEND, EVAL_RESULTS_PATH_BACKEND, DEVICE, Task
from src.backend.manage_requests import EvalRequest
from src.leaderboard.read_evals import EvalResult

from src.envs import QUEUE_REPO, RESULTS_REPO, API, DEBUG_QUEUE_REPO, DEBUG_RESULTS_REPO
from src.utils import my_snapshot_download, analyze_gpu_stats, parse_nvidia_smi, monitor_gpus, get_gpu_details

from src.leaderboard.read_evals import get_raw_eval_results

from typing import Optional
import GPUtil
import time

import pprint
import logging

from lm_eval.filters.extraction import RegexFilter


# Configure the root logger
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARNING,
)

# Get the 'lm-eval' logger from the third-party library
eval_logger = logging.getLogger("lm-eval")

# Explicitly set the level for 'lm-eval' logger to WARNING
eval_logger.setLevel(logging.WARNING)

def tuple_input_decorator(func):
    def wrapper(self, resps, docs):
        stripped_resps = [[resp_data[0] for resp_data in group] for group in resps]

        filtered_resps = func(self, stripped_resps, docs)
        
        combined_resps = []
        for original_group, new_group in zip(resps, filtered_resps):
            combined_group = [(new_resp,) + rest_of_data[1:] for new_resp, rest_of_data in zip(new_group, original_group)]
            combined_resps.append(combined_group)

        return combined_resps
    return wrapper


def my_set_eval_request(api, eval_request, set_to_status, hf_repo, local_dir):
    for i in range(10):
        try:
            set_eval_request(
                api=api, eval_request=eval_request, set_to_status=set_to_status, hf_repo=hf_repo, local_dir=local_dir
            )
            return
        except Exception as e:
            print(f"Error setting eval request to {set_to_status}: {e}. Retrying in 60 seconds")
            time.sleep(60)
    return


logging.getLogger("openai").setLevel(logging.WARNING)

logging.basicConfig(level=logging.ERROR)
pp = pprint.PrettyPrinter(width=80)

PENDING_STATUS = "PENDING"
RUNNING_STATUS = "RUNNING"
FINISHED_STATUS = "FINISHED"
FAILED_STATUS = "FAILED"

TASKS_HARNESS = [task.value for task in Tasks]


my_snapshot_download(
    repo_id=RESULTS_REPO, revision="main", local_dir=EVAL_RESULTS_PATH_BACKEND, repo_type="dataset", max_workers=60
)
my_snapshot_download(
    repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60
)


def sanity_checks():
    print(f"Device: {DEVICE}")

    # pull the eval dataset from the hub and parse any eval requests
    # check completed evals and set them to finished
    my_snapshot_download(
        repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60
    )
    check_completed_evals(
        api=API,
        checked_status=RUNNING_STATUS,
        completed_status=FINISHED_STATUS,
        failed_status=FAILED_STATUS,
        hf_repo=QUEUE_REPO,
        local_dir=EVAL_REQUESTS_PATH_BACKEND,
        hf_repo_results=RESULTS_REPO,
        local_dir_results=EVAL_RESULTS_PATH_BACKEND,
    )
    return


def request_to_result_name(request: EvalRequest) -> str:
    # Request: EvalRequest(model='meta-llama/Llama-2-13b-hf', private=False, status='FINISHED',
    # json_filepath='./eval-queue-bk/meta-llama/Llama-2-13b-hf_eval_request_False_False_False.json',
    # weight_type='Original', model_type='pretrained', precision='float32', base_model='', revision='main',
    # submitted_time='2023-09-09T10:52:17Z', likes=389, params=13.016, license='?')
    #
    # EvalResult(eval_name='meta-llama_Llama-2-13b-hf_float32', full_model='meta-llama/Llama-2-13b-hf',
    # org='meta-llama', model='Llama-2-13b-hf', revision='main',
    # results={'nq_open': 33.739612188365655, 'triviaqa': 74.12505572893447},
    # precision=<Precision.float32: ModelDetails(name='float32', symbol='')>,
    # model_type=<ModelType.PT: ModelDetails(name='pretrained', symbol='🟢')>,
    # weight_type=<WeightType.Original: ModelDetails(name='Original', symbol='')>,
    # architecture='LlamaForCausalLM', license='?', likes=389, num_params=13.016, date='2023-09-09T10:52:17Z', still_on_hub=True)
    #
    org_and_model = request.model.split("/", 1)
    if len(org_and_model) == 1:
        model = org_and_model[0]
        res = f"{model}_{request.precision}"
    else:
        org = org_and_model[0]
        model = org_and_model[1]
        res = f"{org}_{model}_{request.precision}"
    return res


def process_evaluation(task: Task, eval_request: EvalRequest, limit: Optional[int] = None) -> dict:
    batch_size = 1
    batch_size = eval_request.batch_size

    init_gpu_info = analyze_gpu_stats(parse_nvidia_smi())
    # if init_gpu_info['Mem(M)'] > 500:
    #     assert False, f"This machine is not empty: {init_gpu_info}"
    gpu_stats_list = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpus, args=(stop_event, 5, gpu_stats_list))
    monitor_thread.start()
    
    original_apply = RegexFilter.apply
    if task.benchmark in ["gsm8k", "gsm8k_cot", "gsm8k_cot_self_consistency", "gsm8k_custom"]:
        RegexFilter.apply = tuple_input_decorator(RegexFilter.apply)
    else:
        RegexFilter.apply = original_apply

    try:
        results = run_evaluation(
            eval_request=eval_request,
            task_names=[task.benchmark],
            num_fewshot=task.num_fewshot,
            batch_size=batch_size,
            device=DEVICE,
            use_cache=None,
            limit=limit,
        )
    except RuntimeError as e:
        if "No executable batch size found" in str(e):
            batch_size = 1
            results = run_evaluation(
                eval_request=eval_request,
                task_names=[task.benchmark],
                num_fewshot=task.num_fewshot,
                batch_size=batch_size,
                device=DEVICE,
                use_cache=None,
                limit=limit,
            )
        else:
            raise

    # print("RESULTS", results)
    stop_event.set()
    monitor_thread.join()
    gpu_info = analyze_gpu_stats(gpu_stats_list)
    for task_name in results['results'].keys():
        for key, value in gpu_info.items():
            if "GPU" not in key:
                results['results'][task_name][f"{key},none"] = int(value)
            else:
                results['results'][task_name][f"{key},none"] = value

        results['results'][task_name]['batch_size,none'] = batch_size
        results['results'][task_name]['precision,none'] = eval_request.precision
    print(f"gpu_stats_list: {gpu_stats_list}")
    print("GPU Usage:", gpu_info)

    dumped = json.dumps(results, indent=2, default=lambda o: "<not serializable>")
    # print(dumped)

    output_path = os.path.join(
        EVAL_RESULTS_PATH_BACKEND, *eval_request.model.split("/"), f"results_{datetime.now()}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(dumped)

    my_snapshot_download(
        repo_id=RESULTS_REPO, revision="main", local_dir=EVAL_RESULTS_PATH_BACKEND, repo_type="dataset", max_workers=60
    )
    API.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=f"{eval_request.model}/results_{datetime.now()}.json",
        repo_id=RESULTS_REPO,
        repo_type="dataset",
    )
    
    RegexFilter.apply = original_apply
    return results


def process_finished_requests(thr: int, hard_task_lst: Optional[list[str]] = None) -> bool:
    sanity_checks()

    current_finished_status = [FINISHED_STATUS, FAILED_STATUS]

    # Get all eval request that are FINISHED, if you want to run other evals, change this parameter
    eval_requests: list[EvalRequest] = get_eval_requests(
        job_status=current_finished_status, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND
    )
    # Sort the evals by priority (first submitted, first run)
    eval_requests: list[EvalRequest] = sort_models_by_priority(api=API, models=eval_requests)

    random.shuffle(eval_requests)

    eval_results: list[EvalResult] = get_raw_eval_results(EVAL_RESULTS_PATH_BACKEND, EVAL_REQUESTS_PATH_BACKEND)

    result_name_to_request = {request_to_result_name(r): r for r in eval_requests}
    result_name_to_result = {r.eval_name: r for r in eval_results}

    for eval_request in eval_requests:
        if eval_request.likes >= thr:
            result_name: str = request_to_result_name(eval_request)

            # Check the corresponding result
            eval_result: Optional[EvalResult] = (
                result_name_to_result[result_name] if result_name in result_name_to_result else None
            )

            # breakpoint()

            task_lst = TASKS_HARNESS.copy()
            random.shuffle(task_lst)

            # Iterate over tasks and, if we do not have results for a task, run the relevant evaluations
            for task in task_lst:
                task_name = task.benchmark

                do_run_task = False
                if hard_task_lst is None or any(ss in task_name for ss in hard_task_lst):
                    do_run_task = True

                if (eval_result is None or task_name not in eval_result.results) and do_run_task:
                    eval_request: EvalRequest = result_name_to_request[result_name]

                    my_snapshot_download(
                        repo_id=QUEUE_REPO,
                        revision="main",
                        local_dir=EVAL_REQUESTS_PATH_BACKEND,
                        repo_type="dataset",
                        max_workers=60,
                    )
                    my_set_eval_request(
                        api=API,
                        eval_request=eval_request,
                        set_to_status=RUNNING_STATUS,
                        hf_repo=QUEUE_REPO,
                        local_dir=EVAL_REQUESTS_PATH_BACKEND,
                    )

                    results = process_evaluation(task, eval_request)

                    my_snapshot_download(
                        repo_id=QUEUE_REPO,
                        revision="main",
                        local_dir=EVAL_REQUESTS_PATH_BACKEND,
                        repo_type="dataset",
                        max_workers=60,
                    )
                    my_set_eval_request(
                        api=API,
                        eval_request=eval_request,
                        set_to_status=FINISHED_STATUS,
                        hf_repo=QUEUE_REPO,
                        local_dir=EVAL_REQUESTS_PATH_BACKEND,
                    )

                    return True

    return False


def maybe_refresh_results(thr: int, hard_task_lst: Optional[list[str]] = None) -> bool:
    sanity_checks()

    current_finished_status = [PENDING_STATUS, FINISHED_STATUS, FAILED_STATUS]

    # Get all eval request that are FINISHED, if you want to run other evals, change this parameter
    eval_requests: list[EvalRequest] = get_eval_requests(
        job_status=current_finished_status, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND
    )
    # Sort the evals by priority (first submitted, first run)
    eval_requests: list[EvalRequest] = sort_models_by_priority(api=API, models=eval_requests)

    random.shuffle(eval_requests)

    eval_results: list[EvalResult] = get_raw_eval_results(EVAL_RESULTS_PATH_BACKEND, EVAL_REQUESTS_PATH_BACKEND)

    result_name_to_request = {request_to_result_name(r): r for r in eval_requests}
    result_name_to_result = {r.eval_name: r for r in eval_results}

    for eval_request in eval_requests:
        if eval_request.likes >= thr:
            result_name: str = request_to_result_name(eval_request)

            # Check the corresponding result
            eval_result: Optional[EvalResult] = (
                result_name_to_result[result_name] if result_name in result_name_to_result else None
            )

            task_lst = TASKS_HARNESS.copy()
            random.shuffle(task_lst)

            # Iterate over tasks and, if we do not have results for a task, run the relevant evaluations
            for task in task_lst:
                task_name = task.benchmark

                do_run_task = False
                if hard_task_lst is None or any(ss in task_name for ss in hard_task_lst):
                    do_run_task = True

                task_lst = ["nq", "trivia", "tqa", "self"]
                if (
                    eval_result is None
                    or do_run_task
                    or task_name not in eval_result.results
                    or any(ss in task_name for ss in task_lst)
                ):
                    eval_request: EvalRequest = result_name_to_request[result_name]

                    my_snapshot_download(
                        repo_id=QUEUE_REPO,
                        revision="main",
                        local_dir=EVAL_REQUESTS_PATH_BACKEND,
                        repo_type="dataset",
                        max_workers=60,
                    )
                    my_set_eval_request(
                        api=API,
                        eval_request=eval_request,
                        set_to_status=RUNNING_STATUS,
                        hf_repo=QUEUE_REPO,
                        local_dir=EVAL_REQUESTS_PATH_BACKEND,
                    )

                    results = process_evaluation(task, eval_request)

                    my_snapshot_download(
                        repo_id=QUEUE_REPO,
                        revision="main",
                        local_dir=EVAL_REQUESTS_PATH_BACKEND,
                        repo_type="dataset",
                        max_workers=60,
                    )
                    my_set_eval_request(
                        api=API,
                        eval_request=eval_request,
                        set_to_status=FINISHED_STATUS,
                        hf_repo=QUEUE_REPO,
                        local_dir=EVAL_REQUESTS_PATH_BACKEND,
                    )

                    return True

    return False

def process_pending_requests() -> bool:
    sanity_checks()
    print("Processing pending requests")
    current_pending_status = [PENDING_STATUS]

    # Get all eval request that are PENDING, if you want to run other evals, change this parameter
    eval_requests = get_eval_requests(
        job_status=current_pending_status, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND
    )
    # Sort the evals by priority (first submitted, first run)
    eval_requests = sort_models_by_priority(api=API, models=eval_requests)

    random.shuffle(eval_requests)

    print(f"Found {len(eval_requests)} {','.join(current_pending_status)} eval requests")

    if len(eval_requests) == 0:
        return False

    eval_request = eval_requests[0]
    pp.pprint(eval_request)
    
    gpu_type = eval_request.gpu_type
    curr_gpu_type = get_gpu_details()
    if gpu_type != curr_gpu_type:
        print(f"GPU type mismatch: {gpu_type} vs {curr_gpu_type}")
        return False

    my_snapshot_download(
        repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60
    )
    my_set_eval_request(
        api=API,
        eval_request=eval_request,
        set_to_status=RUNNING_STATUS,
        hf_repo=QUEUE_REPO,
        local_dir=EVAL_REQUESTS_PATH_BACKEND,
    )

    task_lst = TASKS_HARNESS.copy()
    random.shuffle(task_lst)

    for task in task_lst:
        results = process_evaluation(task, eval_request)

    my_snapshot_download(
        repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60
    )
    my_set_eval_request(
        api=API,
        eval_request=eval_request,
        set_to_status=FINISHED_STATUS,
        hf_repo=QUEUE_REPO,
        local_dir=EVAL_REQUESTS_PATH_BACKEND,
    )

    return True


def get_args():
    parser = argparse.ArgumentParser(description="Run the backend")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    # debug parameters
    parser.add_argument("--task", type=str, default="selfcheckgpt,mmlu, gsm8k", help="Task to debug")
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1,mistralai/Mixtral-8x7B-v0.1", help="Model to debug")
    parser.add_argument("--precision", type=str, default="float32,bfloat16,8bit,4bit", help="Precision to debug")
    parser.add_argument("--inference-framework", type=str, default="hf-chat", help="Inference framework to debug")
    parser.add_argument("--limit", type=int, default=None, help="Limit for the number of samples")
    parser.add_argument("--gpu-type", type=str, default="NVIDIA-A100-PCIe-80GB", 
                        help="GPU type. NVIDIA-A100-PCIe-80GB; NVIDIA-RTX-A5000-24GB; NVIDIA-H100-PCIe-80GB")
    parser.add_argument("--debug_repo", action="store_true", help="Use debug repo")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    local_debug = args.debug
    # debug specific task by ping
    if local_debug and not args.debug_repo:
        # debug_model_names = [args.model]  # Use model from arguments
        # debug_task_name = [args.task]  # Use task from arguments
        debug_model_names = args.model.split(",")
        debug_task_name = args.task.split(",")
        precisions = args.precision.split(",")
        print(f"debug_model_names: {debug_model_names}, debug_task_name: {debug_task_name}, precisions: {precisions}")
        task_lst = TASKS_HARNESS.copy()
        RESULTS_REPO = DEBUG_RESULTS_REPO
        for precision in precisions:
            for debug_model_name in debug_model_names:
                for task in task_lst:
                    task_name = task.benchmark
                    if task_name not in debug_task_name:
                        continue
                    # try:
                    eval_request = EvalRequest(
                        model=debug_model_name,
                        private=False,
                        status="",
                        json_filepath="",
                        precision=precision,  # Use precision from arguments
                        inference_framework=args.inference_framework,  # Use inference framework from arguments
                        gpu_type=args.gpu_type,
                        batch_size=args.batch_size,
                    )
                    curr_gpu_type = get_gpu_details()
                    if eval_request.gpu_type != curr_gpu_type:
                        print(f"GPU type mismatch: {eval_request.gpu_type} vs {curr_gpu_type}")
                        raise Exception("GPU type mismatch")
                    results = process_evaluation(task, eval_request, limit=args.limit)
                    # except Exception as e:
                    #     print(f"debug running error: {e}")
    elif local_debug and args.debug_repo:
        QUEUE_REPO = DEBUG_QUEUE_REPO
        RESULTS_REPO = DEBUG_RESULTS_REPO
        while True:
            res = False
            # if random.randint(0, 10) == 0:
            res = process_pending_requests()
            print(f"waiting for 60 seconds")
            time.sleep(60)
            # if res is False:
            #     if random.randint(0, 5) == 0:
            #         res = maybe_refresh_results(100)
            #     else:
            #         res = process_finished_requests(100)
            # time.sleep(60)
            # if res is False:
            #     if random.randint(0, 5) == 0:
            #         res = maybe_refresh_results(0)
            #     else:
            #         res = process_finished_requests(0)
    elif not local_debug and not args.debug_repo:
        while True:
           res = False
           # if random.randint(0, 10) == 0:
           res = process_pending_requests()
           print(f"waiting for 60 seconds")
           time.sleep(60)
           # if res is False:
           #     if random.randint(0, 5) == 0:
           #         res = maybe_refresh_results(100)
           #     else:
           #         res = process_finished_requests(100)
           # time.sleep(60)
           # if res is False:
           #     if random.randint(0, 5) == 0:
           #         res = maybe_refresh_results(0)
           #     else:
           #         res = process_finished_requests(0)
    else:
        raise Exception("Cannot use debug_repo without local debug flag")