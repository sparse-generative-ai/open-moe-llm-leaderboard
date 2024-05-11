#!/usr/bin/env python

from huggingface_hub import snapshot_download

from src.backend.envs import EVAL_REQUESTS_PATH_BACKEND
from src.backend.manage_requests import get_eval_requests
from src.backend.manage_requests import EvalRequest
from src.backend.run_eval_suite import run_evaluation

from src.backend.tasks.xsum.task import XSum
from src.backend.tasks.xsum.task_v2 import XSumv2

from src.backend.tasks.cnndm.task import CNNDM
from src.backend.tasks.cnndm.task_v2 import CNNDMv2

from src.backend.tasks.selfcheckgpt.task import SelfCheckGPT

from lm_eval.tasks import TaskManager
from lm_eval import tasks, evaluator, utils

from src.backend.envs import Tasks, EVAL_REQUESTS_PATH_BACKEND, EVAL_RESULTS_PATH_BACKEND, DEVICE, LIMIT, Task
from src.envs import QUEUE_REPO

from lm_eval.models.huggingface import HFLM


def main():
    # snapshot_download(repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60)

    PENDING_STATUS = "PENDING"
    RUNNING_STATUS = "RUNNING"
    FINISHED_STATUS = "FINISHED"
    FAILED_STATUS = "FAILED"

    status = [PENDING_STATUS, RUNNING_STATUS, FINISHED_STATUS, FAILED_STATUS]

    # Get all eval request that are FINISHED, if you want to run other evals, change this parameter
    eval_requests: list[EvalRequest] = get_eval_requests(
        job_status=status, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND, do_download=False
    )
    # eval_request = [r for r in eval_requests if 'bloom-560m' in r.model][0]
    eval_request = [r for r in eval_requests if "meta-llama/Llama-2-7b-hf" in r.model][0]

    # my_task = Task("memo-trap", "acc", "memo-trap", 0)
    # my_task = Task("selfcheckgpt", "avg-selfcheckgpt", "SGPT", 2)
    # my_task = Task("ifeval", "prompt_level_strict_acc", "IFEval", 0)
    # my_task = Task("truefalse_cieacf", "acc", "TrueFalse", 5)
    # my_task = Task("faithdial_hallu", "acc", "FaithDIAL", 2)

    # my_task = Task("nq_swap", "exact_match", "NQ-Swap", 2)
    # my_task = Task("memo-trap_v2", "acc", "XXX", 2)
    my_task = Task("xsum_v2", "rougeL", "XXX", 0)
    # my_task = Task("squadv2", "exact", "XXX", 0)
    # my_task = Task("scrolls_qasper", "f1", "XXX", 0)

    eval_logger = utils.eval_logger
    import logging

    eval_logger.setLevel(getattr(logging, "DEBUG"))

    TASKS_HARNESS = [my_task]
    # task_names = ['triviaqa']
    # TASKS_HARNESS = [task.value for task in Tasks]

    # include_task_folder("src/backend/tasks/")
    task_manager = TaskManager(include_path="./src/backend/tasks/")
    # task_manager.initialize_tasks(include_path="src/backend/tasks/")

    # breakpoint()

    print(task_manager.all_tasks)

    for task in TASKS_HARNESS:
        print(f"Selected Tasks: [{task}]")
        import torch

        # breakpoint()
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=eval_request.get_model_args(),
            tasks=[task.benchmark],
            num_fewshot=task.num_fewshot,
            batch_size=1,
            device="mps",
            use_cache=None,
            limit=2,
            write_out=True,
            task_manager=task_manager,
        )
        print("AAA", results["results"])

        breakpoint()


if __name__ == "__main__":
    main()
