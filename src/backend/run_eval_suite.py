from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.api.metrics import mean
from lm_eval.api.task import ConfigurableTask

from src.backend.manage_requests import EvalRequest


orig_process_results = ConfigurableTask.process_results
orig_aggregation = ConfigurableTask.aggregation
orig_higher_is_better = ConfigurableTask.higher_is_better

def process_results_decorator(func):
    def wrapper(self, doc, results, *args, **kwargs):
        processed_results = [r[0] for r in results]

        end_to_end_time = sum([r[1] for r in results]) / len(results)
        prefilling_time = sum([r[2] for r in results]) / len(results)
        decoding_throughput = sum([r[3] for r in results]) / len(results)
        mfu = sum([r[4] for r in results]) / len(results)
        mbu = sum([r[5] for r in results]) / len(results)
        # print(f"end_to_end_time: {end_to_end_time}, prefilling_time: {prefilling_time}, decoding_throughput: {decoding_throughput}")

        result_dict = func(self, doc, processed_results, *args, **kwargs)
        result_dict["end_to_end_time"] = end_to_end_time
        result_dict["prefilling_time"] = prefilling_time
        result_dict["decoding_throughput"] = decoding_throughput
        result_dict["mfu"] = mfu
        result_dict["mbu"] = mbu
        return result_dict
    return wrapper
ConfigurableTask.process_results = process_results_decorator(orig_process_results)

def aggregation_decorator(func):
    def wrapper(self, *args, **kwargs):
        aggregation_list = func(self, *args, **kwargs)
        aggregation_list["end_to_end_time"] = mean
        aggregation_list["prefilling_time"] = mean
        aggregation_list["decoding_throughput"] = mean
        aggregation_list["mfu"] = mean
        aggregation_list["mbu"] = mean
        return aggregation_list
    return wrapper
ConfigurableTask.aggregation = aggregation_decorator(orig_aggregation)

def higher_is_better_decorator(func):
    def wrapper(self, *args, **kwargs):
        higher_is_better_dict = func(self, *args, **kwargs)
        higher_is_better_dict["end_to_end_time"] = False
        higher_is_better_dict["prefilling_time"] = False
        higher_is_better_dict["decoding_throughput"] = True
        higher_is_better_dict["mfu"] = True
        higher_is_better_dict["mbu"] = True
        return higher_is_better_dict
    return wrapper
ConfigurableTask.higher_is_better = higher_is_better_decorator(orig_higher_is_better)

# from src.backend.tasks.xsum.task import XSum
# from src.backend.tasks.xsum.task_v2 import XSumv2

# from src.backend.tasks.cnndm.task import CNNDM
# from src.backend.tasks.cnndm.task_v2 import CNNDMv2

from src.backend.tasks.selfcheckgpt.task import SelfCheckGPT

from src.backend.huggingface_generate_until import HFLMwithChatTemplate
from src.backend.moe_infinity import MoEHFLM
from src.backend.vllm import VLLM_MOE

def run_evaluation(
    eval_request: EvalRequest,
    task_names,
    num_fewshot,
    batch_size,
    device,
    use_cache=None,
    limit=None,
    max_nb_samples=100,
) -> dict:
    if limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    # include_task_folder("src/backend/tasks/")
    # initialize_tasks('INFO')

    print(f"Allocating task manager for: {task_names}")

    task_manager = TaskManager(include_path="./src/backend/tasks/")
    # task_manager.initialize_tasks('INFO')

    print(f"Considered Tasks: {task_names}")
    # print(f"Allowed Tasks: {tasks.ALL_TASKS}")

    # task_names = utils.pattern_match(task_names, tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")
    print(f"Eval Request: {eval_request}")
    print(
        f"Num Fewshot: {num_fewshot}, Batch Size: {batch_size}, Device: {device}, Use Cache: {use_cache}, Limit: {limit}"
    )
    # hf-chat is implemented to use apply_chat_template
    results = evaluator.simple_evaluate(
        model=eval_request.inference_framework,  # "hf-chat", "moe-infinity"
        model_args=eval_request.get_model_args(),
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=8,
        device=device,
        use_cache=use_cache,
        limit=limit,
        write_out=True,
        task_manager=task_manager,
        verbosity="WARNING",
    )

    results["config"]["model_dtype"] = eval_request.precision
    results["config"]["model_name"] = eval_request.model
    results["config"]["model_sha"] = eval_request.revision
    results["config"]["inference_framework"] = eval_request.inference_framework

    if max_nb_samples is not None:
        if "samples" in results:
            samples = results["samples"]
            for task_name in samples.keys():
                if len(samples[task_name]) > max_nb_samples:
                    results["samples"][task_name] = results["samples"][task_name][:max_nb_samples]

    # print(evaluator.make_table(results))

    return results
