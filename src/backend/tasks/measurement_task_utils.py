import functools
from lm_eval.api.metrics import mean


def process_results_decorator(func):
    # This decorator processes the results of a task before passing them to the original process_results function
    @functools.wraps(func)
    def wrapper(self, doc, results, *args, **kwargs):
        # We process the results here
        processed_results = [r[0] for r in results]
        
        end_to_end_time = sum([r[1] for r in results]) / len(results)
        prefilling_time = sum([r[2] for r in results]) / len(results)
        decoding_throughput = sum([r[3] for r in results]) / len(results)
        decoding_mfu = sum([r[4] for r in results]) / len(results)
        decoding_mbu = sum([r[5] for r in results]) / len(results)
        prefill_throughput = sum([r[6] for r in results]) / len(results)
        prefill_mfu = sum([r[7] for r in results]) / len(results)
        prefill_mbu = sum([r[8] for r in results]) / len(results)

        
        # print(f"end_to_end_time: {end_to_end_time}, prefilling_time: {prefilling_time}, decoding_throughput: {decoding_throughput}")

        # Now call the original process_results with the processed results
        result_dict = func(self, doc, processed_results, *args, **kwargs)
        result_dict["end_to_end_time"] = end_to_end_time
        result_dict["prefilling_time"] = prefilling_time
        result_dict["decoding_throughput"] = decoding_throughput
        result_dict["decoding_mfu"] = decoding_mfu
        result_dict["decoding_mbu"] = decoding_mbu
        result_dict["prefill_throughput"] = prefill_throughput
        result_dict["prefill_mfu"] = prefill_mfu
        result_dict["prefill_mbu"] = prefill_mbu
        return result_dict
    return wrapper


def aggregation_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        aggregation_list = func(self, *args, **kwargs)
        aggregation_list["end_to_end_time"] = mean
        aggregation_list["prefilling_time"] = mean
        aggregation_list["decoding_throughput"] = mean
        aggregation_list["decoding_mfu"] = mean
        aggregation_list["decoding_mbu"] = mean
        aggregation_list["prefill_throughput"] = mean
        aggregation_list["prefill_mfu"] = mean
        aggregation_list["prefill_mbu"] = mean
        return aggregation_list
    return wrapper


def higher_is_better_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        higher_is_better_dict = func(self, *args, **kwargs)
        higher_is_better_dict["end_to_end_time"] = False
        higher_is_better_dict["prefilling_time"] = False
        higher_is_better_dict["decoding_throughput"] = True
        higher_is_better_dict["decoding_mfu"] = True
        higher_is_better_dict["decoding_mbu"] = True
        higher_is_better_dict["prefill_throughput"] = True
        higher_is_better_dict["prefill_mfu"] = True
        higher_is_better_dict["prefill_mbu"] = True
        return higher_is_better_dict
    return wrapper


def measure_system_metrics(cls):
    method_decorators = {
        'process_results': [process_results_decorator],
        'aggregation': [aggregation_decorator],
        'higher_is_better': [higher_is_better_decorator],
    }
    for method_name, decorators in method_decorators.items():
        if callable(getattr(cls, method_name, None)):
            original_method = getattr(cls, method_name)
            for decorator in reversed(decorators):
                original_method = decorator(original_method)
            setattr(cls, method_name, original_method)
    return cls
