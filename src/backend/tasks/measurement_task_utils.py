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
        # print(f"end_to_end_time: {end_to_end_time}, prefilling_time: {prefilling_time}, decoding_throughput: {decoding_throughput}")

        # Now call the original process_results with the processed results
        result_dict = func(self, doc, processed_results, *args, **kwargs)
        result_dict["end_to_end_time"] = end_to_end_time
        result_dict["prefilling_time"] = prefilling_time
        result_dict["decoding_throughput"] = decoding_throughput
        return result_dict
    return wrapper


def aggregation_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        aggregation_list = func(self, *args, **kwargs)
        aggregation_list["end_to_end_time"] = mean
        aggregation_list["prefilling_time"] = mean
        aggregation_list["decoding_throughput"] = mean
        return aggregation_list
    return wrapper


def higher_is_better_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        higher_is_better_dict = func(self, *args, **kwargs)
        higher_is_better_dict["end_to_end_time"] = False
        higher_is_better_dict["prefilling_time"] = False
        higher_is_better_dict["decoding_throughput"] = True
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
