import os
from typing import Union, List

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance

# from lm_eval.api.registry import register_task
from lm_eval.api.metrics import mean

from src.backend.tasks.measurement_task_utils import measure_system_metrics
import json

from typing import (
    List,
    Union,
)

from datasets import Dataset
from src.backend.tasks.eval_metrics import eval_answer


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions

def download_wrapper(func):
    def download(self, *args, **kwargs):
        print("Using MATH, No need to download")
    return download

def load_jsonl(question_file: str):
    """Load questions from a file."""
    questions = []
    model_answers = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line)["problem"])
                model_answers.append(json.loads(line)["solution"])
    return questions, model_answers


original_download = ConfigurableTask.download
ConfigurableTask.download = download_wrapper(original_download)
# @register_task("selfcheckgpt")
@measure_system_metrics
class MATH(ConfigurableTask):
    VERSION = 0.0
    OUTPUT_TYPE = "generate_until"
    data_path = os.path.join(os.path.dirname(__file__), 'competition_math_test.jsonl')

    questions, model_answers = load_jsonl(data_path)

    def __init__(self):
        super().__init__(config={"metadata": {"version": self.VERSION}})
        # these end tokens are hard coded because of the current limitaion of the llm-eval.
        # self.generation_kwargs = {"until": ["\n\n", "<unk>", "<|im_end|>", "</s>", "<|endoftext|>"], "max_length": 512}
        self.generation_kwargs = {"until": ["</s>", "<|im_end|>"], "max_gen_toks": 512, "temperature": 0.0}
        # self.generation_kwargs_sampling_number = 5  # the number of sampling for self-consistence
        # self.generation_kwargs_sampling = {
        #     "temperature": 0.99,
        #     "do_sample": True,
        #     "until": ["<im_end>", "<im_end>"],
        #     "max_length": 1024,
        # }

    def transform_data(self, questions, answers):
        transformed_data = []
        for i, q in enumerate(questions):
            transformed_item = {
                "question_id": i,
                "content": q,  # Assuming you want the first turn's content
                "model_answer": self.remove_boxed(self.last_boxed_only_string(answers[i]))
            }
            transformed_data.append(transformed_item)
        return transformed_data
    
    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        self.dataset = self.transform_data(self.questions, self.model_answers)
        self.dataset = Dataset.from_dict({"question_id": [item["question_id"] for item in self.dataset],
                             "content": [item["content"] for item in self.dataset],
                             "model_answer": [item["model_answer"] for item in self.dataset]})
        return self.dataset

    def doc_to_text(self, doc):
        instruction = (
            "Please output only the final answer in the format \\boxed{answer}. "
            "Do not include any explanation or reasoning. "
            "For example, if the answer is 42, write \\boxed{42}. "
            "If the answer is 2, write \\boxed{2} — not '\\boxed{Two}' or '\\boxed{two}'. "
            "If the answer is a fraction like one-half, write it as \\boxed{\\frac{1}{2}}. "
            "If the answer is a decimal like 0.5, write it as \\boxed{\\frac{1}{2}}. Do not use decimal answers."
        )
        return f"{doc} {instruction}"



    def doc_to_target(self, doc):
        q_id = doc["question_id"]
        return q_id

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> Union[List[Instance], Instance]:
        arguments = (ctx, self.generation_kwargs)
        request_list = [
            Instance(request_type="generate_until", doc=doc, arguments=arguments, idx=0, **kwargs),
        ]
        # sampling_arguments = (ctx, self.generation_kwargs_sampling)
        # request_list.extend(
        #     [
        #         Instance(request_type="generate_until", doc=doc, arguments=sampling_arguments, idx=idx, **kwargs)
        #         for idx in range(1, self.generation_kwargs_sampling_number + 1)
        #     ]
        # )
        return request_list

    def process_results(self, doc, results):
        response_temperature_0 = results[0]
        response = self.remove_boxed(self.last_boxed_only_string(response_temperature_0))
        answer = doc["model_answer"]
        em, f1, _, _ = eval_answer(
            prediction=response,
            gold=answer
        )
        return {
            "em": em,
            "f1": f1
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {k: mean for k in ["em", "f1"]}
        

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {k: True for k in ["em", "f1"]}
    
    # from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
    def remove_boxed(self, s):
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[: len(left)] == left
            return s[len(left) :]

        left = "\\boxed{"

        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]


    def last_boxed_only_string(self, string):
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

