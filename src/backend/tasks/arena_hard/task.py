import os
from typing import Union, List

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance

# from lm_eval.api.registry import register_task
from lm_eval.api.metrics import mean

from src.backend.envs import DEVICE

import pandas as pd

from src.backend.tasks.measurement_task_utils import measure_system_metrics
import json

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from datasets import Dataset
import re

from src.backend.tasks.arena_hard.arena_utils import (
    load_questions,
    load_questions,
    load_model_answers,
    make_config,
)

from src.backend.tasks.arena_hard.arena_judgment import (
    judgment,
    get_battles_from_scores,
    compute_mle_elo,
    predict_win_rate,
    get_win_rate_column
)

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
        print("Using Arena Hard, No need to download")
    return download

original_download = ConfigurableTask.download
ConfigurableTask.download = download_wrapper(original_download)
# @register_task("selfcheckgpt")
@measure_system_metrics
class ArenaHard(ConfigurableTask):
    VERSION = 0.0
    OUTPUT_TYPE = "generate_until"
    data_path = os.path.join(os.path.dirname(__file__), 'question.jsonl')
    judge_config_path = os.path.join(os.path.dirname(__file__), "configs/judge_config.yaml")
    configs = make_config(judge_config_path)
    model_ans_dir = os.path.join(os.path.dirname(__file__), "model_answer")
    model_answers = load_model_answers(model_ans_dir)
    data = load_questions(data_path)

    def __init__(self):
        super().__init__(config={"metadata": {"version": self.VERSION}})
        # these end tokens are hard coded because of the current limitaion of the llm-eval.
        # self.generation_kwargs = {"until": ["\n\n", "<unk>", "<|im_end|>", "</s>", "<|endoftext|>"], "max_length": 512}
        self.generation_kwargs = {"until": ["</s>", "<|im_end|>"], "max_length": 1024}
        # self.generation_kwargs_sampling_number = 5  # the number of sampling for self-consistence
        # self.generation_kwargs_sampling = {
        #     "temperature": 0.99,
        #     "do_sample": True,
        #     "until": ["<im_end>", "<im_end>"],
        #     "max_length": 1024,
        # }

    def transform_data(self, data):
        transformed_data = []
        for i in range(len(data)):
            if self.configs["baseline"]:
                baseline_answer = self.model_answers[self.configs["baseline_model"]][data[i]["question_id"]]
            else:
                baseline_answer = None
            transformed_item = {
                "question_id": data[i]["question_id"],
                "content": data[i]["turns"][0]["content"],  # Assuming you want the first turn's content
                "model_answer": baseline_answer
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
        self.dataset = self.transform_data(self.data)
        self.dataset = Dataset.from_dict({"question_id": [item["question_id"] for item in self.dataset],
                             "content": [item["content"] for item in self.dataset],
                             "model_answer": [item["model_answer"] for item in self.dataset]})
        return self.dataset

    def doc_to_text(self, doc):
        sentence = doc["content"]
        doc_text = f"{sentence}\n"
        return doc_text

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
        # other_responses = results[1:]
        api_config_path = os.path.join(os.path.dirname(__file__), "configs/api_config.yaml")
        endpoint_list = make_config(api_config_path)
        
        if self.configs["regex_pattern"]:
            pattern = re.compile(self.configs["regex_pattern"])

        ref_answer_dir = os.path.join(os.path.dirname(__file__), "reference_answer")
        
        ref_answers = None
        if self.configs["reference"]:
            ref_answers = load_model_answers(ref_answer_dir)
            ref_answers = [ref_answers[model] for model in self.configs["ref_model"]]
        
        # output_files = {}
        # models = ["custom_model"]
        # output_dir = f"{os.path.join(os.path.dirname(__file__))}/model_judgments/{self.configs['judge_model']}"
        # for model in models:
        #     output_files[model] = os.path.join(
        #         output_dir,
        #         f"{model}.jsonl",
        #     )

        # for output_file in output_files.values():
        #     os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        endpoint_info = endpoint_list[self.configs["judge_model"]]
        
        question = doc
        kwargs = {}
        kwargs["question"] = question
        kwargs["answer"] = response_temperature_0
        if ref_answers:
            kwargs["reference"] = [ref_answer[doc["question_id"]] for ref_answer in ref_answers]
            assert len(kwargs["reference"]) == len(self.configs["ref_model"])
        else:
            kwargs["reference"] = None
        
        if self.configs["baseline"]:
            kwargs["baseline_answer"] = doc["model_answer"]
        else:
            kwargs["baseline_answer"] = None
        kwargs["configs"] = self.configs
        kwargs["endpoint_dict"] = endpoint_info
        # kwargs["output_file"] = output_files["custom_model"]
        kwargs["regex_pattern"] = pattern
    
        scores = judgment(**kwargs)
        return {"score": scores}           

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        ##TODO implement the aggregation function to calculate elo for score
        def get_win_rate(score_list):
            battles = get_battles_from_scores(score_list)
            bootstrap_online_elo = compute_mle_elo(battles)
            stats = pd.DataFrame()
            stats["results"] = None
            stats["results"] = stats['results'].astype('object')
            for i, model in enumerate(bootstrap_online_elo.index):
                stats.at[i, "model"] = model
                stats.at[i, "score"] = bootstrap_online_elo[model]

            stats.sort_values(by="model", inplace=True)
            stats["score"] = get_win_rate_column(stats, "score", "gpt-4-0314").tolist()
            
            return stats["score"][1]
            
        return {k: get_win_rate for k in ["score"]}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {k: True for k in ["score"]}
