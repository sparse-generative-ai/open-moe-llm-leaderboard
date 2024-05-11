import os
from typing import Union, List

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance

# from lm_eval.api.registry import register_task
from lm_eval.api.metrics import mean

from src.backend.envs import DEVICE

import spacy
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckNLI, SelfCheckBERTScore, SelfCheckNgram

from src.backend.tasks.measurement_task_utils import measure_system_metrics


# @register_task("selfcheckgpt")
@measure_system_metrics
class SelfCheckGPT(ConfigurableTask):
    VERSION = 0.0
    DATASET_PATH = "potsawee/wiki_bio_gpt3_hallucination"
    DATASET_NAME = None
    OUTPUT_TYPE = "generate_until"

    def __init__(self):
        super().__init__(config={"metadata": {"version": self.VERSION}})
        # these end tokens are hard coded because of the current limitaion of the llm-eval.
        # self.generation_kwargs = {"until": ["\n\n", "<unk>", "<|im_end|>", "</s>", "<|endoftext|>"], "max_length": 512}
        self.generation_kwargs = {"until": ["<im_end>"], "max_length": 1024}
        self.generation_kwargs_sampling_number = 5  # the number of sampling for self-consistence
        self.generation_kwargs_sampling = {
            "temperature": 0.99,
            "do_sample": True,
            "until": ["<im_end>", "</s>"],
            "max_length": 1024,
        }

        self.selfcheckgpt_type = os.environ.get("SELFCHECKGPTTYPE", "SelfCheckNLI")
        self.selfcheckgpt_device = os.environ.get("SELFCHECKGPTDEVICE", DEVICE)
        self.selfcheckgpt_nlp = spacy.load("en_core_web_sm")

        if self.selfcheckgpt_type == "SelfCheckNgram":
            self.selfcheckgpt = SelfCheckNgram(n=1)
        elif self.selfcheckgpt_type == "SelfCheckBERTScore":
            self.selfcheckgpt = SelfCheckBERTScore(rescale_with_baseline=True)
        elif self.selfcheckgpt_type == "SelfCheckMQAG":
            self.selfcheckgpt = SelfCheckMQAG(device=self.selfcheckgpt_device)
        elif self.selfcheckgpt_type == "SelfCheckNLI":
            self.selfcheckgpt = SelfCheckNLI(device=self.selfcheckgpt_device)
        self.SelfCheckNLI_error_cnt = 0

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["evaluation"]

    def doc_to_text(self, doc):
        if not hasattr(self, "selfcheckgpt_nlp"):
            self.selfcheckgpt_nlp = spacy.load("en_core_web_sm")

        sentences = [x.text.strip() for x in self.selfcheckgpt_nlp(doc["wiki_bio_text"]).sents]
        if len(sentences) < 2:
            raise ValueError("This wikipedia passage is too short for self-consistency check: {sentences}")
            # disscussed with Potsawee

        doc_text = f"Please generate a Wikipedia passage that consists of at least two sentences, starting with the following sentence: {sentences[0]}\n"
        return doc_text

    def doc_to_target(self, doc):
        answer = doc["wiki_bio_text"]
        return answer

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> Union[List[Instance], Instance]:
        arguments = (ctx, self.generation_kwargs)
        request_list = [
            Instance(request_type="generate_until", doc=doc, arguments=arguments, idx=0, **kwargs),
        ]
        sampling_arguments = (ctx, self.generation_kwargs_sampling)
        request_list.extend(
            [
                Instance(request_type="generate_until", doc=doc, arguments=sampling_arguments, idx=idx, **kwargs)
                for idx in range(1, self.generation_kwargs_sampling_number + 1)
            ]
        )
        return request_list

    def process_results(self, doc, results):
        response_temperature_0 = results[0]
        other_responses = results[1:]
        passage = self.doc_to_target(doc)

        sentences = self.selfcheckgpt_nlp(response_temperature_0)
        sentences = [sent.text.strip() for sent in sentences.sents]
        if self.selfcheckgpt_type == "SelfCheckNgram":
            selfcheckgpt_scores = self.selfcheckgpt.predict(
                sentences=sentences, passage=response_temperature_0, sampled_passages=other_responses
            )
            return {
                "avg-selfcheckgpt": selfcheckgpt_scores["doc_level"]["avg_neg_logprob"],
                "max-selfcheckgpt": selfcheckgpt_scores["doc_level"]["avg_max_neg_logprob"],
            }

        elif self.selfcheckgpt_type == "SelfCheckBERTScore":
            selfcheckgpt_scores = self.selfcheckgpt.predict(sentences=sentences, sampled_passages=other_responses)
        elif self.selfcheckgpt_type == "SelfCheckMQAG":
            selfcheckgpt_scores = self.selfcheckgpt.predict(
                sentences=sentences,
                passage=response_temperature_0,
                sampled_passages=other_responses,
                num_questions_per_sent=5,  # number of questions to be drawn
                scoring_method="bayes_with_alpha",  # options = 'counting', 'bayes', 'bayes_with_alpha'
                beta1=0.8,
                beta2=0.8,
            )  # additional params depending on scoring_method
        elif self.selfcheckgpt_type == "SelfCheckNLI":
            selfcheckgpt_scores = self.selfcheckgpt.predict(sentences=sentences, sampled_passages=other_responses)

            if len(selfcheckgpt_scores) < 2:
                # at least two sentences
                self.SelfCheckNLI_error_cnt += 1
                result = {"avg-selfcheckgpt": 0.0, "max-selfcheckgpt": 0.0}

            else:
                threshold = 0.7  # https://huggingface.co/blog/dhuynh95/automatic-hallucination-detection
                # passage is hallucianted if one sentence is hallucinated. It's very strict.
                selfcheckgpt_scores_max = 0.0 if max(selfcheckgpt_scores) > threshold else 1.0
                # passage is hallucianted if average score of all sentences is hallucinated.
                selfcheckgpt_scores_avg = (
                    0.0 if sum(selfcheckgpt_scores) / len(selfcheckgpt_scores) > threshold else 1.0
                )
                result = {"avg-selfcheckgpt": selfcheckgpt_scores_avg, "max-selfcheckgpt": selfcheckgpt_scores_max}

            return result

        selfcheckgpt_scores_avg = (
            sum(selfcheckgpt_scores) / len(selfcheckgpt_scores) if len(selfcheckgpt_scores) > 0 else 0
        )
        selfcheckgpt_scores_max = max(selfcheckgpt_scores)

        return {"avg-selfcheckgpt": selfcheckgpt_scores_avg, "max-selfcheckgpt": selfcheckgpt_scores_max}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {k: mean for k in ["avg-selfcheckgpt", "max-selfcheckgpt"]}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {k: True for k in ["avg-selfcheckgpt", "max-selfcheckgpt"]}
