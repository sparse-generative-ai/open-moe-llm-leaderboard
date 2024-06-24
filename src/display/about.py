from src.display.utils import ModelType

TITLE = """<h1 align="center" id="space-title">OPEN-MOE-LLM-LEADERBOARD</h1>"""

INTRODUCTION_TEXT = """
The OPEN-MOE-LLM-LEADERBOARD is specifically designed to assess the performance and efficiency of various Mixture of Experts (MoE) Large Language Models (LLMs). 
This initiative, driven by the open-source community, aims to comprehensively evaluate these advanced MoE LLMs. 

The OPEN-MOE-LLM-LEADERBOARD includes generation and multiple choice tasks to measure the performance and efficiency of MOE LLMs.


Tasks:
- **Multiple Choice Performance** -- [MMLU](https://arxiv.org/abs/2009.03300)
- **Mathematics Problem-Solving Performance** -- [GSM8K](https://arxiv.org/abs/2110.14168)
- **AI Judgment Scores for Responses to Complex User Queries** -- [Arena_Hard](https://lmsys.org/blog/2024-04-19-arena-hard/)

Columns and Metrics:
- Method: The MOE LLMs inference framework.
- E2E(s): Average End to End generation time in seconds.
- PRE(s): Prefilling Time of input prompt in seconds.
- T/s: Tokens throughout per second.
- S-MBU(%): Sparse Model Bandwidth Utilization.
- S-MFU(%): Sparse Model FLOPs Utilization.
- Precision: The precison of used model.

"""

ACKNOWLEDGEMENT_TEXT = """
<div>
    <h4>Acknowledgements</h4>
    {image_html}
    <p>We express our sincere gratitude to <a href="https://netmind.ai/home">NetMind.AI</a> for their generous donation of GPUs, which plays a crucial role in ensuring the continuous operation of our Leaderboard.</p>
</div>
"""

LLM_BENCHMARKS_TEXT = f"""

"""
LLM_BENCHMARKS_DETAILS = f"""

"""

FAQ_TEXT = """
---------------------------
# FAQ
## 1) Submitting a model
XXX
## 2) Model results
XXX
## 3) Editing a submission
XXX
"""

EVALUATION_QUEUE_TEXT = """

"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""

"""
