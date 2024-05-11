import os

from huggingface_hub import HfApi

# clone / pull the lmeh eval data
H4_TOKEN = os.environ.get("H4_TOKEN", None)

# REPO_ID = "pminervini/sparse-generative-ai"
REPO_ID = "sparse-generative-ai/open-moe-llm-leaderboard"

QUEUE_REPO = "sparse-generative-ai/requests"
QUEUE_REPO_OPEN_LLM = "open-llm-leaderboard/requests"
RESULTS_REPO = "sparse-generative-ai/results"

DEBUG_QUEUE_REPO = "sparse-generative-ai/debug_requests"
DEBUG_RESULTS_REPO = "sparse-generative-ai/debug_results"

IS_PUBLIC = bool(os.environ.get("IS_PUBLIC", True))

CACHE_PATH = os.getenv("HF_HOME", ".")

EVAL_REQUESTS_PATH = os.path.join(CACHE_PATH, "eval-queue")
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval-results")
EVAL_REQUESTS_PATH_OPEN_LLM = os.path.join(CACHE_PATH, "eval-queue-open-llm")

EVAL_REQUESTS_PATH_PRIVATE = "eval-queue-private"
EVAL_RESULTS_PATH_PRIVATE = "eval-results-private"

PATH_TO_COLLECTION = "sparse-generative-ai/llm-leaderboard-best-models-652d6c7965a4619fb5c27a03"

# Rate limit variables
RATE_LIMIT_PERIOD = 7
RATE_LIMIT_QUOTA = 5
HAS_HIGHER_RATE_LIMIT = ["TheBloke"]

API = HfApi(token=H4_TOKEN)
