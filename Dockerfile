# Use specific python image
FROM registry.hf.space/sparse-generative-ai-open-moe-llm-leaderboard:latest

RUN pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ moe-infinity --no-cache-dir
# To fix pydantic version
RUN pip install pydantic==2.6.4 --no-cache-dir
# To fix selfcheck (selfchatgpt) dataset missing
RUN python -m spacy download en