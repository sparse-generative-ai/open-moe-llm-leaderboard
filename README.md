---
title: OPEN-MOE-LLM-LEADERBOARD
emoji: 🔥
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: true
license: apache-2.0
fullWidth: true
tags:
  - leaderboard
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Contributing to Open-MOE-LLM-Leaderboard

Thank you for your interest in contributing to the Open-MOE-LLM-Leaderboard project! We welcome contributions from everyone. Below you'll find guidance on how to set up your development environment, understand our architecture, and contribute effectively. If you have any questions or wish to discuss your contributions, please reach out to Yao Fu via email at [Y.Fu@ed.ac.uk](mailto:y.fu@ed.ac.uk).

## What We're Looking For in Contributions

We are looking for contributions in several key areas to enhance the Open-MOE-LLM-Leaderboard project:

1. **General Bug Fixes/Reports**: We welcome reports of any bugs found in the frontend interface or backend, as well as fixes for these issues.

2. **Adding New Tasks (Benchmark Datasets)**: If you have ideas for new benchmark datasets that could be added, your contributions would be greatly appreciated.

3. **Supporting New Inference Frameworks**: Expanding our project to support new inference frameworks is crucial for our growth. If you can contribute in this area, please reach out.

4. **Testing More Models**: To make our leaderboard as comprehensive as possible, we need to test a wide range of models. Contributions in this area are highly valuable.

Documentation is currently of lower priority, but if you have thoughts or suggestions, please feel free to raise them.

Your contributions are crucial to the success and improvement of the Open-MOE-LLM-Leaderboard project. We look forward to collaborating with you.


## Development Setup

To start contributing, set up your development environment as follows:

```bash
conda create -n leaderboard python=3.10
conda activate leaderboard
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ moe-infinity
pip install pydantic==2.6.4 # Resolves a dependency conflict with moe-infinity
python -m spacy download en # Required for selfcheckgpt
```

## Architecture Overview

The Open-MOE-LLM-Leaderboard project uses the following architecture:

- **User Interface (Gradio)** ->upload-> **HuggingFace Dataset (Request)** ->download-> **Backend GPU Server** ->upload-> **HuggingFace Dataset (Result)** ->download-> **User Interface (Gradio)**

In brief:
1. Users submit model benchmarking requests through the Gradio interface ([app.py](./app.py)). These requests are then recorded in a HuggingFace dataset ([sparse-generative-ai/requests](https://huggingface.co/datasets/sparse-generative-ai/requests)).
2. The backend ([backend-cli.py](./backend-cli.py)), running on a GPU server, processes these requests, performs the benchmarking tasks, and uploads the results to another HuggingFace dataset ([sparse-generative-ai/results](https://huggingface.co/datasets/sparse-generative-ai/results)).
3. Finally, the Gradio interface retrieves and displays these results to the users.

## Running the Gradio Interface

To launch the Gradio interface, execute:

```bash
python app.py
```

Then, open your browser and navigate to http://127.0.0.1:7860.

## Running the Backend

To start the backend process, use:

```bash
python backend-cli.py --debug
```

For additional details, please consult the [backend-cli.py](./backend-cli.py) script.

---

We look forward to your contributions and are here to help guide you through the process. Thank you for supporting the Open-MOE-LLM-Leaderboard project!