import os
import json
import glob

from tqdm import tqdm
from huggingface_hub import HfApi, snapshot_download
from src.backend.manage_requests import EvalRequest
from src.backend.envs import EVAL_REQUESTS_PATH_BACKEND_SYNC
from src.envs import QUEUE_REPO, API
from src.envs import EVAL_REQUESTS_PATH_OPEN_LLM, QUEUE_REPO_OPEN_LLM
from src.utils import my_snapshot_download


def my_set_eval_request(api, json_filepath, hf_repo, local_dir):
    for i in range(10):
        try:
            set_eval_request(api=api, json_filepath=json_filepath, hf_repo=hf_repo, local_dir=local_dir)
            return
        except Exception:
            time.sleep(60)
    return


def set_eval_request(api: HfApi, json_filepath: str, hf_repo: str, local_dir: str):
    """Updates a given eval request with its new status on the hub (running, completed, failed, ...)"""

    with open(json_filepath) as fp:
        data = json.load(fp)

    with open(json_filepath, "w") as f:
        f.write(json.dumps(data))

    api.upload_file(
        path_or_fileobj=json_filepath,
        path_in_repo=json_filepath.replace(local_dir, ""),
        repo_id=hf_repo,
        repo_type="dataset",
    )


def get_request_file_for_model(data, requests_path):
    model_name = data["model"]
    precision = data["precision"]
    """Selects the correct request file for a given model. Only keeps runs tagged as FINISHED and RUNNING"""
    request_files = os.path.join(
        requests_path,
        f"{model_name}_eval_request_*.json",
    )
    request_files = glob.glob(request_files)

    # Select correct request file (precision)
    request_file = ""
    request_files = sorted(request_files, reverse=True)

    for tmp_request_file in request_files:
        with open(tmp_request_file, "r") as f:
            req_content = json.load(f)
            if req_content["precision"] == precision.split(".")[-1]:
                request_file = tmp_request_file
    return request_file


def update_model_type(data, requests_path):
    open_llm_request_file = get_request_file_for_model(data, requests_path)

    try:
        with open(open_llm_request_file, "r") as f:
            open_llm_request = json.load(f)
            data["model_type"] = open_llm_request["model_type"]
            return True, data
    except:
        return False, data


def read_and_write_json_files(directory, requests_path_open_llm):
    # Walk through the directory
    for subdir, dirs, files in tqdm(os.walk(directory), desc="updating model type according to open llm leaderboard"):
        for file in files:
            # Check if the file is a JSON file
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                # Open and read the JSON file
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                sucess, data = update_model_type(data, requests_path_open_llm)
                if sucess:
                    with open(file_path, "w") as json_file:
                        json.dump(data, json_file)
                    my_set_eval_request(
                        api=API, json_filepath=file_path, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND_SYNC
                    )


if __name__ == "__main__":
    my_snapshot_download(
        repo_id=QUEUE_REPO_OPEN_LLM,
        revision="main",
        local_dir=EVAL_REQUESTS_PATH_OPEN_LLM,
        repo_type="dataset",
        max_workers=60,
    )
    my_snapshot_download(
        repo_id=QUEUE_REPO,
        revision="main",
        local_dir=EVAL_REQUESTS_PATH_BACKEND_SYNC,
        repo_type="dataset",
        max_workers=60,
    )
    read_and_write_json_files(EVAL_REQUESTS_PATH_BACKEND_SYNC, EVAL_REQUESTS_PATH_OPEN_LLM)
