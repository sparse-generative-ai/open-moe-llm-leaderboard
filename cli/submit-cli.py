#!/usr/bin/env python

import json
import os
import time

from datetime import datetime, timezone

from src.envs import API, EVAL_REQUESTS_PATH, H4_TOKEN, QUEUE_REPO
from src.submission.check_validity import already_submitted_models, get_model_size, is_model_on_hub

from huggingface_hub import snapshot_download
from src.backend.envs import EVAL_REQUESTS_PATH_BACKEND
from src.backend.manage_requests import get_eval_requests
from src.backend.manage_requests import EvalRequest


def add_new_eval(
    model: str, base_model: str, revision: str, precision: str, private: bool, weight_type: str, model_type: str
):
    REQUESTED_MODELS, USERS_TO_SUBMISSION_DATES = already_submitted_models(EVAL_REQUESTS_PATH)

    user_name = ""
    model_path = model
    if "/" in model:
        tokens = model.split("/")
        user_name = tokens[0]
        model_path = tokens[1]

    precision = precision.split(" ")[0]
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if model_type is None or model_type == "":
        return print("Please select a model type.")

    # Does the model actually exist?
    if revision == "":
        revision = "main"

    # Is the model on the hub?
    if weight_type in ["Delta", "Adapter"]:
        base_model_on_hub, error, _ = is_model_on_hub(
            model_name=base_model, revision=revision, token=H4_TOKEN, test_tokenizer=True
        )
        if not base_model_on_hub:
            print(f'Base model "{base_model}" {error}')
            return

    if not weight_type == "Adapter":
        model_on_hub, error, _ = is_model_on_hub(model_name=model, revision=revision, test_tokenizer=True)
        if not model_on_hub:
            print(f'Model "{model}" {error}')
            return

    # Is the model info correctly filled?
    try:
        model_info = API.model_info(repo_id=model, revision=revision)
    except Exception:
        print("Could not get your model information. Please fill it up properly.")
        return

    model_size = get_model_size(model_info=model_info, precision=precision)

    license = "none"
    try:
        license = model_info.cardData["license"]
    except Exception:
        print("Please select a license for your model")
        # return

    # modelcard_OK, error_msg = check_model_card(model)
    # if not modelcard_OK:
    #     print(error_msg)
    #     return

    # Seems good, creating the eval
    print("Adding new eval")

    eval_entry = {
        "model": model,
        "base_model": base_model,
        "revision": revision,
        "private": private,
        "precision": precision,
        "weight_type": weight_type,
        "status": "PENDING",
        "submitted_time": current_time,
        "model_type": model_type,
        "likes": model_info.likes,
        "params": model_size,
        "license": license,
    }

    # Check for duplicate submission
    if f"{model}_{revision}_{precision}" in REQUESTED_MODELS:
        print("This model has been already submitted.")
        return

    print("Creating eval file")
    OUT_DIR = f"{EVAL_REQUESTS_PATH}/{user_name}"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{model_path}_eval_request_{private}_{precision}_{weight_type}.json"

    with open(out_path, "w") as f:
        f.write(json.dumps(eval_entry))

    print("Uploading eval file")
    API.upload_file(
        path_or_fileobj=out_path,
        path_in_repo=out_path.split("eval-queue/")[1],
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Add {model} to eval queue",
    )

    # Remove the local file
    os.remove(out_path)

    print(
        "Your request has been submitted to the evaluation queue!\nPlease wait for up to an hour for the model to show in the PENDING list."
    )
    return


def main():
    from huggingface_hub import HfApi

    api = HfApi()
    model_lst = api.list_models()

    model_lst = [m for m in model_lst]

    def custom_filter(m) -> bool:
        # res = m.pipeline_tag in {'text-generation'} and 'en' in m.tags and m.private is False
        # res = m.pipeline_tag in {'text-generation'} and 'en' in m.tags and m.private is False and 'mistralai/' in m.id
        res = "mistralai/" in m.id
        return res

    filtered_model_lst = sorted([m for m in model_lst if custom_filter(m)], key=lambda m: m.downloads, reverse=True)

    snapshot_download(
        repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH_BACKEND, repo_type="dataset", max_workers=60
    )

    PENDING_STATUS = "PENDING"
    RUNNING_STATUS = "RUNNING"
    FINISHED_STATUS = "FINISHED"
    FAILED_STATUS = "FAILED"

    status = [PENDING_STATUS, RUNNING_STATUS, FINISHED_STATUS, FAILED_STATUS]

    # Get all eval requests
    eval_requests: list[EvalRequest] = get_eval_requests(
        job_status=status, hf_repo=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH_BACKEND
    )

    requested_model_names = {e.model for e in eval_requests}

    # breakpoint()

    for i in range(min(200, len(filtered_model_lst))):
        model = filtered_model_lst[i]

        print(f"Considering {model.id} ..")

        is_finetuned = any(tag.startswith("base_model:") for tag in model.tags)

        model_type = "pretrained"
        if is_finetuned:
            model_type = "fine-tuned"

        is_instruction_tuned = "nstruct" in model.id
        if is_instruction_tuned:
            model_type = "instruction-tuned"

        if model.id not in requested_model_names:

            if "mage" not in model.id:
                add_new_eval(
                    model=model.id,
                    base_model="",
                    revision="main",
                    precision="float32",
                    private=False,
                    weight_type="Original",
                    model_type=model_type,
                )
                time.sleep(10)
        else:
            print(f"Model {model.id} already added, not adding it to the queue again.")


if __name__ == "__main__":
    main()
