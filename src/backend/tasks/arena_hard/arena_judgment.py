'''
This file is part of Open-MoE-LLM-Leaderboard and is modified based on work
under the Apache 2.0 License from the arena-hard project.
(https://github.com/lm-sys/arena-hard)
Original Copyright (c) 2024 Tianle Li*, Wei-Lin Chiang*, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, Ion Stoica
See the NOTICE file distributed with this work for additional
information regarding copyright ownership.
'''

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
import math
from collections import defaultdict
from tqdm import tqdm

from src.backend.tasks.arena_hard.arena_utils import (
    chat_completion_openai,
    chat_completion_openai_azure,
    chat_completion_anthropic,
    load_questions,
    load_model_answers,
    get_endpoint,
    make_config,
)


def get_score(judgment, pattern, pairwise=True):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False


# get answer from model
def get_answer(model, conv, temperature, max_tokens, endpoint_dict=None):
    api_dict = get_endpoint(endpoint_dict["endpoints"])

    if endpoint_dict["api_type"] == "anthropic":
        output = chat_completion_anthropic(model, conv, temperature, max_tokens)
    elif endpoint_dict["api_type"] == "azure":
        output = chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict)
    else:
        output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict)
    return output


def judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]
    # output_file = args["output_file"]
    model = configs["judge_model"]

    num_games = 2 if configs["pairwise"] else 1

    # output = {
    #     "question_id":question["question_id"],
    #     "judge": model,
    #     "model": "custom_model",
    #     "games":[]
    #     }
    output = [question["question_id"]]

    for game in range(num_games):
        conv = [{"role": "system", "content": configs["system_prompt"]}]

        for template in configs["prompt_template"]:
            prompt_args = {}

            prompt_args[f"question_{1}"] = question["content"]
            base = 1

            if baseline:
                if game % 2 == 1: # swap position
                    temp = baseline
                    baseline = answer
                    answer = temp
                
                if game == 0:
                    for i, turn in enumerate(baseline["choices"][0]["turns"]):
                        prompt_args[f"answer_{i+1}"] = turn["content"]
                        base += 1
                
                if game == 1:
                    prompt_args[f"answer_{1}"] = baseline
                    base += 1
            
            if answer:
                prompt_args[f"answer_{base}"] = answer

            if reference:
                for j, ref_answer in enumerate(reference):
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]
            
            user_prompt = template.format(**prompt_args)
            conv.append({"role": "user", "content": user_prompt})

        judgment = ""
        for _ in range(2):
            new_judgment = get_answer(
                model,
                conv,
                configs["temperature"],
                configs["max_tokens"],
                args["endpoint_dict"],
            )

            judgment += ("\n" + new_judgment)

            score, try_again = get_score(judgment, args["regex_pattern"])

            conv.append({"role": "assistant", "content": new_judgment})

            if not try_again:
                break

            conv.append({"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})
        print("Finish judgment!!!")
        # result = {
        #     "user_prompt": conv[1]["content"],
        #     "judgment": judgment,
        #     "score":score
        # }
        output.append(score)
        
    return output

def get_battles_from_scores(score_list, first_game_only=False, WEIGHT=3):
    arena_hard_battles = pd.DataFrame()

    print("Turning score list into battles...")

    for scores in tqdm(score_list):
        question_id, score1, score2 = scores

        # Process game 1
        output = {"question_id": question_id,
                  "model_a": "gpt-4-0314",
                  "model_b": f"custom_model"}  # Unique identifier for model
        weight = 1
        if score1 == "A=B":
            output["winner"] = "tie"
        elif score1 == "A>B":
            output["winner"] = "model_a"
        elif score1 == "A>>B":
            output["winner"] = "model_a"
            weight = WEIGHT
        elif score1 == "B>A":
            output["winner"] = "model_b"
        elif score1 == "B>>A":
            output["winner"] = "model_b"
            weight = WEIGHT
        else:
            weight = 0

        if weight:
            arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

        if not first_game_only:
            # Process game 2
            output = {"question_id": question_id,
                      "model_a": "gpt-4-0314",
                      "model_b": f"custom_model"}  # Unique identifier for model
            weight = 1
            if score2 == "A=B":
                output["winner"] = "tie"
            elif score2 == "A>B":
                output["winner"] = "model_b"
            elif score2 == "A>>B":
                output["winner"] = "model_b"
                weight = WEIGHT
            elif score2 == "B>A":
                output["winner"] = "model_a"
            elif score2 == "B>>A":
                output["winner"] = "model_a"
                weight = WEIGHT
            else:
                weight = 0

            if weight:
                arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

    # arena_hard_battles.to_json("./arena_hard_battles.jsonl", lines=True, orient="records")
    return arena_hard_battles

def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    LOW_RATING = 100
    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    if len(np.unique(Y)) == 1:
        # If there's only one class in the data, assign default ratings
        elo_scores = np.full(p, LOW_RATING)
        elo_scores[models["gpt-4-0314"]] = INIT_RATING
    else:
        lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
        lr.fit(X,Y)

        elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt-4-0314 = 1000
    if "gpt-4-0314" in models.index:
        elo_scores += 1000 - elo_scores[models["gpt-4-0314"]]
    return pd.Series(elo_scores, index = models.index).sort_values(ascending=False)

def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.NAN for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T

def get_win_rate_column(df, column, baseline="gpt-4-0314"):
    to_dict = df[["model", column]].set_index("model").to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))