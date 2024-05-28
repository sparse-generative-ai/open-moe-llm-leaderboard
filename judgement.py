import pickle
from src.backend.tasks.arena_hard.arena_judgment import (
    judgment,
    get_battles_from_scores,
    compute_mle_elo,
    predict_win_rate,
    get_win_rate_column
)
import os
import concurrent
import pandas as pd

kwargss = pickle.load(open(os.path.join(os.path.dirname(__file__), "judgment_kwargs.pkl"), "rb"))

print("Judging...")
score_list = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    for kwargs in kwargss:
        scores = judgment(**kwargs)
        score_list.append(scores)


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

score = get_win_rate(score_list)
print(f"Arena Hard score: {score}")