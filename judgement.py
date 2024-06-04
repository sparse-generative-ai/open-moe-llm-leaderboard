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
import argparse
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment_kwargs_dir", type=str, required=True, help="Path to the judgment kwargs")
    return parser.parse_args()

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

args = get_args()
file_data = []

# Loop through each file in the directory
for filename in sorted(os.listdir(args.judgment_kwargs_dir)):
    file_path = os.path.join(args.judgment_kwargs_dir, filename)
    file_name = filename[:-20]
    kwargss = pickle.load(open(file_path, "rb"))

    print("Judging...")
    score_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for kwargs in kwargss[:2]:
            scores = judgment(**kwargs)
            score_list.append(scores)
        score = get_win_rate(score_list)
        output = [file_name, score]
        print(f"Arena Hard score: {score}")
    
    output_csv_path = 'arena_hard_output.csv'
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(['Model', 'Score'])
        # Write file data
        writer.writerows(output)