"""
STEP 2 — Build features
=======================

A machine-learning model can't read "Team_07 vs Team_12". It needs NUMBERS that
describe each match. Turning raw data into those numbers is called FEATURE
ENGINEERING, and it's where most of the real work (and the real accuracy) lives.

We build ONE ROW PER MATCH, always written from the home team's point of view,
with these columns:

  elo_diff   : home Elo  minus away Elo      (positive => home is stronger)
  rank_diff  : away rank minus home rank      (positive => home is better ranked)
  neutral    : 1 if neutral site, else 0
  result     : what actually happened -> 0 = away win, 1 = draw, 2 = home win
               (this is the TARGET: the thing the model learns to predict)

Notice we use DIFFERENCES, not raw values. A model that learns "bigger Elo gap
=> stronger team usually wins" works for any teams, even ones it's never seen.
A model that memorized team names would be useless for the knockout rounds.
"""

import pandas as pd

# --- load the data from step 1 (swap these paths for the real files later) ---
matches = pd.read_csv("data/sample_matches.csv")
elo = pd.read_csv("data/sample_elo.csv")

# Make quick lookups from team name -> rating, so we can attach ratings to matches
elo_lookup = dict(zip(elo["team"], elo["elo"]))
rank_lookup = dict(zip(elo["team"], elo["fifa_rank"]))

# --- attach each team's rating onto every match ---
matches["home_elo"] = matches["home_team"].map(elo_lookup)
matches["away_elo"] = matches["away_team"].map(elo_lookup)
matches["home_rank"] = matches["home_team"].map(rank_lookup)
matches["away_rank"] = matches["away_team"].map(rank_lookup)

# --- build the feature columns ---
matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]
matches["rank_diff"] = matches["away_rank"] - matches["home_rank"]
matches["neutral"] = matches["neutral"].astype(int)

# --- build the TARGET column from the scoreline ---
def result_label(row):
    if row["home_score"] > row["away_score"]:
        return 2   # home win
    elif row["home_score"] < row["away_score"]:
        return 0   # away win
    else:
        return 1   # draw

matches["result"] = matches.apply(result_label, axis=1)

# --- keep only the columns the model needs, and save ---
feature_cols = ["elo_diff", "rank_diff", "neutral"]
features = matches[feature_cols + ["result"]].dropna()
features.to_csv("data/features.csv", index=False)

print(f"Built feature table: {len(features)} rows, columns = {feature_cols}")
print("\nFirst few rows:")
print(features.head())
print("\nHow often each result happens (0=away win, 1=draw, 2=home win):")
print(features["result"].value_counts(normalize=True).sort_index().round(3))
