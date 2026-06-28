"""
STEP 2 — Build features
=======================

A machine-learning model can't read "Team_07 vs Team_12". It needs NUMBERS that
describe each match. Turning raw data into those numbers is called FEATURE
ENGINEERING, and it's where most of the real work (and the real accuracy) lives.

We build ONE ROW PER MATCH, always written from the home team's point of view,
with these columns:

  elo_diff   : home Elo minus away Elo       (positive => home is stronger)
  form_diff  : home recent form minus away recent form
               (avg points over each team's last N games, BEFORE this match)
  neutral    : 1 if neutral site, else 0
  result     : what actually happened -> 0 = away win, 1 = draw, 2 = home win
               (this is the TARGET: the thing the model learns to predict)

Notice we use DIFFERENCES, not raw values. A model that learns "bigger Elo gap
=> stronger team usually wins" works for any teams, even ones it's never seen.
A model that memorized team names would be useless for the knockout rounds.
"""

import numpy as np
import pandas as pd

#1) --- load the data from step 1 (swap these paths for the real files later) ---
matches = pd.read_csv("data/matches.csv")
elo_history = pd.read_csv("data/elo_history.csv")


#the three columns that make a match a match
matches = pd.merge(matches, elo_history, on=["date", "home_team", "away_team"])
print(f"rows after merge: {len(matches)}")   # should still be ~49,459


#3) --- recent form (leakage-safe: only games BEFORE each match count) ---
# turn each scoreline into points for both sides (3 win / 1 draw / 0 loss)
matches["home_pts"] = np.select(
    [matches["home_score"] > matches["away_score"],
     matches["home_score"] == matches["away_score"]],
    [3, 1], default=0)
matches["away_pts"] = np.select(
    [matches["away_score"] > matches["home_score"],
     matches["away_score"] == matches["home_score"]],
    [3, 1], default=0)

# stack into a LONG table: one row per team per match
home_long = matches[["date", "home_team", "home_pts"]].rename(
    columns={"home_team": "team", "home_pts": "points"})
away_long = matches[["date", "away_team", "away_pts"]].rename(
    columns={"away_team": "team", "away_pts": "points"})
long = pd.concat([home_long, away_long]).sort_values("date")

# form going INTO each match = avg points over last N games, current one excluded
N = 5
long["form"] = (
    long.groupby("team")["points"]
        .transform(lambda s: s.shift(1).rolling(N, min_periods=1).mean())
)

# merge that form back onto the wide table, once per side
home_form = long.rename(columns={"team": "home_team", "form": "home_form"})[
    ["date", "home_team", "home_form"]]
away_form = long.rename(columns={"team": "away_team", "form": "away_form"})[
    ["date", "away_team", "away_form"]]
matches = matches.merge(home_form, on=["date", "home_team"], how="left")
matches = matches.merge(away_form, on=["date", "away_team"], how="left")
matches = matches.drop_duplicates(subset=["date", "home_team", "away_team"])


#4) --- build the feature columns ---
matches["elo_diff"] = matches["home_elo_pre"] - matches["away_elo_pre"]
matches["neutral"] = matches["neutral"].astype(int)
# 0 = no form info either way (e.g. a team's very first games)
matches["form_diff"] = (matches["home_form"] - matches["away_form"]).fillna(0)



#5) --- build the TARGET column from the scoreline ---
def result_label(row):
    if row["home_score"] > row["away_score"]:
        return 2   # home win
    elif row["home_score"] < row["away_score"]:
        return 0   # away win
    else:
        return 1   # draw

matches["result"] = matches.apply(result_label, axis=1)

#6) --- keep only the columns the model needs, and save ---
feature_cols = ["elo_diff", "form_diff", "neutral"]
features = matches[feature_cols + ["date", "result"]].dropna()
features.to_csv("data/features.csv", index=False)

print(f"Built feature table: {len(features)} rows, columns = {feature_cols}")
print("\nFirst few rows:")
print(features.head())
print("\nHow often each result happens (0=away win, 1=draw, 2=home win):")
print(features["result"].value_counts(normalize=True).sort_index().round(3))

# --- save each team's most recent form for step 4 ---
home_side = matches[["date", "home_team", "home_form"]].rename(
    columns={"home_team": "team", "home_form": "form"})
away_side = matches[["date", "away_team", "away_form"]].rename(
    columns={"away_team": "team", "away_form": "form"})
long_form = pd.concat([home_side, away_side], ignore_index=True)

(long_form.sort_values("date")
          .groupby("team")["form"].last()
          .reset_index()
          .to_csv("data/current_form.csv", index=False))
print("Wrote data/current_form.csv")