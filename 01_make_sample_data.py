"""
STEP 1 — Make sample data
==========================

This creates FAKE data so the whole pipeline runs end-to-end today, before the
real 2026 group stage has even finished. Once you download the real datasets,
you delete this step and point step 2 at the real files instead (the README
tells you exactly which columns to match).

What it produces (all saved in ./data/):
  - sample_matches.csv : historical international matches (your TRAINING data)
  - sample_elo.csv     : a current strength rating per team
  - sample_bracket.csv : the 32 knockout teams, in bracket order

The trick that makes this useful for learning: each fake team gets a hidden
"true strength". Match results are generated FROM that strength, so there is a
real pattern in the data for the model to discover. When your backtest in step 3
shows the model beating the baseline, that's it finding this pattern.
"""

import numpy as np
import pandas as pd

# A "seed" makes the random numbers reproducible: you get the SAME fake data
# every run. Handy while learning so results don't jump around on you.
rng = np.random.default_rng(seed=42)

# ----------------------------------------------------------------------------
# 1. Invent some national teams, each with a hidden "true strength"
# ----------------------------------------------------------------------------
N_TEAMS = 48
team_names = [f"Team_{i:02d}" for i in range(N_TEAMS)]

# True strength ~ a bell curve centered at 0. Higher = better team.
true_strength = rng.normal(loc=0.0, scale=1.0, size=N_TEAMS)

# Turn strength into an Elo-style rating (just a friendlier 1300–2100 scale).
# In the REAL project this column comes from eloratings.net instead.
elo = 1700 + true_strength * 120
# A FIFA-style rank: rank 1 = strongest. argsort gives the ordering.
fifa_rank = (-true_strength).argsort().argsort() + 1  # 1 = best

teams = pd.DataFrame({
    "team": team_names,
    "elo": elo.round(0),
    "fifa_rank": fifa_rank,
})
teams.to_csv("data/sample_elo.csv", index=False)
strength_lookup = dict(zip(team_names, true_strength))

# ----------------------------------------------------------------------------
# 2. Simulate a few thousand historical matches between these teams
# ----------------------------------------------------------------------------
# We invent a fake calendar of matches over several "years". For each match we
# generate a scoreline using a Poisson distribution (the standard way to model
# goal counts in soccer). A stronger team gets a higher expected goal count.
N_MATCHES = 3000
rows = []
for _ in range(N_MATCHES):
    # pick two different teams at random
    a, b = rng.choice(team_names, size=2, replace=False)
    sa, sb = strength_lookup[a], strength_lookup[b]

    # About 45% of matches are played at the home team's ground (not neutral).
    # Playing at home is worth a small goal boost. This gives the 'neutral'
    # feature real predictive value, so the model can beat a pure-Elo baseline
    # that ignores it -- just like home advantage matters in real football.
    is_neutral = rng.random() > 0.45
    home_boost = 0.0 if is_neutral else 0.45

    # expected goals: base of 1.3, nudged by the strength gap (+ home boost).
    # np.clip keeps it from going negative.
    lam_a = np.clip(1.3 + 0.8 * (sa - sb) + home_boost, 0.15, 6)
    lam_b = np.clip(1.3 + 0.8 * (sb - sa), 0.15, 6)

    goals_a = rng.poisson(lam_a)
    goals_b = rng.poisson(lam_b)

    rows.append({
        "home_team": a,
        "away_team": b,
        "home_score": goals_a,
        "away_score": goals_b,
        "neutral": is_neutral,
    })

matches = pd.DataFrame(rows)
matches.to_csv("data/sample_matches.csv", index=False)

# ----------------------------------------------------------------------------
# 3. Build a sample 32-team knockout bracket (in bracket order)
# ----------------------------------------------------------------------------
# Bracket order matters: teams[0] plays teams[1], teams[2] plays teams[3], and
# so on. Winners meet in the next round following the same pairing rule. When
# the real Round of 32 is set on Saturday, you just replace these 32 names.
bracket_teams = rng.choice(team_names, size=32, replace=False)
bracket = teams[teams["team"].isin(bracket_teams)].copy()
# put them in a random bracket order for the demo
bracket = bracket.sample(frac=1, random_state=1).reset_index(drop=True)
bracket["bracket_position"] = range(32)
bracket.to_csv("data/sample_bracket.csv", index=False)

print("Sample data written to ./data/")
print(f"  sample_matches.csv : {len(matches)} historical matches")
print(f"  sample_elo.csv     : {len(teams)} teams with ratings")
print(f"  sample_bracket.csv : {len(bracket)} knockout teams")
