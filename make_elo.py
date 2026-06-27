"""
make_elo.py  —  generate data/elo.csv from data/matches.csv

Elo can't be downloaded or typed by hand. You GENERATE it: start every team at
1500, walk through every match in DATE ORDER, and after each one nudge the two
teams' ratings based on who was expected to win vs. who actually did.

Run it with:   python make_elo.py
Re-run it any time matches.csv changes (e.g. after the group stage) — it just
reprocesses everything and overwrites elo.csv with fresh numbers.
"""
import pandas as pd

# 1. Load matches and put them in chronological order. Order is everything here:
#    a 2014 result must be processed before a 2026 one.
m = pd.read_csv("data/matches.csv").sort_values("date")

K = 30                                  # how fast ratings move per match
rating = {}                             # team -> current rating, built as we go
def get(t): return rating.get(t, 1500.0)   # new teams start at 1500

# 2. One pass through history.
for r in m.itertuples():
    home, away = r.home_team, r.away_team
    Rh, Ra = get(home), get(away)

    # expected score for the home team, from the rating gap
    Eh = 1 / (1 + 10 ** ((Ra - Rh) / 400))

    # what actually happened, from the home team's side
    if   r.home_score > r.away_score: Sh = 1.0     # home win
    elif r.home_score < r.away_score: Sh = 0.0     # home loss
    else:                            Sh = 0.5     # draw

    # the update: both teams move by the "surprise" (actual minus expected)
    rating[home] = Rh + K * (Sh - Eh)
    rating[away] = Ra + K * ((1 - Sh) - (1 - Eh))

# 3. Save one row per team, strongest first.
elo = (pd.DataFrame(rating.items(), columns=["team", "elo"])
         .sort_values("elo", ascending=False)
         .reset_index(drop=True))
elo.to_csv("data/elo.csv", index=False)

print(f"Generated data/elo.csv from {len(m)} matches  ->  {len(elo)} teams")
print(elo.head(5).to_string(index=False))
