"""
make_elo.py  —  generate data/elo.csv and data/elo_history.csv from data/matches.csv

Elo can't be downloaded or typed by hand. You GENERATE it: start every team at
1500, walk through every match in DATE ORDER, and after each one nudge the two
teams' ratings based on who was expected to win vs. who actually did.

This version adds three eloratings.net-style refinements:
  - variable K        : big matches (World Cup) move ratings more than friendlies
  - home advantage    : the home side gets +100 when we work out who was expected
                        to win (skipped at neutral sites)
  - goal-margin boost : a 4-0 moves ratings more than a 1-0

Run it with:   python make_elo.py
Re-run any time matches.csv changes; it reprocesses everything and overwrites
both output files.
"""
import pandas as pd

# 1. Load matches in chronological order. Order is everything: a 2014 result
#    must be processed before a 2026 one.
m = pd.read_csv("data/matches.csv").sort_values("date")

HOME_ADVANTAGE = 100        # rating points added to the home side for expectation
START = 1500.0

rating = {}                 # team -> current rating, built as we go
def get(t): return rating.get(t, START)

# how fast ratings move, by how important the match is
def k_factor(tournament):
    t = str(tournament).lower()
    if "friendly" in t:                          return 20
    if "qualification" in t or "qualif" in t:    return 40
    if "world cup" in t:                         return 60   # the finals themselves
    if any(x in t for x in ["euro", "copa", "cup of nations", "gold cup", "asian cup"]):
        return 50                                            # major continental finals
    return 30                                                 # sensible default

# margin-of-victory multiplier: bigger wins move ratings more
def goal_multiplier(gd):
    gd = abs(gd)
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    return (11 + gd) / 8                                      # 3 -> 1.75, 4 -> 1.875, ...

# 2. One pass through history.
history = []                # pre-match ratings, one row per match
for r in m.itertuples():
    home, away = r.home_team, r.away_team
    Rh, Ra = get(home), get(away)            # TRUE pre-match ratings

    # record the genuine pre-match Elo BEFORE we change anything (no leakage)
    history.append({
        "date": r.date,
        "home_team": home,
        "away_team": away,
        "home_elo_pre": Rh,
        "away_elo_pre": Ra,
    })

    # home advantage only counts when it's actually a home game
    adj = 0 if int(r.neutral) == 1 else HOME_ADVANTAGE

    # expected score for the home team, from the (adjusted) rating gap
    Eh = 1 / (1 + 10 ** ((Ra - (Rh + adj)) / 400))

    # what actually happened, from the home team's side
    if   r.home_score > r.away_score: Sh = 1.0    # home win
    elif r.home_score < r.away_score: Sh = 0.0    # home loss
    else:                             Sh = 0.5    # draw

    # how big a step this match earns: importance (K) x margin (G)
    step = k_factor(r.tournament) * goal_multiplier(r.home_score - r.away_score)

    # both teams move by the "surprise" (actual minus expected)
    rating[home] = Rh + step * (Sh - Eh)
    rating[away] = Ra + step * ((1 - Sh) - (1 - Eh))

# 3a. Save current ratings, strongest first.
elo = (pd.DataFrame(rating.items(), columns=["team", "elo"])
         .sort_values("elo", ascending=False)
         .reset_index(drop=True))
elo.to_csv("data/elo.csv", index=False)

# 3b. Save the pre-match ratings that 02_features.py merges on.
pd.DataFrame(history).to_csv("data/elo_history.csv", index=False)

print(f"Generated data/elo.csv and data/elo_history.csv from {len(m)} matches "
      f"->  {len(elo)} teams")
print(elo.head(5).to_string(index=False))