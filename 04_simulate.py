"""
STEP 4 — Simulate the knockout bracket
======================================

This is the payoff. The model from step 3 only knows how to predict a SINGLE
match. The tournament is 31 matches in a knockout tree. So we do a MONTE CARLO
SIMULATION: play the whole bracket thousands of times, letting chance decide
each game according to the model's probabilities, and count how often each team
ends up champion.

  champion probability = (times a team won) / (number of simulations)

That fraction is your final answer: "Team X has a 14% chance to win it all."

PERFORMANCE NOTE (a useful habit): there are only 32 teams, so only a few
hundred possible matchups. Instead of asking the model mid-simulation (slow,
hundreds of thousands of times), we ask it ONCE for every possible pairing up
front and store the answers in a dictionary. The simulation then just looks
them up -- the same work runs in a second instead of minutes. Precompute the
expensive thing once, reuse it many times.

Knockout games can't end in a draw, but the model can predict one. When a
simulated game comes back a draw, we send it to a "penalty shootout" and pick
the winner with a coin flip weighted by the two teams' win probabilities.
"""

import itertools

import joblib
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=7)

# --- load everything from earlier steps ---
model = joblib.load("data/model.joblib")
elo = pd.read_csv("data/elo.csv")
bracket = pd.read_csv("data/bracket.csv").sort_values("bracket_position")
current_form = pd.read_csv("data/current_form.csv")   # written by 02: one row per team, latest form

elo_lookup = dict(zip(elo["team"], elo["elo"]))
form_lookup = dict(zip(current_form["team"], current_form["form"]))

# Column order MUST match how the model was trained in step 3.
FEATURE_COLS = ["elo_diff", "form_diff", "neutral"]

bracket_order = bracket["team"].tolist()

# Fail loudly on name mismatches (e.g. "USA" vs "United States") instead of a
# bare KeyError deep inside the simulation loop.
missing = [t for t in bracket_order if t not in elo_lookup or t not in form_lookup]
assert not missing, f"Bracket teams missing from elo/form lookups: {missing}"

# ----------------------------------------------------------------------------
# Precompute the probabilities for EVERY possible matchup, in one prediction
# ----------------------------------------------------------------------------
# itertools.permutations gives every ordered pair (a, b) with a != b.
pairs = list(itertools.permutations(bracket_order, 2))

feature_rows = [{
    "elo_diff": elo_lookup[a] - elo_lookup[b],
    "form_diff": form_lookup[a] - form_lookup[b],
    "neutral": 1,  # World Cup knockout games are at neutral venues
} for (a, b) in pairs]

feature_table = pd.DataFrame(feature_rows)[FEATURE_COLS]
all_probs = model.predict_proba(feature_table)   # one fast batched call

# Your target encoding (from 02) is: 0 = away win, 1 = draw, 2 = home win.
# team_a sits in the HOME slot (elo_diff = elo[a] - elo[b]), so:
#   home win (label 2) -> team_a wins -> col_a
#   draw      (label 1)               -> col_draw
#   away win  (label 0) -> team_b wins -> col_b
classes = list(model.classes_)
col_b, col_draw, col_a = classes.index(0), classes.index(1), classes.index(2)

# ----------------------------------------------------------------------------
# Squad-value prior (your belief, applied ON TOP of the model — not a feature)
# ----------------------------------------------------------------------------
squad = pd.read_csv("data/squad_values.csv")
sv = dict(zip(squad["Team"], squad["squad_value_GBP_million"]))

# A name that doesn't match would silently get value 0 below and have its odds
# crushed. Fail loudly instead so you know exactly which spelling to fix.
sv_missing = [t for t in bracket_order if t not in sv]
assert not sv_missing, f"No squad value for: {sv_missing}"

BELIEF = 0.0005   # your knob: 0 = trust the model, higher = trust squad value more

# Store results: win_prob[(a, b)] = (p_a_win, p_draw, p_b_win)
win_prob = {}
for (a, b), probs in zip(pairs, all_probs):
    p_a, p_draw, p_b = probs[col_a], probs[col_draw], probs[col_b]
    nudge = BELIEF * (sv[a] - sv[b])                     # + favors the richer squad
    la, ld, lb = np.log([p_a, p_draw, p_b])
    la, lb = la + nudge, lb - nudge                      # shift home/away logits
    e = np.exp([la, ld, lb]); e /= e.sum()              # renormalize to sum 1
    win_prob[(a, b)] = (e[0], e[1], e[2])


def play_match(team_a, team_b):
    """Simulate ONE knockout game and return the winner's name."""
    p_a, p_draw, p_b = win_prob[(team_a, team_b)]

    outcome = rng.choice(3, p=[p_a, p_draw, p_b])  # 0=a, 1=draw, 2=b
    if outcome == 0:
        return team_a
    if outcome == 2:
        return team_b

    # Draw -> penalty shootout, weighted by the two win chances.
    total = p_a + p_b
    pen_a = p_a / total if total > 0 else 0.5
    return team_a if rng.random() < pen_a else team_b


def simulate_one_tournament(teams_in_bracket_order):
    """Play the whole bracket once; return the champion's name."""
    remaining = list(teams_in_bracket_order)
    while len(remaining) > 1:
        # pair them up 2 at a time: (0v1), (2v3), (4v5), ... winners advance
        remaining = [play_match(remaining[i], remaining[i + 1])
                     for i in range(0, len(remaining), 2)]
    return remaining[0]


# ----------------------------------------------------------------------------
# Run the simulation many times
# ----------------------------------------------------------------------------
N_SIMS = 10000
champions = [simulate_one_tournament(bracket_order) for _ in range(N_SIMS)]

# Tally and turn counts into probabilities
counts = pd.Series(champions).value_counts()
champ_prob = (counts / N_SIMS).round(4)

# Attach Elo so you can sanity-check (stronger teams SHOULD rise to the top)
results = pd.DataFrame({"win_probability": champ_prob})
results["elo"] = results.index.map(elo_lookup)
results = results.sort_values("win_probability", ascending=False)

print(f"=== World Cup winner probabilities ({N_SIMS:,} simulations) ===\n")
print(results.head(12).to_string())
print(f"\n(Probabilities sum to {results['win_probability'].sum():.2f} across all 32 teams.)")