"""
STEP 5 — Visualize the bracket as a round-by-round probability heatmap
=====================================================================

Step 4 only told you who wins it all. This script answers the richer question:
"how likely is each team to REACH each round?" To get that, we run the same
Monte Carlo simulation but track, in every simulated tournament, how many games
each team won. A team that won 3 games reached the semifinal; 5 wins = champion.
Average those outcomes over 10,000 tournaments and you get a survival curve per
team, which we draw as a heatmap (one row per team, one column per round).

Run AFTER 04 works, with the same data files in place. Produces:
    bracket_probabilities.png

IMPORTANT: this MUST use the same win_prob as 04, including the squad-value
nudge. If 04 and 05 disagree, it's because the nudge is in one but not the other.
"""

import itertools

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

rng = np.random.default_rng(seed=7)

# --- load everything (same as step 4) ---
model = joblib.load("data/model.joblib")
elo = pd.read_csv("data/elo.csv")
bracket = pd.read_csv("data/bracket.csv").sort_values("bracket_position")
current_form = pd.read_csv("data/current_form.csv")

elo_lookup = dict(zip(elo["team"], elo["elo"]))
form_lookup = dict(zip(current_form["team"], current_form["form"]))
FEATURE_COLS = ["elo_diff", "form_diff", "neutral"]
bracket_order = bracket["team"].tolist()

missing = [t for t in bracket_order if t not in elo_lookup or t not in form_lookup]
assert not missing, f"Bracket teams missing from lookups: {missing}"

# --- precompute every matchup's probabilities (same as step 4) ---
pairs = list(itertools.permutations(bracket_order, 2))
feature_rows = [{
    "elo_diff": elo_lookup[a] - elo_lookup[b],
    "form_diff": form_lookup[a] - form_lookup[b],
    "neutral": 1,
} for (a, b) in pairs]
feature_table = pd.DataFrame(feature_rows)[FEATURE_COLS]
all_probs = model.predict_proba(feature_table)

# Encoding from 02: 0 = away win, 1 = draw, 2 = home win. team_a = home slot.
classes = list(model.classes_)
col_b, col_draw, col_a = classes.index(0), classes.index(1), classes.index(2)

# --- squad-value nudge: MUST match 04 exactly (same column names + same BELIEF) ---
squad = pd.read_csv("data/squad_values.csv")
sv = dict(zip(squad["Team"], squad["squad_value_GBP_million"]))
sv_missing = [t for t in bracket_order if t not in sv]
assert not sv_missing, f"No squad value for: {sv_missing}"

BELIEF = 0.0005   # keep IDENTICAL to the value in 04_simulate.py

win_prob = {}
for (a, b), probs in zip(pairs, all_probs):
    p_a, p_draw, p_b = probs[col_a], probs[col_draw], probs[col_b]
    nudge = BELIEF * (sv[a] - sv[b])
    la, ld, lb = np.log([p_a, p_draw, p_b])
    la, lb = la + nudge, lb - nudge
    e = np.exp([la, ld, lb]); e /= e.sum()
    win_prob[(a, b)] = (e[0], e[1], e[2])


def play_match(team_a, team_b):
    p_a, p_draw, p_b = win_prob[(team_a, team_b)]
    outcome = rng.choice(3, p=[p_a, p_draw, p_b])
    if outcome == 0:
        return team_a
    if outcome == 2:
        return team_b
    total = p_a + p_b
    pen_a = p_a / total if total > 0 else 0.5
    return team_a if rng.random() < pen_a else team_b


def simulate_with_tracking(order):
    """Play one bracket; return {team: number of games it won}."""
    remaining = list(order)
    wins = {t: 0 for t in order}
    while len(remaining) > 1:
        nxt = []
        for i in range(0, len(remaining), 2):
            w = play_match(remaining[i], remaining[i + 1])
            wins[w] += 1
            nxt.append(w)
        remaining = nxt
    return wins


# --- run sims and tally how often each team reaches each round ---
N_SIMS = 10000
n_rounds = int(np.log2(len(bracket_order)))
labels = ["Reach R16", "Reach QF", "Reach SF", "Reach Final", "Champion"][:n_rounds]

reach = {t: np.zeros(n_rounds) for t in bracket_order}
for _ in range(N_SIMS):
    wins = simulate_with_tracking(bracket_order)
    for t, w in wins.items():
        for k in range(n_rounds):
            if w >= k + 1:
                reach[t][k] += 1

P = pd.DataFrame({t: reach[t] / N_SIMS for t in bracket_order}, index=labels).T
P = P.sort_values("Champion", ascending=False)


# --- render the heatmap ---
def render(P, path, n_sims):
    cols = list(P.columns)
    M = P.values
    cmap = LinearSegmentedColormap.from_list(
        "wc", ["#fbfcfe", "#cfe8e3", "#7fc6bd", "#2f9c8f", "#1f6f73", "#243b6b"])
    fig, ax = plt.subplots(figsize=(7.4, 11.6))
    fig.subplots_adjust(top=0.90, left=0.20, right=0.97, bottom=0.02)
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("Reach ", "") for c in cols], fontsize=11)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", pad=6)
    ax.set_yticks(range(len(P)))
    ax.set_yticklabels(P.index, fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(P), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "" if v < 0.005 else ("<1%" if v < 0.01 else f"{v*100:.0f}%")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if v > 0.5 else "#1a1a1a",
                    fontweight="bold" if j == len(cols) - 1 else "normal")
    fig.text(0.035, 0.955, "2026 World Cup — chance of reaching each round",
             fontsize=14, fontweight="bold", ha="left")
    fig.text(0.035, 0.928, f"{n_sims:,} Monte Carlo simulations · squad-value weighted",
             fontsize=9.5, color="#666", ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {path}")


render(P, "bracket_probabilities.png", N_SIMS)