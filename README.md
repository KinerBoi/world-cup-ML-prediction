# World Cup Winner Predictor

Predicts each team's chance of winning the World Cup by (1) training a model on
historical matches to estimate **who wins a single game**, then (2) **simulating
the knockout bracket** thousands of times and counting champions.

It works **right now** on auto-generated fake data, so you can see the whole
thing run before the real group stage even finishes. You then swap in real data
at the end.

---

## Setup (one time)

```bash
pip install scikit-learn pandas numpy joblib
```

## Run it

Run the four scripts in order. Each saves a file that the next one reads.

```bash
python 01_make_sample_data.py   # invents fake data in ./data/
python 02_features.py           # turns matches into model inputs
python 03_train.py              # trains the model + checks it on unseen games
python 04_simulate.py           # simulates the bracket -> winner probabilities
```

If step 4 prints a sensible-looking table of teams and percentages that add up
to 1.00, the machine works. Now you make it real.

---

## The big idea (read this once)

You will only ever get ONE real World Cup to "test" a champion prediction on, so
you can't train a model to predict the champion directly. Instead:

1. **Match model** — train on thousands of past matches to predict a single
   game: probabilities of `away win / draw / home win`.
2. **Simulation** — play the 31-game knockout bracket 10,000 times, letting each
   game be decided randomly by those probabilities. The share of simulations a
   team wins *is* its championship probability.

This is the standard, sane way to do it, and it's why the project is doable by
Sunday.

---

## What each file does

| File | Job |
|------|-----|
| `01_make_sample_data.py` | Creates fake stand-in data. **You delete/replace this once you have real data.** |
| `02_features.py` | Builds one row per match with numeric features + the actual result. |
| `03_train.py` | Trains logistic regression, checks it against a baseline, saves the model. |
| `04_simulate.py` | Loads the model, simulates the bracket many times, prints winner odds. |
| `data/` | Where all the CSVs and the saved model live. |

---

## Swapping in REAL data (do this Fri–Sat)

You only need to make your real files **look like the fake ones**. Open the fake
CSVs in `data/` to see the exact columns, then produce real versions with the
same column names. After that, steps 2–4 run unchanged.

**1. Historical matches** → replace `data/sample_matches.csv`
Needs columns: `home_team, away_team, home_score, away_score, neutral`.
Source: the Kaggle dataset *"International football results from 1872 to present"*
(by Mart Jürisoo). It already has these columns — you mostly just rename a couple.

**2. Team ratings** → replace `data/sample_elo.csv`
Needs columns: `team, elo, fifa_rank`.
Source: Elo from eloratings.net; FIFA ranking from FIFA's site. The single most
important thing here is that **team names match exactly** between your matches
file and your ratings file (e.g. decide once whether it's "USA" or "United
States" and make everything agree). Mismatched names are the #1 bug in this kind
of project.

**3. The real 32-team bracket** → replace `data/sample_bracket.csv`
Needs columns: `team, elo, fifa_rank, bracket_position`.
The groups finish **Sat June 27** and the Round of 32 is set then. List the 32
teams so that `bracket_position` 0 plays 1, 2 plays 3, etc., following the
official bracket. This is your last step before running the final prediction
Sunday morning.

> Tip: build and fully test everything on the fake data first. Treat the real
> numbers as a last-minute swap-in, not something you wait around for.

---

## Ideas to make it better (only if you have time)

- **Recent form**: in `02_features.py`, add each team's points-per-game over its
  last ~5 matches as another feature. (Your earlier instinct — group-stage form —
  goes here. Test whether it actually helps; over 3 games it can be noisy.)
- **Squad strength** (your "club data" idea): add a `squad_value` column from
  Transfermarkt market values and feed the difference as a feature. This is the
  most time-consuming part to collect — leave it for last.
- **Stronger model**: swap `LogisticRegression` for
  `HistGradientBoostingClassifier` (also built into scikit-learn — change one
  line in `03_train.py`). Often a bit more accurate.
- **Confidence**: because the bracket is random, run step 4 a few times; if the
  top teams' percentages jump around a lot, raise `N_SIMS`.

---

## Common gotchas

- *KeyError on a team name* → a name in your bracket/matches isn't in your
  ratings file. Standardize names everywhere.
- *Probabilities don't sum to 1* in step 4 → you have fewer/more than 32 teams,
  or a duplicate; the bracket must be exactly 32 distinct teams.
- *Model loses to the baseline* in step 3 → your features have a bug (often a
  name-matching issue making `elo_diff` come out as 0/NaN). Fix before simulating.