"""
STEP 3 — Train and check the model
==================================

Now we teach a model to predict a match result from the features. We use
LOGISTIC REGRESSION: the most beginner-friendly classifier. Despite the name
it's used for classification, and it naturally outputs PROBABILITIES (e.g.
"61% home win, 24% draw, 15% away win"), which is exactly what the bracket
simulator in step 4 needs.

The most important habit in ML: never trust a score on the data you trained on.
So instead of a random split, we split the matches BY DATE — train on 2015–2023
and test on 2024 onward. That mirrors the real task: learn from the past, then
predict matches that haven't happened yet. We judge the model only on the test
years, the same way it'll face genuinely unseen 2026 games.

We also compare against a dumb BASELINE that ignores the features entirely and
just predicts the historical result rates (roughly 28% away, 23% draw, 49% home)
for every match. The score we use is LOG-LOSS, which rewards confident-and-right
predictions and punishes confident mistakes — lower is better. If our model
can't beat that no-features baseline on log-loss, the features aren't adding
anything yet, and we should fix them before moving on to step 4.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# --- load features from step 2 ---
# --- load features from step 2 ---
data = pd.read_csv("data/features.csv", parse_dates=["date"])

feature_cols = ["elo_diff", "form_diff", "neutral"]

train = data[(data["date"] >= "2015-01-01") & (data["date"] < "2024-01-01")]
test  = data[data["date"] >= "2024-01-01"]

X_train, y_train = train[feature_cols], train["result"]
X_test,  y_test  = test[feature_cols],  test["result"]

# always sanity-check the split actually has data in it
print(f"train: {len(X_train):>5} matches  "
      f"({train['date'].min().date()} → {train['date'].max().date()})")
print(f"test : {len(X_test):>5} matches  "
      f"({test['date'].min().date()} → {test['date'].max().date()})")

# --- train the model ---
# max_iter is just how long it's allowed to keep improving; 1000 is plenty here.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- evaluate on the held-out test set ---
pred = model.predict(X_test)                 # most likely class per match
proba = model.predict_proba(X_test)          # probabilities per class

acc = accuracy_score(y_test, pred)
# log_loss rewards being confident AND right, and punishes confident mistakes.
# Lower is better. It matters more than accuracy for a simulation, because the
# simulation uses the probabilities, not just the single top pick.
ll = log_loss(y_test, proba)

# --- baseline: ignore the features, just predict the historical class rates ---
# how often each result happened in the TRAINING data (don't peek at test)
class_rates = y_train.value_counts(normalize=True).sort_index()

# give every test match those same three probabilities
baseline_proba = np.tile(class_rates.values, (len(y_test), 1))
baseline_ll = log_loss(y_test, baseline_proba, labels=class_rates.index)

print("=== Performance on unseen test matches (lower log-loss = better) ===")
print(f"  Model log-loss    : {ll:.3f}")
print(f"  Baseline log-loss : {baseline_ll:.3f}   (ignores features, just class rates)")
print(f"  Model accuracy    : {acc:.3f}")
print()
if ll < baseline_ll:
    print("  Good: the model beats the no-features baseline, so elo_diff and")
    print("  neutral are carrying real signal.")
else:
    print("  Warning: model isn't beating the baseline. Fix features before step 4.")

# --- what did the model learn? peek at the coefficients ---
# Bigger positive number => that feature pushes toward a home win.
print("\n=== What the model weighs (coefficients for the 'home win' class) ===")
home_win_class = list(model.classes_).index(2)
for name, coef in zip(feature_cols, model.coef_[home_win_class]):
    print(f"  {name:10s}: {coef:+.4f}")

# --- save the trained model so step 4 can load and reuse it ---
joblib.dump(model, "data/model.joblib")
print("\nSaved trained model to data/model.joblib")
