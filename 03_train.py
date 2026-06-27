"""
STEP 3 — Train and check the model
==================================

Now we teach a model to predict a match result from the features. We use
LOGISTIC REGRESSION: the most beginner-friendly classifier. Despite the name
it's used for classification, and it naturally outputs PROBABILITIES (e.g.
"61% home win, 24% draw, 15% away win"), which is exactly what the bracket
simulator in step 4 needs.

The most important habit in ML: never trust accuracy on the data you trained on.
So we split the matches into a TRAIN set (the model studies these) and a TEST
set (held back, the model never sees them during training). We judge the model
only on the test set, the same way it'll face genuinely unseen 2026 matches.

We also compare against a dumb BASELINE ("the higher-Elo team always wins, no
draws"). If your fancy model can't beat the baseline, the model isn't adding
anything yet and you should fix the features before going further.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# --- load features from step 2 ---
data = pd.read_csv("data/features.csv")
feature_cols = ["elo_diff", "rank_diff", "neutral"]

X = data[feature_cols]   # the inputs
y = data["result"]       # the answer we want to predict

# --- split into train (80%) and test (20%) ---
# random_state just fixes the split so results are repeatable while you learn.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

# --- baseline: higher Elo wins, otherwise call it for away ---
# (elo_diff > 0 means home is stronger -> predict home win = class 2)
baseline_pred = np.where(X_test["elo_diff"] > 0, 2, 0)
baseline_acc = accuracy_score(y_test, baseline_pred)

print("=== Model performance on unseen test matches ===")
print(f"  Model accuracy   : {acc:.3f}")
print(f"  Baseline accuracy: {baseline_acc:.3f}   (higher-Elo-wins rule)")
print(f"  Model log-loss   : {ll:.3f}   (lower is better)")
print()
if acc >= baseline_acc - 0.01:
    print("  Good: the model is matching or beating the baseline on accuracy,")
    print("  AND (unlike the baseline) it outputs calibrated probabilities and")
    print("  handles draws -- which is exactly what the step 4 simulation needs.")
else:
    print("  Warning: model is well below baseline. Improve features before step 4.")

# --- what did the model learn? peek at the coefficients ---
# Bigger positive number => that feature pushes toward a home win.
print("\n=== What the model weighs (coefficients for the 'home win' class) ===")
home_win_class = list(model.classes_).index(2)
for name, coef in zip(feature_cols, model.coef_[home_win_class]):
    print(f"  {name:10s}: {coef:+.4f}")

# --- save the trained model so step 4 can load and reuse it ---
joblib.dump(model, "data/model.joblib")
print("\nSaved trained model to data/model.joblib")
