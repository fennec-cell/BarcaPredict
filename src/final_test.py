import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluate import split3, scores, TARGET_COL, CLASSES

df = pd.read_csv("data/processed/matches_features.csv")
train, val, test = split3(df)
fit_data = pd.concat([train, val])  # обучаем на train + val

feats = ["elo_diff", "form_diff", "gd_diff", "h_elo", "a_elo"]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(fit_data[feats], fit_data[TARGET_COL])

pred = model.predict(test[feats])
proba = model.predict_proba(test[feats])
print("модель:", scores(test[TARGET_COL], pred, proba))

# baseline «всегда H» на тех же матчах
freq = fit_data[TARGET_COL].value_counts(normalize=True)
b_pred = np.array(["H"] * len(test))
b_proba = np.tile([freq[c] for c in CLASSES], (len(test), 1))
print("baseline:", scores(test[TARGET_COL], b_pred, b_proba))