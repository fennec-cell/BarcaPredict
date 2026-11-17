import pandas as pd
from sklearn.linear_model import LogisticRegression
from evaluate import split3, scores, TARGET_COL

df = pd.read_csv("data/processed/matches_elo.csv")
train, val, test = split3(df)

feats = ["elo_diff"]
model = LogisticRegression(max_iter=1000)
model.fit(train[feats], train[TARGET_COL])

pred = model.predict(val[feats])
proba = model.predict_proba(val[feats])
print(scores(val[TARGET_COL], pred, proba))
