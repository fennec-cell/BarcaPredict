import pandas as pd
from sklearn.linear_model import LogisticRegression
from evaluate import split3, scores, TARGET_COL

df = pd.read_csv("data/processed/matches_features.csv")
train, val, test = split3(df)

sets = {
    "elo": ["elo_diff"],
    "elo + форма": ["elo_diff", "form_diff"],
    "elo + форма + голы": ["elo_diff", "form_diff", "gd_diff"],
}

for name, feats in sets.items():
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train[feats], train[TARGET_COL])
    pred = model.predict(val[feats])
    proba = model.predict_proba(val[feats])
    s = scores(val[TARGET_COL], pred, proba)
    print(f"{name}: acc={s['accuracy']:.3f} f1={s['macro_f1']:.3f} "
          f"logloss={s['log_loss']:.3f}")