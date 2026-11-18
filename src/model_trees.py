import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from evaluate import split3, scores, TARGET_COL

df = pd.read_csv("data/processed/matches_features.csv")
train, val, test = split3(df)

feats = ["elo_diff", "form_diff", "gd_diff", "h_elo", "a_elo"]

models = {
    "логрегрессия": make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000)),
    "случайный лес": RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=20, random_state=42),
    "градиентный бустинг": HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=100, random_state=42),
}

for name, model in models.items():
    model.fit(train[feats], train[TARGET_COL])
    pred = model.predict(val[feats])
    proba = model.predict_proba(val[feats])
    s = scores(val[TARGET_COL], pred, proba)
    print(f"{name}: acc={s['accuracy']:.3f} f1={s['macro_f1']:.3f} "
          f"logloss={s['log_loss']:.3f}")