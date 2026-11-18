import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from evaluate import split3, TARGET_COL

FEATS = ["elo_diff", "form_diff", "gd_diff", "h_elo", "a_elo"]

df = pd.read_csv("data/processed/matches_features.csv")
train, val, test = split3(df)
val = val.reset_index(drop=True)

model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, random_state=42))
model.fit(train[FEATS], train[TARGET_COL])

proba = model.predict_proba(val[FEATS])
classes = model.classes_
val["pred"] = classes[proba.argmax(axis=1)]
for i, c in enumerate(classes):
    val["p" + c] = proba[:, i]
y = val[TARGET_COL]

print("=== 1. По классам (проверочный сезон) ===")
print(classification_report(y, val["pred"], zero_division=0))

print("=== 2. Средние вероятности при реальном исходе ===")
print(val.groupby(TARGET_COL)[["pA", "pD", "pH"]].mean().round(3))
print("Максимальная вероятность ничьей в сезоне:", round(val["pD"].max(), 3))

print("\n=== 3. Уверенность модели и точность ===")
val["conf"] = proba.max(axis=1)
val["bin"] = pd.cut(val["conf"], [0, 0.4, 0.5, 0.6, 1.0])
val["ok"] = (val["pred"] == y)
print(val.groupby("bin", observed=True)["ok"].agg(["count", "mean"]).round(3))

print("\n=== 4. Матчи Барселоны ===")
bar = val[(val["home"] == "Barcelona") | (val["away"] == "Barcelona")]
print("Матчей:", len(bar), "| точность:", round(bar["ok"].mean(), 3))
print(bar[["date", "home", "away", TARGET_COL, "pred", "pH", "pD", "pA"]]
      .round(2).to_string(index=False))
