import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluate import split3, scores, TARGET_COL

df = pd.read_csv("data/processed/matches_features.csv")
train, val, test = split3(df)

# Берём «сырые» признаки: у первых матчей команды форма пустая (NaN)
feats = ["elo_diff", "h_elo", "a_elo", "h_form", "a_form", "h_gd", "a_gd"]

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42),
)
model.fit(train[feats], train[TARGET_COL])

pred = model.predict(val[feats])
proba = model.predict_proba(val[feats])
print(scores(val[TARGET_COL], pred, proba))