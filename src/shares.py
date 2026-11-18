import pandas as pd
from evaluate import split3, TARGET_COL

df = pd.read_csv("data/processed/matches_clean.csv")
train, val, test = split3(df)
for name, part in [("train", train), ("val", val), ("test", test)]:
    print(name, len(part))
    print(part[TARGET_COL].value_counts(normalize=True).round(3))