import numpy as np
import pandas as pd
from evaluate import split3, scores, TARGET_COL

df = pd.read_csv("data/processed/matches_features.csv")
train, val, test = split3(df)

pred = np.where(val["elo_diff"] + 100 > 0, "H", "A")
print(scores(val[TARGET_COL], pred))