import numpy as np
import pandas as pd
from evaluate import split, scores, TARGET_COL, CLASSES

df = pd.read_csv("data/processed/matches_clean.csv")
train, test = split(df)
y = test[TARGET_COL]

# Прогноз: всегда победа хозяев
pred = np.array(["H"] * len(test))

# Вероятности: доли исходов в train (для log loss)
freq = train[TARGET_COL].value_counts(normalize=True)
proba = np.tile([freq[c] for c in CLASSES], (len(test), 1))

print(scores(y, pred, proba))