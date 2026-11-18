import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from evaluate import (split, scores, TRAIN_SEASONS,
                      TEST_SEASONS, TARGET_COL)

rng = np.random.default_rng(42)
df = pd.DataFrame(rng.normal(size=(1000, 5)),
                  columns=[f"f{i}" for i in range(5)])
df[TARGET_COL] = rng.choice(["H", "D", "A"], size=1000)
df["season"] = rng.choice(TRAIN_SEASONS + TEST_SEASONS, size=1000)

train, test = split(df)
assert set(train["season"]).isdisjoint(set(test["season"]))

feats = [f"f{i}" for i in range(5)]
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(train[feats], train[TARGET_COL])

pred = model.predict(test[feats])
proba = model.predict_proba(test[feats])
print(scores(test[TARGET_COL], pred, proba))
