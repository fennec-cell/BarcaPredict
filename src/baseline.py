import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from evaluate import split3, scores, TARGET_COL

df = pd.read_csv("data/processed/matches_clean.csv")
train, val, test = split3(df)

# «всегда H»: DummyClassifier предсказывает самый частый исход в train
dummy = DummyClassifier(strategy="prior")
dummy.fit(np.zeros((len(train), 1)), train[TARGET_COL])

for name, part in [("val", val), ("test", test)]:
    X = np.zeros((len(part), 1))
    pred = dummy.predict(X)
    proba = dummy.predict_proba(X)
    print(name, scores(part[TARGET_COL], pred, proba))