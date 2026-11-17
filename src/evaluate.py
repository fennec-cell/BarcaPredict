import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, confusion_matrix
)

# --- константы ---
SEASON_COL = "season"
TARGET_COL = "result"
CLASSES = ["A", "D", "H"]  # по алфавиту, как у sklearn

# Подставь по факту: что выведет sorted(df["season"].unique())
TRAIN_SEASONS = [1415, 1516, 1617, 1718, 1819,
                 1920, 2021, 2122, 2223]
TEST_SEASONS = [2324, 2425, 2526]

def split(df):
    """Делит таблицу по сезонам: старые → train, новые → test."""
    seasons = df[SEASON_COL].astype(int)
    train = df[seasons.isin(TRAIN_SEASONS)]
    test = df[seasons.isin(TEST_SEASONS)]
    return train, test

def scores(y_true, y_pred, y_proba=None):
    """Возвращает словарь метрик."""
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion": confusion_matrix(y_true, y_pred, labels=CLASSES),
    }
    if y_proba is not None:
        res["log_loss"] = log_loss(y_true, y_proba, labels=CLASSES)
    return res