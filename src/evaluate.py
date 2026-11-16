import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

FEATURES = ['h_form5', 'a_form5', 'h_gf5', 'a_gf5', 'h_ga5', 'a_ga5',
            'h_elo', 'a_elo', 'elo_diff']

TRAIN = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223']
VAL = ['2324']
TEST = ['2425', '2526']   # открываем ОДИН раз, на неделе 6

SEED = 42


def split(df, seasons):
    d = df[df.season.isin(seasons)]
    return d[FEATURES], d['result']


def scores(y, pred, proba=None, classes=None):
    out = {'accuracy': accuracy_score(y, pred),
           'macro_f1': f1_score(y, pred, average='macro')}
    out['log_loss'] = log_loss(y, proba, labels=classes) if proba is not None else np.nan
    return out


def elo_rule(X):
    # базовая стратегия: побеждает тот, у кого рейтинг выше (хозяевам +100)
    return np.where(X['h_elo'] + 100 > X['a_elo'], 'H', 'A')