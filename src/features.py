"""Единая точка сборки таблицы признаков: Эло -> форма -> matches_features.csv.

Сами расчёты лежат в elo_features.py и form_features.py (роль 3).
Здесь они только соединяются, проверяются и сохраняются одной командой.
Запуск из корня проекта:  py src/features.py
"""
import pandas as pd

from elo_features import build_elo_features
from form_features import add_form

SRC = "data/processed/matches_clean.csv"
OUT = "data/processed/matches_features.csv"

REQUIRED = ["date", "home", "away", "home_goals", "away_goals", "result"]
# признаки, на которых обучается модель (см. FEATS в predict.py): пропусков быть не должно
MODEL_FEATURES = ["elo_diff", "form_diff", "gd_diff", "h_elo", "a_elo"]
# эти колонки пустые у первых трёх матчей каждой команды (так и задумано)
MAY_BE_EMPTY = ["h_form", "a_form", "h_gd", "a_gd"]


def build_features(matches):
    """Чистая таблица матчей -> таблица с признаками Эло и формы."""
    missing = [c for c in REQUIRED if c not in matches.columns]
    if missing:
        raise ValueError(f"В таблице нет колонок: {missing}")
    return add_form(build_elo_features(matches))


def check_features(features, n_rows):
    """Проверки готовой таблицы. Возвращает число пустых значений, которые допустимы."""
    assert len(features) == n_rows, "число строк изменилось"
    for col in MODEL_FEATURES:
        assert col in features.columns, f"нет колонки {col}"
        assert features[col].notna().all(), f"пропуски в {col}"
    return int(features[MAY_BE_EMPTY].isna().sum().sum())


if __name__ == "__main__":
    matches = pd.read_csv(SRC)
    features = build_features(matches)
    empty = check_features(features, len(matches))
    features.to_csv(OUT, index=False)
    print(f"Готово: {len(features)} строк, {OUT}")
    print(f"Допустимых пустых значений (форма у первых матчей): {empty}")