from pathlib import Path

import pandas as pd
import pytest

from src.predict import build_state, DATA


@pytest.mark.skipif(not Path(DATA).exists(), reason="нет data/processed/matches_features.csv")
def test_state_equals_features_of_next_match():
    """Состояние команды из predict.build_state должно совпадать с признаками,
    которые таблица (обучение) посчитала для её СЛЕДУЮЩЕГО матча."""
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    checked_form = 0
    for k in range(300, len(df), 150):
        state = build_state(df.iloc[:k])
        nxt = df.iloc[k]
        h, a = state.get(nxt["home"]), state.get(nxt["away"])
        if h is None or a is None:
            continue  # команда ещё не играла

        # Эло: сравниваем с рейтингами, записанными ДО матча k
        assert abs(h["elo"] - nxt["h_elo"]) < 1e-6
        assert abs(a["elo"] - nxt["a_elo"]) < 1e-6

        # форма и разница голов: сравниваем, когда у обеих команд есть >= 5 матчей
        played = pd.concat([df.iloc[:k]["home"], df.iloc[:k]["away"]]).value_counts()
        if played[nxt["home"]] >= 5 and played[nxt["away"]] >= 5:
            assert abs((h["form"] - a["form"]) - nxt["form_diff"]) < 1e-6
            assert abs((h["gd"] - a["gd"]) - nxt["gd_diff"]) < 1e-6
            checked_form += 1

    assert checked_form > 0  # тест реально что-то проверил