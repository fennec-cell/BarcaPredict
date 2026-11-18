import numpy as np
import pandas as pd

from elo_features import build_elo_features
from form_features import add_form


def fake_matches(n=200, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for i in range(n):
        home, away = rng.choice(teams, size=2, replace=False)
        hg, ag = int(rng.poisson(1.5)), int(rng.poisson(1.1))
        rows.append({
            "date": pd.Timestamp("2023-08-15") + pd.Timedelta(days=i),  # по матчу в день
            "home": home,
            "away": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": "H" if hg > ag else "D" if hg == ag else "A",
        })
    return pd.DataFrame(rows)


def change_result(m, i):
    """Копия таблицы, где в матче i результат и счёт заменены."""
    m2 = m.copy()
    m2.loc[i, ["home_goals", "away_goals", "result"]] = [0, 5, "A"] if m.loc[i, "result"] != "A" else [5, 0, "H"]
    return m2


# ---------- Эло ----------

def test_first_match_starts_at_1500():
    out = build_elo_features(fake_matches())
    assert out.loc[0, "h_elo"] == 1500
    assert out.loc[0, "a_elo"] == 1500


def test_elo_diff_is_home_minus_away():
    out = build_elo_features(fake_matches())
    assert np.allclose(out["elo_diff"], out["h_elo"] - out["a_elo"])


def test_elo_no_leak():
    # меняем результат матча №100: рейтинги в матчах 0..100 не должны измениться,
    # потому что рейтинг записывается ДО матча
    m = fake_matches()
    m2 = change_result(m, 100)
    a = build_elo_features(m).loc[:100, ["h_elo", "a_elo"]]
    b = build_elo_features(m2).loc[:100, ["h_elo", "a_elo"]]
    assert np.allclose(a.values, b.values)
    # а после матча №100 рейтинги должны отличаться (иначе тест ничего не проверяет)
    a2 = build_elo_features(m).loc[101:, ["h_elo", "a_elo"]]
    b2 = build_elo_features(m2).loc[101:, ["h_elo", "a_elo"]]
    assert not np.allclose(a2.values, b2.values)


# ---------- Форма ----------

def test_form_first_three_matches_are_empty():
    # у каждой команды первые три матча без формы (NaN), с четвёртого форма есть
    out = add_form(fake_matches())
    seen = {}
    for r in out.itertuples():
        for team, form in ((r.home, r.h_form), (r.away, r.a_form)):
            seen[team] = seen.get(team, 0) + 1
            if seen[team] <= 3:
                assert pd.isna(form)
            else:
                assert not pd.isna(form)


def test_form_no_leak():
    # меняем результат и счёт матча №100: форма в матчах 0..100 не должна измениться
    cols = ["h_form", "a_form", "h_gd", "a_gd", "form_diff", "gd_diff"]
    m = fake_matches()
    m2 = change_result(m, 100)
    a = add_form(m).loc[:100, cols]
    b = add_form(m2).loc[:100, cols]
    assert np.allclose(a.values, b.values, equal_nan=True)
    # после матча №100 форма должна отличаться (иначе тест ничего не проверяет)
    a2 = add_form(m).loc[101:, cols]
    b2 = add_form(m2).loc[101:, cols]
    assert not np.allclose(a2.values, b2.values, equal_nan=True)


def test_form_value_example():
    # ручной пример: X играет дома 4 раза, в первых трёх матчах получает 3, 1, 0 очков
    rows = []
    for i, (h, a, hg, ag) in enumerate([
        ("X", "A", 1, 0), ("X", "B", 1, 1), ("X", "C", 0, 2), ("X", "D", 2, 0),
    ]):
        rows.append({"date": pd.Timestamp("2023-08-15") + pd.Timedelta(days=i),
                     "home": h, "away": a, "home_goals": hg, "away_goals": ag,
                     "result": "H" if hg > ag else "D" if hg == ag else "A"})
    out = add_form(pd.DataFrame(rows))
    # форма в 4-м матче = (3 + 1 + 0) / 3
    assert abs(out.loc[3, "h_form"] - 4 / 3) < 1e-9

# ---------- Сборка таблицы (src/features.py) ----------

def test_build_features_equals_two_steps():
    from features import build_features, check_features
    m = fake_matches()
    out = build_features(m)
    expected = add_form(build_elo_features(m))
    pd.testing.assert_frame_equal(out, expected)
    # пропусков нет в признаках модели, форма пустая только у первых матчей команд
    empty = check_features(out, n_rows=len(m))
    assert empty > 0


def test_build_features_needs_columns():
    import pytest
    from features import build_features
    with pytest.raises(ValueError):
        build_features(fake_matches().drop(columns=["home_goals"]))