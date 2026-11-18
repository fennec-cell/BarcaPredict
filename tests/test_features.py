import numpy as np
import pandas as pd

from elo_features import build_elo_features


def fake_matches(n=200, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for i in range(n):
        home, away = rng.choice(teams, size=2, replace=False)
        rows.append({
            "date": pd.Timestamp("2023-08-15") + pd.Timedelta(days=i),  # по матчу в день
            "home": home,
            "away": away,
            "result": rng.choice(["H", "D", "A"]),
        })
    return pd.DataFrame(rows)


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
    m2 = m.copy()
    m2.loc[100, "result"] = "A" if m.loc[100, "result"] != "A" else "H"
    a = build_elo_features(m).loc[:100, ["h_elo", "a_elo"]]
    b = build_elo_features(m2).loc[:100, ["h_elo", "a_elo"]]
    assert np.allclose(a.values, b.values)
    # а вот после матча №100 рейтинги должны отличаться (иначе тест ничего не проверяет)
    a2 = build_elo_features(m).loc[101:, ["h_elo", "a_elo"]]
    b2 = build_elo_features(m2).loc[101:, ["h_elo", "a_elo"]]
    assert not np.allclose(a2.values, b2.values)