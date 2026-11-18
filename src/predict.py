import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA = "data/processed/matches_features.csv"
MODEL_PATH = Path("models/barca_model.joblib")
FEATS = ["elo_diff", "form_diff", "gd_diff", "h_elo", "a_elo"]
K = 20

def points(gf, ga):
    return 3 if gf > ga else (1 if gf == ga else 0)

def build_state(df):
    """Текущее состояние каждой команды: Elo, форма (5 матчей), разница голов."""
    df = df.sort_values("date")
    elo, rows = {}, {}
    for r in df.itertuples():
        # Elo ПОСЛЕ матча = Elo до матча + обновление
        s = 1.0 if r.result == "H" else (0.5 if r.result == "D" else 0.0)
        delta = K * (s - r.elo_exp)
        elo[r.home] = r.h_elo + delta
        elo[r.away] = r.a_elo - delta
        rows.setdefault(r.home, []).append(
            (points(r.home_goals, r.away_goals), r.home_goals - r.away_goals))
        rows.setdefault(r.away, []).append(
            (points(r.away_goals, r.home_goals), r.away_goals - r.home_goals))
    state = {}
    for team, lst in rows.items():
        last = lst[-5:]
        state[team] = {
            "elo": elo[team],
            "form": sum(p for p, _ in last) / len(last),
            "gd": sum(g for _, g in last) / len(last),
        }
    return state

def train():
    df = pd.read_csv(DATA)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(df[FEATS], df["result"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "state": build_state(df)}, MODEL_PATH)
    print("Модель обучена и сохранена:", MODEL_PATH)

def predict(home, away):
    bundle = joblib.load(MODEL_PATH)
    model, st = bundle["model"], bundle["state"]
    h, a = st[home], st[away]
    x = pd.DataFrame([{
        "elo_diff": h["elo"] - a["elo"],
        "form_diff": h["form"] - a["form"],
        "gd_diff": h["gd"] - a["gd"],
        "h_elo": h["elo"],
        "a_elo": a["elo"],
    }])[FEATS]
    proba = model.predict_proba(x)[0]
    return {c: float(p) for c, p in zip(model.classes_, proba)}

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        train()
    teams = sorted(joblib.load(MODEL_PATH)["state"].keys())
    print("Команд:", len(teams))
    print(teams)
    res = predict("Barcelona", "Real Madrid")
    print("Barcelona - Real Madrid:", res)
    print("Сумма:", round(sum(res.values()), 6))
    