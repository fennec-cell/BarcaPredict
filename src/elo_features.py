import pandas as pd
from elo import elo_expected, elo_update

START = 1500
HOME_ADV = 100

def build_elo_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    ratings = {}
    h_elo, a_elo, exp_home = [], [], []

    for home, away, result in zip(df["home"], df["away"], df["result"]):
        rh = ratings.get(home, START)
        ra = ratings.get(away, START)

        # признаки: рейтинги ДО матча
        h_elo.append(rh)
        a_elo.append(ra)
        exp_home.append(elo_expected(rh, ra, HOME_ADV))

        # и только потом обновляем
        ratings[home], ratings[away] = elo_update(
            rh, ra, result, k=20, home_adv=HOME_ADV
        )

    df["h_elo"] = h_elo
    df["a_elo"] = a_elo
    df["elo_diff"] = df["h_elo"] - df["a_elo"]
    df["elo_exp"] = exp_home
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/matches_clean.csv")
    out = build_elo_features(df)
    out.to_csv("data/processed/matches_elo.csv", index=False)
    print(out[["date", "home", "away", "h_elo", "a_elo", "elo_diff"]].head())
    print(out["h_elo"].iloc[0], out["a_elo"].iloc[0])  # должно быть 1500 1500