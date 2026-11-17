import pandas as pd

def add_form(df, n=5):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["mid"] = df.index

    pts_h = df["result"].map({"H": 3, "D": 1, "A": 0})
    pts_a = df["result"].map({"H": 0, "D": 1, "A": 3})
    home = pd.DataFrame({"mid": df["mid"], "team": df["home"], "pts": pts_h,
                         "gd": df["home_goals"] - df["away_goals"]})
    away = pd.DataFrame({"mid": df["mid"], "team": df["away"], "pts": pts_a,
                         "gd": df["away_goals"] - df["home_goals"]})
    long = pd.concat([home, away]).sort_values("mid")

    g = long.groupby("team")
    for col, name in [("pts", "form"), ("gd", "gdf")]:
        long[name] = g[col].transform(
            lambda s: s.shift(1).rolling(n, min_periods=3).mean())

    f = long[["mid", "team", "form", "gdf"]]
    df = df.merge(f.rename(columns={"team": "home", "form": "h_form", "gdf": "h_gd"}),
                  on=["mid", "home"], how="left")
    df = df.merge(f.rename(columns={"team": "away", "form": "a_form", "gdf": "a_gd"}),
                  on=["mid", "away"], how="left")
    df["form_diff"] = (df["h_form"] - df["a_form"]).fillna(0)
    df["gd_diff"] = (df["h_gd"] - df["a_gd"]).fillna(0)
    return df.drop(columns="mid")

if __name__ == "__main__":
    df = pd.read_csv("data/processed/matches_elo.csv")
    out = add_form(df)
    out.to_csv("data/processed/matches_features.csv", index=False)
    print(out[["home", "away", "h_form", "form_diff"]].head(8))