import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from model_utils import load_matches

st.title("📊 История Барселоны")
st.write("Статистика клуба Barcelona по данным `matches_clean.csv`.")

df = load_matches()
if df.empty:
    st.warning("Данные не найдены: `data/processed/matches_clean.csv`.")
    st.stop()

team = "Barcelona"
barca = df[(df["home"] == team) | (df["away"] == team)].copy()


def match_outcome(row) -> str:
    """Итог матча для Barcelona: W — победа, D — ничья, L — поражение."""
    if row["home"] == team:
        return "W" if row["result"] == "H" else ("D" if row["result"] == "D" else "L")
    return "W" if row["result"] == "A" else ("D" if row["result"] == "D" else "L")


barca["outcome"] = barca.apply(match_outcome, axis=1)

wins = int((barca["outcome"] == "W").sum())
draws = int((barca["outcome"] == "D").sum())
losses = int((barca["outcome"] == "L").sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Матчей", len(barca))
col2.metric("Победы", wins)
col3.metric("Ничьи", draws)
col4.metric("Поражения", losses)

st.markdown("### Победы по сезонам")
wins_by_season = (
    barca.groupby("season")["outcome"].apply(lambda series: int((series == "W").sum()))
)
wins_by_season.index = wins_by_season.index.astype(str)
st.bar_chart(wins_by_season, y_label="Победы")

st.markdown("### Последние 10 матчей")
recent = barca.sort_values("date").tail(10).copy()
recent["Матч"] = recent["home"] + " — " + recent["away"]
recent["Счёт"] = (
    recent["home_goals"].astype(int).astype(str)
    + " : "
    + recent["away_goals"].astype(int).astype(str)
)
recent["Итог"] = recent["outcome"].map({"W": "Победа", "D": "Ничья", "L": "Поражение"})
table = recent[["date", "Матч", "Счёт", "Итог"]].rename(columns={"date": "Дата"})
st.dataframe(table, use_container_width=True, hide_index=True)
