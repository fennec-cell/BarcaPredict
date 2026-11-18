import sys
from pathlib import Path

# Добавляем каталог app/ в путь, чтобы импортировать model_utils из любой страницы.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from model_utils import CLASS_LABELS, get_teams, model_available, predict_proba

st.title("⚽ Прогноз матча")
st.write("Выберите хозяев и гостей — модель оценит вероятности исхода матча Ла Лиги.")

if not model_available():
    st.error(
        "Модель не найдена. Обучите её один раз командой:\n\n"
        "```\npy src/predict.py\n```"
    )
    st.stop()

teams = get_teams()
if not teams:
    st.error("Не удалось загрузить список команд из модели.")
    st.stop()

default_home = "Barcelona" if "Barcelona" in teams else teams[0]
default_away = "Real Madrid" if "Real Madrid" in teams else teams[min(1, len(teams) - 1)]

col_home, col_away = st.columns(2)
with col_home:
    home = st.selectbox("Хозяева", teams, index=teams.index(default_home))
with col_away:
    away = st.selectbox("Гости", teams, index=teams.index(default_away))

if home == away:
    st.warning("Хозяева и гости не могут быть одной командой. Выберите разные команды.")
elif st.button("Спрогнозировать", type="primary", use_container_width=True):
    result = predict_proba(home, away)

    st.markdown(f"### {home} — {away}")
    order = ["H", "D", "A"]
    columns = st.columns(3)
    for col, cls in zip(columns, order):
        prob = result.get(cls, 0.0)
        with col:
            st.metric(CLASS_LABELS[cls], f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))

    best = max(order, key=lambda key: result.get(key, 0.0))
    st.success(f"Наиболее вероятный исход: {CLASS_LABELS[best]} ({result[best]:.1%})")
    st.caption("Учебная модель, не предназначена для ставок.")

with st.expander("Как это работает?"):
    st.write(
        "Модель (логистическая регрессия) учитывает рейтинг Elo команд, форму за "
        "последние 5 матчей и разницу забитых/пропущенных мячей. Вероятности "
        "нормированы: их сумма всегда равна 100%."
    )
