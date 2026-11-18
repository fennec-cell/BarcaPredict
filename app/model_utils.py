"""Общие утилиты BarcaPredict: загрузка модели, данные и прогноз.

Работает с абсолютными путями относительно корня репозитория, поэтому
приложению не важен текущий рабочий каталог при запуске.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MODEL_PATH = REPO_ROOT / "models" / "barca_model.joblib"
MATCHES_PATH = REPO_ROOT / "data" / "processed" / "matches_clean.csv"

# Порядок признаков ДОЛЖЕН совпадать с обучением (src/predict.py).
FEATURES = ["elo_diff", "form_diff", "gd_diff", "h_elo", "a_elo"]
CLASS_LABELS = {"H": "Победа хозяев", "D": "Ничья", "A": "Победа гостей"}


@st.cache_resource(show_spinner=False)
def load_bundle():
    """Загрузить обученную модель и состояние команд (кэшируется)."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def model_available() -> bool:
    """Есть ли файл обученной модели."""
    return MODEL_PATH.exists()


def get_teams() -> list[str]:
    """Отсортированный список команд из состояния модели."""
    bundle = load_bundle()
    if not bundle:
        return []
    return sorted(bundle["state"].keys())


def predict_proba(home: str, away: str) -> dict[str, float]:
    """Вероятности исходов H/D/A для пары команд."""
    bundle = load_bundle()
    if not bundle:
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

    model = bundle["model"]
    state = bundle["state"]
    h, a = state[home], state[away]

    row = pd.DataFrame([
        {
            "elo_diff": h["elo"] - a["elo"],
            "form_diff": h["form"] - a["form"],
            "gd_diff": h["gd"] - a["gd"],
            "h_elo": h["elo"],
            "a_elo": a["elo"],
        }
    ])[FEATURES]

    proba = model.predict_proba(row)[0]
    return {cls: float(p) for cls, p in zip(model.classes_, proba)}


@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame:
    """Загрузить очищенные матчи (кэшируется)."""
    if not MATCHES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MATCHES_PATH, parse_dates=["date"])
