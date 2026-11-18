import streamlit as st

st.set_page_config(
    page_title="BarcaPredict",
    page_icon="⚽",
    layout="wide",
)

pages = [
    st.Page("pages/1_prediction.py", title="Прогноз", icon="⚽", default=True),
    st.Page("pages/2_history.py", title="История", icon="📊"),
    st.Page("pages/3_about.py", title="О проекте", icon="ℹ️"),
]

pg = st.navigation(pages)
pg.run()
