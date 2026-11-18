# BarcaPredict

Учебный проект: прогноз вероятностей исхода матчей испанской Ла Лиги.
Веб-интерфейс на **Streamlit**, модель — логистическая регрессия на признаках
Elo, формы за последние матчи и разницы забитых/пропущенных мячей.

## Быстрый старт

1. Установить зависимости:

   ```bash
   pip install -r requirements.txt
   ```

2. Обучить модель (если файла `models/barca_model.joblib` нет). Из корня репозитория:

   ```bash
   # Windows (cmd)
   set PYTHONPATH=src
   py src/elo_features.py
   py src/form_features.py
   py src/predict.py
   ```

   > В PowerShell вместо `set PYTHONPATH=src` используйте `$env:PYTHONPATH="src"`.

3. Запустить интерфейс:

   ```bash
   streamlit run app/app.py
   ```

4. Открыть в браузере: <http://localhost:8501>

## Страницы приложения

- **Прогноз** — выбор хозяев и гостей, вероятности победы хозяев / ничьей / победы гостей.
- **История** — статистика Barcelona по сезонам из `data/processed/matches_clean.csv`.
- **О проекте** — цель, роли, технологии и источники данных.

## Структура проекта

```
barca-predict/
├── app/
│   ├── app.py               # роутер страниц (st.navigation)
│   ├── model_utils.py       # загрузка модели, прогноз, данные
│   └── pages/
│       ├── 1_prediction.py  # страница «Прогноз»
│       ├── 2_history.py     # страница «История»
│       └── 3_about.py       # страница «О проекте»
├── data/
│   ├── raw/                 # SP1_*.csv (football-data.co.uk)
│   └── processed/           # matches_clean.csv, matches_features.csv
├── models/                  # barca_model.joblib (модель + состояние команд)
├── notebooks/               # исследование данных
├── reports/figures/         # графики EDA
├── src/                     # подготовка данных, признаки, модель
├── tests/                   # тесты
├── docs/                    # документация
├── requirements.txt
└── README.md
```

## Роли проекта

- **Роль 1** — подготовка и очистка данных.
- **Роль 2** — формирование признаков (Elo, форма, голы).
- **Роль 3** — обучение и оценка моделей.
- **Роль 4** — интерфейс Streamlit и документация.

## Данные

Результаты матчей Ла Лиги берутся с [football-data.co.uk](https://www.football-data.co.uk/)
(файлы `SP1_*.csv`), затем очищаются в `data/processed/matches_clean.csv`.

## Дисклеймер

⚠️ Учебная модель: результаты не являются рекомендацией для ставок.
