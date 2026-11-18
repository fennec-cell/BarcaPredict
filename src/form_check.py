import pandas as pd

pts = pd.Series([3, 1, 0, 3, 3, 0, 1, 3])   # очки одной команды в 8 матчах
form = pts.shift(1).rolling(5, min_periods=3).mean()
print(form.round(3).tolist())
