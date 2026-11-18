#Прогон шести матчей кодом. Создайте файл src/toy_check.py:
from src.elo import elo_update

r = {'A': 1500.0, 'B': 1500.0, 'C': 1500.0, 'D': 1500.0}
matches = [('A', 'B', 'H'), ('C', 'D', 'D'), ('A', 'C', 'A'),
           ('B', 'D', 'H'), ('A', 'D', 'D'), ('B', 'C', 'A')]
for home, away, result in matches:
    r[home], r[away] = elo_update(r[home], r[away], result)
    print(home, away, result, {k: round(v, 1) for k, v in r.items()})
print(sum(r.values()))