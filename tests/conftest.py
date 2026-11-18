import sys
from pathlib import Path

# скрипты роли 3 импортируют соседние файлы как `from elo import ...`,
# поэтому добавляем папку src в путь поиска модулей
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))