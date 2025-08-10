"""
Главный файл запуска Dashboard приложения
"""
import sys
import os
from pathlib import Path

# Добавляем путь к исходному коду в PYTHONPATH
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Импортируем основное приложение
from dashboard.app import main

if __name__ == "__main__":
    main()
