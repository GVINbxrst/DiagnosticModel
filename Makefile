# DiagMod - Система диагностики асинхронных двигателей
# Makefile для Windows PowerShell и Unix систем

.PHONY: help install install-dev setup test lint format clean docker-build docker-up docker-down

# Определяем операционную систему
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    PYTHON := python
    PIP := pip
    VENV_ACTIVATE := venv\Scripts\Activate.ps1
    SHELL_CMD := powershell.exe -ExecutionPolicy Bypass -Command
    SETUP_SCRIPT := .\scripts\setup.ps1
    RUN_API := .\scripts\run-api.ps1
    RUN_WORKER := .\scripts\run-worker.ps1
    RUN_DASHBOARD := .\scripts\run-dashboard.ps1
else
    DETECTED_OS := Unix
    PYTHON := python3
    PIP := pip3
    VENV_ACTIVATE := source venv/bin/activate
    SHELL_CMD :=
    SETUP_SCRIPT := ./scripts/setup.sh
    RUN_API := ./scripts/run_api.sh
    RUN_WORKER := ./scripts/run_worker.sh
    RUN_DASHBOARD := ./scripts/run_dashboard.sh
endif

DOCKER_COMPOSE := docker-compose

# Цвета для вывода (только для Unix)
ifneq ($(OS),Windows_NT)
    RED := \033[0;31m
    GREEN := \033[0;32m
    YELLOW := \033[0;33m
    BLUE := \033[0;34m
    NC := \033[0m
endif

help: ## Показать справку по командам
ifeq ($(OS),Windows_NT)
	@echo "DiagMod - Система диагностики асинхронных двигателей (Windows)"
	@echo ""
	@echo "Доступные команды:"
	@echo "  install          Установить зависимости для продакшена"
	@echo "  install-dev      Установить все зависимости"
	@echo "  setup            Полная настройка среды разработки"
	@echo "  venv             Создать виртуальное окружение"
	@echo "  test             Запустить тесты"
	@echo "  lint             Проверка кода"
	@echo "  format           Форматирование кода"
	@echo "  docker-build     Собрать Docker образы"
	@echo "  docker-up        Запустить все сервисы"
	@echo "  docker-down      Остановить сервисы"
	@echo "  dev-api          Запустить API в режиме разработки"
	@echo "  dev-worker       Запустить Worker"
	@echo "  dev-dashboard    Запустить Dashboard"
	@echo "  process-raw      Обработать все необработанные RawSignal"
	@echo "  anomalies        Запустить детекцию аномалий для новых признаков"
	@echo "  forecast         Прогноз трендов для активного оборудования"
	@echo "  report           Сводный аналитический отчёт"
	@echo "  ingest           Загрузить локальные CSV файлы в RawSignal"
	@echo "  pipeline         Полный E2E конвейер"
	@echo "  clean            Очистить временные файлы"
	@echo "  info             Информация о проекте"
else
	@echo "$(BLUE)DiagMod - Система диагностики асинхронных двигателей$(NC)"
	@echo ""
	@echo "$(YELLOW)Доступные команды:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
endif

# =============================================================================
# Установка и настройка
# =============================================================================

install: ## Установить зависимости для продакшена
ifeq ($(OS),Windows_NT)
	@echo "Установка продакшен зависимостей..."
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "Зависимости установлены"
else
	@echo "$(BLUE)Установка продакшен зависимостей...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Зависимости установлены$(NC)"
endif

install-dev: ## Установить все зависимости (продакшен + разработка)
ifeq ($(OS),Windows_NT)
	@echo "Установка всех зависимостей..."
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "Все зависимости установлены"
else
	@echo "$(BLUE)Установка всех зависимостей...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)✓ Все зависимости установлены$(NC)"
endif

setup: ## Полная настройка среды разработки
ifeq ($(OS),Windows_NT)
	$(SHELL_CMD) "$(SETUP_SCRIPT) -Dev"
else
	$(SETUP_SCRIPT) --dev
endif

venv: ## Создать виртуальное окружение
ifeq ($(OS),Windows_NT)
	@echo "Создание виртуального окружения..."
	$(PYTHON) -m venv venv
	@echo "Виртуальное окружение создано"
	@echo "Активируйте его командой: venv\Scripts\Activate.ps1"
else
	@echo "$(BLUE)Создание виртуального окружения...$(NC)"
	$(PYTHON) -m venv venv
	@echo "$(GREEN)✓ Виртуальное окружение создано$(NC)"
	@echo "$(YELLOW)Активируйте его командой: source venv/bin/activate$(NC)"
endif

# =============================================================================
# Тестирование и качество кода
# =============================================================================

test: ## Запустить тесты
ifeq ($(OS),Windows_NT)
	@echo "Запуск тестов..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "Тесты завершены"
else
	@echo "$(BLUE)Запуск тестов...$(NC)"
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Тесты завершены$(NC)"
endif

lint: ## Проверка кода линтерами
ifeq ($(OS),Windows_NT)
	@echo "Проверка кода линтерам��..."
	flake8 src/ tests/
	mypy src/
	@echo "Проверка завершена"
else
	@echo "$(BLUE)Проверка кода линтерами...$(NC)"
	flake8 src/ tests/
	mypy src/
	pylint src/
	@echo "$(GREEN)✓ Проверка линтерами завершена$(NC)"
endif

format: ## Форматирование кода
ifeq ($(OS),Windows_NT)
	@echo "Форматирование кода..."
	black src/ tests/
	isort src/ tests/
	@echo "Код отформатирован"
else
	@echo "$(BLUE)Форматирование кода...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)✓ Код отформатирован$(NC)"
endif

# =============================================================================
# Docker операции
# =============================================================================

docker-build: ## Собрать Docker образы
	@echo "Сборка Docker образов..."
	$(DOCKER_COMPOSE) build
	@echo "Docker образы собраны"

docker-up: ## Запустить все сервисы в Docker
	@echo "Запуск сервисов в Docker..."
	$(DOCKER_COMPOSE) up -d
	@echo "Сервисы запущены"
	@echo "API доступно по адресу: http://localhost:8000"
	@echo "Dashboard доступен по адресу: http://localhost:8501"
	@echo "Prometheus доступен по адресу: http://localhost:9090"

docker-down: ## Остановить все сервисы Docker
	@echo "Остановка сервисов Docker..."
	$(DOCKER_COMPOSE) down
	@echo "Сервисы остановлены"

docker-logs: ## Показать логи всех сервисов
	$(DOCKER_COMPOSE) logs -f

docker-clean: ## Очистить Docker ресурсы
	@echo "Очистка Docker ресурсов..."
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f
	@echo "Docker ресурсы очищены"

# =============================================================================
# База данных
# =============================================================================

db-init: ## Инициализация базы данных
	@echo "Инициализация базы данных..."
	$(PYTHON) -m alembic upgrade head
	@echo "База данных инициализирована"

db-migrate: ## Создать новую миграцию
	@echo "Создание миграции..."
	$(PYTHON) -m alembic revision --autogenerate -m "$(MSG)"
	@echo "Миграция создана"

db-upgrade: ## Применить миграции
	@echo "Применение миграций..."
	$(PYTHON) -m alembic upgrade head
	@echo "Миграции применены"

create-db: ## Создать PostgreSQL базу данных (использует скрипт create_database.py)
	$(PYTHON) scripts/create_database.py --db $(or $(DB),diagmod) --user $(or $(PGUSER,postgres)) --password $(or $(PGPASSWORD,postgres)) --host $(or $(PGHOST,localhost)) --port $(or $(PGPORT,5432))

init-db: ## Инициализация БД (DROP=1 для очистки, SEED=1 для начальных данных)
	$(PYTHON) scripts/init_db.py $(if $(DROP),--drop,) $(if $(SEED),--seed,)

# =============================================================================
# Разработка
# =============================================================================

dev-api: ## Запустить API в режиме разработки
ifeq ($(OS),Windows_NT)
	$(SHELL_CMD) "$(RUN_API)"
else
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
endif

dev-worker: ## Запустить Worker в режиме разработки
ifeq ($(OS),Windows_NT)
	$(SHELL_CMD) "$(RUN_WORKER)"
else
	celery -A src.worker.celery_app worker --loglevel=info
endif

dev-dashboard: ## Запустить Dashboard в режиме разработки
ifeq ($(OS),Windows_NT)
	$(SHELL_CMD) "$(RUN_DASHBOARD)"
else
	streamlit run src/dashboard/main.py
endif

dev-all: ## Запустить все компоненты в режиме разработки
	@echo "Запуск всех компонентов..."
	make docker-up
	@echo "Все компоненты запущены в Docker"

# =============================================================================
# Batch / Pipeline
# =============================================================================

process-raw: ## Обработать все необработанные RawSignal (локально, eager)
	set CELERY_TASK_ALWAYS_EAGER=1 && $(PYTHON) scripts/process_all_rawsignals.py

anomalies: ## Детектировать аномалии для признаков без Prediction
	set CELERY_TASK_ALWAYS_EAGER=1 && $(PYTHON) scripts/run_anomaly_detection_all.py

forecast: ## Прогноз трендов для активного оборудования
	set CELERY_TASK_ALWAYS_EAGER=1 && $(PYTHON) scripts/forecast_all_equipment.py

report: ## Построить сводный отчёт
	$(PYTHON) scripts/analytics_report.py

report-html: ## Построить HTML отчёт (reports/latest_report.html)
	$(PYTHON) scripts/build_html_report.py

ingest: ## Загрузить локальные CSV (DIR=каталог PATTERN=*.csv REC=1 LIMIT=N)
	@if not "$(DIR)"=="" ( set CELERY_TASK_ALWAYS_EAGER=1 && $(PYTHON) scripts/ingest_local_csv.py --path "$(DIR)" --pattern "$(PATTERN)" $(if $(REC),--recursive,) $(if $(LIMIT),--limit $(LIMIT),) ) else ( echo "Укажите DIR= каталог" )

pipeline: ## Полный E2E конвейер (обработка -> модель -> аномалии -> прогноз -> отчёт)
	set CELERY_TASK_ALWAYS_EAGER=1 && $(PYTHON) scripts/e2e_pipeline.py --limit 50

# =============================================================================
# Очистка и служебные команды
# =============================================================================

clean: ## Очистить временные файлы
ifeq ($(OS),Windows_NT)
	@echo "Очистка временных файлов..."
	@if exist "**\*.pyc" del /s /q "**\*.pyc" 2>nul
	@for /d /r %%i in (__pycache__) do @if exist "%%i" rmdir /s /q "%%i" 2>nul
	@if exist ".coverage" del ".coverage" 2>nul
	@if exist "htmlcov" rmdir /s /q "htmlcov" 2>nul
	@if exist ".pytest_cache" rmdir /s /q ".pytest_cache" 2>nul
	@if exist ".mypy_cache" rmdir /s /q ".mypy_cache" 2>nul
	@echo "Временные файлы очищены"
else
	@echo "$(BLUE)Очистка временных файлов...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "$(GREEN)✓ Временные файлы очищены$(NC)"
endif

# =============================================================================
# Информация о проекте
# =============================================================================

info: ## Показать информацию о проекте
	@echo "DiagMod - Система диагностики асинхронных двигателей"
	@echo ""
	@echo "Операционная система: $(DETECTED_OS)"
	@echo ""
	@echo "Структура проекта:"
	@echo "  • src/api/         - FastAPI веб-сервис"
	@echo "  • src/worker/      - Celery задачи"
	@echo "  • src/dashboard/   - Streamlit интерфейс"
	@echo "  • src/ml/          - ML модели"
	@echo "  • src/database/    - Модели БД"
	@echo ""
	@echo "Основные порты:"
	@echo "  • 8000  - FastAPI API"
	@echo "  • 8501  - Streamlit Dashboard"
	@echo "  • 9090  - Prometheus"
	@echo "  • 5432  - PostgreSQL"
	@echo "  • 6379  - Redis"

status: ## Показать статус всех сервисов
	@echo "Статус сервисов:"
	@$(DOCKER_COMPOSE) ps 2>nul || echo "Docker Compose недоступен"

# По умолчанию показываем справку
.DEFAULT_GOAL := help
