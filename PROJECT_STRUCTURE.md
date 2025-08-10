# Структура проекта токовой диагностики асинхронных двигателей

## 1. Полная иерархия папок и файлов

```
DiagMod/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.worker
├── Dockerfile.dashboard
├── Makefile
│
├── src/                           # Основной исходный код
│   ├── __init__.py
│   ├── config/                    # Конфигурации
│   │   ├── __init__.py
│   │   ├── settings.py           # Основные настройки приложения
│   │   ├── database.py           # Настройки БД
│   │   ├── celery_config.py      # Настройки Celery
│   │   ├── logging_config.py     # Настройки логирования
│   │   ├── monitoring.py         # Настройки Prometheus/Grafana
│   │   └── security.py           # JWT, шифрование
│   │
│   ├── database/                  # Работа с базой данных
│   │   ├── __init__.py
│   │   ├── models.py             # SQLAlchemy модели
│   │   ├── schemas.py            # Pydantic схемы
│   │   ├── connection.py         # Подключение к БД
│   │   └── migrations/           # Alembic миграции
│   │       ├── versions/
│   │       ├── env.py
│   │       ├── script.py.mako
│   │       └── alembic.ini
│   │
│   ├── data_processing/           # Обработка данных
│   │   ├── __init__.py
│   │   ├── csv_loader.py         # Загрузка CSV файлов
│   │   ├── data_validator.py     # Валидация данных
│   │   ├── feature_extractor.py  # Извлечение признаков
│   │   ├── signal_processing.py  # Обработка сигналов
│   │   └── utils.py              # Утилиты обработки
│   │
│   ├── ml/                       # Машинное обучение
│   │   ├── __init__.py
│   │   ├── anomaly_detection.py  # Модели обнаружения аномалий
│   │   ├── prediction.py         # Модели прогнозирования
│   │   ├── feature_engineering.py # Инженерия признаков
│   │   ├── model_training.py     # Обучение моделей
│   │   ├── model_evaluation.py   # Оценка моделей
│   │   └── model_registry.py     # Реестр моделей
│   │
│   ├── api/                      # FastAPI приложение
│   │   ├── __init__.py
│   │   ├── main.py              # Основное FastAPI приложение
│   │   ├── dependencies.py      # Зависимости API
│   │   ├── middleware.py        # Middleware (аутентификация, CORS)
│   │   └── routes/              # API роуты
│   │       ├── __init__.py
│   │       ├── auth.py          # Аутентификация
│   │       ├── data.py          # Работа с данными
│   │       ├── models.py        # Управление моделями
│   │       ├── diagnostics.py   # Диагностика двигателей
│   │       └── monitoring.py    # Мониторинг системы
│   │
│   ├── worker/                   # Celery задачи
│   │   ├── __init__.py
│   │   ├── celery_app.py        # Celery приложение
│   │   ├── tasks/               # Задачи
│   │   │   ├── __init__.py
│   │   │   ├── data_tasks.py    # Задачи обработки данных
│   │   │   ├── ml_tasks.py      # Задачи ML
│   │   │   └── monitoring_tasks.py # Задачи мониторинга
│   │   └── utils.py             # Утилиты worker
│   │
│   ├── dashboard/               # Streamlit дашборд
│   │   ├── __init__.py
│   │   ├── main.py             # Основное приложение Streamlit
│   │   ├── pages/              # Страницы дашборда
│   │   │   ├── __init__.py
│   │   │   ├── data_overview.py # Обзор данных
│   │   │   ├── diagnostics.py   # Диагностика
│   │   │   ├── monitoring.py    # Мониторинг
│   │   │   └── admin.py         # Администрирование
│   │   ├── components/          # Компоненты UI
│   │   │   ├── __init__.py
│   │   │   ├── charts.py        # Графики
│   │   │   ├── tables.py        # Таблицы
│   │   │   └── widgets.py       # Виджеты
│   │   └── utils.py            # Утилиты дашборда
│   │
│   └── utils/                   # Общие утилиты
│       ├── __init__.py
│       ├── logger.py           # Логирование
│       ├── security.py         # Безопасность
│       ├── exceptions.py       # Исключения
│       └── helpers.py          # Вспомогательные функции
│
├── data/                        # Директория данных
│   ├── raw/                    # Сырые CSV файлы
│   ├── processed/              # Обработанные данные
│   ├── features/               # Извлеченные признаки
│   └── exports/                # Экспорты данных
│
├── models/                      # Сохраненные ML модели
│   ├── anomaly_detection/      # Модели обнаружения аномалий
│   │   ├── v1.0.0/
│   │   │   ├── model.pkl
│   │   │   ├── scaler.pkl
│   │   │   ├── metadata.json
│   │   │   └── config.yaml
│   │   └── latest/             # Симлинк на последнюю версию
│   │
│   ├── prediction/             # Модели прогнозирования
│   │   ├── v1.0.0/
│   │   │   ├── model.pkl
│   │   │   ├── scaler.pkl
│   │   │   ├── metadata.json
│   │   │   └── config.yaml
│   │   └── latest/
│   │
│   └── registry.json           # Реестр всех моделей
│
├── sql/                        # SQL схемы и скрипты
│   ├── schema/                 # Схемы БД
│   │   ├── 001_initial.sql
│   │   ├── 002_motor_data.sql
│   │   ├── 003_features.sql
│   │   └── 004_predictions.sql
│   ├── procedures/             # Хранимые процедуры
│   ├── views/                  # Представления
│   └── seed/                   # Начальные данные
│
├── scripts/                    # Скрипты автоматизации
│   ├── setup_database.py      # Настройка БД
│   ├── load_csv_data.py       # Загрузка CSV данных
│   ├── train_models.py        # Обучение моделей
│   ├── backup_data.py         # Резервное копирование
│   └── monitoring_setup.py    # Настройка мониторинга
│
├── configs/                    # Конфигурационные файлы
│   ├── development.env         # Конфиг разработки
│   ├── production.env          # Конфиг продакшена
│   ├── logging.yaml           # Конфиг логирования
│   ├── prometheus.yml         # Конфиг Prometheus
│   ├── grafana/               # Конфиги Grafana
│   │   ├── dashboards/
│   │   └── datasources/
│   └── nginx/                 # Конфиги Nginx
│       └── nginx.conf
│
├── docker/                     # Docker файлы
│   ├── api/
│   │   └── Dockerfile
│   ├── worker/
│   │   └── Dockerfile
│   ├── dashboard/
│   │   └── Dockerfile
│   ├── postgres/
│   │   ├── Dockerfile
│   │   └── init.sql
│   └── redis/
│       └── Dockerfile
│
├── tests/                      # Тесты
│   ├── __init__.py
│   ├── conftest.py            # Конфигурация pytest
│   ├── unit/                  # Юнит-тесты
│   │   ├── test_data_processing.py
│   │   ├── test_ml.py
│   │   ├── test_api.py
│   │   └── test_worker.py
│   ├── integration/           # Интеграционные тесты
│   │   ├── test_api_integration.py
│   │   └── test_database.py
│   └── fixtures/              # Тестовые данные
│       └── sample_data.csv
│
├── docs/                       # Документация
│   ├── api/                   # API документация
│   ├── deployment/            # Документация по развертыванию
│   ├── user_guide/            # Руководство пользователя
│   ├── architecture.md        # Архитектура системы
│   ├── data_format.md         # Формат данных
│   └── ml_models.md           # Описание ML моделей
│
└── monitoring/                 # Мониторинг и метрики
    ├── prometheus/
    │   └── rules/
    ├── grafana/
    │   └── dashboards/
    └── alerts/
```

## 2. Назначение папок и файлов

### src/ - Основной исходный код
- **config/**: Централизованные настройки всех компонентов
- **database/**: Модели данных, схемы, подключения к БД
- **data_processing/**: Обработка CSV, валидация, извлечение признаков
- **ml/**: Все ML компоненты - модели, обучение, оценка
- **api/**: FastAPI приложение с роутами
- **worker/**: Celery задачи для фоновой обработки
- **dashboard/**: Streamlit интерфейс
- **utils/**: Общие утилиты для всех модулей

### data/ - Данные
- **raw/**: Исходные CSV файлы по двигателям
- **processed/**: Очищенные и подготовленные данные
- **features/**: Извлеченные признаки для ML
- **exports/**: Экспорты для анализа

### models/ - ML модели
- Версионирование моделей по семантическому версионированию
- Каждая версия содержит: модель, скейлер, метаданные, конфиг
- **latest/**: Симлинки на актуальные версии

### sql/ - База данных
- **schema/**: DDL скрипты создания таблиц
- **procedures/**: Хранимые процедуры для сложной логики
- **views/**: Представления для отчетности
- **seed/**: Начальные данные

### scripts/ - Автоматизация
- Скрипты для развертывания, загрузки данных, обучения
- Утилиты администрирования и мониторинга

## 3. Расположение компонентов

### SQL схемы
- `sql/schema/` - DDL скрипты
- `src/database/models.py` - SQLAlchemy модели
- `src/database/migrations/` - Alembic миграции

### Скрипты загрузки CSV
- `src/data_processing/csv_loader.py` - основной модуль
- `scripts/load_csv_data.py` - CLI скрипт
- `src/worker/tasks/data_tasks.py` - асинхронные задачи

### Извлечение признаков
- `src/data_processing/feature_extractor.py` - FFT, RMS, kurtosis, skewness
- `src/data_processing/signal_processing.py` - обработка сигналов
- `src/ml/feature_engineering.py` - продвинутая инженерия признаков

### ML-модели и прогнозирование
- `src/ml/anomaly_detection.py` - модели аномалий
- `src/ml/prediction.py` - прогнозные модели
- `src/ml/model_training.py` - обучение
- `models/` - хранение обученных моделей

### API и роуты
- `src/api/main.py` - FastAPI приложение
- `src/api/routes/` - группировка роутов по функциональности
- `src/api/middleware.py` - аутентификация, CORS, логирование

### Фоновые задачи Celery
- `src/worker/celery_app.py` - настройка Celery
- `src/worker/tasks/` - задачи по категориям
- Redis как брокер сообщений

### Дашборд Streamlit
- `src/dashboard/main.py` - мультистраничное приложение
- `src/dashboard/pages/` - отдельные страницы
- `src/dashboard/components/` - переиспользуемые компоненты

### Конфиги
- `src/config/` - Python модули конфигурации
- `configs/` - файлы конфигурации (YAML, ENV)
- `src/config/logging_config.py` - настройки логирования
- `configs/prometheus.yml` - метрики Prometheus
- `src/config/security.py` - JWT, шифрование

## 4. Взаимодействие модулей

```
CSV Files → Data Processing → Database → Feature Extraction → ML Models
                ↓                ↓              ↓              ↓
            Worker Tasks    → API Routes → Dashboard → Monitoring
                ↓                ↓              ↓              ↓
            Celery Queue    → Redis Cache → Streamlit → Prometheus/Grafana
```

**Поток данных:**
1. CSV файлы загружаются через `csv_loader.py`
2. Данные валидируются и сохраняются в PostgreSQL
3. Worker задачи извлекают признаки асинхронно
4. ML модели обучаются на признаках
5. API предоставляет доступ к данным и прогнозам
6. Dashboard отображает результаты
7. Prometheus собирает метрики, Grafana визуализирует

**Связи между компонентами:**
- Database модели используются во всех модулях
- Config модули импортируются везде
- Utils предоставляют общую функциональность
- API вызывает ML модели и Worker задачи
- Dashboard обращается к API
- Все компоненты логируют в единую систему

## 5. Запуск через Docker

```yaml
# docker-compose.yml
services:
  postgres:    # База данных
  redis:       # Брокер сообщений и кеш
  api:         # FastAPI приложение
  worker:      # Celery worker
  dashboard:   # Streamlit дашборд
  nginx:       # Reverse proxy
  prometheus:  # Метрики
  grafana:     # Мониторинг
```

**Последовательность запуска:**
1. `docker-compose up postgres redis` - инфраструктура
2. `docker-compose up api worker` - основные сервисы  
3. `docker-compose up dashboard nginx` - фронтенд
4. `docker-compose up prometheus grafana` - мониторинг

**Команды управления:**
```bash
make build     # Сборка всех образов
make up        # Запуск всех сервисов
make migrate   # Применение миграций
make seed      # Загрузка тестовых данных
make test      # Запуск тестов
```

## 6. Хранение и именование моделей

**Структура версионирования:**
```
models/
├── anomaly_detection/
│   ├── v1.0.0/
│   │   ├── model.pkl          # Основная модель
│   │   ├── scaler.pkl         # Препроцессор
│   │   ├── metadata.json      # Метаданные (метрики, дата)
│   │   └── config.yaml        # Гиперпараметры
│   ├── v1.1.0/               # Следующая версия
│   └── latest -> v1.1.0/     # Симлинк на актуальную
├── prediction/
└── registry.json             # Глобальный реестр моделей
```

**Именование:**
- Семантическое версионирование (MAJOR.MINOR.PATCH)
- MAJOR: Breaking changes в API модели
- MINOR: Новая функциональность, обратно совместимо
- PATCH: Исправления багов

**Метаданные модели:**
```json
{
  "version": "1.0.0",
  "created_at": "2025-01-10T10:00:00Z",
  "model_type": "IsolationForest",
  "metrics": {
    "precision": 0.95,
    "recall": 0.87,
    "f1_score": 0.91
  },
  "features": ["rms_R", "fft_peak_R", "kurtosis_R"],
  "training_data_hash": "abc123...",
  "git_commit": "a1b2c3d4"
}
```

## 7. Зависимости

### requirements.txt (основные)
```
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
streamlit==1.28.1

# Database
sqlalchemy[asyncio]==2.0.23
asyncpg==0.29.0
alembic==1.12.1
psycopg2-binary==2.9.9

# Queue
celery[redis]==5.3.4
redis==5.0.1

# ML & Data Science
numpy==1.24.3
scipy==1.11.4
pandas==2.1.3
scikit-learn==1.3.2
xgboost==2.0.2
statsmodels==0.14.0
prophet==1.1.5

# Signal Processing
librosa==0.10.1
pywavelets==1.4.1

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Utilities
pydantic==2.5.0
python-dotenv==1.0.0
pyyaml==6.0.1
```

### pyproject.toml (современный подход)
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "diagmod"
version = "1.0.0"
description = "Токовая диагностика асинхронных двигателей"
authors = [{name = "DiagMod Team"}]
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    # ... остальные зависимости
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.7.0",
]

[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 88
```

## 8. Диаграмма взаимодействия компонентов

```
                           ┌─────────────────┐
                           │   CSV Files     │
                           └─────────┬───────┘
                                     │
                           ┌─────────▼───────┐
                           │  Data Loader    │
                           │  (csv_loader)   │
                           └─────────┬───────┘
                                     │
                           ┌─────────▼───────┐      ┌─────────────────┐
                           │   PostgreSQL    │◄────►│   Redis Cache   │
                           │   (raw_data)    │      │   (sessions)    │
                           └─────────┬───────┘      └─────────────────┘
                                     │
                           ┌─────────▼───────┐
                           │ Feature Extract │
                           │ (Celery Worker) │
                           └─────────┬───────┘
                                     │
                ┌────────────────────▼────────────────────┐
                │                ML Models                │
                ├─────────────┬───────────────────────────┤
                │ Anomaly     │ Prediction  │ Forecasting │
                │ Detection   │ Models      │ (Prophet)   │
                └─────────────┴─────────────┬─────────────┘
                                           │
                ┌─────────────────────────▼─────────────────────────┐
                │                FastAPI Server                     │
                ├─────────┬─────────┬─────────┬──────────┬─────────┤
                │ Auth    │ Data    │ Models  │ Diagnostics│ Monitor │
                │ Routes  │ Routes  │ Routes  │ Routes   │ Routes  │
                └─────────┴─────────┴─────────┴──────────┴─────────┘
                                           │
                ┌─────────────────────────▼─────────────────────────┐
                │              Streamlit Dashboard                  │
                ├─────────┬─────────┬─────────┬──────────┬─────────┤
                │ Data    │ Anomaly │ Trends  │ Motor    │ Admin   │
                │ Overview│ Detection│ Analysis│ Status   │ Panel   │
                └─────────┴─────────┴─────────┴──────────┴─────────┘
                                           │
                ┌─────────────────────────▼─────────────────────────┐
                │                 Monitoring                        │
                ├─────────────────┬───────────────────────────────────┤
                │   Prometheus    │           Grafana               │
                │   (Metrics)     │         (Dashboards)            │
                └─────────────────┴───────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                            Security Layer                           │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┤
│ JWT Auth        │ Role-based      │ API Rate        │ Data            │
│ (API Access)    │ Access Control  │ Limiting        │ Encryption      │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

**Легенда:**
- `│` - прямая связь
- `◄────►` - двусторонняя связь
- `▼` - поток данных вниз
- Прямоугольники - компоненты системы
- Горизонтальные разделители - логические группы

**Описание потоков:**
1. **Данные**: CSV → Loader → PostgreSQL → Features → ML
2. **API**: ML Models → FastAPI → Dashboard → Users  
3. **Фоновые задачи**: PostgreSQL → Celery → ML Training
4. **Мониторинг**: All Components → Prometheus → Grafana
5. **Безопасность**: JWT → RBAC → Rate Limiting → Encryption

Эта структура обеспечивает:
- **Масштабируемость**: Микросервисная архитектура
- **Модульность**: Четкое разделение ответственности
- **Безопасность**: Многоуровневая защита
- **Мониторинг**: Полная наблюдаемость системы
- **Развертывание**: Docker-compose для легкого запуска
