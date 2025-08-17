# DiagMod - Система токовой диагностики асинхронных двигателей

## Описание

DiagMod - это комплексная система для диагностики асинхронных двигателей на основе анализа токовых сигналов с использованием методов машинного обучения. Система обеспечивает обнаружение аномалий, прогнозирование деградации и мониторинг состояния двигателей в режиме реального времени.

## Особенности

- 📊 **Анализ токовых сигналов**: Обработка CSV данных с токами трех фаз (R, S, T)
- 🤖 **Машинное обучение**: Модели обнаружения аномалий и прогнозирования
- 🔍 **Извлечение признаков**: FFT, RMS, kurtosis, skewness и другие статистические метрики
- 🌐 **Web API**: RESTful API на FastAPI для интеграции
- 📱 **Dashboard**: Интерактивный веб-интерфейс на Streamlit
- ⚡ **Асинхронная обработка**: Celery для фоновых задач
- 📈 **Мониторинг**: Prometheus + Grafana для наблюдаемости
- 🐳 **Контейнеризация**: Docker для легкого развертывания
- 🔒 **Безопасность**: JWT аутентификация и шифрование

## Архитектура

```
CSV Files → Data Processing → PostgreSQL → Feature Extraction → ML Models
                ↓               ↓             ↓                 ↓
           Worker Tasks → Redis Cache → API Routes → Dashboard → Monitoring
```

## Быстрый старт

### 1. Клонирование и настройка

```bash
git clone <repository-url>
cd DiagMod
make setup  # Создание .env и папок
```

### 2. Конфигурация

Отредактируйте файл `.env` согласно вашему окружению:

```bash
cp .env.example .env
# Настройте параметры базы данных, Redis, API и т.д.
```

### 3. Запуск через Docker

```bash
make build  # Сборка образов
make up     # Запуск всех сервисов
make migrate # Применение миграций БД
make seed   # Загрузка тестовых данных
```

### 4. Доступ к сервисам

- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501  
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)

### 5. Локальный E2E (без Docker, SQLite + Celery eager)

```bash
# 1. Установить зависимости
make install-dev

# 2. Включить синхронный режим выполнения задач (без Redis)
set CELERY_TASK_ALWAYS_EAGER=1  # PowerShell/Windows
# export CELERY_TASK_ALWAYS_EAGER=1  # bash/Linux/macOS

# 3. Выполнить шаги по отдельности
make process-raw    # обработка сырых сигналов -> извлечение признаков
make anomalies      # запуск детекции аномалий
make forecast       # прогноз трендов
make report         # сводный отчёт в консоли

# Или полный конвейер одним шагом
make pipeline
```

### 6. Быстрый запуск с PostgreSQL (ручная инициализация без Alembic)

```bash
make install-dev
# Убедитесь что PostgreSQL запущен и доступен по DATABASE_URL (.env)
make init-db          # создаёт таблицы (Base.metadata.create_all)
set CELERY_TASK_ALWAYS_EAGER=1  # или export ...
python scripts/e2e_pipeline.py --csv ./data/raw --pattern "*.csv" --train-full
make report-html      # HTML отчёт -> reports/latest_report.html
```

Если в CSV только три колонки токов (current_R,current_S,current_T) — этого достаточно для минимального пайплайна.

Переменная окружения `CELERY_TASK_ALWAYS_EAGER=1` позволяет выполнять Celery-задачи синхронно
в текущем процессе для быстрой локальной отладки без запуска брокера Redis.

## Использование

### Загрузка данных

```bash
# Загрузка CSV файлов
make load-csv

# Извлечение признаков
make extract-features

# Обучение моделей
make train-models

### Загрузка локальных CSV через Docker

Для удобной загрузки локальных CSV-файлов из вашей файловой системы в БД используйте сервис `loader` из `docker-compose.dev.yml`.

1) Поднимите Postgres и Redis:

```bash
docker compose -f docker-compose.dev.yml up -d postgres redis
```

2) Загрузка одного файла:

```bash
docker compose -f docker-compose.dev.yml run --rm loader \
     python src/data_processing/csv_loader.py /data/csvs/sample.csv --sample-rate 25600
```

3) Загрузка всех файлов из директории:

```bash
docker compose -f docker-compose.dev.yml run --rm loader \
     python src/data_processing/csv_loader.py /data/csvs --sample-rate 25600 --batch-size 10000
```

Примечания:
- В `docker-compose.dev.yml` сервис `loader` монтирует локальную директорию с CSV в `/data/csvs` контейнера (путь слева в volume замените на абсолютный путь к вашей папке CSV).
- `DATABASE_URL` и `REDIS_URL` берутся из конфига dev-компоуз; сервис пишет напрямую в Postgres из состава `docker-compose.dev.yml`.
```

### API примеры

```python
import requests

# Получение статуса двигателя
response = requests.get("http://localhost:8000/api/v1/motor/status/motor_001")

# Обнаружение аномалий
response = requests.post("http://localhost:8000/api/v1/detect/anomaly", 
                        json={"motor_id": "motor_001"})

# Прогнозирование
response = requests.post("http://localhost:8000/api/v1/predict/degradation",
                        json={"motor_id": "motor_001", "horizon_days": 30})
```

## Структура проекта

```
DiagMod/
├── src/                    # Исходный код
│   ├── api/               # FastAPI приложение
│   ├── dashboard/         # Streamlit интерфейс  
│   ├── worker/            # Celery задачи
│   ├── ml/                # ML модели
│   ├── data_processing/   # Обработка данных
│   └── database/          # Модели БД
├── data/                  # Данные (сырые, обработанные)
├── models/                # Обученные ML модели
├── sql/                   # SQL схемы и скрипты
├── docker/                # Docker файлы
├── configs/               # Конфигурации
└── tests/                 # Тесты
```

## Формат данных

CSV файлы должны иметь следующий формат:

```csv
current_R,current_S,current_T
1.23,1.45,1.67
1.24,,1.68
1.25,1.47,
...
```

- Первая строка: заголовки фаз
- Остальные строки: значения токов через запятую
- Пустые значения допускаются для фаз S и T

## Разработка

### Установка зависимостей

```bash
make install-dev  # Установка с dev зависимостями
```

### Запуск тестов

```bash
make test          # Все тесты
make test-unit     # Только unit тесты
make test-coverage # Тесты с покрытием
```

### Качество кода

```bash
make format       # Форматирование кода
make lint         # Проверка линтерами
```

### Разработка в режиме hot-reload

```bash
make up-dev  # Запуск с автоперезагрузкой
```

## ML Модели

### Обнаружение аномалий
- **Isolation Forest**: Для выявления выбросов
- **One-Class SVM**: Для обнаружения новизны
- **Autoencoder**: Для сложных паттернов

### Прогнозирование
- **XGBoost**: Для краткосрочных прогнозов
- **Prophet**: Для временных рядов
- **LSTM**: Для долгосрочных трендов
 - **TCN (Temporal Convolutional Network)**: Прогноз вероятности возникновения дефекта и степени его развития на горизонте H.

### Sequence / TCN Обучение

Минимальный запуск обучения TCN (использует эмбеддинги и агрегаты признаков):

```bash
python -c "import asyncio,uuid; from src.ml.tcn_forecasting import train_tcn; from src.config.settings import get_settings; eid='00000000-0000-0000-0000-000000000000'; asyncio.run(train_tcn(uuid.UUID(eid)))"
```

Параметры настраиваются через переменные окружения или `.env`:

```
TCN_WINDOW_SIZE=32
TCN_PREDICTION_HORIZON=4
TCN_CHANNELS=64,64,64
TCN_KERNEL_SIZE=3
TCN_DROPOUT=0.1
TCN_MAX_EPOCHS=10
TCN_LEARNING_RATE=0.001
```

После обучения модель сохраняется в директорию `models/tcn/model_<equipment_id>.pkl`.

### Кеширование моделей

Включено через настройку `USE_MODEL_CACHE=true`.

Кешируются:
 - Prophet модель для RMS (`models/prophet_rms/<equipment_id>.pkl`)
 - LSTM sequence (`models/lstm_seq/<equipment_id>.pkl`)
 - TCN (`models/tcn/model_<equipment_id>.pkl`)

Метрика `model_cache_events_total{model_name, event}` отслеживает события:
 - `hit` — модель найдена в памяти / на диске
 - `miss` — модель отсутствовала и была обучена заново
 - `store` — успешное сохранение модели в кеш

Пример просмотра метрики:
```
curl http://localhost:8001/metrics | findstr model_cache_events_total
```

### Признаки
- **Временная область**: RMS, среднее, дисперсия, kurtosis, skewness
- **Частотная область**: FFT пики, спектральная энтропия, MFCC
- **Статистические**: Гистограммы, квантили, автокорреляция

## Мониторинг

### Метрики
- Производительность системы (CPU, память, диск)
- Качество моделей ML (точность, полнота, F1-score)
- Бизнес-метрики (количество обработанных файлов, аномалий)

### Метрики готовности данных / кластеризации
Добавлены Prometheus метрики:
- `features_with_embedding_total` – количество Feature с embedding
- `embedding_coverage_ratio` – доля features с embedding
- `clustering_noise_ratio` – доля точек без cluster_id
- `clustering_label_coverage_ratio` – покрытие кластеров метками дефектов
- `min_cluster_size_labeled`, `p50_cluster_size`, `p90_cluster_size` – статистика размеров кластеров
- `data_recency_seconds` – давность последнего окна признаков
- `clustering_drift_score` – оценка дрейфа распределения кластеров (JS divergence 0..1)
- `training_snapshot_timestamp` – время последнего snapshot датасета

### Data Readiness & Snapshot Endpoints
- `GET /admin/clustering/data-readiness` – агрегированные метрики готовности
- `GET /admin/clustering/summary` – дистрибуция кластеров + manifest + label coverage
- `POST /admin/clustering/snapshot` – формирование parquet snapshot (embeddings+labels)
- `GET /admin/clustering/embedding-manifest` – manifest embedding автоэнкодера (код hash, dim, версия)

### Манифесты моделей
Хранятся в каталоге `models/`:
- `models/anomaly_detection/latest/manifest.json` – streaming anomaly модель (threshold, p98, outlier_ratio, version)
- `models/clustering/manifest.json` – результаты последней кластеризации (clusters_found, labeled, params)
- `models/embeddings/manifest.json` – автоэнкодер эмбеддингов (embedding_dim, code_hash, trained_at, version, final_loss)
- `models/clustering/knn_manifest.json` – semi-supervised kNN состояние (coverage)
- `models/snapshots/<ts>/manifest.json` – snapshot обучающего набора

### Алерты
- Падение качества модели
- Высокая нагрузка на систему
- Критические аномалии в двигателях

## Безопасность

- JWT токены для API аутентификации
- RBAC для управления доступом
- Шифрование чувствительных данных
- Rate limiting для защиты от DoS
- HTTPS в продакшене

## Производительность

### Рекомендуемые ресурсы
- **CPU**: 8+ ядер для ML обучения
- **RAM**: 16+ GB для обработки больших CSV
- **Диск**: SSD для быстрого доступа к данным
- **GPU**: Опционально для глубокого обучения

### Оптимизация
- Индексы PostgreSQL для быстрых запросов
- Redis кеширование для частых операций
- Батчевая обработка CSV файлов
- Асинхронные задачи для тяжелых операций
 - Партиционирование таблиц `raw_signals` и `features` по месяцам (`sql/schema/005_partitioning.sql`)
 - Почасовые агрегаты RMS (`hourly_feature_summary`) + Feature Store (db/redis)
 - Прореживание и агрегация данных для визуализаций (часовой тренд)
 - Кэширование результатов sequence risk (Redis / DB fallback)

## Dashboard: Тренды, Риск и Feature Store

Добавлена вкладка **"Тренды"**:
 - Почасовой график среднего RMS (агрегаты из HourlyFeatureSummary или Redis Feature Store)
 - Кнопка расчёта краткосрочного риска (sequence LSTM) через эндпоинт `GET /api/v1/signals/sequence_risk/{equipment_id}`
 - Эндпоинт тренда: `GET /api/v1/equipment/{equipment_id}/rms/hourly?limit=168`

Feature Store (`src/utils/feature_store.py`):
 - Backend `db`: чтение из БД без дополнительного хранения
 - Backend `redis`: хранение последних N точек в Redis списке
 - Авто-запись RMS при сохранении признаков
 - Параметры управления: FEATURE_STORE_BACKEND (db|redis), MAX_FEATURE_STORE_POINTS (лимит точек для redis)
 - Метрики: feature_store_hit_total{backend}, feature_store_miss_total{backend,reason}

Партиционирование:
 - Скрипт `sql/schema/005_partitioning.sql` преобразует таблицы в декларативно партиционированные и создаёт месячные партиции ±12 месяцев
 - Выполнение: `psql "$DATABASE_URL" -f sql/schema/005_partitioning.sql`


## Развертывание

### Development
```bash
make up-dev
```

### Staging
```bash
make deploy-staging
```

### Production
```bash
make deploy-prod
```

## Документация

- [API документация](docs/api/README.md)
- [Руководство по развертыванию](docs/deployment/README.md)
- [Руководство пользователя](docs/user_guide/README.md)

## Поддержка

При возникновении проблем:

1. Проверьте логи: `make logs`
2. Проверьте статус: `make status`
3. Проверьте здоровье: `make health`


## Команда

Цифровой Синтез
