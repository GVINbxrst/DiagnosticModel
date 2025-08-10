# Система логирования и мониторинга DiagMod

## Обзор

Система DiagMod включает комплексную систему логирования и мониторинга, основанную на:
- **JSON-логирование** с структурированными данными
- **Prometheus метрики** для мониторинга производительности
- **Алерты** для критических событий
- **Аудит безопасности** для отслеживания действий пользователей

## Компоненты системы мониторинга

### 1. JSON-логирование (`src/utils/logger.py`)

#### Основные возможности:
- Структурированное JSON-логирование
- Ротация логов (10MB, 5 backup файлов)
- Различные уровни логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Специализированный аудит-логгер для безопасности

#### Типы логов:
```python
# Обычное логирование
logger.info("Обработка CSV файла", extra={
    'equipment_id': 123,
    'file_size': 1024000,
    'processing_time': 15.5
})

# Аудит действий пользователей
audit_logger.log_user_action(
    user_id="user123",
    username="engineer1",
    action="view_anomalies",
    resource="/anomalies/123",
    result="success"
)

# Аудит API запросов
audit_logger.log_api_request(
    user_id="user123",
    endpoint="/signals/456",
    method="GET",
    status_code=200,
    response_time=0.5,
    ip_address="192.168.1.100"
)
```

### 2. Prometheus метрики (`src/utils/metrics.py`)

#### Ключевые метрики:

**API метрики:**
- `api_requests_total` - счетчик HTTP запросов по методам, endpoints, статусам
- `api_request_duration_seconds` - гистограмма времени ответа API

**Метрики машинного обучения:**
- `anomalies_detected_total` - счетчик обнаруженных аномалий
- `anomaly_detection_duration_seconds` - время выполнения детекции
- `forecast_latency_seconds` - время генерации прогнозов

**Метрики обработки данных:**
- `csv_files_processed_total` - счетчик обработанных CSV файлов
- `csv_processing_duration_seconds` - время обработки файлов
- `data_points_processed_total` - количество обработанных точек данных

**Worker метрики:**
- `worker_tasks_total` - счетчик выполненных задач Celery
- `worker_task_duration_seconds` - время выполнения задач
- `worker_active_tasks` - количество активных задач

**Системные метрики:**
- `system_cpu_usage_percent` - использование CPU
- `system_memory_usage_bytes` - использование памяти
- `system_disk_usage_bytes` - использование диска

### 3. Middleware для автоматического сбора метрик

#### API Middleware (`src/api/middleware/metrics.py`)
Автоматически собирает метрики для всех HTTP запросов:
- Время выполнения
- Статус коды
- Информация о пользователе
- IP адреса клиентов

#### Worker Monitoring (`src/worker/monitoring.py`)
Отслеживает выполнение Celery задач:
- Время выполнения задач
- Статус завершения
- Ошибки и исключения
- Активные задачи

### 4. Endpoints мониторинга

#### `/metrics` - Prometheus метрики
```bash
curl http://api:8000/metrics
```
Возвращает все метрики в формате Prometheus.

#### `/health` - Базовая проверка здоровья
```bash
curl http://api:8000/health
```
Возвращает статус сервиса.

#### `/health/detailed` - Детальная проверка
```bash
curl http://api:8000/health/detailed
```
Проверяет состояние всех компонентов:
- База данных PostgreSQL
- Redis
- Worker (Celery)
- Системные ресурсы

#### `/stats` - Системная статистика
```bash
curl http://api:8000/stats
```
Возвращает подробную статистику системы.

### 5. Система алертов (`configs/prometheus/rules/diagmod_alerts.yml`)

#### Критические алерты:
- **HighAPIErrorRate** - более 10% ошибок 5xx в API
- **WorkerTaskFailures** - более 10% неудачных задач Worker
- **HighMemoryUsage** - использование памяти > 90%
- **ServiceDown** - недоступность сервисов
- **CriticalEquipmentAnomaly** - критические аномалии оборудования

#### Предупреждения:
- **SlowAPIResponse** - медленные API ответы (>5 сек)
- **HighAnomalyDetectionRate** - более 100 аномалий/час
- **HighCPUUsage** - использование CPU > 85%
- **CSVProcessingErrors** - ошибки обработки CSV файлов

## Конфигурация

### Prometheus (`configs/prometheus/prometheus.yml`)
```yaml
scrape_configs:
  - job_name: 'diagmod-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'diagmod-worker'
    static_configs:
      - targets: ['worker:8002']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Переменные окружения
```bash
# Логирование
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_PATH=/app/logs/diagmod.log

# Метрики
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8001
WORKER_METRICS_PORT=8002
```

## Использование

### 1. Декораторы для автоматического отслеживания

```python
from src.utils.metrics import track_anomaly_detection, track_csv_processing

@track_csv_processing(equipment_id=123)
def process_file(file_path):
    # Код обработки файла
    pass

@track_anomaly_detection(equipment_id=123, model_name="isolation_forest")
def detect_anomaly(features):
    # Код обнаружения аномалий
    pass
```

### 2. Ручное обновление метрик

```python
from src.utils.metrics import increment_counter, observe_histogram

# Увеличение счетчика
increment_counter('csv_files_processed_total', {
    'equipment_id': '123',
    'status': 'success'
})

# Добавление в гистограмму
observe_histogram('processing_duration_seconds', 15.5, {
    'operation': 'feature_extraction'
})
```

### 3. Аудит действий пользователей

```python
from src.utils.logger import audit_logger

# Логирование действия пользователя
audit_logger.log_user_action(
    user_id="user123",
    username="engineer1", 
    action="download_report",
    resource="/reports/456",
    result="success",
    ip_address="192.168.1.100"
)

# Логирование обнаружения аномалии
audit_logger.log_anomaly_detection(
    equipment_id=123,
    model_name="isolation_forest",
    is_anomaly=True,
    confidence=0.85,
    features={'rms_a': 25.5, 'rms_b': 24.8}
)
```

## Запуск и развертывание

### 1. Запуск с Docker Compose
```bash
# Запуск всей системы мониторинга
docker-compose up -d

# Проверка состояния
docker-compose ps
```

### 2. Доступ к компонентам
- **API метрики**: http://localhost:8000/metrics
- **Worker метрики**: http://localhost:8002/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **API health**: http://localhost:8000/health/detailed

### 3. Просмотр логов
```bash
# Логи API
docker-compose logs -f api

# Логи Worker
docker-compose logs -f worker

# Логи в JSON формате
tail -f logs/diagmod.log | jq
```

## Мониторинг в продакшене

### 1. Рекомендуемые дашборды Grafana
- Системные метрики (CPU, память, диск)
- API производительность
- ML модели и аномалии
- Обработка данных
- Алерты и события

### 2. Настройка алертов
- Интеграция с Slack/Email через Alertmanager
- Эскалация критических алертов
- Группировка связанных алертов

### 3. Ретенция данных
- Метрики: 30 дней (настраивается в prometheus.yml)
- Логи: ротация каждые 10MB, 5 backup файлов
- Аудит: рекомендуется архивирование в долгосрочное хранилище

## Безопасность

### 1. Аудит логирование
Все действия пользователей логируются с следующей информацией:
- ID и имя пользователя
- Выполненное действие
- Ресурс (endpoint)
- Результат (успех/ошибка)
- IP адрес и User-Agent
- Временная метка

### 2. Защищенные endpoint'ы
- `/metrics` - доступ ограничен внутренней сетью
- `/health/detailed` - требует аутентификации
- `/stats` - только для администраторов

### 3. Политика хранения аудит-логов
- Логи безопасности хранятся в неизменяемой форме
- Регулярное архивирование в защищенное хранилище
- Мониторинг попыток несанкционированного доступа
