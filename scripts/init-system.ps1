# DiagMod - Полная инициализация системы
# Этот скрипт выполняет первый запуск: создание БД, загрузка данных, обучение моделей

# Настройка кодировки
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

param(
    [switch]$SkipDocker,
    [switch]$SkipTraining,
    [switch]$Help
)

if ($Help) {
    Write-Host "DiagMod System Initialization" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Параметры:"
    Write-Host "  -SkipDocker     Пропустить запуск Docker сервисов"
    Write-Host "  -SkipTraining   Пропустить обучение ML моделей"
    Write-Host "  -Help           Показать эту справку"
    Write-Host ""
    Write-Host "Примеры:"
    Write-Host "  .\init-system.ps1                  # Полная инициализация"
    Write-Host "  .\init-system.ps1 -SkipDocker      # Без Docker"
    Write-Host "  .\init-system.ps1 -SkipTraining    # Без обучения моделей"
    exit 0
}

Write-Host "🚀 Инициализация системы DiagMod..." -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""

# Функция проверки команды
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# 1. Проверка зависимостей
Write-Host "1. 🔍 Проверка системных зависимостей..." -ForegroundColor Yellow

# Проверка Python
if (Test-Command python) {
    $pythonCmd = "python"
    $pythonVersion = & python --version
    Write-Host "   ✅ Python: $pythonVersion" -ForegroundColor Green
} elseif (Test-Command py) {
    $pythonCmd = "py"
    $pythonVersion = & py --version
    Write-Host "   ✅ Python: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "   ❌ Python не найден!" -ForegroundColor Red
    Write-Host "   Установите Python и запустите .\scripts\install-simple.ps1" -ForegroundColor Yellow
    exit 1
}

# Проверка Docker
if (-not $SkipDocker) {
    if (Test-Command docker) {
        Write-Host "   ✅ Docker найден" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Docker не найден!" -ForegroundColor Red
        Write-Host "   Установите Docker Desktop или используйте -SkipDocker" -ForegroundColor Yellow
        exit 1
    }

    if (Test-Command docker-compose) {
        Write-Host "   ✅ Docker Compose найден" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Docker Compose не найден!" -ForegroundColor Red
        exit 1
    }
}

# Проверка виртуального окружения
if ($env:VIRTUAL_ENV) {
    Write-Host "   ✅ Виртуальное окружение активно" -ForegroundColor Green
} else {
    Write-Host "   ⚠️ Виртуальное окружение не активно" -ForegroundColor Yellow
    Write-Host "   Запустите: venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}

Write-Host ""

# 2. Создание необходимых директорий
Write-Host "2. 📁 Создание структуры директорий..." -ForegroundColor Yellow

$directories = @(
    "data\raw",
    "data\processed",
    "data\features",
    "data\exports",
    "models\anomaly_detection",
    "models\prediction",
    "logs",
    "configs\ssl"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   ✅ Создана директория: $dir" -ForegroundColor Green
    } else {
        Write-Host "   ℹ️ Директория уже существует: $dir" -ForegroundColor Cyan
    }
}

Write-Host ""

# 3. Создание конфигурационного файла .env
Write-Host "3. ⚙️ Настройка конфигурации..." -ForegroundColor Yellow

if (-not (Test-Path ".env")) {
    $envContent = @"
# DiagMod Environment Configuration

# Database
DATABASE_URL=postgresql+asyncpg://diagmod_user:diagmod_password@localhost:5432/diagmod

# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# API
API_HOST=0.0.0.0
API_PORT=8000

# Security
SECRET_KEY=diagmod-secret-key-change-in-production-$(Get-Random)
JWT_SECRET_KEY=diagmod-jwt-secret-$(Get-Random)
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_PATH=logs/diagmod.log

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8001
WORKER_METRICS_PORT=8002

# Environment
ENVIRONMENT=development

# ML Models
MODEL_PATH=models
RETRAIN_INTERVAL_HOURS=24
ANOMALY_THRESHOLD=0.7

# Data Processing
MAX_FILE_SIZE_MB=100
BATCH_SIZE=1000
FEATURE_WINDOW_SIZE=1024
"@
    $envContent | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "   ✅ Создан файл .env" -ForegroundColor Green
} else {
    Write-Host "   ℹ️ Файл .env уже существует" -ForegroundColor Cyan
}

Write-Host ""

# 4. Запуск Docker сервисов
if (-not $SkipDocker) {
    Write-Host "4. 🐳 Запуск Docker сервисов..." -ForegroundColor Yellow

    try {
        Write-Host "   Запуск PostgreSQL и Redis..." -ForegroundColor Cyan
        & docker-compose up -d postgres redis

        Write-Host "   Ожидание готовности сервисов..." -ForegroundColor Cyan
        Start-Sleep -Seconds 10

        # Проверка состояния сервисов
        $pgStatus = & docker-compose ps postgres --format "table"
        $redisStatus = & docker-compose ps redis --format "table"

        Write-Host "   ✅ Docker сервисы запущены" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Ошибка запуска Docker: $_" -ForegroundColor Red
        Write-Host "   Попробуйте запустить вручную: docker-compose up -d postgres redis" -ForegroundColor Yellow
    }
} else {
    Write-Host "4. ⏭️ Пропуск Docker сервисов (флаг -SkipDocker)" -ForegroundColor Yellow
}

Write-Host ""

# 5. Инициализация базы данных
Write-Host "5. 🗄️ Инициализация базы данных..." -ForegroundColor Yellow

try {
    Write-Host "   Применение миграций Alembic..." -ForegroundColor Cyan
    & $pythonCmd -m alembic upgrade head
    Write-Host "   ✅ Схема базы данных создана" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Ошибка инициализации БД: $_" -ForegroundColor Red
    Write-Host "   Убедитесь, что PostgreSQL запущен и доступен" -ForegroundColor Yellow
}

Write-Host ""

# 6. Создание тестовых данных
Write-Host "6. 📊 Создание тестовых данных..." -ForegroundColor Yellow

try {
    Write-Host "   Загрузка начальных данных..." -ForegroundColor Cyan
    & $pythonCmd scripts\load_initial_data.py
    Write-Host "   ✅ Тестовые данные загружены" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️ Ошибка загрузки тестовых данных: $_" -ForegroundColor Yellow
    Write-Host "   Это не критично, можно загрузить данные позже" -ForegroundColor Cyan
}

Write-Host ""

# 7. Обучение ML моделей
if (-not $SkipTraining) {
    Write-Host "7. 🧠 Обучение ML моделей..." -ForegroundColor Yellow

    try {
        Write-Host "   Обучение моделей детекции аномалий..." -ForegroundColor Cyan
        & $pythonCmd scripts\train_models.py
        Write-Host "   ✅ Модели обучены и сохранены" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️ Ошибка обучения моделей: $_" -ForegroundColor Yellow
        Write-Host "   Модели можно обучить позже командой: python scripts\train_models.py" -ForegroundColor Cyan
    }
} else {
    Write-Host "7. ⏭️ Пропуск обучения моделей (флаг -SkipTraining)" -ForegroundColor Yellow
}

Write-Host ""

# 8. Проверка готовности системы
Write-Host "8. ✅ Проверка готовности системы..." -ForegroundColor Yellow

$allReady = $true

# Проверка импорта основных модулей
try {
    & $pythonCmd -c "import src.api.main, src.worker.tasks, src.ml.train; print('Модули успешно импортированы')"
    Write-Host "   ✅ Python модули готовы" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Ошибка импорта модулей: $_" -ForegroundColor Red
    $allReady = $false
}

# Проверка подключения к БД (если Docker запущен)
if (-not $SkipDocker) {
    try {
        & $pythonCmd -c "from src.database.connection import test_connection; import asyncio; asyncio.run(test_connection())"
        Write-Host "   ✅ Подключение к БД работает" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Ошибка подключения к БД: $_" -ForegroundColor Red
        $allReady = $false
    }
}

Write-Host ""

# 9. Итоговый статус и инструкции
Write-Host "=" * 50 -ForegroundColor Green
if ($allReady) {
    Write-Host "🎉 Система DiagMod успешно инициализирована!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Система DiagMod частично готова к работе" -ForegroundColor Yellow
}
Write-Host "=" * 50 -ForegroundColor Green

Write-Host ""
Write-Host "📋 Следующие шаги для запуска:" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Запуск API сервера:" -ForegroundColor White
Write-Host "   .\scripts\run-api.ps1" -ForegroundColor Gray
Write-Host "   Доступен по адресу: http://localhost:8000" -ForegroundColor Gray
Write-Host ""

Write-Host "2. Запуск Worker (в новом окне PowerShell):" -ForegroundColor White
Write-Host "   .\scripts\run-worker.ps1" -ForegroundColor Gray
Write-Host ""

Write-Host "3. Запуск Dashboard (в третьем окне PowerShell):" -ForegroundColor White
Write-Host "   .\scripts\run-dashboard.ps1" -ForegroundColor Gray
Write-Host "   Доступен по адресу: http://localhost:8501" -ForegroundColor Gray
Write-Host ""

Write-Host "4. Или запуск всех сервисов в Docker:" -ForegroundColor White
Write-Host "   docker-compose up -d" -ForegroundColor Gray
Write-Host ""

Write-Host "5. Проверка статуса системы:" -ForegroundColor White
Write-Host "   .\scripts\status.ps1 -Detailed" -ForegroundColor Gray
Write-Host ""

Write-Host "📚 Полезные ссылки:" -ForegroundColor Cyan
Write-Host "   API документация: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "   Метрики API: http://localhost:8000/metrics" -ForegroundColor Gray
Write-Host "   Метрики Worker: http://localhost:8002/metrics" -ForegroundColor Gray
Write-Host ""

Write-Host "🔧 Дополнительные команды:" -ForegroundColor Cyan
Write-Host "   Загрузка CSV данных: python scripts\load_csv_data.py <file>" -ForegroundColor Gray
Write-Host "   Переобучение моделей: python scripts\train_models.py" -ForegroundColor Gray
Write-Host "   Генерация отчетов: python scripts\generate_reports.py" -ForegroundColor Gray
