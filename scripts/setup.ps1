# DiagMod - Система диагностики асинхронных двигателей
# PowerShell скрипт для Windows

# Параметры
param(
    [switch]$Dev,
    [switch]$Help
)

# Цвета для вывода
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Cyan"
    NC = "White"
}

# Функции для вывода
function Write-Info {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param($Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param($Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

# Справка
function Show-Help {
    Write-Host "DiagMod Setup Script для Windows PowerShell" -ForegroundColor $Colors.Blue
    Write-Host ""
    Write-Host "Использование:"
    Write-Host "  .\setup.ps1            # Установка основных зависимостей"
    Write-Host "  .\setup.ps1 -Dev       # Установка с зависимостями для разработки"
    Write-Host "  .\setup.ps1 -Help      # Показать эту справку"
    Write-Host ""
    Write-Host "Примеры:"
    Write-Host "  .\setup.ps1 -Dev       # Полная установка для разработки"
    Write-Host ""
}

# Проверка системы
function Test-SystemRequirements {
    Write-Info "Проверка системных требований..."

    # Проверяем Python
    try {
        $pythonVersion = python --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Python не найден"
        }
        Write-Info "Найден $pythonVersion"
    }
    catch {
        Write-Error "Python 3 не найден. Установите Python 3.8 или выше с python.org"
        Write-Error "Убедитесь, что Python добавлен в PATH"
        exit 1
    }

    # Проверяем pip
    try {
        pip --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "pip не найден"
        }
    }
    catch {
        Write-Error "pip не найден. Переустановите Python с включенным pip"
        exit 1
    }

    # Проверяем Docker
    try {
        docker --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker не найден"
        }
        Write-Info "Docker найден"
    }
    catch {
        Write-Warning "Docker не найден. Для полноценной работы установите Docker Desktop"
    }

    # Проверяем Docker Compose
    try {
        docker-compose --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose не найден"
        }
        Write-Info "Docker Compose найден"
    }
    catch {
        Write-Warning "Docker Compose не найден. Установите Docker Desktop или docker-compose"
    }

    Write-Success "Системные требования проверены"
}

# Создание виртуального окружения
function New-VirtualEnvironment {
    Write-Info "Создание виртуального окружения..."

    if (-not (Test-Path "venv")) {
        python -m venv venv
        Write-Success "Виртуальное окружение создано"
    }
    else {
        Write-Info "Виртуальное окружение уже существует"
    }
}

# Активация виртуального окружения
function Enable-VirtualEnvironment {
    Write-Info "Активация виртуального окружения..."

    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
        Write-Success "Виртуальное окружение активировано"
    }
    else {
        Write-Error "Не найден скрипт активации виртуального окружения"
        exit 1
    }
}

# Установка зависимостей
function Install-Dependencies {
    param([bool]$IncludeDev = $false)

    Write-Info "Обновление базовых инструментов..."
    python -m pip install --upgrade pip setuptools wheel
    Write-Success "Базовые инструменты обновлены"

    Write-Info "Установка основных зависимостей..."
    pip install -r requirements.txt
    Write-Success "Основные зависимости установлены"

    if ($IncludeDev) {
        Write-Info "Установка зависимостей для разработки..."
        pip install -r requirements-dev.txt
        Write-Success "Зависимости для разработки установлены"
    }
}

# Создание директорий
function New-ProjectDirectories {
    Write-Info "Создание необходимых директорий..."

    $directories = @(
        "logs",
        "data\raw",
        "data\processed",
        "data\features",
        "data\exports",
        "models\anomaly_detection",
        "models\prediction",
        "configs\ssl"
    )

    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }

    Write-Success "Директории созданы"
}

# Настройка конфигурации
function Set-Configuration {
    Write-Info "Настройка конфигурации..."

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
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production

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
"@
        $envContent | Out-File -FilePath ".env" -Encoding UTF8
        Write-Success "Файл .env создан"
    }
    else {
        Write-Info "Файл .env уже существует"
    }
}

# Проверка установки
function Test-Installation {
    Write-Info "Проверка установки..."

    try {
        python -c @"
import fastapi
import streamlit
import celery
import sqlalchemy
import numpy
import pandas
import sklearn
import prometheus_client
print('✓ Все основные модули доступны')
"@
        Write-Success "Установка проверена"
    }
    catch {
        Write-Error "Ошибка при проверке модулей: $_"
        exit 1
    }
}

# Показать инструкции
function Show-Instructions {
    Write-Success "Установка завершена!"
    Write-Host ""
    Write-Info "Следующие шаги:"
    Write-Host ""
    Write-Host "1. Активируйте виртуальное окружение:" -ForegroundColor $Colors.Yellow
    Write-Host "   venv\Scripts\Activate.ps1" -ForegroundColor $Colors.NC
    Write-Host ""
    Write-Host "2. Запустите Docker сервисы:" -ForegroundColor $Colors.Yellow
    Write-Host "   docker-compose up -d" -ForegroundColor $Colors.NC
    Write-Host ""
    Write-Host "3. Инициализируйте базу данных:" -ForegroundColor $Colors.Yellow
    Write-Host "   python -m alembic upgrade head" -ForegroundColor $Colors.NC
    Write-Host ""
    Write-Host "4. Запустите компоненты для разработки:" -ForegroundColor $Colors.Yellow
    Write-Host "   # API сервер" -ForegroundColor $Colors.Blue
    Write-Host "   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor $Colors.NC
    Write-Host ""
    Write-Host "   # Worker (в новом окне PowerShell)" -ForegroundColor $Colors.Blue
    Write-Host "   celery -A src.worker.celery_app worker --loglevel=info" -ForegroundColor $Colors.NC
    Write-Host ""
    Write-Host "   # Dashboard (в новом окне PowerShell)" -ForegroundColor $Colors.Blue
    Write-Host "   streamlit run src/dashboard/main.py" -ForegroundColor $Colors.NC
    Write-Host ""
    Write-Host "Или запустите всё сразу в Docker:" -ForegroundColor $Colors.Yellow
    Write-Host "   docker-compose up -d" -ForegroundColor $Colors.NC
    Write-Host ""
    Write-Info "Веб-интерфейсы (после запуска):"
    Write-Host "   http://localhost:8000     # API" -ForegroundColor $Colors.NC
    Write-Host "   http://localhost:8000/docs # API документация" -ForegroundColor $Colors.NC
    Write-Host "   http://localhost:8501     # Dashboard" -ForegroundColor $Colors.NC
    Write-Host "   http://localhost:9090     # Prometheus" -ForegroundColor $Colors.NC
    Write-Host ""
    Write-Info "Полезные команды PowerShell:"
    Write-Host "   .\scripts\run-api.ps1      # Запуск API" -ForegroundColor $Colors.NC
    Write-Host "   .\scripts\run-worker.ps1   # Запуск Worker" -ForegroundColor $Colors.NC
    Write-Host "   .\scripts\run-dashboard.ps1 # Запуск Dashboard" -ForegroundColor $Colors.NC
    Write-Host ""
}

# Основная функция
function Main {
    if ($Help) {
        Show-Help
        return
    }

    Write-Host "╔═══════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor $Colors.Blue
    Write-Host "║                    DiagMod Setup - Установка системы                         ║" -ForegroundColor $Colors.Blue
    Write-Host "║                  Диагностика асинхронных двигателей                          ║" -ForegroundColor $Colors.Blue
    Write-Host "╚═══════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor $Colors.Blue
    Write-Host ""

    # Проверяем политику выполнения
    $executionPolicy = Get-ExecutionPolicy
    if ($executionPolicy -eq "Restricted") {
        Write-Warning "Политика выполнения PowerShell ограничена."
        Write-Info "Выполните команду как администратор:"
        Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor $Colors.Yellow
        Write-Info "Или запустите этот скрипт с параметром -ExecutionPolicy Bypass"
        return
    }

    try {
        Test-SystemRequirements
        New-VirtualEnvironment
        Enable-VirtualEnvironment
        Install-Dependencies -IncludeDev $Dev
        New-ProjectDirectories
        Set-Configuration
        Test-Installation
        Show-Instructions
    }
    catch {
        Write-Error "Ошибка при установке: $_"
        exit 1
    }
}

# Запуск
Main
